# Taking the Human Out of the Loop: A Review of Bayesian Optimization

Bobak Shahriari, et. al.

## 0. Abstract

Big data applications are typically associated with systems involving large numbers of users, massive complex software systems, and large-scale heterogeneous computing and storage architectures. The construction of such systems involves many distributed design choices. The end products (e.g., recommendation systems, medical analysis tools, real-time game engines, speech recognizers) thus involves many tunable configuration parameters. These parameters are often specified and hard-coded into the software by various developers or teams. If optimized jointly, these parameters can result in significant improvements. Bayesian optimization is a powerful tool for the joint optimization of design choices that is gaining great popularity in recent years. It promises greater automation so as to increase both product quality and human productivity. This review paper introduces Bayesian optimization, highlights some of its methodological aspects, and showcases a wide range of applications.

一般来说，大数据应用有关的系统，通常都涉及到大量用户，复杂的软件系统，大规模的异质计算和存储架构。构建这种系统涉及到很多分布式的设计选择。终端产品（如，推荐系统，医学分析工具，实时游戏引擎，语音识别器）因此涉及到很多可调节的配置参数。这些参数通常由各种开发者或团队指定并编码到软件中。如果联合优化，这些参数可以得到显著的改进。贝叶斯优化是一个很强的工具，可以进行设计选择的联合优化，最近几年越来越流行。可以得到更大的自动化程度，增加产品的质量和人类生产力。本综述介绍了贝叶斯优化，强调了一些方法学的方面，展示了很多应用。

## 1. Introduction

Design problems are pervasive in scientific and industrial endeavours: scientists design experiments to gain insights into physical and social phenomena, engineers design machines to execute tasks more efficiently, pharmaceutical researchers design new drugs to fight disease, companies design websites to enhance user experience and increase advertising revenue, geologists design exploration strategies to harness natural resources, environmentalists design sensor networks to monitor ecological systems, and developers design software to drive computers and electronic devices. All these design problems are fraught with choices, choices that are often complex and high-dimensional, with interactions that make them difficult for individuals to reason about.

设计问题在科学和工业中广泛存在：科学家设计试验以得到对物理现象和社会现象的洞见，工程师设计机器以更高效的执行任务，药剂研究者设计新药以对抗疾病，公司设计网站以增强用户体验，增加广告收益，地理学家设计探索策略以利用自然资源，环境学家设计传感器网络，以监控生态系统，开发者设计软件，以驱动计算机和电子设备。所有这些设计问题都充满了选项，选项通常是复杂的，高维的，个人进行推理是非常困难的。

For example, many organizations routinely use the popular mixed integer programming solver IBM ILOG CPLEX for scheduling and planning. This solver has 76 free parameters, which the designers must tune manually – an overwhelming number to deal with by hand. This search space is too vast for anyone to effectively navigate.

比如，很多组织会例行使用流行的混合整数规划求解器IBM ILOG CPLEX，以进行调度和规划。这个求解器有76个自由参数，设计者必须手动调节，这是非常大的数量。这个搜索空间是非常大的，任何人都很难高效的探索。

More generally, consider teams in large companies that develop software libraries for other teams to use. These libraries have hundreds or thousands of free choices and parameters that interact in complex ways. In fact, the level of complexity is often so high that it becomes impossible to find domain experts capable of tuning these libraries to generate a new product.

更一般的，考虑大公司中开发软件库给其他团队用的团队。这些库有成百上千个自由选项和参数，以复杂的方式互动。实际上，复杂程度通常非常高，不可能找到领域专家能够调节这些库以生成一个新的产品。

As a second example, consider massive online games involving the following three parties: content providers, users, and the analytics company that sits between them. The analytics company must develop procedures to automatically design game variants across millions of users; the objective is to enhance user experience and maximize the content provider’s revenue.

第二个例子，考虑大型线上游戏，设计到下面三方：内容提供者，用户，和在两者之间的分析公司。分析公司需要开发出过程来在数百万用户中自动设计游戏变量；目标是增强用户体验，最大化内容提供者的利润。

The preceding examples highlight the importance of automating design choices. For a nurse scheduling application, we would like to have a tool that automatically chooses the 76 CPLEX parameters so as to improve healthcare delivery. When launching a mobile game, we would like to use the data gathered from millions of users in real-time to automatically adjust and improve the game. When a data scientist uses a machine learning library to forecast energy demand, we would like to automate the process of choosing the best forecasting technique and its associated parameters.

前面的例子强调的是自动设计选择的重要性。对于一个护士调度应用，我们希望有一个工具，自动的选择76个CPLEX参数，以改进健康服务。当发布一个移动游戏时，我们希望使用从数百万用户那里收集到的数据，实时的自动的调整和改进游戏。当数据科学家使用机器学习库预测能量需求，我们希望自动化选择最佳预测技术和其相关参数的过程。

Any significant advances in automated design can result in immediate product improvements and innovation in a wide area of domains, including advertising, health-care informatics, banking, information mining, life sciences, control engineering, computing systems, manufacturing, e-commerce, and entertainment.

自动设计中任何显著的改进，都会得到很多领域中立刻的产品改进和创新，包括广告，健康信息服务，银行业，信息挖掘，生命科学，控制工程，计算系统，制造业，电子商务，和娱乐。

Bayesian optimization has emerged as a powerful solution for these varied design problems. In academia, it is impacting a wide range of areas, including interactive user-interfaces [26], robotics [101], [110], environmental monitoring [106], information extraction [158], combinatorial optimisation [79], [159], automatic machine learning [16], [143], [148], [151], [72], sensor networks [55], [146], adaptive Monte Carlo [105], experimental design [11] and reinforcement learning [27].

贝叶斯优化是这些多样的设计问题的有力解决工具。在学术中，它正在影响很多领域，包括交互式用户界面，机器人，环境监控，信息提取，组合优化，自动机器学习，传感器网络，自适应蒙特卡洛，试验设计和强化学习。

When software engineers develop programs, they are often faced with myriad choices. By making these choices explicit, Bayesian optimization can be used to construct optimal programs [74]: that is to say, programs that run faster or compute better solutions. Furthermore, since different components of software are typically integrated to build larger systems, this framework offers the opportunity to automate integrated products consisting of many parametrized software modules.

当软件工程师开发了程序，他们通常面临着很多选择。将这些选项显式化，就可以利用贝叶斯优化来构建最优程序：即，可以运行的更快或计算得到更好的解的程序。而且，由于软件的不同组成部分一般集成以构建更大的系统，这个框架就有可能自动化集成产品过程，包含很多参数化的软件模块。

Mathematically, we are considering the problem of finding a global maximizer (or minimizer) of an unknown objective function f:

数学书来说，我们考虑找到未知目标函数f的一个全局最大化器（或最小化器）的问题：

$$x* = argmax_{x∈X} f(x)$$(1)

where X is some design space of interest; in global optimization, X is often a compact subset of R^d but the Bayesian optimization framework can be applied to more unusual search spaces that involve categorical or conditional inputs, or even combinatorial search spaces with multiple categorical inputs. Furthermore, we will assume the black-box function f has no simple closed form, but can be evaluated at any arbitrary query point x in the domain. This evaluation produces noise-corrupted (stochastic) outputs y ∈ R such that E[y | f(x)] = f(x). In other words, we can only observe the function f through unbiased noisy point-wise observations y. Although this is the minimum requirement for Bayesian optimization, when gradients are available, they can be incorporated in the algorithm as well; see for example Sections 4.2.1 and 5.2.4 of [99]. In this setting, we consider a sequential search algorithm which, at iteration n, selects a location x_n+1 at which to query f and observe y_n+1. After N queries, the algorithm makes a final recommendation $\bar x_N$, which represents the algorithm’s best estimate of the optimizer.

其中X是某感兴趣的设计空间；在全局优化中，X通常是R^d的一个紧凑子集，但贝叶斯优化框架也可以应用于更不同寻常的搜索空间，比如涉及到类别输入或有条件的输入，或有多个类别输入的组合搜索空间。而且，我们会假设黑箱函数f没有简单的闭合形式，但可以由领域中的任意查询点x评估得到。这个评估会产生含有噪声的（随机的）输出y ∈ R，这样E[y|f(x)] = f(x)。换句话说，我们只可以通过无偏的含噪逐点的观察y，来观察到f。虽然这是贝叶斯优化的最小要求，但是当有梯度可用时，是可以整合到算法中的；比如可以参考[99]中的4.2.1节和5.2.4节。在这个设置下，我们考虑一个顺序搜索算法，在第n次迭代时，选择一个位置x_n+1，在此查询f值，观察y_n+1。在N次查询后，算法进行最终的推荐$\bar x_N$，表示算法优化器的最佳估计。

In the context of big data applications for instance, the function f can be an object recognition system (e.g., deep neural network) with tunable parameters x (e.g., architectural choices, learning rates, etc) with a stochastic observable classification accuracy y = f(x) on a particular dataset such as ImageNet. Because the Bayesian optimization framework is very data efficient, it is particularly useful in situations like these where evaluations of f are costly, where one does not have access to derivatives with respect to x, and where f is non-convex and multimodal. In these situations, Bayesian optimization is able to take advantage of the full information provided by the history of the optimization to make this search efficient.

比如，在大数据应用的上下文中，函数f可以是一个目标识别系统（如，深度神经网络），有可调节的参数x（如，架构选择，学习速率，等），有随机的可观察的在特定数据集上的分类准确率y = f(x)，如ImageNet。因为贝叶斯优化框架需要的数据并不多，所以在一些情况中就特别有用，比如函数f的评估非常昂贵，比如对x的导数是不可用的，或f是非凸的、多峰的。在这些情况中，贝叶斯优化会完全利用优化过程的历史提供的信息，以使搜索更加高效。

Fundamentally, Bayesian optimization is a sequential model-based approach to solving problem (1). In particular, we prescribe a prior belief over the possible objective functions and then sequentially refine this model as data are observed via Bayesian posterior updating. The Bayesian posterior represents our updated beliefs – given data – on the likely objective function we are optimizing. Equipped with this probabilistic model, we can sequentially induce acquisition functions α_n : X → R that leverage the uncertainty in the posterior to guide exploration. Intuitively, the acquisition function evaluates the utility of candidate points for the next evaluation of f; therefore x_n+1 is selected by maximizing α_n, where the index n indicates the implicit dependence on the currently available data. Here the “data” refers to previous locations where f has been evaluated, and the corresponding noisy outputs.

基础上，贝叶斯优化是一个顺序的基于模型的求解问题(1)的方法。特别是，我们对可能的目标函数指定一个先验的置信度，然后随着数据通过贝叶斯后验更新时，顺序的精炼这个模型。贝叶斯后验表示我们更新的对很可能的我们正在优化的目标函数的置信度。有了这个概率模型后，我们可以顺序的引入采集函数α_n : X → R，利用了后验中的不确定性，以引导探索。直觉上，采集函数评估的是候选点对下一次f的评估的效用；因此，通过最大化α_n来选择x_n+1，其中索引n表示对可用数据的隐式依赖性。这里数据是指之前对f进行评估的位置，以及对应的含噪的输出。

In summary, the Bayesian optimization framework has two key ingredients. The first ingredient is a probabilistic surrogate model, which consists of a prior distribution that captures our beliefs about the behavior of the unknown objective function and an observation model that describes the data generation mechanism. The second ingredient is a loss function that describes how optimal a sequence of queries are; in practice, these loss functions often take the form of regret, either simple or cumulative. Ideally, the expected loss is then minimized to select an optimal sequence of queries. After observing the output of each query of the objective, the prior is updated to produce a more informative posterior distribution over the space of objective functions; see Figure 1 and Algorithm 1 for an illustration and pseudo-code of this framework. See Section 4 of [64] for another introduction.

总结起来，贝叶斯优化框架有两个关键组成部分。第一个组成部分是一个概率代理模型，包含先验分布，代表我们关于未知目标函数的行为的置信度，和一个观察模型，描述了数据生成的机制。第二个组成部分是一个损失函数，描述了一系列查询的最优程度有多少；在实践中，这些损失函数通常是regret的形式，要么简单，或是累积的。理想的，期望的损失进行最小化，以选择一个最优的查询序列。在观察目标的每个查询的输出后，更新先验以在目标函数空间中产生一个更有信息的后验分布；见图1和算法1，是这个框架的描述和伪代码。见[64]的第4部分是另一个介绍。

One problem with this minimum expected risk framework is that the true sequential risk, up to the full evaluation budget, is typically computationally intractable. This has led to the introduction of many myopic heuristics known as acquisition functions, e.g., Thompson sampling, probability of improvement, expected improvement, upper-confidence-bounds, and entropy search. These acquisition functions trade off exploration and exploitation; their optima are located where the uncertainty in the surrogate model is large (exploration) and/or where the model prediction is high (exploitation). Bayesian optimization algorithms then select the next query point by maximizing such acquisition functions. Naturally, these acquisition functions are often even more multimodal and difficult to optimize, in terms of querying efficiency, than the original black-box function f. Therefore it is critical that the acquisition functions be cheap to evaluate or approximate: cheap in relation to the expense of evaluating the black-box f. Since acquisition functions have analytical forms that are easy to evaluate or at least approximate, it is usually much easier to optimize them than the original objective function.

这种最小期望风险框架的一个问题是，在完全的评估运算下，真正的顺序风险是典型的计算上不可行的。这带来了很多缺少远见的启发式，称为采集函数，如，Thompson采样，改进概率，期望改进，置信度上限，和熵搜索。这些采集函数在探索和利用之间折中；其最优值定位在，代理模型中的不确定性很大（探索），和/或模型预测是很高（利用）。贝叶斯优化算法选择下一个查询点，是最大化这样的采集函数。自然的，这些采集函数通常是更加多峰的，难以优化，在查询效率上比原始的黑盒函数f更甚之。因此采集函数评估或近似起来要很容易，这就很关键：与评估黑箱函数f相比，要更加容易评估。由于采集函数有解析形式，容易评估或近似，通常会比原始的目标函数更加容易优化。

### 1.1. Paper overview

In this paper, we introduce the ingredients of Bayesian optimization in depth. Our presentation is unique in that we aim to disentangle the multiple components that determine the success of Bayesian optimization implementations. In particular, we focus on statistical modelling as this leads to general algorithms to solve a broad range tasks. We also provide an extensive comparison among popular acquisition functions. We will see that the careful choice of statistical model is often far more important than the choice of acquisition function heuristic.

本文中，我们深度介绍了贝叶斯优化的组成部分。我们的目标是拆分多个组成部分，这些决定了贝叶斯优化实现的成功。特别是，我们聚焦在统计建模中，这会得到求解很多任务的一般性算法。我们还给出了流行的采集函数的广泛比较。我们还会看到，仔细选择统计模型通常比采集函数启发式的选择更加重要。

We begin in Sections II and III, with an introduction to parametric and non-parametric models, respectively, for binary- and real-valued objective functions. In Section IV, we will introduce many acquisition functions, compare them, and even combine them into portfolios. Several practical and implementation details, including available software packages, are discussed in Section V. A survey of theoretical results and a brief history of model-based optimization are provided in Sections VI and VII, respectively. Finally, we introduce more recent developments in Section VIII.

我们从第2部分和第3部分开始，对二值和实值目标函数，分别介绍参数化和非参数化模型。在第4部分，我们介绍很多采集函数，进行比较，甚至将其合并到portfolios中。第5部分讨论了几个实际的和实现的细节，包括可用的软件包。第6部分和第7部分，分别总结了理论结果，和基于模型的优化的历史。最后，我们在第8部分介绍了更多的最近发展。

### 1.2. Applications of Bayesian optimization

Before embarking on a detailed introduction to Bayesian optimization, the following sections provide an overview of the many and varied successful applications of Bayesian optimization that should be of interest to data scientists.

在开始仔细介绍贝叶斯优化之前，我们先给出贝叶斯优化成功应用的很多领域，数据科学家应当对此感兴趣。

1) A/B testing: Though the idea of A/B testing dates back to the early days of advertising in the form of so-called focus groups, the advent of the internet and smartphones has given web and app developers a new forum for implementing these tests at unprecedented scales. By redirecting small fractions of user traffic to experimental designs of an ad, app, game, or website, the developers can utilize noisy feedback to optimize any observable metric with respect to the product’s configuration. In fact, depending on the particular phase of a product’s life, new subscriptions may be more valuable than revenue or user retention, or vice versa; the click-through rate might be the relevant objective to optimize for an ad, whereas for a game it may be some measure of user engagement.

A/B测试：其思想可以追溯到广告的早期，形式为所谓的聚焦组，但互联网和智能手机的出现，给了网络和app开发者新的论坛，可以在很大的规模上实现这些测试。将用户流量的一小部分重新导向到一个广告，app，游戏或网站的试验设计上，开发者可以利用含噪的反馈来对产品的配置优化任意可观察的度量。实际上，依赖于一个产品生命的特定阶段，新的订阅可能比利润或用户保留更加珍贵，或是反之；点击率可能是一个广告的相关优化目标，而对于游戏，可能是用户粘性的某种度量。

The crucial problem is how to optimally query these subsets of users in order to find the best product with high probability within a predetermined query budget, or how to redirect traffic sequentially in order to optimize a cumulative metric while incurring the least opportunity cost [88], [135], [38].

关键问题是，怎样最优的查询这些用户子集，以很高的概率找到最佳的产品，在预先确定的查询预算内，或怎样顺序的重新导向流量，以最优化累积度量，同时带来最少的机会价值。

2) Recommender systems: In a similar setting, online content providers make product recommendations to their subscribers in order to optimize either revenue in the case of e-commerce sites, readership for news sites, or consumption for video and music streaming websites. In contrast to A/B testing, the content provider can make multiple suggestions to any given subscriber. The techniques reviewed in this work have been successfully used for the recommendation of news articles [97], [38], [153].

推荐系统：在类似的设置中，在电子商务网站中，在新闻网站的读者中，或消费视频和音乐流的网站中，在线内容提供者对其订阅者进行产品推荐，以最优化其利润。与A/B测试形成对比的是，内容提供者可以对任意给定的订阅者进行多次建议。在本文中回顾的技术，已经成功的应用于新闻文章的推荐中。

3) Robotics and Reinforcement learning: Bayesian optimization has also been successfully applied to policy search. For example, by parameterizing a robot’s gait it is possible to optimize it for velocity or smoothness as was done on the Sony AIBO ERS-7 in [101]. Similar policy parameterization and search techniques have been used to navigate a robot through landmarks, minimizing uncertainty about its own location and map estimate [110], [108]. See [27] for an example of applying Bayesian optimization to hierarchical reinforcement learning, where the technique is used to automatically tune the parameters of a neural network policy and to learn value functions at higher levels of the hierarchy. Bayesian optimization has also been applied to learn attention policies in image tracking with deep networks [44].

机器人和强化学习：贝叶斯优化也成功的应用于策略搜索。比如，通过将机器人的步态参数化，可能优化其速度或平滑度，Sony AIBO ERS-7在[101]中就是这么做的。类似的策略参数化和搜索技术用于让一个机器人通过地标进行导航，最小化关于其自身位置和地图估计的不确定性。[27]中有一个例子，将贝叶斯优化应用到层次化强化学习中，其中该技术用于自动调节神经网络策略的参数，在更高层次的层级中学习值函数。贝叶斯优化还应用于在深度学习的图像追踪中，学习注意力策略。

4) Environmental monitoring and sensor networks: Sensor networks are used to monitor environmentally relevant quantities: temperature, concentration of pollutants in the atmosphere, soil, oceans, etc. Whether inside a building or at a planetary scale, these networks make noisy local measurements that are interpolated to produce a global model of the quantity of interest. In some cases, these sensors are expensive to activate but one can answer important questions like what is the hottest or coldest spot in a building by activating a relatively small number of sensors. Bayesian optimization was used for this task and the similar one of finding the location of greatest highway traffic congestion [146]. Also, see [55] for a meteorological application.

环境检测和传感器网络：传感器网络用于监控环境相关的量：温度，大气中污染物的聚集，土壤，海洋，等。不管是在一个建筑的尺度上，或是在行星的尺度上，这些网络进行的局部测量是含噪的，要进行插值，以产生感兴趣的量的全局模型。在一些情况中，这些传感器激发起来很昂贵，但只需要激活相对较少的传感器，就可以回答一些问题，如哪里是最热，或最冷的点。贝叶斯优化可以用在这个任务以及类似的任务上，如找到高速上最大流量拥塞的地点。同时，见[55]中气象学上的应用。

When the sensor is mobile, there is a cost associated with making a measurement which relates to the distance travelled by a vehicle on which the sensor is mounted (e.g., a drone). This cost can be incorporated in the decision making process as in [106].

当传感器是移动的，在测量时就会遇到这样的代价，即安装传感器的交通工具平台移动的距离（如，无人机）。这个代价可能会纳入到决策制定的过程中，如[106]。

5) Preference learning and interactive interfaces: The computer graphics and animation fields are filled with applications that require the setting of tricky parameters. In many cases, the models are complex and the parameters unintuitive for non-experts. In [28], [26], the authors use Bayesian optimization to set the parameters of several animation systems by showing the user examples of different parametrized animations and asking for feedback. This interactive Bayesian optimization strategy is particulary effective as humans can be very good at comparing examples, but unable to produce an objective function whose optimum is the example of interest.

倾向性学习和交互式界面：计算机图形学和动画领域的应用，需要设置很多参数。在很多情况中，模型非常复杂，参数的设置对于非专家来说，很不直观。在[28,26]中，作者使用贝叶斯优化来设置几个动画系统的参数，展示给用户不同的参数化动画的例子，并要求反馈。这种交互式贝叶斯优化的策略非常有效，因为人类非常擅于比较例子，但是不能得出一个目标函数，其最优值是感兴趣的例子。

6) Automatic machine learning and hyperparameter tuning: In this application, the goal is to automatically select the best model (e.g., random forests, support vector machines, neural networks, etc.) and its associated hyperparameters for solving a task on a given dataset. For big datasets or when considering many alternatives, cross-validation is very expensive and hence it is important to find the best technique within a fixed budget of cross-validation tests. The objective function here is the generalization performance of the models and hyperparameter settings; a noisy evaluation of the objective corresponds to training a single model on all but one cross-validation folds and returning, e.g., the empirical error on the held out fold. 

自动机器学习和超参数调节：在这个应用中，目标是自动选择最佳模型（如，随机森林，支持矢量机，神经网络，等），及其相关的超参数，在给定的数据集上求解一个任务。对于大型数据集，或考虑很多替代品时，交叉验证是非常昂贵的，因此在交叉验证测试是固定的预算时，找到最好的技术就很重要的。这里的目标函数是模型和超参数的泛化性能；目标函数的含噪评估对应着在所有的（除了一个）交叉验证折数上训练单个模型并返回，如保留fold的经验误差。

The traditional alternatives to cross-validation include racing algorithms that use conservative concentration bounds to rule out underperforming models [107], [113]. Recently, the Bayesian optimization approach for the model selection and tuning task has received much attention in tuning deep belief networks [16], Markov chain Monte Carlo methods [105], [65], convolutional neural networks [143], [148], and automatically selecting among WEKA and scikit-learn offerings [151], [72].

对交叉验证的传统替代品包括竞争算法，使用保守的聚焦界限，将性能没那么好的模型剔除出去。最近，对模型选择和调节任务的贝叶斯优化方法受到了很多关注，如调节深度置信网络，MCMC方法，卷积神经网络，和在WEKA和scikit-learn中自动选择。

7) Combinatorial optimization: Bayesian optimization has been used to solve difficult combinatorial optimization problems in several applications. One notable approach is called empirical hardness models (EHMs) that use a set of problem features to predict the performance of an algorithm on a specific problem instance [96]. Bayesian optimization with an EHM amounts to finding the best algorithm and configuration for a given problem. This concept has been applied to e.g., tuning mixed integer solvers [78], [159], and tuning approximate nearest neighbour algorithms [109]. Bayesian optimization has also been applied to fast object localization in images [163].

组合优化：贝叶斯优化曾用于在几个应用中求解困难的组合优化问题。一个值得注意的方法是经验难度模型(EHMs)，使用问题特征集合来预测一个算法在特定问题实例上的性能。带有EHM的贝叶斯优化，可以给一个给定的问题找到最佳算法和配置。这个概念已经应用到了，如，调节混合整数求解器，调节近似最近邻算法。贝叶斯优化还用于图像中的快速目标定位。

8) Natural language processing and text: Bayesian optimization has been applied to improve text extraction in [158] and to tune text representations for more general text and language tasks in [162].

自然语言处理和文本：贝叶斯优化还应用于改进文本提取，和在更一般性的文本和语言任务中，调节文本表示。

## 2. Bayesian Optimization with Parametric Models

The central idea of Bayesian optimization is to build a model that can be updated and queried to drive optimization decisions. In this section, we cover several such models, but for the sake of clarity, we first consider a generic family of models parameterized by w. Let D denote the available data. We will generalize to the non-parametric situation in the proceeding section.

贝叶斯优化的中心思想是，构建一个可以更新和查询的模型，以驱动优化决策。在本节中，我们考虑几个这种模型，但为了明确，我们首先考虑一种通用模型族，参数为w。令D表示可用的数据。我们会在下一章中泛化到无参数的情况。

Since w is an unobserved quantity, we treat it as a latent random variable with a prior distribution p(w), which captures our a priori beliefs about probable values for w before any data is observed. Given data D and a likelihood model p(D|w), we can then infer a posterior distribution p(w|D) using Bayes’ rule:

由w是未观测的量，我们认为这是一个隐随机变量，先验分布为p(w)，是在任何数据被观测之前，我们对w的可能值的先验信念。给定数据D，和似然模型p(D|w)，我们可以使用贝叶斯准则推理一个后验分布：

$$p(w|D) = \frac {p(D|w)p(w)} {p(D)}$$(2)

This posterior represents our updated beliefs about w after observing data D. The denominator p(D) is the marginal likelihood, or evidence, and is usually computationally intractable. Fortunately, it does not depend on w and is therefore simply a normalizing constant. A typical modelling choice is to use conjugacy to match the prior and likelihood so that the posterior (and often the normalizing constant) can be computed analytically.

后验表示我们在观测数据D后，关于w更新的信念。分母p(D)是边缘似然，通常是无法通过计算进行处理的。幸运的是，它不依赖于w，因此只是一个归一化常数。典型的建模选择是，使用对偶性来匹配先验和似然，这样后验（通常是归一化常数）可以解析的进行计算。

### 2.1. Thompson sampling in the Beta-Bernoulli bandit model

We begin our discussion with a treatment of perhaps the simplest statistical model, the Beta-Bernoulli. Imagine that there are K drugs that have unknown effectiveness, where we define “effectiveness” as the probability of a successful cure. We wish to cure patients, but we must also identify which drugs are effective. Such a problem is often called a Bernoulli (or binomial) bandit problem by analogy to a group of slot machines, which each yield a prize with some unknown probability. In addition to clinical drug settings, this formalism is useful for A/B testing [135], advertising, and recommender systems [97], [38], among a wide variety of applications. The objective is to identify which arm of the bandit to pull, e.g., which drug to administer, which movie to recommend, or which advertisement to display. Initially, we consider the simple case where the arms are independent insofar as observing the success or failure of one provides no information about another.

我们以最简单的统计模型开始我们的讨论，即Beta-Bernoulli模型。假设有K个药物，效果未知，这里的效果是指成功治愈的概率。我们希望治愈病患，但我们必须识别出哪些药是有效的。这样一个问题我们通常称之为Bernoulli老虎机问题，与一组老虎机可以相类比，每个都以未知的概率产生一些奖品。除了临床药物的情况，这种表述对于A/B测试，广告，推荐系统也是有用的，有很广的应用。目标是识别出拉哪个老虎机的臂，如，给哪个药，推荐哪部电影，展示哪个广告。最初，我们考虑简单情况，这些臂是独立的，一个的成功与失败，与另一个毫无关系。

Returning to the drug application, we can imagine the effectiveness of different drugs (arms on the bandit) as being determined by a function f that takes an index a ∈ 1, . . . , K and returns a Bernoulli parameter in the interval (0, 1). With y_i ∈ {0, 1}, we denote the Bernoulli outcome of the treatment of patient i, and this has mean parameter f(a_i) if the drug administered was a_i. Note that we are assuming stochastic feedback, in contrast to deterministic or adversarial feedback [9], [10]. With only K arms, we can fully describe the function f with a parameter w ∈ (0, 1)^K so that f_w(a) := w_a.

回到药物的应用中，我们可以假设不同的药物的效果（老虎机上的臂）由于一个函数f确定，以索引a ∈ 1, . . . , K为输入，在区间(0,1)中返回一个Bernoulli参数。我们用y_i ∈ {0, 1}表示病人i治疗的Bernoulli输出，如果给的是药物a_i，则有均值参数f(a_i)。注意，我们假设是随机的反馈，这与确定性的或对抗性的反馈形成对比。有了K个臂，我们可以用参数w ∈ (0, 1)^K完全描述函数f，这样f_w(a) := w_a。

Over time, we will see outcomes from different patients and different drugs. We can denote these data as a set of tuples D_n = {(a_i, y_i)}^n_{i=1}, where a_i indicates which of the K drugs was administered and y_i is 1 if the patient was cured and 0 otherwise. In a Bayesian setting, we will use these data to compute a posterior distribution over w. A natural choice for the prior distribution is a product of K beta distributions:

随着治疗过程的进行，我们会看到不同的患者和不同药物的结果。我们表示这些数据为元组的集合D_n = {(a_i, y_i)}^n_{i=1}，这里a_i表示给的K个药的每一个，y_i是患者治疗的效果，如果治愈了就是1，如果没有治愈就是0。在贝叶斯的设置下，我们会用这些数据计算w的后验分布。先验分布的一个自然选择是，K个beta分布的乘积：

$$p(w|α, β) = \prod_{a=1}^K Beta(w_a | α, β)$$(3)

as this is the conjugate prior to the Bernoulli likelihood, and it leads to efficient posterior updating. We denote by n_{a,1} the number of patients cured by drug a and by n_{a,0} the number of patients who received a but were unfortunately not cured; that is

由于这是Bernoulli似然的共轭先验，会得到高效的后验更新。我们用n_{a,1}表示用药物a治愈的患者数量，n_{a,0}表示使用了药物a但是没有作用的患者的数量；即

$$n_{a,0} = \sum_{i=1}^n 1(y_i=0, a_i = a)$$(4)

$$n_{a,1} = \sum_{i=1}^n 1(y_i=1, a_i = a)$$(5)

The convenient conjugate prior then leads to a posterior distribution which is also a product of betas:

便捷的共轭先验然后可以得到后验分布，也是beta分布的乘积：

$$p(w|D) = \prod_{a=1}^K Beta(w_a|α + n_{a,1}, β + n_{a,0})$$(6)

Note that this makes it clear how the hyperparameters α, β > 0 in the prior can be interpreted as pseudo-counts. Figure 2 provides a visualization of the posterior of a three-armed Beta-Bernoulli bandit model with a Beta(2, 2) prior.

这就清楚了，先验中的超参数α, β > 0可以怎样解释为伪技术。图2展示了一个三臂Beta-Bernoulli老虎机模型，在先验为Beta(2,2)时的后验。

In Section IV, we will introduce various strategies for selecting the next arm to pull within models like the Beta-Bernoulli, but for the sake of illustration, we introduce Thompson sampling [150], the earliest and perhaps the simplest nontrivial bandit strategy. This strategy is also commonly known as randomized probability matching [135] because it selects the arm based on the posterior probability of optimality, here given by a beta distribution. In simple models like the Beta-Bernoulli, it is possible to compute this distribution in closed form, but more often it must be estimated via, e.g., Monte Carlo.

在第4部分，我们会介绍选择取拉的下一个臂的各种策略，模型就是类似Beta-Bernoulli的，但为了描述方便，我们介绍Thompson采样[150]，最早的，可能也是最简单的老虎机策略。这个策略也称为随机概率匹配[135]，因为会基于最优的后验概率来选择臂，这里是由beta分布来给出。在简单模型中，如Beta-Bernoulli，可能计算出分布的闭合形式，但更多的需要通过比如Monte Carlo算法来进行估计。

After observing n patients in our drug example, we can think of a bandit strategy as being a rule for choosing which drug to administer to patient n + 1, i.e., choosing a_{n+1} among the K options. In the case of Thompson sampling, this can be done by drawing a single sample $\tilde w$ from the posterior and then maximizing the resulting surrogate $f_{\tilde w}$, i.e.,

在我们的药物例子中，在观察了n个病人之后，我们可以想出一个老虎机策略，作为对患者n+1给什么药的规则，如，在K个选项中选择a_{n+1}。在Thompson采样的情况下，可以从后验中选择一个样本$\tilde w$，然后最大化结果代理$f_{\tilde w}$，即

$$a_{n+1} = argmax_a f_{\tilde w} (a), where \tilde w ~ p(w|D_n)$$(7)

For the Beta-Bernoulli, this corresponds to simply drawing $\tilde w$ from (6) and then choosing the action with the largest $\tilde w_a$. This procedure, shown in pseudo-code in Algorithm 2, is also commonly called posterior sampling [127]. It is popular for several reasons: 1) there are no free parameters other than the prior hyperparameters of the Bayesian model, 2) the strategy naturally trades off between exploration and exploitation based on its posterior beliefs on w; arms are explored only if they are likely (under the posterior) to be optimal, 3) the strategy is relatively easy to implement as long as Monte Carlo sampling mechanisms are available for the posterior model, and 4) the randomization in Thompson sampling makes it particularly appropriate for batch or delayed feedback settings where many selections a_{n+1} are based on the identical posterior [135], [38].

对于Beta-Bernoulli，这就是从(6)中抽出$\tilde w$，然后选择最大$\tilde w_a$的行为。这个过程如算法2中的伪代码，通常称为后验采样。这非常流行，有几个原因：1)除了贝叶斯模型的先验超参数以外，没有其他自由参数了；2)这个策略很自然的在探索和挖掘之间进行折中，基于w的后验置信；在这些臂上进行探索只有其很可能是最优（在后验中）的情况下进行；3)这个策略相对容易实现，只要Monte Carlo采样机制对于后验模型是可用的，4)Thompson采样中的随机化使其对于批次反馈或延迟反馈的设置非常合适，这里很多选择a_{n+1}是基于相同的后验的。

### 2.2. Linear models

In many applications, the designs available to the experimenter have components that can be varied independently. For example, in designing an advertisement, one has choices such as artwork, font style, and size; if there are five choices for each, the total number of possible configurations is 125. In general, this number grows combinatorially in the number of components. This presents challenges for approaches such as the independent Beta-Bernoulli model discussed in the previous section: modelling the arms as independent will lead to strategies that must try every option at least once. This rapidly becomes infeasible in the large spaces of real-world problems. In this section, we discuss a parametric approach that captures dependence between the arms via a linear model. For simplicity, we first consider the case of real-valued outputs y and generalize this model to binary outputs in the succeeding section.

在很多应用中，实验者可用的设计，有很多组成部分，可以独立的进行变化。比如，在设计一个广告时，有一些选项，比如艺术作品，字体，和大小；如果每个都有5个选项，可能的配置的总量就有125个。总体上，这个数量随着组成部分的数量呈指数级增长。这对一些方法给出了挑战，比如前一节提出的独立Beta-Bernoulli模型：将这些臂建模为独立的，会得到每个选项都至少尝试一次的策略。这在真实世界的问题中，有很大的搜索空间，会迅速的变得不可行。本节中，我们讨论了一个参数化的方法，通过一个线性模型来表示臂之间的依赖关系。简化来说，我们首先考虑实值输出y的情况，然后将这个模型在下一节泛化到二值输出的情况。

As before, we begin by specifying a likelihood and a prior. In the linear model, it is natural to assume that each possible arm a has an associated feature vector x_a ∈ R^d. We can then express the expected payout (reward) of each arm as a function of this vector, i.e., f(a) = f(x_a). Our objective is to learn this function f : R^d → R for the purpose of choosing the best arm, and in the linear model we require f to be of the form f_w(a) = x_a^T w, where the parameters w are now feature weights. This forms the basis of our likelihood model, in which the observations for arm a are drawn from a Gaussian distribution with mean x_a^T w and variance σ^2.

像之前一样，我们以指定一个似然和先验开始。在线性模型中，可以很自然的假设，每个可能的臂a都有一个相关的特征向量x_a ∈ R^d。我们可以将每个臂的期望的回报表示为这个向量的一个函数，即f(a) = f(x_a)。我们的目标是，学习这个函数f : R^d → R，以选择最好的臂，在线性模型中，我们要求f具有下面的形式f_w(a) = x_a^T w，其中参数w现在是特征权重。这形成了我们的似然模型的基础，对于臂a的观察是从高斯分布中抽取得到的，均值x_a^T w，方差σ^2。

We use X to denote the n × d design matrix in which row i is the feature vector associated with the arm pulled in the i-th iteration, x_{a_i}. We denote by y the n-vector of observations. In this case, there is also a natural conjugate prior for w and σ^2: the normal-inverse-gamma, with density given by

我们使用X来表示n × d设计矩阵，行i是在第i次迭代中拉下的臂相关的特征向量x_{a_i}。我们用y来表示观察的n维向量。在这个情况中，对w和σ^2也有一个自然的共轭先验，normal-inverse-gamma，其密度由下式给出

$$NIG(w, σ^2 | w_0, V_0, α_0, β_0) = |2πσ^2V_0|^{-1/2} exp\{-\frac{1}{2σ^2} (w − w_0)^T V^{−1}_0 (w − w_0)\} × \frac {β_0^{α_0}} {Γ(α_0)(σ^2)^{α_0+1}} exp\{ -β_0/σ^2 \}$$(8)

There are four prior hyperparameters in this case, w0, V0, α0, and β0. As in the Beta-Bernoulli case, this conjugate prior enables the posterior distribution to be computed easily, leading to another normal-inverse-gamma distribution, now with parameters

在这个情况中有四个先验超参数。在Beta-Bernoulli的情况中，这个共轭先验使后验分布可以很容易的计算，得到另一个normal-inverse-gamma分布，现在的参数是

$$w_n = V_n(V_0^{−1} w_0 + X^T y)$$(9)
$$V_n = (V_0^{−1} + X^T X)^{−1}$$(10)
$$α_n = α_0 + n/2$$(11)
$$β_n = β_0 + (w_0^T V_0^{−1} w_0 + y^T y − w_n^T V_n^{−1} w_n)/2$$(12)

Integrating out the weight parameter w leads to coupling between the arms and makes it possible for the model to generalize observations of reward from one arm to another.

集成权重参数w，会将臂与臂之间耦合起来，使模型可能将回报的观察从一个臂泛化到另一个。

In this linear model, Thompson sampling draws a $\tilde w$ from the posterior p(w|D_n) and selects the arm with the highest expected reward under that parameter, i.e.,

在这个线性模型中，Thompson采样从后验p(w|D_n)中抽取一个$\tilde w$，选择在这个参数下期望回报最大的臂，即

$$a_{n+1} = argmax_a x_a^T \tilde w, where \tilde w ~ p(w|D_n)$$(13)

After arm a_{n+1} is pulled and y_{n+1} is observed, the posterior model can be readily updated using equations (9–12).

在拉下了臂a_{n+1}，观察到y_{n+1}之后，后验模型就可以使用式(9-12)进行更新了。

Various generalizations can be immediately seen. For example, by embedding the arms of a multi-armed bandit into a feature space denoted X , we can generalize to objective functions f defined on the entire domain X, thus unifying the multi-armed bandit problem with that of general global optimization:

可以立刻看到各种泛化。比如，通过将多臂老虎机的臂嵌入到特征空间X中，我们泛化到在整个领域X中定义的目标函数中，因此将多臂老虎机问题与通用全局优化问题统一起来：

$$maximize f(x) s.t. x ∈ X$$(14)

In the multi-armed bandit, the optimization is over a discrete and finite set {x_a}^K_{a=1} ⊂ X, while global optimization seeks to solve the problem on, e.g., a compact set X ⊂ R^d.

在多臂老虎机问题中，优化是在一个离散有限集{x_a}^K_{a=1} ⊂ X中，而全局优化则是在一个紧凑集X ⊂ R^d上求解这个问题。

As in other forms of regression, it is natural in increase the expressiveness of the model with non-linear basis functions. In particular, we can use J basis functions φ_j : X → R, for j = 1, . . . , J, and model the function f with a linear combination

至于其他形式的回归，用非线性基底函数来增加模型的表示能力，这是很自然的。特别的，我们可以使用J基底函数φ_j : X → R, for j = 1, . . . , J，用一个线性组合来建模这个函数f

$$f(x) = Φ(x)^T w$$(15)

where Φ(x) is the column vector of concatenated features {φ_j (x)}^J_{j=1}. Common classical examples of such φ_j include radial basis functions such as

其中Φ(x)是拼接特征{φ_j (x)}^J_{j=1}的列向量。这样的φ_j的常见的经典例子包括径向基函数，比如

$$φ_j (x) = exp \{ −(x − z_j)^T Λ(x − z_j)/2 \}$$(16)

where Λ and {z_j}^J_{j=1} are model hyperparameters, and Fourier bases 其中Λ和{z_j}^J_{j=1}是模型超参数，和Fourier基底

$$φ_j (x) = exp \{ −ix^T ω_j \}$$(17)

with hyperparameters {ω_j}^J_{j=1}. 超参数为{ω_j}^J_{j=1}。

Recently, such basis functions have also been learned from data by training deep belief networks [71], deep neural networks [93], [144], or by factoring the empirical covariance matrix of historical data [146], [72]. For example, in [34] each sigmoidal layer of an L layer neural network is defined as L_l(x) := σ(W_lx + B_l) where σ is some sigmoidal non-linearity, and W_l and B_l are the layer parameters. Then the feature map Φ : R^d → R^J can be expressed as Φ(x) = L_L ◦ · · · ◦ L_1(x), where the final layer L_L has J output units. In [144], the weights of the last layer of a deep neural network are integrated out to result in a tractable Bayesian model with flexible learned basis functions.

最近，这样的基底函数还通过训练深度置信网络、深度神经网络从数据中学习到，或分解历史数据的经验协方差矩阵。比如，在[34]中，L层神经网络的每个sigmoidal层都定义为，L_l(x) := σ(W_lx + B_l)，其中σ是某sigmoidal非线性函数，W_l和B_l是层的超参数。那么特征图Φ : R^d → R^J可以表示为as Φ(x) = L_L ◦ · · · ◦ L_1(x)，其中最终层L_L有J个输出单元。在[144]中，深度神经网络最后一层的权重积分掉，得到一个贝叶斯模型的结果中，有灵活的学习的基函数。

Regardless of the feature map Φ, when conditioned on these basis functions, the posterior over the weights w can be computed analytically using (9-12). Let Φ(X) denote the n × J matrix where [Φ(X)]_{i,j} = φ_j (x_i); then the posterior is as in Bayesian linear regression, substituting Φ(X) for the design matrix X.

不管是什么特征图Φ，当以这些基函数为条件时，对权重w的后验可以使用(9-12)进行解析计算。令Φ(X)表示n × J矩阵，其中[Φ(X)]_{i,j} = φ_j (x_i)；那么后验就像在贝叶斯线性回归中一样，将设计矩阵X替换为Φ(X)。

### 2.3. Generalized linear models

While simple linear models capture the dependence between bandit arms in a straightforward and expressive way, the model as described does not immediately apply to other types of observations, such as binary or count data. Generalized linear models (GLMs) [119] allow more flexibility in the response variable through the introduction of a link function. Here we examine the GLM for binary data such as might arise from drug trials or AB testing.

简单的线性模型，以一种直接和有表示力的方式，捕获了老虎机臂之间的依赖关系，但描述的模型不能立刻应用到其他类型的观察中，比如二值数据或计数数据中。泛化线性模型(GLMs)引入了连接函数，在响应变量中有更多的灵活性。这里，我们检查一下GLM在二值数据中的应用，这会在药物试验或AB测试中出现。

The generalized linear model introduces a link function g that maps from the observation space into the reals. Most often, we consider the mean function g^{−1}, which defines the expected value of the response as a function of the underlying linear model: E[y|x] = g^{−1}(x^T w) = f(x). In the case of binary data, a common choice is the logit link function, which leads to the familiar logistic regression model in which g^{−1}(z) = 1/(1 + exp(z)). In probit regression, the logistic mean function is replaced with the CDF of a standard normal. In either case, the observations yi are taken to be Bernoulli random variables with parameter g^{−1}(x^T_i w).

泛化线性模型引入了连接函数g，从观测空间映射到实际空间。多数时候，我们考虑均值函数g^{−1}，将响应的期望值定义为潜在的线性模型的函数：E[y|x] = g^{−1}(x^T w) = f(x)。在二值数据的情况下，常见的选择是logit连接函数，带来的就是熟悉的logistic回归模型，其中g^{−1}(z) = 1/(1 + exp(z))。在probit回归中，logistic均值函数替换为了标准正态的CDF。在任一种情况中，观测yi都被认为是Bernoulli随机变量，参数为g^{−1}(x^T_i w)。

Unfortunately, there is no conjugate prior for the parameters w when such a likelihood is used and so we must resort to approximate inference. Markov chain Monte Carlo (MCMC) methods [4] approximate the posterior with a sequence of samples that converge to the posterior; this is the approach taken in [135] on the probit model. In contrast, the Laplace approximation fits a Gaussian distribution to the posterior by matching the curvature of the posterior distribution at the mode. For example in [38], Bayesian logistic regression with a Laplace approximation was used to model click-throughs for the recommendation of news articles in a live experiment. In the generalized linear model, Thompson sampling draws a $\tilde w$ from the posterior p(w|D_n) using MCMC or a Laplace approximation, and then selects the arm with the highest expected reward given the sampled parameter $\tilde w$, i.e., a_{n+1} = argmax_a g^{−1}(x^T_a \tilde w).

不幸的是，当使用这样的似然时，对参数w没有对应的共轭先验，所以我们必须寻求近似的推理。MCMC方法用一系列收敛到后验的样本近似了后验；这是[135]对probit模型采用的方法。对比起来，Laplace近似对后验拟合一个高斯分布，匹配后验在这个模式下的曲率。比如在[38]中，带有Laplace近似的贝叶斯logistic回归，用于对新闻文章在现场试验的推荐的点击率进行建模。在泛化线性模型中，Thompson采样从后验p(w|D_n)中抽取一个$\tilde w$，使用MCMC或Laplace近似，然后在给定采样参数$\tilde w$的情况下，选择最高期望回报的臂，即a_{n+1} = argmax_a g^{−1}(x^T_a \tilde w)。

### 2.4. Related literature

There are various strategies beyond Thompson sampling for Bayesian optimization that will be discussed in succeeding sections of the paper. However, before we can reason about which selection strategy is optimal, we need to establish what the goal of the series of sequential experiments will be. Historically, these goals have been quantified using the principle of maximum expected utility. In this framework, a utility function U is prescribed over a set of experiments X := {xi}^n_{i=1}, their outcomes y := {yi}^n_{i=1}, and the model parameter w. The unknown model parameter and outcomes are marginalized out to produce the expected utility

除了Thompson采样，还有很多策略可以用于贝叶斯优化，本文后面的小节中会进行讨论。但是，在我们推理哪个选择策略是最优的之前，我们需要确定顺序试验系列的目标是什么。历史上来说，这些目标已经使用最大期望效用进行量化过。在这个框架中，效用函数U指定给试验X := {xi}^n_{i=1}的集合上，其输出y := {yi}^n_{i=1}，模型参数w。未知的模型参数和输出进行边缘化，以产生期望的效用

$$α(X) := E_w E_{y|X,w} [U(X, y, w)]$$((18))

which is then maximized to obtain the best set of experiments with respect to the given utility U and the current posterior. The expected utility α is related to acquisition functions in Bayesian optimization, reviewed in Section IV. Depending on the literature, researchers have focussed on different goals which we briefly discuss here.

然后最大化以得到对给定效用U和当前后验的最佳的试验集。期望效用α与贝叶斯优化中的采集函数相关，在第4节中会回顾。在不同的文献中，研究者聚焦在不同的目标中，我们这里进行简要讨论。

1) Active learning and experimental design: In this setting, we are usually concerned with learning about w, which can be framed in terms of improving an estimator of w given the data. One popular approach is to select points that are expected to minimize the differential entropy of the posterior distribution p(w | X, y), i.e., maximize:

主动学习和试验设计：在这个设置下，我们通常关心w的学习，即在给定数据的情况下改进w的估计器。一种流行的方法是，选择的点最好要最小化后验分布p(w | X, y)的微分熵，即，最大化

$$α(X) := E_w E_{y|X,w} [\int p(w'|X, y) log p(w'|X, y) dw']$$

In the Bayesian experimental design literature, this criterion is known as the D-optimality utility and was first introduced by Lindley [98]. Since this seminal work, many alternative utilities have been proposed in the experimental design literature. See [37] for a detailed survey.

在贝叶斯试验设计的文献中，这个准则称为D最优性效用，由Lindley[98]首先提出。在这个工作之后，在试验设计文献中提出了很多其他效用。见[37]的详细综述。

In the context of A/B testing, following this strategy would result in exploring all possible combinations of artwork, font, and sizes, no matter how bad initial outcomes were. This is due to the fact that the D-optimality utility assigns equal value to any information provided about any advertisement configuration, no matter how effective.

在A/B测试的上下文中，按照这个策略会得到对所有的艺术品，字体和大小的组合进行探索，不论最初的输出多差。这是因为，D最优性效用对任何广告配置的任何信息都指定相同的值，不管其效果如何。

In contrast to optimal experimental design, Bayesian optimization explores uncertain arms a ∈ {1, . . . , K}, or areas of the search space X , only until they can confidently be ruled out as being suboptimal. Additional impressions of suboptimal ads would be a waste of our evaluation budget. In Section IV, we will introduce another differential entropy based utility that is better suited for the task of optimization and that partially bridges the gap between optimization and improvement of estimator quality.

与最优试验设计相比，贝叶斯优化探索不确定的臂a ∈ {1, . . . , K}，或搜索空间X中的区域，直到很有信心这是次优的，从而剔除掉。次优的广告的额外印象是，浪费了我们的评估预算。在第4部分中，我们会引入另一种基于微分熵的效用，更适用于优化任务，弥补了优化和估计器质量改进的空白。

2) Multi-armed bandit: Until recently, the multi-armed bandit literature has focussed on maximizing the sum of rewards y_i, possibly discounted by a discount factor γ ∈ (0, 1]:

多臂老虎机：直到最近，多臂老虎机的文献关注的是最大化回报y_i的和，也可能有一个折扣因子γ ∈ (0, 1]：

$$α(X) := E_w E_{y|X,w} [\sum_{i=1}^n γ^{i−1} y_i]$$(19)

When γ < 1, a Bayes-optimal sequence X can be computed for the Bernoulli bandit via dynamic programming, due to Gittins [59]. However, this solution is intractable for general reward distributions, and so in practice sequential heuristics are used and analyzed in terms of a frequentist measure, namely cumulative regret [92], [135], [146], [38], [127].

当γ < 1时，可以对Bernoulli老虎机通过动态规划计算出一个贝叶斯最优序列X。但是，这个解对于通用回报分布是很难处理的，所以在实践中，使用顺序启发式，并用频率学派的度量进行分析，即累积regret。

Cumulative regret is a frequentist measure defined as 累积regret是一个频率学派的度量，定义为

$$R_n(w) = \sum_{i=1}^n f_w^* - f_w(x_{a_i})$$(20)

where f^*_w := max_a f_w(x_a) denotes the best possible expected reward. Whereas the D-optimality utility leads to too much exploration, the cumulative regret encourages exploitation by including intermediate selections a_i in the final loss function R_n. For certain tasks, this is an appropriate loss function: for example, when sequentially selecting ads, each impression incurs an opportunity cost. Meanwhile, for other tasks such as model selection, we typically have a predetermined evaluation budget for optimization and only the performance of the final recommended model should be assessed by the loss function.

其中f^*_w := max_a f_w(x_a)，表示最佳期望回报。鉴于D最优性效用带来了太多探索，累积regret鼓励挖掘，在最终的损失函数R_n中包括了中间的选择a_i。对于特定的任务，这是一个合适的损失函数：比如，当顺序选择广告，每个印象带来一个机会价值。同时，对于其他任务，比如模型选择，我们典型的对优化有预先确定的评估预算，只有最终推荐的模型的性能，才会被损失函数评估。

Recently, there has been growing interest in the best arm identification problem, which is more suitable for the model selection task [104], [30], [7], [51], [50], [72]. When using Bayesian surrogate models, this is equivalent to performing Bayesian optimization on a finite, discrete domain. In this so-called pure exploration settings, in addition to a selection strategy, a recommendation strategy ρ is specified to recommend an arm (or ad or drug) at the end of the experimentation based on observed data. The experiment is then judged via the simple regret, which depends on the recommendation \bar a = ρ(D):

最近，在识别最佳臂的问题上，有越来越多的工作，这更适用于模型选择任务。当使用贝叶斯代理模型时，这等价于在一个有限的离散域上进行贝叶斯优化。在这个所谓的纯探索设置中，除了选择策略，指定一个推荐策略ρ，在基于试验的观测数据的最后，来推荐一个臂（或广告，或药物）。这个试验然后通过简单的regret进行评估，这依赖于推荐\bar a = ρ(D)：

$$r_n(w) = f^*_w - f_w(x_{\bar a})$$(21)

## 3. Non-parametric Models

In this section, we show how it is possible to marginalize away the weights in Bayesian linear regression and apply the kernel trick to construct a Bayesian non-parametric regression model. As our starting point, we assume the observation
variance σ^2 is fixed and place a zero-mean Gaussian prior on the regression coefficients p(w | V_0) = N (0, V_0). In this case, we notice that it possible to analytically integrate out the weights, and in doing so we preserve Gaussianity:

本节中，我们展示了怎样将贝叶斯线性回归中的权重边缘化掉，采用核的技巧来构建一个贝叶斯非参数回归模型。作为我们的起始点，我们假设观察方差σ^2是固定的，对回归系数加入一个零均值的高斯先验p(w|V_0) = N (0, V_0)。在这个情况中，我们注意到是可能解析的积分掉权重的，这样做我们也保持了高斯的性质：

$$p(y|X, σ^2) = \int p(y|X, w, σ^2) p(w|0, V_0) dw = \int N(y|Xw, σ^2 I) N(w|0, V_0) dw = N (y|0, XV_0X^T + σ^2I)$$(22)

As noted earlier, it can be useful to introduce basis functions φ and in the context of Bayesian linear regression we in effect replace the design matrix X with a feature mapping matrix Φ = Φ(X). In Equation (22), this results in a slightly different Gaussian for weights in feature space:

之前提到过，引入基准函数φ是有用的，在贝叶斯线性回归的上下文中，我们实际上将设计矩阵X替换为特征映射矩阵Φ = Φ(X)。在式(22)中，对于在特征空间中的权重，这得到了略微不同的高斯函数：

$$p(y|X, σ^2) = N (y|0, ΦV_0Φ^T + σ^2I)$$(23)

Note that ΦV_0Φ^T ∈ R^{n×n} is a symmetric positive semi-definite matrix made up of pairwise inner products between each of the data in their basis function representations. The celebrated kernel trick emerges from the observation that these inner products can be equivalently computed by evaluating the corresponding kernel function k for all pairs to form the matrix K

注意ΦV_0Φ^T ∈ R^{n×n}是一个对称的半正定矩阵，由用基函数表示的数据的成对的内积组成的。著名的核技巧来自于，这些内积可以等价的由评估对应的所有对的核函数k而计算得到，以形成矩阵K

$$K_{i,j} = k(x_i, x_j) = Φ(x_i)V_0Φ(x_j)^T$$(24)
$$= <Φ(x_i), Φ(x_j)>_{V_0}$$(25)

The kernel trick allows us to specify an intuitive similarity between pairs of points, rather than a feature map Φ, which in practice can be hard to define. In other words, we can either think of predictions as depending directly on features Φ, as in the linear regression problem, or on kernels k, as in the lifted variant, depending on which paradigm is more interpretable or computationally tractable. Indeed, the former requires a J × J matrix inversion compared to the latter’s n × n.

核的技巧使我们可以在成对的点之间指定一个直觉上的相似性，而不是一个特征图Φ，这在实践中是很难定义的。换句话说，我们可以认为预测直接依赖于特征Φ，就像在线性回归问题中，或依赖于核k，就像在lifted变体中，依赖于哪个范式更可解释，或计算上更可行。确实，前者需要一个J×J的矩阵逆，后者需要的是n×n。

Note also that this approach not only allows us to compute the marginal likelihood of data that have already been seen, but it enables us to make predictions of outputs $y_*$ at new locations $X_*$. This can be done by observing that

注意，这个方法使我们不仅可以计算已经看到的数据的边缘似然，还使我们可以在新的位置$X_*$预测输出$y_*$。

$$p(y_*|X_*, X, y, σ^2) = \frac {p(y_*, y|X_*, X, σ^2)} {p(y|X, σ^2)}$$(26)

Both the numerator and the denominator are Gaussian with the form appearing in Equation (23), and so the predictions are jointly Gaussian and can be computed via some simple linear algebra. Critically, given a kernel k, it becomes unnecessary to explicitly define or compute the features Φ because both the predictions and the marginal likelihood only depend on K.

分子和分母都是高斯函数，形式如式(23)所示，所以这个预测是联合高斯的，可以通过一些简单的线性代数计算得到。给定一个核k，显式的定义或计算特征Φ变得不再需要，因为预测和边缘似然都依赖于K。

### 3.1. The Gaussian process

By kernelizing a marginalized version of Bayesian linear regression, what we have really done is construct an object called a Gaussian process. The Gaussian process GP(µ_0, k) is a non-parametric model that is fully characterized by its prior mean function µ_0 : X → R and its positive-definite kernel, or covariance function, k : X ×X → R [126]. Consider any finite collection of n points x_{1:n}, and define variables f_i := f(x_i) and y_{1:n} to represent the unknown function values and noisy observations, respectively. In Gaussian process regression, we assume that f := f_{1:n} are jointly Gaussian and the observations y := y_{1:n} are normally distributed given f, resulting in the following generative model:

将边缘版的贝叶斯线性回归核化，我们真正做的是，构建一个物体，称为高斯过程。高斯过程GP(µ_0, k)是一个无参数模型，可以完全由其先验均值函数µ_0 : X → R和其正定的核，或其协方差函数k : X ×X → R [126]，来刻画。考虑n个点的任意有限的集合，x_{1:n}，定义变量f_i := f(x_i)和y_{1:n}来分别表示未知的函数值和含噪的观测。在高斯过程回归中，我们假设f := f_{1:n}是联合高斯的，观测y := y_{1:n}在给定f的情况下是正态分布的，这可以得到下面的生成式模型：

$$f | X ∼ N (m, K)$$(27)
$$y |f, σ^2 ∼ N (f, σ^2 I)$$(28)

where the elements of the mean vector and covariance matrix are defined as m_i:= µ_0(x_i) and K_{i,j} := k(x_i, x_j), respectively. Equation (27) represents the prior distribution p(f) induced by the GP.

其中均值向量和协方差矩阵的元素分别定义为m_i:= µ_0(x_i)和K_{i,j} := k(x_i, x_j)。式(27)表示由GP引入的先验分布p(f)。

Let D_n = {(xi, yi)}^n_{i=1} denote the set of observations and x denote an arbitrary test point. As mentioned when kernelizing linear regression, the random variable f(x) conditioned on observations D_n is also normally distributed with the following posterior mean and variance functions

令D_n = {(xi, yi)}^n_{i=1}表示观测集合，x表示任意一个测试点。在线性回归核化的时候提到过，以观测D_n为条件的随机变量f(x)也是正态分布的，后验均值和方差函数如下

$$µ_n(x) = µ_0(x) + k(x)^T (K + σ^2I)^{−1} (y − m)$$(29)
$$σ^2_n(x) = k(x, x) − k(x)^T (K + σ^2I)^{−1} k(x)$$(30)

where k(x) is a vector of covariance terms between x and x_{1:n}. 这里k(x)是x和x_{1:n}之间的协方差向量。

The posterior mean and variance evaluated at any point x represent the model’s prediction and uncertainty, respectively, in the objective function at the point x. These posterior functions are used to select the next query point x_{n+1} as detailed in Section IV.

在任意点x处评估的后验均值和方差，分别表示模型在目标函数中在点x处的预测和不确定性。这些后验函数用于选择下一个查询点x_{n+1}，在第4部分详述。

### 3.2. Common kernels

In Gaussian process regression, the covariance function k dictates the structure of the response functions we can fit. For instance, if we expect our response function to be periodic, we can prescribe a periodic kernel. In this review, we focus on stationary kernels, which are shift invariant.

在高斯过程回归中，协方差函数k决定了我们可以拟合的响应函数的结构。比如，如果我们期待我们的响应函数是周期性的，我们可以规定一个周期性的核。在本回顾中，我们聚焦在平稳性的核中，这是平移不变的。

Matern kernels are a very flexible class of stationary kernels. These kernels are parameterized by a smoothness parameter ν > 0, so called because samples from a GP with such a kernel are differentiable [ν − 1] times [126]. The exponential kernel is a special case of the Matern kernel with ν = 1/2, and the squared exponential kernel is the limiting kernel when ν → ∞. The following are the most commonly used kernels, labelled by the smoothness parameter, omitting the factor of 1/2。

Matern核是一种非常灵活的平稳核。这些核有一个平滑度参数ν > 0，因为有这样一个核的GP过程的样本，是可以微分[ν − 1]次的。指数核是Matern核的一种特殊情况，其中ν = 1/2，平方指数核是当ν → ∞时的有限核。下面是最常用的核，由平滑参数标记，忽略了因子1/2。

$$k_{MATERN1}(x, x') = θ_0^2 exp(−r)$$(31)
$$k_{MATERN3}(x, x') = θ_0^2 exp(−\sqrt 3r)(1+\sqrt 3r)$$(32)
$$k_{MATERN5}(x, x') = θ_0^2 exp(−\sqrt 5r)(1+\sqrt 5r+5r^2/3)$$(33)
$$k_{SQ-EXP} (x, x') = θ_0^2 exp(−r^2/2)$$(34)

where r^2 = (x − x')^T Λ (x − x') and Λ is a diagonal matrix of d squared length scales θ^2_i. This family of covariance functions are therefore parameterized by an amplitude and d length scale hyperparameters, jointly denoted θ. Covariance functions with learnable length scale parameters are also known as automatic relevance determination (ARD) kernels. Figure 3 provides a visualization of the kernel profiles and samples from the corresponding priors and posteriors.

其中r^2 = (x − x')^T Λ (x − x')，Λ是一个对角矩阵，有d个平方长度尺度θ^2_i。这一族协方差函数因此其参数是一个幅度和d个长度尺度超参数，联合表示为θ。带有可学习长度尺度参数的协方差函数，也称为自动相关确定(ARD)核。图3给出了核profiles的可视化效果，和从对应的先验和后验中的采样。

### 3.3. Prior mean functions

While the kernel function controls the smoothness and amplitude of samples from the GP, the prior mean provides a possible offset. In practice, this function is set to a constant µ0(x) ≡ µ0 and inferred from data using techniques covered in Section V-A. Unless otherwise specified, in what follows we assume a constant prior mean function for convenience. However, the prior mean function is a principled way of incorporating expert knowledge of the objective function, if it is available, and the following analysis can be readily applied to non-constant functions µ0.

核函数控制着GP中样本的平滑性和幅度，先验均值给出了一个可能的偏移。在实践中，这个函数设置为一个常数µ0(x) ≡ µ0，从数据中使用5.1中的技术推理得到。除非另有指定，在接下来的篇幅中，我们假设先验均值函数为常数，这样很方便。但是，先验均值函数是结合目标函数的专家知识的一个主要方法，后续的分析可以随时应用到非常数函数µ0中。

### 3.4. Marginal likelihood

Another attractive property of the Gaussian process model is that it provides an analytical expression for the marginal likelihood of the data, where marginal refers to the fact that the unknown latent function f is marginalized out. The expression for the log marginal likelihood is simply given by:

高斯过程模型的另一个吸引人的性质是，提供了数据的边缘似然的解析表达式，其中边缘是指，未知的隐函数f可以被边缘化掉。log边缘似然的表达式由下式给出

$$log p(y|x_{1:n}, θ) = −(y − m_θ)^T (K^θ + σ^2I)^{−1} (y − m_θ)/2 - log |K^θ + σ^2I|/2 − nlog(2π)/2$$(35)

where in a slight abuse of notation we augment the vector θ := (θ_{0:d}, µ_0, σ^2); and the dependence on θ is made explicit by adding a superscript to the covariance matrix K^θ. The marginal likelihood is very useful in learning the hyperparameters, as we will see in Section V-A. The right hand side of (35) can be broken into three terms: the first term quantifies how well the model fits the data, which is simply a Mahalanobis distance between the model predictions and the data; the second term quantifies the model complexity – smoother covariance matrices will have smaller determinants and therefore lower complexity penalties; finally, the last term is simply a linear function of the number of data points n, indicating that the likelihood of data tends to decrease with larger datasets.

我们扩充了向量θ := (θ_{0:d}, µ_0, σ^2)，对协方差矩阵加上了上标K^θ，表示对θ有依赖。边缘似然在学习超参数方面是非常有用的，这在5.1节可以看到。(35)式的右边可以分为3项：第一项量化了模型对数据的拟合程度，就是模型预测与数据之间的Mahalanobis距离；第二项量化了模型的复杂度，更平滑的协方差矩阵有更小的行列式，因此更低的复杂度惩罚；最后，最后一项是数据点数量n的简单的线性函数，表明数据的似然对更大型的数据集会变小。

Conveniently, as long as the kernel is differentiable with respect to its hyperparameters θ, the marginal likelihood can be differentiated and can therefore be optimized in an off-the-shelf way to obtain a type II maximum likelihood (MLII) or empirical Bayes estimate of the kernel parameters. When data is scarce this can overfit the available data. In Section V-A we will review various practical strategies for learning hyperparameters which all use the marginal likelihood.

只要核对于其超参数θ是可微分的，边缘似然就是可微分的，因此可以很容易的进行优化，以得到II型最大似然，或核参数的经验贝叶斯估计。当数据缺少时，会对可用的数据过拟合。在5.1节中，我们会回顾学习超参数的各种实践中的策略，都使用了边缘似然。

### 3.5. Computational costs and other regression models

Although we have analytic expressions, exact inference in Gaussian process regression is O(n^3) where n is the number of observations. This cost is due to the inversion of the covariance matrix. In practice, the Cholesky decomposition can be computed once and saved so that subsequent predictions are O(n^2). However, this Cholesky decomposition must be recomputed every time the kernel hyperparameters are changed, which usually happens at every iteration (see Section V-A). For large datasets, or large function evaluation budgets in the Bayesian optimization setting, the cubic cost of exact inference is prohibitive and there have been many attempts at reducing this computational burden via approximation techniques. In this section we review two sparsification techniques for Gaussian processes and the alternative random forest regression.

虽然我们有解析表达式，但是高斯过程回归的严格推理的复杂度是O(n^3)，其中n是观测的数量。这个复杂度是由于协方差矩阵的逆的计算。在实践中，可以用Cholesky分解计算，这样后续的预测的复杂度是O(n^2)。但是，这个Cholesky分解在核超参数变化的时候，必须重新计算一次，也就是说在每次迭代中都要重新计算一次（见5.1节）。对于大型数据集，或贝叶斯优化设置中的大型函数评估预算，严格推理的三次方的代价是不可行的，有很多通过近似技术降低这个计算工作量的工作。本节中，我们回顾两种对高斯过程的稀疏化技术和随机森林回归。

1) Sparse pseudo-input Gaussian processes (SPGP).

2) Sparse spectrum Gaussian processes (SSGP).

3) Random forests.

## 4. Acquisition Functions

Thus far, we have described the statistical model used to represent our belief about the unknown function f at iteration n. We have not described the exact mechanism or policy for selecting the sequence of query points x_{1:n}. One could select these arbitrarily but this would be wasteful. Instead, there is a rich literature on selection strategies that utilize the posterior model to guide the sequential search, i.e., the selection of the next query point x_{n+1} given D_n.

迄今，我们描述了统计模型，用于表示未知函数f在迭代n时我们的信念。我们还没有描述选择查询点序列x_{1:n}的严格机制或策略。可以任意选择，但是这就很浪费。在选择策略上有很多文献，利用后验模型来引导顺序搜索，即，在给定D_n的情况下，选择下一个查询点x_{n+1}。

Consider the utility function U : R^d × R × Θ → R which maps an arbitrary query point x, its corresponding function value v = f(x), and a setting of the model hyperparameters θ to a measure of quality of the experiment, e.g., how much information this query will provide as in [98]. Given some data accumulated thus far, we can marginalize the unseen outcome y and the unknown model hyperparameters θ to obtain the expected utility of a query point x:

考虑效用函数U：R^d × R × Θ → R，将任意一个查询点x，其对应的函数值v=f(x)，和模型超参数θ的设置，映射到试验质量的度量，如，这个查询会给出多少信息，如[98]。给定了迄今累积的一些数据，我们可以边缘化未看到的输出y，和未知模型超参数θ，以得到一个查询点x的期望效用：

$$α(x; D_n) = E_θ E_{v|x,θ} [U(x, v, θ)]$$(41)

For simplicity, in this section we will mostly ignore the θ dependence and we will discuss its marginalization in Section V-A.

为简化，在本节中，我们大多数情况下会忽略θ，在5.1节中会讨论其边缘化。

Whereas in experimental design and decision theory, the function α is called the expected utility, in Bayesian optimization it is often called the acquisition or infill function. These acquisition functions are carefully designed to trade off exploration of the search space and exploitation of current promising areas. We first present traditional improvement-based and optimistic acquisition functions, followed by more recent information-based approaches.

在试验设计和决策理论中，函数α称之为期望效用，在贝叶斯优化中，通常称为采集函数或填充函数。这些采集函数是仔细设计的，以平衡搜索空间的探索，和当前有希望区域的挖掘。我们首先给出传统的基于改进的和乐观的采集函数，然后给出最近的基于信息的方法。

### 4.1. Improvement-based policies

Improvement-based acquisition functions favour points that are likely to improve upon an incumbent target τ. An early strategy in the literature, probability of improvement (PI) [91], measures the probability that a point x leads to an improvement upon τ. Since the posterior distribution of v = f(x) is Gaussian, we can analytically compute this probability as follows:

基于改进的采集函数，倾向于在现有的目标τ上很可能改进的点。文献中一个早期的策略，改进概率(PI)，度量的是一个点x在τ的基础上得到改进的概率。由于v = f(x)的后验分布是高斯的，我们可以解析的计算这个概率如下：

$$α_{PI}(x; D_n) := P[v > τ] = Φ (\frac {µ_n(x) − τ} {σ_n(x)})$$(42)

where Φ is the standard normal cumulative distribution function. Recall that α_{PI} is then maximized to select the next query point. For this criterion, the utility function is simply an indicator of improvement U(x, v, θ) = I[v > τ], where the utility function is expressed (and marginalized) with respect to the latent variable v. Therefore, all improvements are treated equal and PI simply accumulates the posterior probability mass above τ at x.

这里Φ是标准正态累积分布函数。回忆一下，α_{PI}进行最大化，以选择下一个查询点。由于这个规则，效用函数就是改进的标志U(x, v, θ) = I[v > τ]，其中效用函数相对于隐变量v进行表示。因此，所有的改进都是平等对待的，PI只是将在x点处高于τ的后验概率质量累积起来。

Although probability of improvement can perform very well when the target is known, in general the heuristic used for an unknown target τ causes PI to exploit quite aggressively [81].

虽然改进概率在目标是已知的时候可以表现很好，总体上说，用于未知目标τ的启发式导致PI会挖掘的相当激进。

One could instead measure the expected improvement (EI) [115] which incorporates the amount of improvement. This new criterion corresponds to a different utility that is called the improvement function, denoted by I(x). Formally, the improvement function I is defined as follows

这时可以度量期望的改进(EI)[115]，纳入了改进的量。这个新的准则对应一个不同的效用函数，称为改进函数，表示为I(x)。形式上，改进函数I定义如下：

$$I(x, v, θ) := (v − τ) I(v > τ)$$(43)

Note that I > 0 only if there is an improvement. Once again, because the random variable v is normally distributed, the expectation can be computed analytically as follows

注意只有在有改进的时候，I > 0。再一次，因为随机变量v是正态分布的，期望可以进行解析的计算如下：

$$α_{EI}(x; D_n) := E[I(x, v, θ)] = (µ_n(x) − τ) Φ(\frac {µ_n(x) − τ} {σ_n(x)}) + σ_n(x) φ(\frac {µ_n(x) − τ} {σ_n(x)})$$(44)

when σ_n > 0 and vanishes otherwise. Here, not to be confused with the previous section, φ is the standard normal probability density function. These improvement strategies have been empirically studied in the literature [82], [81], [27] and recently convergence rates have been proven for EI [32].

当σ_n > 0，否则就消失了。这里，不要与前一节混淆了，φ是标准正态概率密度函数。这些改进策略在文献中已经研究过了，最近EI的收敛速度也得到了证明。

Finally, although the target objective value (i.e., the best reachable objective value) is often unknown, in practice τ is adaptively set to the best observed value y^+ = max_{i=1:n} y_i. Whereas for PI this heuristic can lead to an overly greedy optimization [81], it works reasonably with EI in practice [143]. When the objective function being minimized is very noisy, using the lowest mean value as the target is reasonable [157].

最近，虽然目标目标值（即，能够达到的最佳目标值）通常是未知的，但在实际中τ自适应的设置为最佳观测值y^+ = max_{i=1:n} y_i。尽管对于PI来说，这个启发式可以带来过度贪婪的优化，但在实践中其效果与EI还不错。当要最小化的目标函数噪声很大，使用最低均值作为目标是很合理的[157]。

### 4.2. Optimistic policies

Dating back to the seminal work of Lai & Robbins [92] on the multi-armed bandit problem, the upper confidence bound criterion has been a popular way of negotiating exploration and exploitation, often with provable cumulative regret bounds. The guiding principle behind this class of strategies is to be optimistic in the face of uncertainty. Indeed, using the upper confidence for every query point x corresponds to effectively using a fixed probability best case scenario according to the model. Originally, the upper confidence was given by frequentist Chernoff–Hoeffding bounds [8].

上置信度界限准则(UCB)可以追溯到Lai&Robbins[92]在多臂老虎机问题上的开创性工作，是在探索和挖掘上折中的流行方式，通常都有可证明的累积regret界限。在这类策略上的引导准则是，在面对不确定性时，要乐观。确实，对每个查询点x使用上置信度，对应着根据模型有效的使用固定概率最佳案例场景。最初，上置信度是由频率学派Chernoff-Hoeffding界限给出。

More recently, the Gaussian process upper confidence bound (GP-UCB [146]) algorithm was proposed as a Bayesian optimistic algorithm with provable cumulative regret bounds. In the deterministic case, a branch-and-bound extension to GP-UCB was proven to have exponentially vanishing instantaneous regret [43]. The GP-UCB algorithm has since been generalized to other Bayesian models by considering upper quantiles [84] instead of Equation (45) defined below, which is more reminiscent of frequentist concentration bounds. In the GP case, since the posterior at any arbitrary point x is a Gaussian, any quantile of the distribution of f(x) is computed with its corresponding value of βn as follows:

最近提出了高斯过程上置信度界限(GP-UCB)算法[146]，作为一种贝叶斯优化算法，有可证明的累积regret界限。在确定性的案例中，GP-UCB的分支界定拓展被证明有指数速度消失的即时regret。GP-UCB算法自从那时就被泛化到其他贝叶斯模型中，考虑了上分位数，而不是下面式(45)定义的，这更让人想起频率学派聚焦界限。在GP的情况中，由于在任意点x处的后验是高斯，分布f(x)的任意分位数都用对应的βn值计算如下：

$$α_{UCB}(x; D_n) := µ_n(x) + β_nσ_n(x)$$(45)

There are theoretically motivated guidelines for setting and scheduling the hyperparameter βn to achieve optimal regret [146] and, as with τ in the improvement policies, tuning this parameter within these guidelines can offer a performance boost.

有理论上的指引，对超参数βn进行设置和调度，以获得最优regret，至于改进策略中的τ，在这些引导中调节这个参数，会给出性能提升。

Finally, there also exist variants of these algorithms for the contextual bandits [153] (see Section VIII-D) and parallel querying [45] (see Section V-E).

最后，对于上下文老虎机，和并行查询的情况，有这些算法的变体。

### 4.3. Information-based policies

In contrast to the acquisition functions introduced so far, information-based policies consider the posterior distribution over the unknown minimizer x^* , denoted p_*(x|D_n). This distribution is implicitly induced by the posterior over objective functions f. There are two policies in this class, namely Thompson sampling and entropy search.

与目前介绍的采集函数对比，基于信息的策略考虑对未知最小化器x^* 的后验分布，表示为p_*(x|D_n)。这个分布是由对目标函数f的后验隐式引入的。在这个类别中有两个策略，即Thompson采样和熵搜索。

Though it was introduced in 1933 [150], Thompson sampling has attracted renewed interest in the multi-armed bandit community, producing empirical evaluations [135], [38] as well as theoretical results [85], [2], [127]. Thompson sampling (TS) is a randomized strategy which samples a reward function from the posterior and selects the arm with the highest simulated reward. Therefore the selection made by TS can be expressed as the randomized acquisition function x_{n+1} ∼ p_*(x|Dn).

Thompson采样是在1933年提出的，但在多臂老虎机团体中吸引了新的兴趣，得到了经验评估和理论结果。TS是一个随机化的策略，从后验中采样一个回报函数，选择最高仿真回报的臂。因此，TS进行的选择可以表述为随机化的采集函数x_{n+1} ∼ p_*(x|Dn)。

However, in continuous search spaces, the analog of Thompson sampling is to draw a continuous function f^{(n)} from the posterior GP and optimize it to obtain x_{n+1}. In order to be optimized, the sample f^{(n)} needs to be fixed so it can be queried at arbitrary points; unfortunately, it is not clear how to fix an exact sample from the GP. However, using recent spectral sampling techniques [20], [125], [94], we can draw an approximate sample from the posterior that can be evaluated at any arbitrary point x [69], which extends TS to continuous search spaces. As an acquisition function, TS can be formulated as

但是，在连续的搜索空间，Thompson采样的类比，是从后验GP中抽取一个连续函数f^{(n)}，优化以得到x_{n+1}。为进行优化，样本f^{(n)}需要进行修复，这样可以在任意点进行查询；不幸的是，怎样从GP中修复一个精确的样本，并不明确。但是，使用最近的谱采样技术，我们可以从后验中抽取一个近似的样本，可以在任意点x处进行评估，将TS拓展到了连续搜索空间。作为一个采集函数，TS可以表述为

$$α_{TS}(x; D_n) := f^{(n)}(x), where f^{(n)} ~ s.s. GP(µ_0, k|D_n)$$(46)

where ∼ s.s. indicates approximate simulation via spectral sampling. Empirical evaluations show good performance which, however, seems to deteriorate in high dimensional problems, likely due to aggressive exploration [139].

这里∼ s.s.表示通过谱采样的近似仿真。经验评估展示了很好的性能，但是在高维问题中性能似乎会恶化，很可能是因为激进的探索。

Instead of sampling the distribution p_*(x|D_n), entropy search (ES) techniques aim to reduce the uncertainty in the location x^* by selecting the point that is expected to cause the largest reduction in entropy of the distribution p_*(x|D_n) [156], [67], [69]. In terms of utility, entropy search methods use the information gain defined as follows

熵搜索(ES)没有采样分布p_*(x|D_n)，而是要降低在位置x^* 处的不确定性，选择的点要使分布p_*(x|D_n)的熵降最大。以效用来说，熵降方法使用的信息增加定义如下

$$U(x, y, θ) = H(x^* |D_n) − H(x^*| Dn ∪ \{(x, y)\})$$(47)

where the θ implicitly parameterizes the distribution of y. 其中θ隐式的参数化了y的分布。

In other words, ES measures the expected information gain from querying an arbitrary point x and selects the point that offers the most information about the unknown x_*. The acquisition function for ES can be expressed formally as

换句话说，ES度量的是从查询任意点x中得到的期望信息增加，选择关于未知点x_*能够给出最多信息的点。ES的采集函数可以正式的表达为

$$α_{ES}(x; D_n) := H(x^* | Dn) − E_{y|Dn,x} H(x^* | D_n ∪ \{(x, y)\})$$

where H(x^* | Dn) denotes the differential entropy of the posterior distribution p_*(x | Dn), and the expectation is over the distribution of the random variable y ∼ N (µ_n(x), σ^2_n(x) + σ^2).

这里H(x^* | Dn)表示后验分布p_*(x | Dn)的微分熵，期望是随机变量y的，分布为y ∼ N (µ_n(x), σ^2_n(x) + σ^2)。

Once again, this function is not tractable for continuous search spaces X so approximations must be made. Early work discretized the space X and computed the conditional entropy via Monte Carlo sampling [156]. More recent work uses a discretization of the X to obtain a smooth approximation to p_* and its expected information gain [67]. This method is unfortunately O(M^4) where M is the number of discrete so-called representer points.

再一次，这个函数对于连续搜索空间X是不可处理的，所以必须进行近似。早期的工作将空间X离散化，通过MC采样计算其条件熵。最近的工作使用X的离散化，得到对p_*及其期望信息增的平滑近似。这种方法的复杂度为O(M^4)，其中M是离散表示点的数量。

Finally, predictive entropy search (PES) removes the need for a discretization and approximates the acquisition function in O((n + d)^3) time, which, for d < n is of the same order as EI [69]. This is achieved by using the symmetric property of mutual information to rewrite α_{ES}(x) as

最后，预测性熵搜索(PES)不需要进行离散化，近似采集函数，复杂度为O((n + d)^3)，对于d < n，与EI的复杂度是一样的。使用互信息的对称性质重写α_{ES}(x)，可以得到

$$α_{PES}(x; D_n) := H(y | D_n, x) − E_{x^* | D_n} [H(y | D_n, x, x^*)]$$

The expectation can be approximated via Monte Carlo with Thompson samples; and three simplifying assumptions are made to compute H(y | Dn, x, x^*). Empirically, this algorithm has been shown to perform as well or better than the discretized version without the unappealing quartic term [69], making it arguably the state of the art in entropy search approximation.

这个期望可以通过带有Thompson样本的MC进行近似；进行三种简化假设，以计算H(y | Dn, x, x^*)。经验上来说，这个算法比离散化版本的表现一样好，或更好，而且没有二次项，使其称为熵搜索近似中最好的方法。

### 4.4. Portfolios of acquisition functions

No single acquisition strategy provides better performance over all problem instances. In fact, it has been empirically observed that the preferred strategy can change at various stages of the sequential optimization process. To address this issue, [73] proposed the use of a portfolio containing multiple acquisition strategies. At each iteration, each strategy in the portfolio provides a candidate query point and meta-criterion is used to select the next query point among these candidates. The meta-criterion is analogous to an acquisition function at a higher level; whereas acquisition functions are optimized in the entire input space, a meta-criterion is only optimized within the set of candidates suggested by its base strategies.

没有哪个采集函数在所有的问题中能够给出更好的性能。实际上，通过经验观察得到，在顺序优化过程的不同阶段，会有不同的策略更好。为处理这个问题，[73]提出了使用一个组合，包含不同的采集策略。在每次迭代中，组合中的每个策略给出一个候选查询点，使用元规则来在这些候选中选择下一个查询点。元规则类似于更高层的采集函数；而采集函数是在整个输入空间中优化，元规则是在候选集中优化，候选是由基策略建议得到的。

The earlier approach of Hoffman et al. is based on a modification of the well-known Hedge algorithm [9], designed for the full-information adversarial multi-armed bandit. This particular portfolio algorithm relies on using the past performance of each acquisition function to predict future performance, where performance is measured by the objective function. However, this performance metric does not account for valuable information that is gained through exploration.

Hoffman等的更早的方法是基于著名的Hedge算法的修正，这是为完全信息的对抗多臂老虎机设计的算法。这个特殊的组合算法依赖于使用每个组合函数的过去的性能，来预测未来的性能，其中性能是由目标函数来度量的。但是，这个性能度量没有计入通过探索得到的宝贵的信息。

A more recent approach, the so-called entropy search portfolio (ESP), considers the use of an information-based metric instead [139]. In contrast to the GP-Hedge portfolio, ESP selects among different candidates by considering the gain of information towards the optimum. Removing the constant entropy at the current time, the ESP meta-criterion reduces to

最近的一种方法是所谓的熵搜索组合(ESP)，考虑了使用基于信息的度量。与GP-Hedge组合形成对比，ESP从不同的候选中选择，考虑向最优的信息增加。去除掉当前时间的常数熵，ESP元规则退化为

$$α_{ESP}(x; Dn) = −E_{y | D_n,x} [ H [x_* | D_n ∪ \{(x, y)\} ] ]$$(48)
$$x_n = argmax_{x_{1:K,n}} α_{ESP}(x; D_n)$$(49)

where x_{1:K,n} represent the candidates provided by the K base acquisition functions. In other words the candidate selected by this criterion is the one that results in the greatest expected reduction in entropy about the minimizer x_*. If the meta-criterion α_{ESP}(x|D_n) were minimized over the entire space X, ESP reduces to the acquisition functions proposed by [156], [67], [69]. However, ESP restricts this minimization to the set of candidates made by each portfolio member.

其中x_{1:K,n}表示候选，由K个基准采集函数给出。换句话说，通过这个规则选择的候选，是关于最小化器x_*的熵的最大期望下降的结果。如果元规则α_{ESP}(x|D_n)在整个空间X中最小化，ESP退化成了[156,67,69]提出的采集函数。但是，ESP将这种最小化局限到了每个组合成员给出的候选的集合。

## 5. Practical Considerations

### 5.1 Handling hyperparameters

### 5.2 Optimizing acquisition functions

### 5.3 Conditional Spaces

### 5.4 Non-stationarity

### 5.5 Parallelization

### 5.6 Software implementation

## 6. Theory of Bayesian Optimization

There exist a vast literature on the theoretical properties of bandit algorithms in general. Theoretical properties of Bayesian optimization, however, have only been established recently. In this section, we focus on the results concerning Gaussian process based Bayesian optimization and defer detailed discussions of bandit algorithms to other dedicated surveys [29], [117].

有很多关于老虎机算法的理论性质的文献。但是，贝叶斯优化的理论性质，在最近才被确立。本节中，我们关注基于高斯过程的贝叶斯优化的结果，关于老虎机算法的详细讨论请参考其他综述[29,117]。

There exist several early consistency proofs for Gaussian process based Bayesian optimization algorithms, in the one-dimensional setting [102] and one for a simplification of the algorithm using simplicial partitioning in higher dimensions [164]. The consistency of the algorithm using multivariate Gaussian processes has been established in [155].

基于高斯过程的贝叶斯优化算法有几个早期的一致性证明，是在一维的设置下，还有一个是对更高维的单形分割的算法简化的情况。多变量高斯过程算法的一致性在[155]中被证明。

More recently, [146] provided the first finite sample bound for Gaussian process based Bayesian optimization. In this work, the authors showed that the GP-UCB algorithm suffers from sub-linear cumulative regret in the stochastic setting. The regret bounds, however, allow only fixed hyperparameters. In [32], Bull provided both upper and lower bounds of simple regret for the EGO algorithm [82] in the deterministic setting. In addition to regret bounds concerning fixed hyperparameters, the author also provided simple regret bounds while allowing varying hyperparameters.

最近，[146]为基于高斯过程的贝叶斯优化提供了第一个有限样本界限。在本文中，作者展示了GP-UCB算法在随机的设置下会有亚线性累积regret的缺陷。但是，regret界限只允许固定的超参数。在[32]中，Bull对EGO算法在确定性的设置下给出了简单regret的下限和上限。除了有关固定超参数的regret界限，作者还在变化超参数的情况下给出了简单的regret界限。

Since the pioneering work of [146] and [32], there emerged a large body of results on this topic including, exponentially vanishing simple regret bounds in the deterministic setting [43]; bounds for contextual Gaussian process bandits [89]; Bayes regret bounds for Thompson sampling [85], [127]; bounds for high-dimensional problems with a underlying low-rank structure [46]; bounds for parallel Bayesian optimization [45]; and improved regret bounds using mutual information [41].

自从[146,32]的先驱工作，在这个主题下涌现了很多工作，包括，在确定性设置下指数速度消失的简单regret界限[43]，上下文高斯过程老虎机的界限[89]，Thompson采样的贝叶斯regret界限[85,127]，高维问题低秩结构的界限[46]，并行贝叶斯优化的界限[45]，利用互信息的改进regret界限[41]。

Despite the recent surge in theoretical contributions, there is still a wide gap between theory and practice. Regret bounds or even consistency results, for example, have not been established for approaches that use a full Bayesian treatment of hyperparameters [143]. Such theoretical results could advance the field of Bayesian optimization and provide insight for practitioners.

尽管最近理论贡献很多，但在理论和实践之间仍然有很宽的空白。比如，对于使用完整贝叶斯处理的超参数，还没有确定regret界限或甚至一致性结果。这样的理论结果会推进贝叶斯优化这个领域，为参与者提供洞见。

## 7. History of Bayesian Optimization and Related Approaches

Arguably the earliest work related to Bayesian optimization was that of William Thompson in 1933 where he considered the likelihood that one unknown Bernoulli probability is greater than another given observational data [150]. In his article, Thompson argues that when considering, for example, two alternative medical treatments one should not eliminate the worst one based on a single clinical trial. Instead, he proposes, one should estimate the probability that one treatment is better than the other and weigh future trials in favour of the seemingly better treatment while still trying the seemingly suboptimal one. Thompson rightly argues that by adopting a single treatment following a clinical trial, there is a fixed chance that all subsequent patients will be given suboptimal treatment. In contrast, by dynamically selecting a fraction of patients for each treatment, this sacrifice becomes vanishingly small.

与贝叶斯优化有关的最早的工作可能是，William Thompson在1933年，他在考虑，在给定观察数据的情况下，一个未知的Bernoulli概率比另外一个要高。在他的文章中，Thompson认为，比如，当考虑两个医学治疗情况，不应当根据一次临床试验就排除掉最坏的那个结果。他提出，一个人应当估计一个治疗比另一个治疗好的概率，权衡未来的尝试时，可以倾向于似乎更好的治疗，但仍然要尝试那个似乎不太好的那个。Thompson正确的认为，在一次临床试验之后采用一个治疗，但后续的病人仍然有固定的概率要进行次优的治疗。比较之下，通过动态的选择一定比例的病人进行每种治疗，这种牺牲会变得越来越小。

In modern terminology, Thompson was directly addressing the exploration–exploitation trade-off, referring to the tension between selecting the best known treatment for every future patient (the greedy strategy) and continuing the clinical trial for longer in order to more confidently assess the quality of both treatments. This is a recurring theme not only in the Bayesian optimization literature, but also the related fields of sequential experimental design, multi-armed bandits, and operations research.

用现代的术语来说，Thompson是在直接处理探索和挖掘的tradeoff，指的是对未来的每个病人选择目前已知的最好的治疗（贪婪策略），或继续进行更长时间的临床试验，对两种治疗的更有信息的质量评估。这是重复发生的主题，不仅在贝叶斯优化文献中存在，而且还在相关的领域中，包括顺序试验设计，多臂老虎机，和操作研究。

Although modern experimental design had been developed a decade earlier by Ronald Fisher’s work on agricultural crops, Thompson introduced the idea of making design choices dynamically as new evidence becomes available; a general strategy known as sequential experimental design or, in the multi-armed bandit literature, adaptive or dynamic allocation rules [92], [59].

虽然现代的试验设计在十年前由Ronald Fisher在农业作物上的工作得到了开发，Thompson引入了在新的证据可用时，动态的进行设计选择的概念；一个通用的策略是，顺序试验设计或，在多臂老虎机的文献中，自适应或动态分配规则。

The term Bayesian optimization was coined in the seventies [115], but a popular version of the method has been known as efficient global optimization in the experimental design literature since the nineties [134]. Since the approximation of the objective function is often obtained using Gaussian process priors, the technique is also referred to as Gaussian process bandits [146].

贝叶斯优化的术语是在70年代造出来的，但这种方法的一个流行版本的名字是，自从90年代以后，在试验设计文献中的高效全局优化。因为目标函数的近似通常是用高斯过程先验得到的，这种技术也称之为高斯过程老虎机。

In the nonparametric setting, Kushner [91] used Wiener processes for unconstrained one-dimensional optimization problems. Kushner’s decision model was based on maximizing the probability of improvement. He also included a parameter that controlled the trade-off between ‘more global’ and ‘more local’ optimization, in the same spirit as the exploration–exploitation trade-off. Meanwhile, in the former Soviet Union, Mockus and colleagues developed a multi-dimensional Bayesian optimization method using linear combinations of Wiener fields [115], [114]. Both of these methods, probility of and expected improvement, were in studied in detail in [81].

在非参数的设置中，Kushner使用Wiener过程在无约束的一维优化问题中。Kushner的决策模型是基于最大化改进概率的。他还包括了一个参数，控制在更加全局和更加局部优化的折中，与探索-挖掘折中的思想一样。同时，在前苏联，Mockus和同时提出了一种多维贝叶斯优化方法，使用的是Wiener域的线性组合。这两种方法，改进概率和期望期望改进，都在[81]中进行了详细研究。

At the same time, a large, related body of work emerged under the name kriging, in honour of the South African student who developed this technique at the University of the Witwatersrand [90], though largely popularized by Matheron and colleagues (e.g., [111]). In kriging, the goal is interpolation of a random field via a linear predictor. The errors on this model are typically assumed to not be independent, and are modelled with a Gaussian process.

同时，很多文献以kriging的名称出现，纪念提出了这种技术的南非学生，但这种技术是从Matheron和其同事那开始流行起来的。在kriging中，目标是通过一个线性预测器来进行随机领域的插值。这个模型的误差一般都假设为不独立的，是用高斯过程建模得到的。

Kriging has been applied to experimental design under the name DACE, after design and analysis of computer experiments, the title of a paper by Sacks et al. [128] (and more recently a book by Santner et al. [130]). In DACE, the regression model is a best linear unbiased predictor (BLUP), and the residual model is a noise-free Gaussian process. The goal is to find a design point or points that optimizes some criterion. Experimental design is usually non-adaptive: the entire experiment is designed before data is collected. However, sequential design is an important and active subfield (e.g., [160], [33]).

Kriging曾应用到试验设计中，名称为DACE，在设计和分析了计算机试验后，这是Sacks等的文章[128]标题。在DACE中，回归模型是一个最佳线性无偏预测器，残差模型是一个无噪声的高斯过程。目标是找到一个设计点，或几个设计点，最优化一些准则。试验设计通常是非自适应的：整个试验是在数据收集之前进行设计的。但是，顺序设计是一个重要的，活跃的子领域。

The efficient global optimization (EGO) algorithm is the combination of DACE model with the sequential expected improvement acquisition criterion. It was published in a paper by Jones et al. [82] as a refinement of the SPACE algorithm (stochastic process analysis of computer experiments) [133]. Since EGO’s publication, there has evolved a body of work devoted to extending the algorithm, particularly in adding constraints to the optimization problem [6], [131], [23], and in modelling noisy functions [14], [75], [76].

高效全局优化(EGO)算法是DACE模型与顺序期望改进采集原则的结合。这是Jones等[82]发表的一篇文章，是SPACE算法[133]（计算机试验的随机过程分析）的改进。自从EGO的发表，演化出了很多工作来拓展这个算法，特别是对优化问题加入了约束，和在含噪函数的建模中。

In the bandits setting, Lai and Robbins [92] introduced upper confidence bounds (UCB) as approximate alternatives to Gittins indices in 1985. Auer studied these bounds using frequentist techniques, and in adversarial multi-armed bandit settings [9], [8].

在老虎机的设置中，Lai和Robbins在1985年[92]引入了UCB，作为Gittins indices的近似替代。Auer使用频率派的技术，在对抗多臂老虎机的设置中，研究了这些界限。

The literature on multi-armed bandits is vast. The book of Cesa-Bianchi [36] is a good reference on the topic of online learning with experts and bandits in adversarial settings. There are many results on exploration [30], [51], [50] and contextual bandits [97], [112], [2]. These contextual bandits, may also be seen as myopic approximations to Markov decision processes.

多臂老虎机中的文献是很多的。专著[36]是与专家和老虎机的对抗设置中的在线学习的领域一个很好的参考。有很多在探索领域和上下文老虎机的结果。这些上下文老虎机，也可以视为对Markov决策过程的近似。

## 8. Extensions and Open Questions

### 8.1 Constrained Bayesian optimization

### 8.2 Cost-sensitivity

### 8.3 High-dimensional problems

### 8.4 Multi-task

### 8.5 Freeze-Thaw

## 9. Concluding Remarks

In this paper we have introduced Bayesian optimization from a modelling perspective. Beginning with the Beta-Bernoulli and linear models, and extending them to non-parametric models, we recover a wide range of approaches to Bayesian optimization that have been introduced in the literature. There has been a great deal of work that has focussed heavily on designing acquisition functions, however we have taken the perspective that the importance of this plays a secondary role to the choice of the underlying surrogate model.

本文中，我们从建模的角度介绍了贝叶斯优化。从Beta-Bernoulli和线性模型开始，将其拓展到非参数模型，我们回顾了文献中提出的很多贝叶斯优化方法。有很多工作聚焦在设计采集函数，但是我们采取的角度是，这个的重要性并没有选择潜在的代理模型重要。

In addition to outlining different modelling choices, we have considered many of the design decisions that are used to build Bayesian optimization systems. We further highlighted relevant theory as well as practical considerations that are used when applying these techniques to real-world problems. We provided a history of Bayesian optimization and related fields and surveyed some of the many successful applications of these methods. We finally discussed extensions of the basic framework to new problem domains, which often require new kinds of surrogate models.

除了列出了不同的建模选项，我们考虑了很多设计决策，用于构建贝叶斯优化系统。我们进一步强调了相关的理论，以及实践的考虑，当这些技术用于真实世界问题时会用到。我们给出贝叶斯优化和相关领域的历史，回顾了这些方法很多成功的应用。我们最后讨论了基本框架拓展到新的问题领域，这通常需要新的代理模型。

Although the underpinnings of Bayesian optimization are quite old, the field itself is undergoing a resurgence, aided by new problems, models, theory, and software implementations. In this paper, we have attempted to summarize the current state of Bayesian optimization methods; however, it is clear that the field itself has only scratched the surface and that there will surely be many new problems, discoveries, and insights in the future.

虽然贝叶斯优化的支柱很老，但这个领域正经历一场复兴，有新的问题，模型，理论和软件实现。在本文中，我们试图总结目前贝叶斯优化算法的目前状态；但是，很清楚，这个领域本身还很浅，未来还会有很多新问题，发现和洞见。