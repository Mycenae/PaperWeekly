# Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures

J. Bergstra et. al. @ Harvard

## 0. Abstract

Many computer vision algorithms depend on configuration settings that are typically hand-tuned in the course of evaluating the algorithm for a particular data set. While such parameter tuning is often presented as being incidental to the algorithm, correctly setting these parameter choices is frequently critical to realizing a method’s full potential. Compounding matters, these parameters often must be re-tuned when the algorithm is applied to a new problem domain, and the tuning process itself often depends on personal experience and intuition in ways that are hard to quantify or describe. Since the performance of a given technique depends on both the fundamental quality of the algorithm and the details of its tuning, it is sometimes difficult to know whether a given technique is genuinely better, or simply better tuned.

很多计算机视觉算法依赖于配置设置，在对特定数据集进行算法评估的过程中，一般是由手工调节的。这样的参数调节目前对于算法来说是次要的，但准确的设置这些参数，能够完全实现一种方法的完全的潜力。更复杂的是，这些参数在算法应用到新的问题领域中时，必须要重新调节，这个调节过程通常依赖于个人经验和直觉，很难量化或描述。由于一种给定技术的性能，依赖于算法的基本质量，和调节的细节，有时候很难知道这种给定的技术是真的更好，还是调节的更好。

In this work, we propose a meta-modeling approach to support automated hyperparameter optimization, with the goal of providing practical tools that replace hand-tuning with a reproducible and unbiased optimization process. Our approach is to expose the underlying expression graph of how a performance metric (e.g. classification accuracy on validation examples) is computed from hyperparameters that govern not only how individual processing steps are applied, but even which processing steps are included. A hyperparameter optimization algorithm transforms this graph into a program for optimizing that performance metric. Our approach yields state of the art results on three disparate computer vision problems: a face-matching verification task (LFW), a face identification task (PubFig83) and an object recognition task (CIFAR-10), using a single broad class of feed-forward vision architectures.

本文中，我们提出了一种元建模方法，支持自动超参数优化，其目标是提供实际的工具，替换掉手工调节，而且是可复现的，无偏的优化过程。我们的方法是要将怎样从超参数中计算得到性能度量（在验证样本上的分类准确率）的潜在表达图暴露出来，这些超参数不仅主宰了单个处理过程是怎样应用的，而且主宰了包含哪个处理步骤。超参数优化算法将这个图变换成优化性能度量的程序。我们的方法在三个不同的计算机视觉问题中得到了目前最好的结果：一个人脸匹配的验证任务(LFW)，一个人脸识别任务(PubFig83)，和一个目标识别任务(CIFAR-10)，使用的是一大类前馈视觉架构。

## 1. Introduction

Many computer vision algorithms depend on hyperparameter choices such as the size of filter bank, the strength of classifier regularization, and positions of quantization levels. These choices can have enormous impact on system performance: e.g. in (Pinto & Cox, 2011), the authors extensively explored a single richly-parameterized model family, yielding classification performance that ranged from chance to state-of-the-art performance, depending solely on hyperparameter choices. This and other recent work show that the question of “how good is this model on that dataset?” is ill-posed. Rather, it makes sense to speak of the quality of the best configuration that can typically be discovered by a particular search procedure in a given amount of time, for a task at hand. From this perspective, the tuning of hyperparameters is an important part of understanding algorithm performance, and should be a formal and quantified part of model evaluation.

很多计算机视觉算法依赖于超参数的选择，比如滤波器组的规模，分类器正则化的强度，量化级的位置。这些选择对系统性能有很大影响：如，在(Pinto & Cox, 2011)中，作者广泛探索了一族参数丰富的模型，产生了很多分类性能，其不同之处只有超参数的选择。这个工作和最近的其他工作表明，问题“这个模型在这个数据集上表现怎样”是病态的。而如果讨论对手头的任务，在给定的时间内，由特定的搜索过程发现的最好的配置的质量，这才是有意义的。从这个角度来说，超参数的调节是理解算法性能的重要部分，应当是模型评估的正式且量化的部分。

On the other hand, ad hoc manual tuning by the algorithm inventor, while generally hard to reproduce or compare with fairly, can be efficient. A system’s designer has expectations for how his or her system should work, and he or she can quickly diagnose unexpected deviations.

另一方面，由算法发明者专门的手工调节，一般来说很难复现，难以进行公平比较，但是会很高效。系统设计者对系统应当怎样工作是有预期的，他可以很快的诊断出未预期的偏离。

In this work we explore the possibility that manual optimization is no longer efficient enough to justify the lack of formalization that it entails. Recent developments in algorithm configuration raise the efficiency of automatic search, even in mathematically awkward search spaces, to a level where the result of hand-tuning can be matched and exceeded in a matter of hours on a small cluster of computers. Using these ideas, we implemented a broad class of feed-forward image feature extraction and classification models in order to formalize the steps of selecting the parameters of a model, and evaluating that model on a task. We compared random search in that model class with a more sophisticated algorithm for hyperparameter optimization, and found that the optimization-based search strategy recovered or improved on the best known configurations for all three image classification tasks in our study. This success motivates us to suggest that questions regarding the utility of modeling ideas should generally be tested in this style. Automatic search is reproducible, and thus supports analysis that is impossible for human researchers to perform fairly (e.g. “How would you have tuned approach Y if you had not already learned to optimize approach X?”) To support research in hyperparameter optimization, we provide our optimization algorithms and specification language for download as free open source software. This software replicates the results presented in this work, and provides a foundation for general algorithm configuration in future work.

本文中，我们探索的可能性是，手工优化效率太低。算法配置最近的进展提升了自动搜索的效率，即使是在数学上比较奇特的搜索空间中，提升的水平到了，在小型计算机集群中，在几小时以内，可以匹配并超过手工调节的结果。使用这些思想，我们实现了一类前馈图像特征提取和分类模型，将选择模型参数，和在任务上评估这个模型的步骤公式化。我们在这类模型中，比较了随机搜索，和更复杂的超参数优化算法，发现基于优化的搜索策略，在我们的研究中，在所有三个图像分类任务中，找到或超过了已知最好的配置。这种成功促使我们认为，利用建模思想的有关问题，一般来说都应当以这种模式进行测试。自动搜索是可复现的，因此以前人类研究者不能公平的进行的分析，也能支持（如，如果尚未学习优化方法X，怎么才能调节方法Y？）。为支持超参数优化的研究，我们的优化算法开源可以下载。软件复现了本文中给出的结果，为未来工作中的通用算法配置打下了基础。

## 2. Previous Work

Our work extends two veins of research with little historical overlap: feed-forward model architectures for computer vision, and techniques for algorithm configuration. 我们的工作拓展了两条研究线，之前的历史重叠很少：计算机视觉的前馈模型架构，和算法配置的技术。

**Feed-forward models in computer vision**. There is a long tradition of basing computer vision systems on models of biological vision (Fukushima, 1980; LeCun et al., 1989; Riesenhuber & Poggio, 1999; Lowe, 1999; Hinton et al., 2006; DiCarlo et al., 2012). Such efforts have arrived at a rough consensus model in which nonlinear image features are computed by a feed-forward neural network. Each layer of the network comprises a relatively standard set of transformations, including: (i) dimensionality expansion (e.g. by convolution with a filter bank), (ii) dynamic-range reduction (e.g. by thresholding), (iii) spatial smoothing (e.g. pooling or soft-max), (iv) local competition (e.g. divisive normalization), and (v) dimensionality reduction (e.g. sub-sampling or PCA). Feature extraction is usually followed by a simple classifier read-out trained on labeled data.

计算机视觉中的前馈模型。计算机视觉系统以生物视觉为基础，有很长的历史。这些努力已经达到了一个大致共识的模型，通过一个前馈神经网络计算非线性的图像特征。网络的每一层都是由相对标准的变换集组成，包括：(i)维度扩张（如，与滤波器组卷积），(ii)动态范围降低（如，阈值），(iii)空域平滑（如，池化或soft-max），(iv)局部竞赛（如，归一化），(v)维度降低（如，降采样或PCA）。特征提取后通常是一个简单的分类器，在标注的数据上进行训练得到。

Beyond this high-level consensus, however, many details remain unresolved: which specific operations should be involved, in what order should they be applied, how many layers should be used, what kinds of classifier(s) should be used, and how (if at all) should the filter values be learned from statistics of input data. Many competing modeling approaches can roughly be thought of as having made different design choices within a larger unformalized space of feed-forward algorithm configurations.

除了这些高层共识，很多细节仍然没有解决：要包括哪些具体的运算，它们应用的顺序应当是怎样的，应当使用多少层，应当使用哪些类别的分类器，从输入数据的统计中怎样学习得到滤波器的值。很多有竞争的模型方法可以大致认为是，在更大的不正规的前馈算法配置空间中，不同的设计选择。

**Algorithm configuration**. Algorithm configuration is a branch of optimization dealing with mathematically difficult search spaces, comprising both discrete and continuous variables as well as conditional variables that are only meaningful for some combinations of other variables. Bayesian optimization approaches have proved useful in these difficult domains (Mockus et al., 1978). A Bayesian optimization approach centers on a probability model for P(score|configuration) that is obtained by updating a prior from a history H of (configuration, score) pairs. This model can be queried more quickly than the original system in order to find promising candidates. Search efficiency comes from only evaluating these most promising candidates on the original system. Gaussian processes (Rasmussen & Williams, 2006) have often been used as the probability model, but other regression models such as decision trees have also proved successful (Hutter, 2009; Brochu, 2010; Bardenet & Kégl, 2010; Hutter et al., 2011; Bergstra et al., 2012). In these approaches, the criterion of Expected Improvement (EI) beyond a threshold μ is a popular heuristic for making proposals (Jones, 2001). In that approach, the optimization algorithm repeatedly suggests a configuration c that optimizes EI(c) = \int_{y<μ} yP(y|c, H) while the experimental history of (score, configuration) pairs, H, accumulates and changes the model. Recently Bergstra et al. (2011) suggested an approach to Bayesian optimization based on a model of P(c|y) instead. Under some assumptions this approach can also be seen to optimize EI.

算法配置。算法配置是优化的一个分支，处理数学上很难的搜索空间，由离散变量，连续变量，以及条件变量组成，条件变量只对与其他变量的一些组合是有意义的。贝叶斯优化方法在这些困难的领域是有用的。贝叶斯优化方法以概率模型P(score|configuration)为中心，用(configuration, score)对的历史H更新一个先验得到。这个模型比原始系统可以更快的查询到，以找到有希望的候选。搜索效率来自于，只在原系统中最有希望的候选中进行评估。高斯过程通常用作概率模型，但其他回归模型也是很成功的，比如决策树。在这些方法中，大于阈值μ的期望改进(EI)的规则是提出建议的一个流行的启发式。在这个方法中，优化算法重复建议一个配置c，使EI(c) = \int_{y<μ} yP(y|c, H)进行最优化，同时(score, configuration)对H的试验历史会累积并修改模型。最近提出一种基于模型P(c|y)的贝叶斯优化方法。在一些假设下，这个方法也可以视作是优化EI。

Hyperparameter optimization in computer vision is typically carried out by hand, by grid search, or by random search. We conjecture that Bayesian optimization is not typically used because it is relatively new technology, and because it requires a layer of abstraction between the researcher toying with settings at a command prompt and the system being optimized. We show that although algorithm configuration is a young discipline, it already provides useful techniques for formalizing the difficult task of simultaneous optimization of many hyperparameters. One of the contributions of our work is to show how useful Bayesian optimization can be, even when optimizing hundreds of hyper-parameters.

计算机视觉中的超参数优化通常是手工进行的，或网格搜索，或随机搜索。我们推测，贝叶斯优化一般并没有使用，因为这是相对更新的技术，也因为在研究者和要优化的系统之间需要一层抽象。我们展示了，虽然算法配置是一个年轻的准则，但对同时优化很多超参数上已经很有用了。我们的一个贡献是，展示了贝叶斯优化有用的地方，即使是在优化数百个超参数的时候。

## 3. Automatic Hyperparameter Optimization

Our approach to hyperparameter optimization has four conceptual components: 我们的超参数优化方法有四个组成部分：

**1. Null distribution specification language**. We propose an expression language for specifying the hyperparameters of a search space. This language describes the distributions that would be used for random, unoptimized search of the configuration space, and encodes the bounds and legal values for any other search procedure. A null prior distribution for a search problem is an expression G written in this specification language, from which sample configurations can be drawn.

我们提出一种表示语言，指定一个搜索空间的超参数。这种语言描述的分布，会用于配置空间的随机，未优化搜索，为任何其他搜索过程编码了边界和合法值。一个搜索问题的null prior分布是用这种指标语言写的一个表达式G，从中可以抽取出样本配置。

For example: G = {a = normal(0, 1), b = choice(0, log(uniform(2, 10)), a)} specifies a joint distribution in which a is distributed normally with mean 0 and variance 1, and b takes either value 0, or a, or a value drawn uniformly between 2 and 10. There are three hyperparameters at play here, shown in bold: the value of a, the value of the choice, and the value of the uniform.

比如：G = {a = normal(0, 1), b = choice(0, log(uniform(2, 10)), a)}指定了一个联合分布，其中a是均值为0方差为1的正态分布，b的值为0，或a，或2到10之间均匀分布的一个值。这里有3个超参数，以粗体显示：a的值，choice的值，和uniform的值。

More generally, the expressions that make up the null distribution specification can be arbitrarily nested, composed into sequences, passed as arguments to deterministic functions, and referenced internally, to form an directed acyclic expression graph (DAG).

更一般的，组成这个null分布指标的这个表达式可以任意嵌套，组成序列，作为参数传递到确定性的函数中，内部引用，以形成一个有向无环表达式图(DAG)。

**2. Loss Function**. The loss function is the criterion we desire to minimize. It maps legal configurations sampled from G to a real value. For example, the loss functions could extract features from a particular image dataset using configuration parameters specified by the random sample from G, and then report mis-classification accuracy for those features. Typically the loss function will be intractable analytically and slow enough to compute that doing so imposes a meaningful cost on the experimenter’s time.

损失函数。损失函数是我们希望最小化的准则。它将从G中采样的合法配置映射到一个实值。比如，损失函数会从一个特定的图像数据集使用从G中随机样本指定的配置参数提取特征，然后给出这些特征的错误分类准确率。典型的，损失函数一般不有解析形式，计算起来很慢，如果计算的话，会消耗相当多的时间。

**3. Hyperparameter Optimization algorithm(HOA)**. The HOA is an algorithm which takes as inputs the null prior expression G and an experimental history H of values of the loss function, and returns suggestions for which configuration to try next. Random sampling from the prior distribution specification G is a perfectly valid HOA. More sophisticated HOAs will generally commandeer the random nodes within the null prior expression graph, replacing them with expressions that use the experimental history in a nontrivial way (e.g. by replacing a uniform node with a Gaussian mixture whose number of components, means, and variances are refined over the course of the experiment).

超参数优化算法。HOA是一个算法，以null prior表达式G，损失函数值的试验历史H为输入，返回要尝试的下一个配置的建议。从先验分布指标G中的随机采样，完全是一个有效的HOA。更复杂的HOAs一般会在null prior表达式图中占据随机的节点，将其替换成使用试验历史的表达式（如，将均匀分布的节点替换成高斯混合，其组成部分数量，均值和方差是在试验过程中提炼过的）。

**4. Database**. Our approach relies on a database to store the experimental history H of configurations that have been tried, and the value of the loss function at each one. As a search progresses, the database grows, and the HOA explores different areas of the search space.

数据库。我们的方法在一个数据库中存储已经尝试过的配置的试验历史H，以及在每一个尝试中的损失函数值。作为一个搜索过程，数据库逐渐增长，HOA探索了搜索空间的不同区域。

The stochastic choice node, which randomly chooses an argument from a list of possibilities, is an important aspect of our approach. Choice nodes make it possible to encode conditional parameters in a search space (Hutter, 2009). To continue the example above, if the choice node is evaluated such that b takes the value of a, then our parameterization of G allows the optimizer to infer that whatever score we obtain has nothing to do with the hyperparameter associated with uniform(2, 10). Visual system models have many configurable components, and entire components can be omitted from a particular pipeline configuration, so it is natural to describe the parameters of an optional component using conditional parameters. The use of conditional parameters makes credit assignment among a set of hyperparameters more efficient.

随机选择节点，从可能的列表中随机选择一个参数，是我们方法中的重要方面。选择节点使其可能在搜索空间中编码条件参数。为继续上面的例子，如果选择节点的评估使b的值为a，那么我们对G的参数化使优化器可以推断，无论我们得到什么分数，都和与uniform(2,10)相关的超参数无关。视觉系统模型有很多可配置的组成部分，整个组成部分在特定的流水线配置中是可以忽略的，所以将可选的组成部分的参数用条件参数进行描述，是很自然的。使用条件参数，使得在超参数集合进行credit指定更加高效。

Our implementation of these four components is available for download as both a general purpose tool for program optimization and a specific visual system model for new image classification data sets (Bergstra, 2013; Bergstra et al., 2013).

我们对这四个部分的实现是可以下载的，可以当做程序优化的通用目的工具用，和一个具体的图像分类的视觉系统模型用。

## 4. Object Recognition Model Family

We evaluate the viability of automatic parameter search by encoding a broad class of feed-forward classification models in terms of the null distribution specification language described in the previous section. This space is a combination of the work of Coates & Ng (2011) and Pinto et al. (2009), and is known to contain parameter settings that achieve the state of the art performance on three data sets (i.e, loss functions): LFW, Pubfig83, and CIFAR-10.

我们评估自动参数搜索的可行性，将很大一类前馈分类模型以前一节描述的null分布规范语言进行编码。这个空间是Coates & Ng (2011)和Pinto et al. (2009)的工作的组合，包含的参数设置可以在三个数据集上得到目前最好的性能：LFW，Pubfig83和CIFAR-10。

The full model family that we explore is illustrated in Figure 1. Like Coates & Ng (2011), we include ZCA-based filter-generation algorithms (Hyvärinen & Oja, 2000) and coarse histogram features (described in their work as the R-T and RP-T algorithms). Like Pinto et al. (2009), we allow for 2-layer and 3-layer sequences of filtering and non-linear spatial pooling. Our search space is configured by a total of 238 hyperparameters – far too large for brute force search, and an order of magnitude larger in dimensionality than the 32-dimensional space searched by Bergstra et al.(2011). The remainder of this section describes the components of our model family. An implementation of the model is available for download (Bergstra et al., 2013).

我们探索的完整模型族，如图1所示。与Coates & Ng (2011)相似，我们包含了基于ZCA的滤波器生成算法，和粗糙直方图特征（在其文章中称为R-T和RP-T算法）。与Pinto et al. (2009)相似，我们允许2层和3层序列的滤波，和非线性空域池化。我们的搜索空间由总计238个参数进行配置，对于暴力搜索来说太大了，比Bergstra et al.(2011)搜索的32维空间也大了一个数量级。本节剩余的部分描述了我们的模型族的组成部分。模型的实现是可以下载的。

The inter-layers (Figure 1a) perform a filter bank normalized cross-correlation, spatial pooling, and possibly sub-sampling. These layers are very much in the spirit of the elements of the Pinto & Cox (2011) model, except that we have combined the normalization and filter bank cross-correlation into a single mathematical operation (fbncc, Equations 1-2).

inter层（图1a）进行滤波器族归一化互相关，空域池化，还有可能的下采样。这些层与Pinto & Cox (2011)模型中的思想元素非常接近，除了我们将归一化和滤波器组互相关结合成了一个数学运算(fbncc, 式1-2)

$$y = fbncc(x,f)$$(1)

$$y_{ijk} = \frac {\hat f_k * ǔ_{ij}} {\sqrt_{ρ max(||ǔ_{ij}||^2, β) + (1-ρ)(||ǔ_ij||^2+β)}}$$(2)

The fbncc operation is a filter bank convolution of each filter fk with a multi-channel image or feature map x, in which each patch x̌ij of x is first shifted by its mean εm̌ (motivating ǔij = x̌ij −εm̌) then scaled to have approximately unit norm. Whereas Pinto & Cox (2011) employed only random uniform filters fk, we include also some of the filter-generation strategies employed in Coates & Ng (2011): namely random projections of ZCA components, and randomly chosen ZCA-filtered image patches. Filter-generation is parametrized by a filter count K ∈ [16, 256]), a filter size Sf ∈ [2, 10], a random seed, and a band-pass parameter in the case of ZCA. The pair-indexed hat-notation x̌ij refers to a patch volume from x at row i and column j that includes Sf rows and columns as well as all channels of x; Our fbncc implementation is controlled by log-normally distributed hyperparameter β which defines a low-variance cutoff, a binary-valued hyperparameter ρ that determines whether that cutoff is soft or hard, and a binary-valued parameter ε that determines whether the empirically-defined patch mean m̌ should be subtracted off or not.

fbncc运算是一个滤波器组卷积，每个滤波器fk和一个多通道图像或特征图x进行卷积，其中x的每个图像块x̌ij首先减去其均值εm̌，然后缩放，达到近似范数为1。尽管Pinto & Cox (2011)只采用了随机均匀滤波器fk，我们还包含了一些Coates & Ng (2011)采用的滤波器生成策略：即ZCA组成部分的随机投影，随机选择的ZCA滤波的图像块。滤波器生成的参数有滤波器数量K ∈ [16, 256])，滤波器大小Sf ∈ [2, 10]，随机种子，在ZCA情况下的带通参数。成对索引的hat符号x̌ij表示一个x中图像块体，从第i行第j列开始，包含Sf行和列，包含x的所有通道；我们的fbncc实现由log-normal分布的超参数ρ控制，决定了截止是软的还是硬的，二值参数ε，决定了经验定义的图像块均值m̌是否要被减去。

Local spatial pooling (lpool, Equation 3) was implemented as in Pinto & Cox (2011).局部空域池化(lpool, Equation 3)的实现和Pinto & Cox (2011)一样。

$$y = lpool(x) ⇔ y_{ijk} = x_{i'j'k} / ||x̌_{i'j'k}||_p$$(3)

The operation is parameterized by a patch size Sp ∈ [2, 8], a sub-sampling stride i'/i = j'/j ∈ {1, 2}, and a log-normally distributed norm parameter p. The triple-indexed x̌_{ijk} refers to a single-channel patch surface from x at row i, column j, and channel k that extends spatially to include Sp rows and columns.

运算的参数有块大小Sp ∈ [2, 8]，下采样步长i'/i = j'/j ∈ {1, 2}，log-normal分布的范数参数p。三索引变量x̌_{ijk}指的是x中在第i行第j列第k通道的单通道块的平面，大小为Sp行和列。

The outer-layers (Figure 1b) combine the fbncc operation of inter-layers with different pooling options. Rather than sampling or optimizing the filter count, the filter count is determined analytically so that the number of image features approaches but does not exceed sixteen thousand (16,000). Pooling is done either (1) with lpool and lnorm (Equation 4) as in Pinto & Cox (2011), or (2) with spatial summation of positive and negative half-rectified filter bank responses (dihist, Equation 5). Within pooling strategy (2) we used two strategies to define the spatial patches used in the summation: either (2a) grid cell summation as in Coates & Ng (2011), or (2b) box filtering. The difference between (2a) and (2b) is a trade-off between spatial resolution and depth of filter bank in making up the output feature set.

outer层（图1b）将inter层的fbncc运算与不同的pooling选项结合到一起。滤波器数量是解析确定的，这样图像特征数量接近但不超过16000个。池化的进行方式如下：(1)用式4中的lpool和lnorm，和Pinto & Cox (2011)一样；(2)half-rectified滤波器组响应（式5中的dihist）的正和负的空域和：要么是(2a)网格单元相加，和Coates & Ng (2011)一样，或(2b)box滤波。(2a)(2b)之间的不同是在空间分辨率和滤波器组深度之间的折中，以组成输出特征集。

$$y = lnorm(x) ⇔ y_{ij} = \frac {x_{ijk}} {x̌_{ij}}, if ||x̌_{ij}||_2 > τ; x_{ijk}, otherwise$$(4)

$$y = dihist(x) ⇔ y_{ijk} = [||max(x̌_{ijk} − α, 0)||_1, ||max(−x̌ _{ijk} − α, 0)||_1]^T$$(5)

Hyperparameter τ of the lnorm operation was log-normally distributed, as was the α hyperparameter of dihist. In approach (2a) we allowed 2x2 or 3x3 grids. In approach (2b) we allowed for sub-sampling by 1, 2, or 3 and square summation regions of side-length 2 to 8.

lnorm运算的超参数τ是log-normal分布，dihist的超参数α也是。在方法(2a)中，我们允许2x2或3x3的网格。在方法(2b)中，我们允许下采样1，2或3倍，或边长2到8的区域的方形求和。

The last step in our image-processing pipeline is a classifier, for which we used an l2-regularized, linear, L2-SVM. For the smaller training sets we used liblinear via sklearn as the solver(Fan et al., 2008; Pedregosa et al., 2011), for larger ones we used a generic L-BFGS algorithm in the primal domain (Bergstra et al., 2010). Training data were column-normalized. The classifier components had just two hyperparameters: the strength of regularization and a cutoff for low-variance feature columns. Code for these experiments was written in Python, with the feature extraction carried out by Theano (Bergstra et al., 2010) and hyperparameter optimization carried out by the hyperopt package (Bergstra, 2013).

我们的图像处理流水中的最后步骤是一个分类器，我们可以采用一个l2正则化的，线性，L2-SVM。对于更小的训练集，我们通过sklearn使用liblinear作为求解器，对于更大的训练集，我们使用了一个通用L-BFGS算法。训练数据是列归一化的。分类器组成部分只有2个超参数：正则化的强度，和低方差特征列的截止值。这些试验的代码是用Python写的，特征提取是用Theano进行的，超参数优化是用hyperopt进行的。

## 5. Results

We evaluate the technique of automatic hyperparameter configuration by comparing two hyperparameter optimization algorithms: random search versus a Tree of Parzen Estimators (TPE) (Bergstra et al., 2011). The TPE algorithm is an HOA that acts by replacing stochastic nodes in the null description language with ratios of Gaussian Mixture Models (GMM). On each iteration, for each hyperparameter, TPE fits one GMM l(x) to the set of hyperparameter values associated with the smallest (best) loss function values, and another GMM g(x) to the remaining hyperparameter values. It chooses the hyperparameter value x that maximizes the ratio l(x)/g(x). Relative to Bergstra et al. (2011) we made two minor modifications to the search algorithm. The first modification was to down-weight trials as they age so that old results do not count for as much as more recent ones. We gave full weight to the most recent 25 trials and applied a linear ramp from 0 to 1.0 to older trials. This is a heuristic concession to TPE’s assumption of hyperparameter independence: as our search moves through space, we use temporal distance within the experiment as a surrogate for distance in the search space. The second modification was to vary the fraction of trials used to estimate l(x) and g(x) with time. Out of T observations of any given variable, we used the top-performing $\sqrt_T/4$ trials to estimate the density of l. We initialized TPE with 50 trials drawn from the null configuration description. These hyper-hyperparameters were chosen manually by observing the shape of optimization trajectories on LFW view 1. We did not reconfigure TPE for the other data sets. The TPE algorithm took up to one or two seconds to suggest new points, so it contributed a negligible computational cost to these experiments.

我们评估了自动超参数配置的技术，比较了两种超参数优化算法：随机搜索，和Tree of Parzen Estimators(TPE)。TPE算法是一种HOA，将null description语言中的随机节点替换为GMM的比率。在每次迭代中，对于每个超参数，TPE对超参数值的集合，与最小（最佳）损失函数值相关的，拟合一个GMM l(x)，对剩下的超参数值拟合另一个GMM g(x)。选择的超参数值x可以最大化比率l(x)/g(x)。与Bergstra et al. (2011)相对，我们对搜索算法进行了两个小的修改。第一个修改是，对老的尝试施加更小的权重，这样最近的尝试的权重更大。我们对最近25次尝试给与完整的权重，对更老的尝试加以线性斜坡，从0到1。这是对TPE的超参数独立性的启发式让步：在我们的搜索在空间中移动时，我们使用试验中的时域距离作为搜索空间中距离的代理。第二个改动是，对用于估计l(x)和g(x)的尝试随着时间进行改变。对任意给定的变量的T个观察，我们使用性能最好的$\sqrt_T/4$次尝试，来估计l的密度。我们用50个从null description配置中抽取的尝试初始化TPE。这些超参数是手工选择的，通过观察在LFW视角1的优化轨迹形状。我们对其他数据集，没有重新配置TPE。TPE算法需要1秒或2秒来建议新的点，所以对这些试验来说，新增加的计算量是可以忽略的。

### 5.1. TPE vs. Random Search: LFW and PubFig83

Random search in a large space of biologically-inspired models has been shown to be an effective approach to face verification (Pinto & Cox, 2011) and identification (Pinto et al., 2011). Our search space is similar to the one used in those works, so LFW (Huang et al., 2007) and PubFig83 (Pinto et al., 2011) provide fair playing fields for comparing TPE with random search.

在很大的空间中用生物启发的模型进行随机搜索，已经证明对人脸验证和识别是非常有效的方法。我们的搜索空间与这些工作中使用的是类似的，所以LFW和PubFig83可以很公平的比较TPE和随机搜索。

For experiments on LFW, we follow Pinto & Cox (2011) in using the aligned image set, and resizing the gray scale images to 200 × 200. We followed the official evaluation protocol – performing model selection on the basis of one thousand images from “view 1” and testing by re-training the classifier on 10 “view 2” splits of six thousand pairs. We transformed image features into features of image pairs by applying an element-by-element comparison function to the left-image and right-image feature vectors. Following Pinto & Cox (2011) we used one comparison function for model selection (square root of absolute difference) and we concatenated four comparison functions for the final “view 2” model evaluation (product, absolute difference, squared difference, square root of absolute difference).

对于在LFW上的试验，我们按照Pinto & Cox (2011)，使用对齐的图像集，将灰度图像改变到200x200大小。我们按照官方的评估协议，用视角1的1000幅图像进行模型选择，用视角2中6000对图像的分割重新训练分类器并进行测试。我们将图像特征变换到图像对的特征，将左图和右图的特征向量进行逐个元素的对比。按照Pinto & Cox (2011)，我们对模型选择进行一次比较函数（绝对差的均方），我们将4个比较函数拼接到一起，用于最终视角2的模型评估（乘积，绝对差，平方差，绝对差的均方根）。

The PubFig83 data set contains 8300 color images of size 100 × 100, with 100 pictures of each of 83 celebrities (Pinto et al., 2011). For our PubFig83 experiments we converted the un-aligned images to gray scale and screened models on the 83-way identification task using 3 splits of 20 train/20 validation examples per class, running two simultaneous TPE optimization processes for a total of 1200 model evaluations. Top-scoring configurations on the screening task were then tested in a second phase, consisting of five training splits of 90 train/10 test images per class. Each of the five second phase training sets of 90 images per class consisted of the 40 images from the first phase and 50 of the 60 remaining images.

PubFig83数据集包含8300幅彩色图像，大小100x100，83位名人每人100幅图像。对于我们的PubFig83试验，我们将未对齐的图像转换到灰度级图像，屏蔽了83路识别任务的模型，使用了每个类别的3分割，20训练/20验证样本，同时运行2个TPU优化过程，共计有1200次模型评估。在屏蔽任务中，评分最高的配置，在第二阶段进行测试，每个类由5个训练分割组成，90幅训练图像，10幅测试图像。5个第二阶段的每个中，其每类的90幅图像由第一阶段的40幅图像和剩余60幅图像中的50幅构成。

The results of our model search on LFW are shown in Figure 2. The TPE algorithm exceeded the best random search view 1 performance within 200 trials, for both our random search and that carried out in Pinto & Cox (2011). TPE converged within 1000 trials to an error rate of 16.2%, significantly lower than the best configuration found by random search (21.9%). On LFW’s test data (view 2) the optimal TPE configuration also beats those found by our random search (84.5% vs. 79.2% accurate). The best configuration found by random search in Pinto & Cox (2011) does well on View 2 relative to View 1 (84.1% vs. approximately 79.5%) and is approximately as accurate as TPE’s best configuration on the test set. On PubFig83, the optimal TPE configuration outperforms the best random configuration found by our random search (86.5% vs 81.0% accurate) and the previous state of the art result (85.2%) Pinto et al. (2011).

在LFW上的模型搜索的结果，如图2所示。TPE算法用200次尝试超过了最好的随机搜索视角1性能，包括我们的随机搜索和Pinto & Cox (2011)中进行的试验。TPE在1000次尝试以内收敛到了错误率16.2%，比随机搜索找到的最好配置（21.9%）都要好。在LFW的测试数据（视角2）上，最优的TPE配置也比我们的随机搜索找到的最好结果要好(84.5% vs. 79.2% accurate)。Pinto & Cox (2011)中随机搜索找到的最好配置在视角2比视角1相对要好一些(84.1% vs. approximately 79.5%)，与TPE的最好配置在测试集上性能几乎一样准确。在PubFig83上，最优的TPE配置超过了我们的随机搜索找到的最好的随机配置(86.5% vs 81.0% accurate)，以及Pinto et al. (2011)的之前最好的结果(85.2%)。

### 5.2. Matching Hand-Tuning: CIFAR-10

Coates & Ng (2011) showed that single-layer approaches are competitive with the best multi-layer alternatives for 10-way object classification using the CIFAR-10 data set (Krizhevsky, 2009). The success of their single-layer approaches depends critically on correct settings for several hyperparameters governing pre-processing and feature extraction. CIFAR-10 images are low-resolution color images (32×32) but there are fifty thousand labeled images for training and ten thousand for testing. We performed model selection on the basis of a single random, stratified subset of ten thousand training examples.

Coates & Ng (2011)展示了，在使用CIFAR-10上的10路目标分类问题上，单层方法与最好的多层方法效果类似。他们的单层方法的成功，严重依赖于几个超参数的正确设置，在预处理和特征提取上很关键。CIFAR-10图像是低分辨率的彩色图像(32×32)，但有50000幅标注的图像用于训练，10000幅用于测试。我们用10000幅训练图像进行模型选择。

The results of TPE and random search are reported in Figure 3. TPE, starting from broad priors over a wide variety of processing pipelines, was able to match the performance of a skilled and motivated domain expert. With regards to the wall time of the automatic approach, our implementation of the pipeline was designed for GPU execution and the loss function required from 0 to 30 minutes. TPE found a configuration very similar to the one found by in Coates & Ng (2011) within roughly 24 hours of processing on 6 GPUs. Random search was not able to approach the same level of performance.

TPE和随机搜索的结果如图3所示。TPE的开始是很宽的先验，很多处理流水线，可以达到熟练的领域专家的性能。至于自动方法的时间，我们的实现是为GPU执行设计的，损失函数需要0到30分钟。TPE找到了一个配置，与Coates & Ng (2011)的配置很像，在6个GPUs上耗费了大约24个小时。随机搜索无法达到类似的性能水平。

## 6. Discussion

In this work, we have described a conceptual framework to support automated hyperparameter optimization, and demonstrated that it can be used to quickly recover state-of-the-art results on several unrelated image classification tasks from a large family of computer vision models, with no manual intervention. On each of three datasets used in our study we compared random search to a more sophisticated alternative: TPE. A priori, random search confers some advantages: it is trivially parallel, it is simpler to implement, and the independence of trials supports more interesting analysis (Bergstra & Bengio, 2012). However, our experiments found that TPE clearly outstrips random search in terms of optimization efficiency. TPE found best known configurations for each data set, and did so in only a small fraction of the time we allocated to random search. TPE, but not random search, was found to match the performance of manual tuning on the CIFAR-10 data set. With regards to the computational overhead of search, TPE took no more than a second or two to suggest new hyperparameter assignments, so it added a negligible computational cost to the experiments overall.

本文中，我们描述了一个概念框架，支持自动超参数搜索，证明了在几个不相关的图像分类任务中，从很大一族计算机视觉模型中，不需要手工干预，就能很快的找到目前最好的结果。用在我们的研究中的三个数据集的每个中，我们比较了随机搜索，和另外一个复杂的替代：TPE。先验，随机搜索有一些优势：并行，实现更简单，与尝试无关，支持更有趣的分析。但是，我们的试验发现，TPE在优化效率上明显超过了随机搜索。TPE对每个数据集都找到了目前最好的配置，而且所需的时间，比随机搜索所需的时间要少很多。在CIFAR-10数据集上，TPE的性能达到了手工调节的效果，但是随机搜索无法达到。至于搜索耗时，TPE建议新的超参数设置的时间是1-2秒，所以对试验增加的计算代价是可以忽略的。

This work opens many avenues for future work. One direction is to enlarge the model class to include a greater variety of components, and configuration strategies for those components. Many filter-learning and feature extraction techniques have been proposed in the literature beyond the core implemented in our experiment code base. Another direction is to improve the search algorithms. The TPE algorithm is conspicuously deficient in optimizing each hyperparameter independently of the others. It is almost certainly the case that the optimal values of some hyperparameters depend on settings of others. Algorithms such as SMAC (Hutter et al., 2011) that can represent such interactions might be significantly more effective optimizers than TPE. It might be possible to extend TPE to profitably employ non-factorial joint densities P(config|score). Relatedly, such optimization algorithms might permit the model description language to include constraints in the form of distributional parameters that are themselves optimizable quantities (e.g. uniform(0, lognormal(0, 1))). Another important direction for research in algorithm configuration is a recognition that not all loss function evaluations are equally expensive in terms of various limited resources, most notably in terms of computation time. All else being equal, configurations that are cheaper to evaluate should be favored (Snoek et al., 2012).

本文对未来的工作开辟了很多道路。一个方向是，增大模型的类别，包括更大范围内的组成部分，和这些部分的配置策略。除了我们试验的代码中实现的核心之外，文献中还提出了很多滤波器学习和特征提取技术。另一个方向是改进搜索算法。TPE算法在独立优化每个参数时效率明显不够高。几乎可以肯定的是，一些超参数的最优值，是依赖于其他值的设置的。像SMAC这样的算法，可以表示这种交互，可能比TPE的优化效率要更高。拓展TPE，采用non-factorial联合密度P(config|score)是可能的。相关的，这种优化算法可以使得模型描述语言包含约束，形式为分布参数，其本身也是可以优化的量，如uniform(0, lognormal(0, 1))。算法配置的另一个重要的方向是，损失函数评估的代价并不都是一样的，包括资源和计算时间。其他所有条件都一样的话，评估起来更廉价的配置，更受到青睐。

Our experiments dealt with the optimization of classification accuracy, but our approach extends quite naturally to the optimization (and constrained optimization via barrier techniques) of any real-valued criterion. We could search instead for the smallest or fastest model that meets a certain level of classification performance, or the best-performing model that meets the resource constraints imposed by a particular mobile platform. Having to perform such searches by hand may be daunting, but when the search space is encoded as a searchable model class, automatic optimization methods can be brought to bear.

我们的试验处理了分类准确率的优化，但该方法可以很自然的拓展到对任何实值规则的优化（通过barrier技术进行约束优化）。我们可以搜索最小的或最快的模型，满足一定水平的分类性能，或性能最好的模型，满足一定的资源约束，可能是由移动平台加上的约束。手工进行这样的搜索是不可行的，但如果搜索空间编码成了可搜索的模型类，就可以借用自动优化方法来进行。