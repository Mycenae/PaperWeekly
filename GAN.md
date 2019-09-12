# Generative Adversarial Nets

Ian J. Goodfellow et al. University of Montreal

## Abstract 摘要

We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.

我们提出一种新的框架，通过一种对抗的过程估计生成式模型，其中我们同时训练两个模型：一个生成式模型G，捕获数据分布，和一个判别式模型D，估计一个样本是来自训练样本，而不是来自G的概率。G的训练过程是使得D犯错的可能性最大化的过程。这个框架对应着一个minmax双玩家游戏。在任意函数G和D的空间中，存在唯一解，使得G恢复出训练数据的分布，D处处等于1/2。在G和D由多层感知机定义的情况下，整个系统可以用反向传播进行训练。在训练或样本生成的过程中，完全不需要用任何Markov链或展开的近似推理网络。试验中对生成的样本进行了定量和定性的评估，表明了这个框架的潜力。

## 1 Introduction

The promise of deep learning is to discover rich, hierarchical models [2] that represent probability distributions over the kinds of data encountered in artificial intelligence applications, such as natural images, audio waveforms containing speech, and symbols in natural language corpora. So far, the most striking successes in deep learning have involved discriminative models, usually those that map a high-dimensional, rich sensory input to a class label [14, 22]. These striking successes have primarily been based on the backpropagation and dropout algorithms, using piecewise linear units [19, 9, 10] which have a particularly well-behaved gradient. Deep generative models have had less of an impact, due to the difficulty of approximating many intractable probabilistic computations that arise in maximum likelihood estimation and related strategies, and due to difficulty of leveraging the benefits of piecewise linear units in the generative context. We propose a new generative model estimation procedure that sidesteps these difficulties. 

深度学习可以发现丰富的层次式的模型，表示人工智能应用中各种数据的概率分布，如自然图像，在自然语言语料库中包含语音、符号的音频。迄今为止，深度学习最令人兴奋的成功中，包含了判别式模型，通常是将高维度的，丰富的传感器输入映射到一个类别标签。这种成功主要是基于反向传播和dropout算法，使用分段线性的单元，有着表现很好的梯度。深度生成式模型的影响略小一些，主要因为近似很多很难处理的概率计算比较困难，这在最大似然估计和相关的策略中都有发现，还因为在生成式上下文中利用分段线性单元也比较困难。我们提出了一种新的生成式模型估计过程，回避了这些困难。

In the proposed adversarial nets framework, the generative model is pitted against an adversary: a discriminative model that learns to determine whether a sample is from the model distribution or the data distribution. The generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to the police, trying to detect the counterfeit currency. Competition in this game drives both teams to improve their methods until the counterfeits are indistiguishable from the genuine articles.

在提出的对抗网络框架中，生成式模型与一个对手进行对抗：一个判别式模型，学习判别一个模型是符合模型分布的，还是符合数据分布的。生成式模型可以类比成造假者，试图生成假币，在使用中不被检测到，而判别式模型可以类比成警察，试图甄别出假币。这个游戏中的竞赛，驱动两个小组来改进其方法，直到造假者与真实的物品无法区分。

This framework can yield specific training algorithms for many kinds of model and optimization algorithm. In this article, we explore the special case when the generative model generates samples by passing random noise through a multilayer perceptron, and the discriminative model is also a multilayer perceptron. We refer to this special case as adversarial nets. In this case, we can train both models using only the highly successful backpropagation and dropout algorithms [17] and sample from the generative model using only forward propagation. No approximate inference or Markov chains are necessary.

这个框架可以对很多模型和优化算法，产生特定的训练算法。在本文中，我们探索了一种特殊情况，即生成式模型通过将噪声输入到多层感知机中，以生成样本，判别器模型也是一个多层感知机。我们称这种特殊情况为对抗网络。在这种情况中，我们可以只使用非常成功的反向传播和dropout算法来训练模型，生成式模型只需要使用前向传播就得到样本。不需要近似估计或Markov链。

## 2 Related work

An alternative to directed graphical models with latent variables are undirected graphical models with latent variables, such as restricted Boltzmann machines (RBMs) [27, 16], deep Boltzmann machines (DBMs) [26] and their numerous variants. The interactions within such models are represented as the product of unnormalized potential functions, normalized by a global summation/integration over all states of the random variables. This quantity (the partition function) and its gradient are intractable for all but the most trivial instances, although they can be estimated by Markov chain Monte Carlo (MCMC) methods. Mixing poses a significant problem for learning algorithms that rely on MCMC [3, 5].

带有隐藏变量的有向图模型的一个替代是带有隐藏变量的无向图模型，如有限Boltzmann机(RBMs)，深度Boltzmann机(DBMs)及其大量变体。这些模型内的相互作用表示为下面的结果，即未归一化的势函数，由全局和/整合，在随机变量的所有状态下进行的归一化。这个量（分割函数）和其梯度在所有情况下都是难以处理的，虽然可以用Markov链蒙特卡洛方法进行估计。混合对依赖MCMC的学习算法提出量一个显著的问题。

Deep belief networks (DBNs) [16] are hybrid models containing a single undirected layer and several directed layers. While a fast approximate layer-wise training criterion exists, DBNs incur the computational difficulties associated with both undirected and directed models.

DBNs是混合模型，包含量单个无向层和几个有向层。虽然逐层训练的快速近似是存在的，但DBNs带来量计算上的困难，这与无向和有向模型都是相关的。

Alternative criteria that do not approximate or bound the log-likelihood have also been proposed, such as score matching [18] and noise-contrastive estimation (NCE) [13]. Both of these require the learned probability density to be analytically specified up to a normalization constant. Note that in many interesting generative models with several layers of latent variables (such as DBNs and DBMs), it is not even possible to derive a tractable unnormalized probability density. Some models such as denoising auto-encoders [30] and contractive autoencoders have learning rules very similar to score matching applied to RBMs. In NCE, as in this work, a discriminative training criterion is employed to fit a generative model. However, rather than fitting a separate discriminative model, the generative model itself is used to discriminate generated data from samples a fixed noise distribution. Because NCE uses a fixed noise distribution, learning slows dramatically after the model has learned even an approximately correct distribution over a small subset of the observed variables.

还提出量另一种不近似或限制log似然的准则，如分数匹配[18]和噪声对比的估计(NCE)[13]。这两种方法都需要学习的概率密度解析上指定一个归一化常数。注意，在很多有趣的带有几层隐藏变量的生成式模型（如DBNs和DBMs）中，推导出一个可行的未归一化的概率密度甚至都是不可能的。一些模型，如去噪的自动编码机和对比自动编码机，其学习规则与分数匹配应用到RBM中非常类似。在NCE中，和在本文中一样，采用了一种判别式训练规则，来适配一个生成式模型。但是，生成式模型没有适配分离的判别式模型，其本身就用于判别生成的数据与样本。因为NCE使用来固定的噪声分布，模型在观察到的变量上的小型子集上学习了近似正确的分布后，仍然速度非常慢。

Finally, some techniques do not involve defining a probability distribution explicitly, but rather train a generative machine to draw samples from the desired distribution. This approach has the advantage that such machines can be designed to be trained by back-propagation. Prominent recent work in this area includes the generative stochastic network (GSN) framework [5], which extends generalized denoising auto-encoders [4]: both can be seen as defining a parameterized Markov chain, i.e., one learns the parameters of a machine that performs one step of a generative Markov chain. Compared to GSNs, the adversarial nets framework does not require a Markov chain for sampling. Because adversarial nets do not require feedback loops during generation, they are better able to leverage piecewise linear units [19, 9, 10], which improve the performance of backpropagation but have problems with unbounded activation when used in a feedback loop. More recent examples of training a generative machine by back-propagating into it include recent work on auto-encoding variational Bayes [20] and stochastic backpropagation [24].

最后，一些技术没有涉及到显式的定义一个概率分布，而是训练一个生成式模型，从期望的分布中抽取样本。这种方法有一些优势，如这种模型可以设计用反向传播训练。这个领域最近的工作包括，生成式随机网络(GSN)，拓展了推广的去噪自动编码机：两个都可以看作是定义了一个参数化的Markov链，即，学习了一个机器的参数，进行一步生成式Markov链。与GSN相比，对抗网络框架不需要Markov链来取样。因为对抗网络在生成时不需要反馈回路，它们可以更好的利用分段线性单元，这改进来反向传播的性能，但在用于一个反馈回路时，有激活无界的问题。用反向传播训练一个生成式机器的最近例子，包括自动编码变分贝叶斯和随机反向传播。

## 3 Adversarial nets 对抗网络

The adversarial modeling framework is most straightforward to apply when the models are both multilayer perceptrons. To learn the generator’s distribution pg over data x, we define a prior on input noise variables pz(z), then represent a mapping to data space as G(z;θg), where G is a differentiable function represented by a multilayer perceptron with parameters θg . We also define a second multilayer perceptron D(x; θd) that outputs a single scalar. D(x) represents the probability that x came from the data rather than pg. We train D to maximize the probability of assigning the correct label to both training examples and samples from G. We simultaneously train G to minimize log(1 − D(G(z))):

当两个模型都是多层感知机时，对抗建模的框架可以得到最直接的应用。为在数据x上学习到生成器的分布pg，我们在输入噪声变量上定义了一个先验pz(z)，然后表示到数据空间的映射为G(z;θg)，其中G是可微分的函数，表示为多层感知机，参数为θg。我们还定义了第二个多层感知机D(x; θd)，输出一个单独标量。D(x)表示x是从数据来的，而不是从pg来的的概率。我们训练D，使其最大化对数据正确分类的概率，数据包括训练样本和G得到的样本。我们同时训练G，以最小化log(1 − D(G(z)))：

In other words, D and G play the following two-player minimax game with value function V (G, D): 换句话说，D和G进行下面的双玩家minmax博弈，价值函数为V(G,D):

$$min_G max_D V(D,G) = E_{x∼pdata(x)}[logD(x)]+E_{z∼pz(z)}[log(1−D(G(z)))]$$(1)

In the next section, we present a theoretical analysis of adversarial nets, essentially showing that the training criterion allows one to recover the data generating distribution as G and D are given enough capacity, i.e., in the non-parametric limit. See Figure 1 for a less formal, more pedagogical explanation of the approach. In practice, we must implement the game using an iterative, numerical approach. Optimizing D to completion in the inner loop of training is computationally prohibitive, and on finite datasets would result in overfitting. Instead, we alternate between k steps of optimizing D and one step of optimizing G. This results in D being maintained near its optimal solution, so long as G changes slowly enough. This strategy is analogous to the way that SML/PCD [31, 29] training maintains samples from a Markov chain from one learning step to the next in order to avoid burning in a Markov chain as part of the inner loop of learning. The procedure is formally presented in Algorithm 1.

下一节中，我们给出对抗网络的理论分析，表明只要G和D有足够的容量（即，在非参数化的限制下），训练准则使得恢复数据生成的分布成为可能。图1给出来一个本方法不太正式的，描述性的解释。实践中，我们必须用一种迭代的数值的方法实现这个博弈。最优化D以完成训练的内层循环，其计算量高到让人无法接受，在有限的数据集上回导致过拟合。我们没有这样做，而是首先进行k步优化D，然后进行一步优化G，交替进行。这样得到的结果是，只要G的变化足够的慢，那么D就维持在最有解附近。这种策略与SML/PCD [31,29]的训练可以类比。正式过程如算法1所示。

Algorithm 1 Minibatch stochastic gradient descent training of generative adversarial nets. The number of steps to apply to the discriminator, k, is a hyperparameter. We used k = 1, the least expensive option, in our experiments.

In practice, equation 1 may not provide sufficient gradient for G to learn well. Early in learning, when G is poor, D can reject samples with high confidence because they are clearly different from the training data. In this case, log(1 − D(G(z))) saturates. Rather than training G to minimize log(1 − D(G(z))) we can train G to maximize log D(G(z)). This objective function results in the same fixed point of the dynamics of G and D but provides much stronger gradients early in learning.

在实践中，式1可能不会提供足够的梯度，使G学习的很好。在学习的早期，当G的效果还很差时，D可能以很高的置信度拒绝样本，因为这些样本与训练数据很明显不一样。在这种情况下，log(1 − D(G(z)))会饱和。我们训练G就不以最小化log(1 − D(G(z)))为目标，而是以最大化log D(G(z))为目标。这个目标函数会对G和D的动力得到相同的定点，但在训练初期却给出强的多的梯度。

## 4 Theoretical Results 理论结果

The generator G implicitly defines a probability distribution pg as the distribution of the samples G(z) obtained when z ∼ pz. Therefore, we would like Algorithm 1 to converge to a good estimator of pdata, if given enough capacity and training time. The results of this section are done in a non- parametric setting, e.g. we represent a model with infinite capacity by studying convergence in the space of probability density functions.

生成器G隐式的定义来一个概率分布pg，作为以z ∼ pz得到的样本G(z)的分布。因此，我们希望，在给定足够的容量和训练时间下，算法1收敛到pdata的很好的估计值。本节的结果是以非参数的设置得到的，即，我们认为模型有无限的容量，研究在概率密度函数空间中的收敛性。

We will show in section 4.1 that this minimax game has a global optimum for pg = pdata. We will then show in section 4.2 that Algorithm 1 optimizes Eq 1, thus obtaining the desired result.

我们在4.1节给出，这个minmax博弈有一个全局最优点pg = pdata。在4.2节我们会表明，算法1会优化式1，因此得到希望的结果。

### 4.1 Global Optimality of pg = pdata

### 4.2 Convergence of Algorithm 1

In practice, adversarial nets represent a limited family of pg distributions via the function G(z; θg ), and we optimize θg rather than pg itself. Using a multilayer perceptron to define G introduces multiple critical points in parameter space. However, the excellent performance of multilayer perceptrons in practice suggests that they are a reasonable model to use despite their lack of theoretical guarantees.

实践中，对抗网络通过函数G(z; θg )来表示有限的pg分布，我们优化θg，而不是优化pg本身。使用一个多层感知机来定义G，会在参数空间中引入多个关键点。但是，多层感知机的优异表现说明，虽然缺少理论上的保证，但这是一个可行的模型。

## 5 Experiments 试验

We trained adversarial nets in a range of datasets including MNIST[23], the Toronto Face Database (TFD) [28], and CIFAR-10 [21]. The generator nets used a mixture of rectifier linear activations [19, 9] and sigmoid activations, while the discriminator net used maxout [10] activations. Dropout [17] was applied in training the discriminator net. While our theoretical framework permits the use of dropout and other noise at intermediate layers of the generator, we used noise as the input to only the bottommost layer of the generator network.

我们在几个数据集上训练对抗网络，包括MNIST，Toronto Face Database，和CIFAR10。生成器使用ReLU和Sigmoid的混合，判别器使用maxout激活。判别器网络的训练中使用了Dropout。在我们的理论框架中，生成器的中间层允许使用dropout和其他噪声，但我们只在生成器网络的最上层使用噪声作为输入。

We estimate probability of the test set data under pg by fitting a Gaussian Parzen window to the samples generated with G and reporting the log-likelihood under this distribution. The σ parameter of the Gaussians was obtained by cross validation on the validation set. This procedure was introduced in Breuleux et al. [8] and used for various generative models for which the exact likelihood is not tractable [25, 3, 5]. Results are reported in Table 1. This method of estimating the likelihood has somewhat high variance and does not perform well in high dimensional spaces but it is the best method available to our knowledge. Advances in generative models that can sample but not estimate likelihood directly motivate further research into how to evaluate such models.

我们估计pg下测试集数据的概率，方法是对G生成的样本适配一个高斯Parzen窗，给出这个分布下的log概率。高斯分布的参数σ通过验证集的交叉验证得到。这个过程由Breuleux等提出，在很多生成式模型中使用，而其确切的似然是不可得到的。结果如表1所示。这种估计似然的方法，其方差会有些高，在高维空间中表现不是很好，但据我们所知，这是最好的可用模型了。生成式模型的进展，可以采样但不能估计似然，这会直接激励未来怎样评估这样的模型的研究。

Table 1: Parzen window-based log-likelihood estimates.

In Figures 2 and 3 we show samples drawn from the generator net after training. While we make no claim that these samples are better than samples generated by existing methods, we believe that these samples are at least competitive with the better generative models in the literature and highlight the potential of the adversarial framework.

在图2和图3中，我们给出生成器网络在训练后，生成的数据的样本。我们并没有说，这些样本比现有的方法生成的样本要好，但我们相信，这些样本与更好的生成式模型是有的一比的，强调的是对抗框架的潜力。

Figure 2: Visualization of samples from the model.

Figure 3: Digits obtained by linearly interpolating between coordinates in z space of the full model.

## 6 Advantages and disadvantages

This new framework comes with advantages and disadvantages relative to previous modeling frameworks. The disadvantages are primarily that there is no explicit representation of pg (x), and that D must be synchronized well with G during training (in particular, G must not be trained too much without updating D, in order to avoid “the Helvetica scenario” in which G collapses too many values of z to the same value of x to have enough diversity to model pdata), much as the negative chains of a Boltzmann machine must be kept up to date between learning steps. The advantages are that Markov chains are never needed, only backprop is used to obtain gradients, no inference is needed during learning, and a wide variety of functions can be incorporated into the model. Table 2 summarizes the comparison of generative adversarial nets with other generative modeling approaches.

这个新的框架与之前的模型框架相比，有优点，也有缺点。其缺点主要是，没有pg(x)的显式表示，以及D在训练过程中必须与G很好的同步（特别是，在没有更新D之前，G不能训练的太多，以避免Helvetica情形，mode collapse）。其优点是再也不需要Markov链了，只需要反向传播来得到梯度，在学习过程中不需要推理，模型可以使用很多类型的函数。表2总结了GAN与其他生成式模型方法的比较。

Table 2: Challenges in generative modeling: a summary of the difficulties encountered by different approaches to deep generative modeling for each of the major operations involving a model.

The aforementioned advantages are primarily computational. Adversarial models may also gain some statistical advantage from the generator network not being updated directly with data examples, but only with gradients flowing through the discriminator. This means that components of the input are not copied directly into the generator’s parameters. Another advantage of adversarial networks is that they can represent very sharp, even degenerate distributions, while methods based on Markov chains require that the distribution be somewhat blurry in order for the chains to be able to mix between modes.

之前提到的优点主要是计算上的。对抗模型还会得到一些统计上的优势，来源是，生成器网络不是直接用数据样本更新的，而只是用通过判别起的梯度。这意味着输入的部分没有直接拷贝到生成器的参数中。对抗网络的另一个优势是，它们可以表示非常尖锐甚至降质的分布，而基于Markov链的方法需要的分布必须有某种成都的模糊，以使得链可以在模式之间进行混合。

## 7 Conclusions and future work

This framework admits many straightforward extensions: 这个框架有一些很直接的拓展：

- A conditional generative model p(x|c) can be obtained by adding c as input to both G and D. 对G和D增加一个条件c，可以得到条件生成式模型p(x|c)。

- Learned approximate inference can be performed by training an auxiliary network to predict z given x. This is similar to the inference net trained by the wake-sleep algorithm [15] but with the advantage that the inference net may be trained for a fixed generator net after the generator net has finished training. 学习到的近似的推理可以通过训练一个辅助网络进行，在给定x的情况下预测z。这与用wake-sleep算法训练得到的推理网络很类似，但其优点是在生成器网络结束训练后，推理网络可以对一个固定的生成器网络训练，

- One can approximately model all conditionals p(xS | x!S) where S is a subset of the indices of x by training a family of conditional models that share parameters. Essentially, one can use adversarial nets to implement a stochastic extension of the deterministic MP-DBM [11].

- Semi-supervised learning: features from the discriminator or inference net could improve performance of classifiers when limited labeled data is available. 半监督学习：在标注的数据有限时，判别器的特征可以改进分类器的性能。

- Efficiency improvements: training could be accelerated greatly by divising better methods for coordinating G and D or determining better distributions to sample z from during training. 效率改进：通过设计更好的协调G和D的方法，或确定更好的从z取样的方法，训练可以得到极大的加速。

This paper has demonstrated the viability of the adversarial modeling framework, suggesting that these research directions could prove useful. 本文给出了对抗模型框架的可行性，说明这些研究方向会是有用的。