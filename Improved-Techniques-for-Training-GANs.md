# Improved Techniques for Training GANs

Tim Salimans et al. OpenAI

## 0 Abstract

We present a variety of new architectural features and training procedures that we apply to the generative adversarial networks (GANs) framework. We focus on two applications of GANs: semi-supervised learning, and the generation of images that humans find visually realistic. Unlike most work on generative models, our primary goal is not to train a model that assigns high likelihood to test data, nor do we require the model to be able to learn well without using any labels. Using our new techniques, we achieve state-of-the-art results in semi-supervised classification on MNIST, CIFAR-10 and SVHN. The generated images are of high quality as confirmed by a visual Turing test: our model generates MNIST samples that humans cannot distinguish from real data, and CIFAR-10 samples that yield a human error rate of 21.3%. We also present ImageNet samples with unprecedented resolution and show that our methods enable the model to learn recognizable features of ImageNet classes.

我们提出很多新的架构特征和训练过程，可以应用在GANs框架中。我们聚焦在GANs的两类应用中：半监督学习，和图像生成。与大多数生成式模型的工作不同，我们的主要目标不是训练得到一个模型，给测试数据以很高的概率，我们也不需要模型不用任何标签就学习的非常好。使用我们的新技术，我们在MNIST、CIFAR-10和SVHN的半监督分类取得了目前最好的效果。生成的图像质量很高，这由视觉图灵测试确认得到：我们的模型生成的MNIST样本，人类不能与真实数据区分开来，生成的CIFAR-10样本得到的人类错误率为21.3%。我们还给出了ImageNet样本，分辨率比之前的都高，表明我们的方法使模型可以学习到ImageNet类别的可识别特征。

## 1 Introduction

Generative adversarial networks [1] (GANs) are a class of methods for learning generative models based on game theory. The goal of GANs is to train a generator network $G(z;θ^{(G)})$ that produces samples from the data distribution, $p_{data}(x)$, by transforming vectors of noise z as $x = G(z; θ^{(G)})$. The training signal for G is provided by a discriminator network D(x) that is trained to distinguish samples from the generator distribution $p_{model}(x)$ from real data. The generator network G in turn is then trained to fool the discriminator into accepting its outputs as being real.

GANs是一类基于博弈论学习生成式模型的方法。GANs的目标是训练一个生成器网络$G(z;θ^{(G)})$，可以从数据分布$p_{data}(x)$中生成样本，将噪声向量z转化成$x = G(z; θ^{(G)})$。G的训练信号是由判别网络D(x)给出的，这个网络是训练用于将生成器分布$p_{model}(x)$的样本与真实数据区分开来。生成器网络G的训练是使判别器无法区分生成的样本是真实的还是合成的。

Recent applications of GANs have shown that they can produce excellent samples [2, 3]. However, training GANs requires finding a Nash equilibrium of a non-convex game with continuous, high-dimensional parameters. GANs are typically trained using gradient descent techniques that are designed to find a low value of a cost function, rather than to find the Nash equilibrium of a game. When used to seek for a Nash equilibrium, these algorithms may fail to converge [4].

最近的GANs应用已经表明，可以生成非常好的样本。但是，训练GANs需要对一个非凸博弈用连续的高维参数找到Nash均衡。GANs一般使用梯度下降技术来训练，而梯度下降是用于找到损失函数的最小值，而不是为了找到博弈的Nash均衡。当用于寻找Nash均衡时，这些算法很可能无法收敛。

In this work, we introduce several techniques intended to encourage convergence of the GANs game. These techniques are motivated by a heuristic understanding of the non-convergence problem. They lead to improved semi-supervised learning peformance and improved sample generation. We hope that some of them may form the basis for future work, providing formal guarantees of convergence.

本文中，我们提出了几个技术，可以增加GANs博弈收敛的可能。这些技术是受非收敛问题的直觉理解推动的。这可以改进半监督学习的性能，改进样本生成。我们希望，其中的一些技术会成为未来工作的基础，给出收敛的正式保证。

All code and hyperparameters may be found at: https://github.com/openai/improved_gan

## 2 Related work

Several recent papers focus on improving the stability of training and the resulting perceptual quality of GAN samples [2, 3, 5, 6]. We build on some of these techniques in this work. For instance, we use some of the “DCGAN” architectural innovations proposed in Radford et al. [3], as discussed below.

最近的几篇文章关注了改进训练的稳定性，以及改进得到GANs样本的感官质量。本文中我们以其中的一些技术为基础。比如，我们使用了DCGAN[3]的一些架构创新，下面会进行讨论。

One of our proposed techniques, feature matching, discussed in Sec. 3.1, is similar in spirit to approaches that use maximum mean discrepancy [7, 8, 9] to train generator networks [10, 11]. Another of our proposed techniques, minibatch features, is based in part on ideas used for batch normalization [12], while our proposed virtual batch normalization is a direct extension of batch normalization.

我们提出的技术之一，特征匹配，在3.1节中进行了讨论，与使用最大平均差异训练生成器网络的思想类似。我们提出的另一项技术，minibatch特征，是部分基于BN的思想，而我们提出的虚拟BN是BN的直接延伸。

One of the primary goals of this work is to improve the effectiveness of generative adversarial networks for semi-supervised learning (improving the performance of a supervised task, in this case, classification, by learning on additional unlabeled examples). Like many deep generative models, GANs have previously been applied to semi-supervised learning [13, 14], and our work can be seen as a continuation and refinement of this effort.

本文的一个目标是，改进GANs对半监督学习的有效性（通过在额外的未标注的数据上学习，改进监督任务的性能，本文的例子为分类）。和很多深度生成式模型类似，GANs之前曾应用于半监督学习，我们的工作可以视作这些努力的延续和改进。

## 3 Toward Convergent GAN Training

Training GANs consists in finding a Nash equilibrium to a two-player non-cooperative game. Each player wishes to minimize its own cost function, $J^{(D)}(θ^{(D)},θ^{(G)})$ for the discriminator and $J^{(G)}(θ^{(D)},θ^{(G)})$ for the generator. A Nash equilibirum is a point $(θ^{(D)},θ^{(G)})$ such that $J^{(D)}$ is at a minimum with respect to $θ^{(D)}$ and $J^{(G)}$ is at a minimum with respect to $θ^{(G)}$. Unfortunately, finding Nash equilibria is a very difficult problem. Algorithms exist for specialized cases, but we are not aware of any that are feasible to apply to the GAN game, where the cost functions are non-convex, the parameters are continuous, and the parameter space is extremely high-dimensional.

训练GANs就是找到一个双玩家非配合博弈的Nash均衡。每个玩家都希望可以最小化其损失函数，对于判别器是$J^{(D)}(θ^{(D)},θ^{(G)})$，对于生成器是$J^{(G)}(θ^{(D)},θ^{(G)})$。Nash均衡是一个点$(θ^{(D)},θ^{(G)})$，在这个点上，损失函数$J^{(D)}$对参数$θ^{(D)}$的变化是最小的，$J^{(G)}$对于参数$θ^{(G)}$的变化是最小的。不幸的是，找到Nash均衡是非常困难的问题。对于特定情况，存在找到的算法，但在GAN博弈的情况下，我们还不知道有哪个可行的解，因为其代价函数是非凸的，参数是连续的，而且参数空间维度非常的高。

The idea that a Nash equilibrium occurs when each player has minimal cost seems to intuitively motivate the idea of using traditional gradient-based minimization techniques to minimize each player’s cost simultaneously. Unfortunately, a modification to $θ^{(D)}$ that reduces $J^{(D)}$ can increase $J^{(G)}$, and a modification to $θ^{(G)}$ that reduces $J^{(G)}$ can increase $J^{(D)}$. Gradient descent thus fails to converge for many games. For example, when one player minimizes xy with respect to x and another player minimizes −xy with respect to y, gradient descent enters a stable orbit, rather than converging to x = y = 0, the desired equilibrium point [15]. Previous approaches to GAN training have thus applied gradient descent on each player’s cost simultaneously, despite the lack of guarantee that this procedure will converge. We introduce the following techniques that are heuristically motivated to encourage convergence:

当每个玩家都其代价函数都最小，就达到Nash均衡的状态，这似乎直觉上可以使用传统的基于梯度的最小化方法，以同时最小化每个玩家的代价函数。不幸的是，使得$J^{(D)}$值降低的参数$θ^{(D)}$的变化，可能使$J^{(G)}$增加，而使$J^{(G)}$值降低的参数$θ^{(D)}$的变化，也可能使$J^{(D)}$增加。对于很多博弈来说，梯度下降都无法收敛。比如，当一个玩家以x为变量最小化xy，而另一个玩家以y为变量最小化-xy，那么梯度下降会进入一个稳定的轨道，而不是收敛到x=y=0这个理想的均衡点。之前的方法训练GAN时，对每个玩家的损失函数同时使用梯度下降，但这个过程是否能够收敛，则是缺少保证的。我们提出了下面的技术，直觉上来说，这会增加收敛的可能性：

### 3.1 Feature matching

Feature matching addresses the instability of GANs by specifying a new objective for the generator that prevents it from overtraining on the current discriminator. Instead of directly maximizing the output of the discriminator, the new objective requires the generator to generate data that matches the statistics of the real data, where we use the discriminator only to specify the statistics that we think are worth matching. Specifically, we train the generator to match the expected value of the features on an intermediate layer of the discriminator. This is a natural choice of statistics for the generator to match, since by training the discriminator we ask it to find those features that are most discriminative of real data versus data generated by the current model.

特征匹配对生成器指定一个新的目标函数，防止在当前的判别器上过度训练，这样就处理了GANs训练的不稳定性。新的目标函数不是直接最大化判别器的输出，而是要求生成器生成的数据与真实数据的统计相匹配，我们使用判别器只指定我们认为值得匹配的统计数据。特别的，我们训练生成器以匹配特征的期望值（在判别器的中间层）。这是生成器要匹配的统计值的自然选择，因为通过训练判别器，我们使其找到那些最能区分真实数据与目前模型生成的数据的特征。

Letting f(x) denote activations on an intermediate layer of the discriminator, our new objective for the generator is defined as: $||E_{x∼p_{data}} f(x) − E_{z∼p_z(z)}f(G(z))||_2^2$. The discriminator, and hence f(x), are trained in the usual way. As with regular GAN training, the objective has a fixed point where G exactly matches the distribution of training data. We have no guarantee of reaching this fixed point in practice, but our empirical results indicate that feature matching is indeed effective in situations where regular GAN becomes unstable.

令f(x)表示在判别器的中间层上的激活，生成器的新目标函数定义为：$||E_{x∼p_{data}} f(x) − E_{z∼p_z(z)}f(G(z))||_2^2$。判别器，也就是f(x)，按照通常的方式进行训练。至于常规的GAN训练，目标函数有一个固定点，在这个点上G与训练数据的分布相匹配。我们不保证能够达到这个固定点，但我们的经验结果表明，特征匹配在常规GAN不稳定的时候，确实是有效的。

### 3.2 Minibatch discrimination

One of the main failure modes for GAN is for the generator to collapse to a parameter setting where it always emits the same point. When collapse to a single mode is imminent, the gradient of the discriminator may point in similar directions for many similar points. Because the discriminator processes each example independently, there is no coordination between its gradients, and thus no mechanism to tell the outputs of the generator to become more dissimilar to each other. Instead, all outputs race toward a single point that the discriminator currently believes is highly realistic. After collapse has occurred, the discriminator learns that this single point comes from the generator, but gradient descent is unable to separate the identical outputs. The gradients of the discriminator then push the single point produced by the generator around space forever, and the algorithm cannot converge to a distribution with the correct amount of entropy. An obvious strategy to avoid this type of failure is to allow the discriminator to look at multiple data examples in combination, and perform what we call minibatch discrimination.

GAN生成器的一个主要的失败模式是，坍缩到一种参数设置，一直生成同一个点。当马上要坍缩到一种模式后，判别器的梯度对于很多类似的点都会指向类似的方向。由于判别器是独立处理每个样本的，所以梯度之间就没有任何协调，所以就没有一种机制，使生成器的输出要彼此不一样。反而是，所有的输出都朝向的点，是判别器目前相信是非常真实的。当达到坍缩的情况时，判别器学习到的情况是，这个单点是从生成器来的，但梯度下降不能将同样的输出区分开来。判别器的梯度使这个生成器产生的单点在这个空间中一直震荡，算法无法收敛到有正确熵的分布。防止这种类型的失误，一个很明显的策略是，使判别器判别的是多个数据样本的组合，进行的是minibatch判别。

Figure 1: Figure sketches how mini-batch discrimination works. Features f(x_i) from sample x_i are multiplied through a tensor T, and cross-sample distance is computed.

The concept of minibatch discrimination is quite general: any discriminator model that looks at multiple examples in combination, rather than in isolation, could potentially help avoid collapse of the generator. In fact, the successful application of batch normalization in the discriminator by Radford et al. [3] is well explained from this perspective. So far, however, we have restricted our experiments to models that explicitly aim to identify generator samples that are particularly close together. One successful specification for modelling the closeness between examples in a minibatch is as follows: Let $f(x_i) ∈ R^A$ denote a vector of features for input $x_i$, produced by some intermediate layer in the discriminator. We then multiply the vector $f(x_i)$ by a tensor $T ∈ R^{A×B×C}$, which results in a matrix $M_i ∈ R^{B×C}$. We then compute the $L_1$-distance between the rows of the resulting matrix $M_i$ across samples i ∈ {1,2,...,n} and apply a negative exponential (Fig. 1): $c_b (x_i, x_j)= exp(−||M_{i,b} − M_{j,b}||_{L_1} )∈ R$. The output $o(x_i)$ for this minibatch layer for a sample $x_i$ is then defined as the sum of the $c_b(x_i, x_j)$’s to all other samples:

minibatch判别的概念是非常一般性的：任何判别器模型，只要对多个样本进行组合判别，而不是单独判别，就可能对防止生成器坍缩。实际上，BN在判别器中的成功应用[3]也从这个观点从得到了很好的解释。但我们将试验局限在目标是对生成器生成的非常接近的样本进行辨别的模型。一个minibatch中，样本的接近性的建模的成功例子如下：如果$f(x_i) ∈ R^A$表示一个输入$x_i$的特征向量，这个输入是判别器的一些中间层生成的。然后我们将向量$f(x_i)$乘以张量$T ∈ R^{A×B×C}$，得到一个矩阵$M_i ∈ R^{B×C}$。然后我们在样本i ∈ {1,2,...,n}之间，计算得到的矩阵$M_i$的行之间的$L_1$-距离，并计算一个负值的指数（图1）：$c_b (x_i, x_j)= exp(−||M_{i,b} − M_{j,b}||_{L_1} )∈ R$。这个minibatch层的一个样本$x_i$的输出$o(x_i)$就定义为与所有其他样本的$c_b(x_i, x_j)$的和：

$$o(x_i)_b = \sum_{j=1}^n c_b(x_i, x_j) ∈ R$$
$$o(x_i) = [o(x_i)_1, o(x_i)_2, …, o(x_i)_B] ∈ R^B$$
$$o(X) ∈ R^{n×B}$$

Next, we concatenate the output $o(x_i)$ of the minibatch layer with the intermediate features $f(x_i)$ that were its input, and we feed the result into the next layer of the discriminator. We compute these minibatch features separately for samples from the generator and from the training data. As before, the discriminator is still required to output a single number for each example indicating how likely it is to come from the training data: The task of the discriminator is thus effectively still to classify single examples as real data or generated data, but it is now able to use the other examples in the minibatch as side information. Minibatch discrimination allows us to generate visually appealing samples very quickly, and in this regard it is superior to feature matching (Section 6). Interestingly, however, feature matching was found to work much better if the goal is to obtain a strong classifier using the approach to semi-supervised learning described in Section 5.

下面，我们将这个minibatch层的输出$o(x_i)$与其输入的中间特征$f(x_i)$拼接在一起，我们将这个结果送入判别器的下一层中。我们计算这些minibatch特征时，对来自生成器的样本和训练数据的样本是分别计算的。和之前一样，还是需要判别器对每个样本输出一个数，即其来自训练数据的可能性：判别器的任务还是将每个样本分类为真实数据或合成的数据，但现在可以使用minibatch中的其他样本作为附加信息。Minibatch判别使我们能很快的生成外观上很好的样本，在这一点上，是优于特征匹配的（第6节）。但有趣的是，如果目标是使用这种方法进行半监督的学习得到一个很强的分类器，特征匹配会得到更好的结果，这会在第5节叙述。

### 3.3 Historical averaging

When applying this technique, we modify each player’s cost to include a term $||θ-\frac{1}{t} \sum_{i=1}^t θ[i]||^2$ where θ[i] is the value of the parameters at past time i. The historical average of the parameters can be updated in an online fashion so this learning rule scales well to long time series. This approach is loosely inspired by the fictitious play [16] algorithm that can find equilibria in other kinds of games. We found that our approach was able to find equilibria of low-dimensional, continuous non-convex games, such as the minimax game with one player controlling x, the other player controlling y, and value function (f(x) − 1)(y − 1), where f(x) = x for x < 0 and $f(x) = x^2$ otherwise. For these same toy games, gradient descent fails by going into extended orbits that do not approach the equilibrium point.

当使用这种技术时，我们修正每个玩家的代价函数，增加了一项$||θ-\frac{1}{t} \sum_{i=1}^t θ[i]||^2$，其中θ[i]是过去时间i时参数的值。参数的历史平均可以以在线的形式更新，这样学习规则可以很好的适用于长期序列。这个方法是部分收到虚拟玩家算法[16]的启发，这个算法可以在其他类型的博弈中找到平衡点。我们发现，我们的方法可以找到低维、连续、非凸博弈的平衡点，比如minimax博弈，一个玩家控制x，另一个玩家控制y，价值函数为(f(x) − 1)(y − 1)，其中x<0时f(x) = x，否则$f(x) = x^2$。对于这样的博弈，梯度下降是不行的，其会在偏离平衡点的轨迹上不断延伸。

### 3.4 One-sided label smoothing

Label smoothing, a technique from the 1980s recently independently re-discovered by Szegedy et. al [17], replaces the 0 and 1 targets for a classifier with smoothed values, like .9 or .1, and was recently shown to reduce the vulnerability of neural networks to adversarial examples [18].

标签平滑是一种1980s的技术，最近被人重新发现利用，将分类器给出的0和1的信息进行平滑，如.9或.1，近期的工作表明，这会降低神经网络对对抗样本的脆弱性。

Replacing positive classification targets with α and negative targets with β, the optimal discriminator becomes $D(x) = \frac {α p_{data}(x) + β p_{model}(x)} {p_{data}(x) + p_{model}(x)}$. The presence of p_model in the numerator is problematic because, in areas where p_data is approximately zero and p_model is large, erroneous samples from
p_model have no incentive to move nearer to the data. We therefore smooth only the positive labels to α, leaving negative labels set to 0.

将正的分类结果替换成α，负的替换成β，最佳判别器变成$D(x) = \frac {α p_{data}(x) + β p_{model}(x)} {p_{data}(x) + p_{model}(x)}$。分子中的p_model是一个问题，因为在p_data近似为0，p_model很大的区域中，p_model中的错误样本没有激励来靠近数据。因此，我们只将正标签平滑为α，负标签设置为0。

### 3.5 Virtual batch normalization

Batch normalization greatly improves optimization of neural networks, and was shown to be highly effective for DCGANs [3]. However, it causes the output of a neural network for an input example x to be highly dependent on several other inputs x' in the same minibatch. To avoid this problem we introduce virtual batch normalization (VBN), in which each example x is normalized based on the statistics collected on a reference batch of examples that are chosen once and fixed at the start of training, and on x itself. The reference batch is normalized using only its own statistics. VBN is computationally expensive because it requires running forward propagation on two minibatches of data, so we use it only in the generator network.

BN极大的改进了神经网络的优化，在DCGANs[3]中非常有用。但是，BN会使神经网络对一个输入样本x的输出，高度依赖于同一个minibatch中的其他输入x'。为避免这个问题，我们提出了虚拟BN(VBN)，其中每个样本x的归一化，是基于一个参考batch的样本的统计数据，这个参考batch是在训练开始前一次性选择的，并固定下来的，而且也基于x本身进行归一化。参考batch的归一化是使用其自己的统计值。VBN计算量很大，因为需要在两个minibatch数据上运行前向传播，所以我们只在生成器网络中使用。

## 4 Assessment of image quality

Generative adversarial networks lack an objective function, which makes it difficult to compare performance of different models. One intuitive metric of performance can be obtained by having human annotators judge the visual quality of samples [2]. We automate this process using Amazon Mechanical Turk (MTurk), using the web interface in figure Fig. 2 (live at http://infinite-chamber-35121.herokuapp.com/cifar-minibatch/), which we use to ask annotators to distinguish between generated data and real data. The resulting quality assessments of our models are described in Section 6.

GAN缺少目标函数，这使其很难比较不同模型的性能。直觉上的一个判断标准是，让人类标注者判断样本的视觉质量。我们使用AMT使这个过程自动化，使用图2所示的网页界面，可以向标注者询问区分生成的数据和真实数据。我们模型得到的质量评估在第6章中叙述。

A downside of using human annotators is that the metric varies depending on the setup of the task and the motivation of the annotators. We also find that results change drastically when we give annotators feedback about their mistakes: By learning from such feedback, annotators are better able to point out the flaws in generated images, giving a more pessimistic quality assessment. The left column of Fig. 2 presents a screen from the annotation process, while the right column shows how we inform annotators about their mistakes.

使用人类标注者的不好的地方是，对不同的任务设置，和标注者动机的度量标准，度量标准会变化。我们还发现，当我们给标注者反馈其错误时，结果会变化很剧烈：标注者得到这样的反馈后，可以更好的指出生成的图像中的缺陷，给出一个更悲观的质量评估。图2的左边是标注过程的界面，而右边是我们告诉标注者其错误的界面。

As an alternative to human annotators, we propose an automatic method to evaluate samples, which we find to correlate well with human evaluation: We apply the Inception model [19] to every generated image to get the conditional label distribution p(y|x). Images that contain meaningful objects should have a conditional label distribution p(y|x) with low entropy. Moreover, we expect the model to generate varied images, so the marginal $\int p(y|x = G(z))dz$ should have high entropy. Combining these two requirements, the metric that we propose is: $exp(E_x KL(p(y|x)||p(y)))$, where we exponentiate results so the values are easier to compare. Our Inception score is closely related to the objective used for training generative models in CatGAN [14]: Although we had less success using such an objective for training, we find it is a good metric for evaluation that correlates very well with human judgment. We find that it’s important to evaluate the metric on a large enough number of samples (i.e. 50k) as part of this metric measures diversity.

作为人类标注者的替代，我们提出了一种自动方法来评估样本，我们发现这种方法与人类评估的相关性很好：我们对每个生成的图像使用Inception模型，得到条件标签分布p(y|x)。包含有意义的目标的图像，其条件标签分布p(y|x)的熵应当较低。而且，我们的模型应当产生不同的图像，所以边缘概率$\int p(y|x = G(z))dz$应当熵值较高。将这两个要求结合到一起，我们提出的度量标准为：$exp(E_x KL(p(y|x)||p(y)))$，这里对结果进行指数运算，所以其值更容易进行比较。Inception值与CatGAN中训练生成式模型的目标函数非常接近：虽然我们使用这样的目标函数进行训练的成功率不多，我们发现这用于评估还是不错的度量标准，与人类的评估结果相关性非常好。我们发现，在足够多的样本(50k)上评估这个度量标准，是很重要的，因为这个度量标准也部分度量了多样性。

## 5 Semi-supervised learning

Consider a standard classifier for classifying a data point x into one of K possible classes. Such a model takes in x as input and outputs a K-dimensional vector of logits {$l_1,...,l_K$}, that can be turned into class probabilities by applying the softmax: $p_{model} (y=j|x) = \frac {exp(l_j)} {\sum_{k=1}^K exp(l_k)}$. In supervised learning, such a model is then trained by minimizing the cross-entropy between the observed labels and the model predictive distribution $p_{model}(y|x)$.

考虑一个标准分类器，将数据点x分类成K个可能的类别。这样的模型以x为输入，输出一个K维向量的logits {$l_1,...,l_K$}，然后使用softmax函数转化成类别概率，$p_{model} (y=j|x) = \frac {exp(l_j)} {\sum_{k=1}^K exp(l_k)}$。在监督学习中，这样一个模型的训练，是对观察到的标签和模型预测的分布$p_{model}(y|x)$之间的交叉熵进行最小化。

We can do semi-supervised learning with any standard classifier by simply adding samples from the GAN generator G to our data set, labeling them with a new “generated” class y = K + 1, and correspondingly increasing the dimension of our classifier output from K to K + 1. We may then use p_model (y = K + 1 | x) to supply the probability that x is fake, corresponding to 1 − D(x) in the original GAN framework. We can now also learn from unlabeled data, as long as we know that it corresponds to one of the K classes of real data by maximizing log $p_{model} (y ∈ {1, . . . , K}|x)$. Assuming half of our data set consists of real data and half of it is generated (this is arbitrary), our loss function for training the classifier then becomes

我们用任意标准分类器，只要通过GAN生成器G为数据集增加样本，并给其一个新的“生成的”标签y=K+1，就可以进行半监督学习，对应的增加我们的分类器输出的维度，从K到K+1。我们可以使用p_model (y = K + 1 | x)来提供x是虚假的概率，对应着在原始GAN框架中的1-D(x)。我们现在就可以从未标注的数据中学习了，我们通过最大化log $p_{model} (y ∈ {1, . . . , K}|x)$，就可以知道对应着K类中的一类。假设我们的数据集的一半是真实数据，一半是生成的，我们的分类器损失函数成为

$$L = -E_{x,y∼p_{data}(x,y)} [log p_{model}(y|x)] - E_{x∼G} [log p_{model} (y=K+1|x)] = L_{supervised} + L_{unsupervised}$$

where:

$$L_{supervised} = -E_{x,y∼p_{data}(x,y)} log p_{model} (y|x,y<K+1)$$
$$L_{unsupervised} = -E_{x∼p_{data}(x)} log[1-p_{model} (y=K+1|x)] - E_{x∼G} log[p_{model} (y=K+1|x)]$$

where we have decomposed the total cross-entropy loss into our standard supervised loss function $L_{supervised}$ (the negative log probability of the label, given that the data is real) and an unsupervised loss $L_{unsupervised}$ which is in fact the standard GAN game-value as becomes evident when we substitute $D(x) = 1 − p_{model}(y = K + 1|x)$ into the expression:

其中我们将总交叉熵损失分解成标准的监督损失函数$L_{supervised}$（在数据是真实的情况下，标签的概率的-log），和一个无监督损失$L_{unsupervised}$，这实际上是标准的GAN博弈值，当我们将$D(x) = 1 − p_{model}(y = K + 1|x)$替换成下式的时候，就变得很明显：

$$L_{unsupervised} = −\{E_{x∼p_{data}(x)} log D(x) + E_{z∼noise} log(1 − D(G(z)))\}$$

The optimal solution for minimizing both $L_{supervised}$ and $L_{unsupervised}$ is to have $exp[l_j (x)] = c(x)p(y=j, x)∀j<K+1$ and $exp[l_{K+1}(x)] = c(x)p_G(x)$ for some undetermined scaling function c(x). The unsupervised loss is thus consistent with the supervised loss in the sense of Sutskever et al. [13], and we can hope to better estimate this optimal solution from the data by minimizing these two loss functions jointly. In practice, $L_{unsupervised}$ will only help if it is not trivial to minimize for our classifier and we thus need to train G to approximate the data distribution. One way to do this is by training G to minimize the GAN game-value, using the discriminator D defined by our classifier. This approach introduces an interaction between G and our classifier that we do not fully understand yet, but empirically we find that optimizing G using feature matching GAN works very well for semi-supervised learning, while training G using GAN with minibatch discrimination does not work at all. Here we present our empirical results using this approach; developing a full theoretical understanding of the interaction between D and G using this approach is left for future work.

要对$L_{supervised}$和$L_{unsupervised}$进行最小化，其最优解是使$exp[l_j (x)] = c(x)p(y=j, x)∀j<K+1$和$exp[l_{K+1}(x)] = c(x)p_G(x)$，c(x)是一些未确定的缩放函数。无监督的损失与监督损失在Sutskever等[13]的意义上是一致的，通过对这个损失函数进行联合最小化，我们可以从数据中更好的估计最优解。在实践中，只有在分类器最小化的过程并不那么容易的情况下，$L_{unsupervised}$才会有用，因此我们需要训练G来对数据分布进行近似。一种方法是，训练G以最小化GAN博弈值，使用分类器定义的判别器D。这种方法引入了G和我们的分类器之间的相互作用，目前我们还没有完全理解，但经验上来说，我们发现使用与GAN匹配的特征，对于半监督来说，效果很好，而使用带有minibatch区别的GAN训练G效果反而不好。这里我们给出使用这种方法得到的经验结果；未来我们还会给出，D和G使用这种方法的互相作用的理论上的理解。

Finally, note that our classifier with K + 1 outputs is over-parameterized: subtracting a general function f(x) from each output logit, i.e. setting $l_j(x) ← l_j(x) − f(x)∀j$, does not change the output of the softmax. This means we may equivalently fix $l_{K+1}(x) = 0, ∀x$, in which case $L_{supervised}$ becomes the standard supervised loss function of our original classifier with K classes, and our discriminator D is given by $D(x) = \frac {Z(x)} {Z(x)+1}$, where $Z(x) = \sum_{k=1}^K exp[l_k(x)]$.

最后，注意我们的K+1输出分类器是参数过多的：从每个输出的logit中减去一个通用函数f(x)，即，设$l_j(x) ← l_j(x) − f(x)∀j$，不会改变softmax的输出。这意味着，我们可以等价的设$l_{K+1}(x) = 0, ∀x$，这种情况下$L_{supervised}$成为K类原始分类器的标准监督损失函数，我们的判别器D由式$D(x) = \frac {Z(x)} {Z(x)+1}$给出，其中$Z(x) = \sum_{k=1}^K exp[l_k(x)]$。

### 5.1 Importance of labels for image quality

Besides achieving state-of-the-art results in semi-supervised learning, the approach described above also has the surprising effect of improving the quality of generated images as judged by human annotators. The reason appears to be that the human visual system is strongly attuned to image statistics that can help infer what class of object an image represents, while it is presumably less sensitive to local statistics that are less important for interpretation of the image. This is supported by the high correlation we find between the quality reported by human annotators and the Inception score we developed in Section 4, which is explicitly constructed to measure the “objectness” of a generated image. By having the discriminator D classify the object shown in the image, we bias it to develop an internal representation that puts emphasis on the same features humans emphasize. This effect can be understood as a method for transfer learning, and could potentially be applied much more broadly. We leave further exploration of this possibility for future work.

除了在半监督学习中得到了目前最好的结果，上述方法还有改进生成图像的质量的效果，人类标注者对其评判会提高。原因可能是，人类视觉系统更喜欢可以推理可以推理出一幅图像表示什么目标的图像统计值，而很可能对局部统计值不太敏感，这些局部统计值对于解释图像并不那么重要。我们发现人类标注者给出的图像质量和Inception分数之间与很高的相关性，支持了这个观点，Inception值就是为估计生成图像的objectness的。用判别器D对图像中的目标进行分类，其内部表示与人类强调的特征是一致的。这个效果可以理解为一种迁移学习的方法，可能得到更广的应用。未来我们将会进一步探索这种可能性。

## 6 Experiments

We performed semi-supervised experiments on MNIST, CIFAR-10 and SVHN, and sample generation experiments on MNIST, CIFAR-10, SVHN and ImageNet. We provide code to reproduce the majority of our experiments.

我们在MNIST, CIFAR-10和SVHN上进行半监督的试验，以及在MNIST, CIFAR-10, SVHN和ImageNet上进行样本生成试验。我们对主要试验都给出了代码。

### 6.1 MNIST

The MNIST dataset contains 60,000 labeled images of digits. We perform semi-supervised training with a small randomly picked fraction of these, considering setups with 20, 50, 100, and 200 labeled examples. Results are averaged over 10 random subsets of labeled data, each chosen to have a balanced number of examples from each class. The remaining training images are provided without labels. Our networks have 5 hidden layers each. We use weight normalization [20] and add Gaussian noise to the output of each layer of the discriminator. Table 1 summarizes our results.

MNIST数据集包含60000标注的数字图像。我们进行半监督训练，使用的是随机选择的小部分图像，只用20，50，100和200个标注的样本。得到的结果在10个标注数据的随机子集进行平均，每个子集都有各个类别的样本，数量比较均衡。剩余的训练数据是不提供标签的。我们的网络每个都有5层隐含层。我们使用权重归一化[20]，并给判别器每一层的输出增加了高斯噪声。表1总结了我们的结果。

Table 1: Number of incorrectly classified test examples for the semi-supervised setting on permuta- tion invariant MNIST. Results are averaged over 10 seeds.

Samples generated by the generator during semi-supervised learning using feature matching (Section 3.1) do not look visually appealing (left Fig. 3). By using minibatch discrimination instead (Section 3.2) we can improve their visual quality. On MTurk, annotators were able to distinguish samples in 52.4% of cases (2000 votes total), where 50% would be obtained by random guessing. Similarly, researchers in our institution were not able to find any artifacts that would allow them to distinguish samples. However, semi-supervised learning with minibatch discrimination does not produce as good a classifier as does feature matching.

生成器在半监督学习中使用特征匹配（3.1节）生成的样本看起来不怎么好看（图3左）。通过使用minibatch区分（3.2节），我们可以改进图像质量。在MTurk中，标注者可以区分52.4%情况下的样本（共2000个投票），其中50%可以通过随机猜测得到。类似的，研究者也没有发现任何artifacts使人们区分样本。但是，使用minibatch区分的半监督学习，与使用特征匹配的相比，并没有那么好。

Figure 3: (Left) samples generated by model dur- ing semi-supervised training. Samples can be clearly distinguished from images coming from MNIST dataset. (Right) Samples generated with minibatch discrimination. Samples are completely indistinguishable from dataset images.

### 6.2 CIFAR-10

CIFAR-10 is a small, well studied dataset of 32 × 32 natural images. We use this data set to study semi-supervised learning, as well as to examine the visual quality of samples that can be achieved. For the discriminator in our GAN we use a 9 layer deep convolutional network with dropout and weight normalization. The generator is a 4 layer deep CNN with batch normalization. Table 2 summarizes our results on the semi-supervised learning task.

CIFAR-10是一个小型数据集，研究的很多，由很多32×32自然图像组成。我们使用这个数据集来研究半监督学习，以及检查生成样本的图像质量。对于GAN中的判别器，我们使用9层深度卷积网络，使用了dropout和权重归一化。生成器是一个4层的深度CNN，使用了BN。表2总结了在这个半监督学习任务上的结果。

When presented with 50% real and 50% fake data generated by our best CIFAR-10 model, MTurk users correctly categorized 78.7% of images correctly. However, MTurk users may not be sufficiently familiar with CIFAR-10 images or sufficiently motivated; we ourselves were able to categorize images with > 95% accuracy. We validated the Inception score described above by observing that MTurk accuracy drops to 71.4% when the data is filtered by using only the top 1% of samples according to the Inception score. We performed a series of ablation experiments to demonstrate that our proposed techniques improve the Inception score, presented in Table 3. We also present images for these ablation experiments—in our opinion, the Inception score correlates well with our subjective judgment of image quality. Samples from the dataset achieve the highest value. All the models that even partially collapse have relatively low scores. We caution that the Inception score should be used as a rough guide to evaluate models that were trained via some independent criterion; directly optimizing Inception score will lead to the generation of adversarial examples [25].

在提供给MTurk用户的数据中，50%是真实数据，50%是我们最好的CIFAR-10模型生成的数据，用户可以正确归类78.7%的图像。但是，MTurk用户可能不太熟悉CIFAR-10图像，或动机不足；我们自己是可以以>95%的准确率将图像归类的。我们验证了上述的Inception分数，计算了样本的Inception分数后，只取出最高分数1%的图像，然后MTurk准确率降到了71.4%。我们进行了一系列分离试验，在表3中展示了我们提出的改进Inception分数的方法。我们还给出了分离试验的图像，我们认为，Inception分数与我们对图像质量的主观判断的相关性很好。数据集中的样本得到了最高的分数。部分坍缩的模型得到了相对较低的分数。应当注意，Inception分数应当用作评估通过不同原则训练出的模型的大致指南；直接对Inception分数进行优化，会导致生成对抗样本。

### 6.3 SVHN

For the SVHN data set, we used the same architecture and experimental setup as for CIFAR-10.

### 6.4 ImageNet

We tested our techniques on a dataset of unprecedented scale: 128 × 128 images from the ILSVRC2012 dataset with 1,000 categories. To our knowledge, no previous publication has applied a generative model to a dataset with both this large of a resolution and this large a number of object classes. The large number of object classes is particularly challenging for GANs due to their tendency to underestimate the entropy in the distribution. We extensively modified a publicly available implementation of DCGANs using TensorFlow [26] to achieve high performance, using a multi-GPU implementation. DCGANs without modification learn some basic image statistics and generate contiguous shapes with somewhat natural color and texture but do not learn any objects. Using the techniques described in this paper, GANs learn to generate objects that resemble animals, but with incorrect anatomy. Results are shown in Fig. 6.

我们在大规模数据集上测试了我们的技术：1000类的ILSVRC2012数据集，图像大小128×128。据我们所知，之前的工作都没有将生成式模型应用于这么大分辨率和这么多类别数量的数据集。这样多目标类别的数量，对于GANs是很有挑战的，因为GANs倾向于低估分布中的熵。我们对DCGANs使用Tensorflow的开源实现进行了大量修改，以得到更高的性能，使用了多GPU实现。没有修改的DCGANs学习到了一些基本的图像统计数据，生成了连续的形状，颜色和纹理也比较自然，但没有学习到任何目标。使用了本文中所述的技术，GANs学习生成的目标比较像动物，但解剖结构不正确。结果如图6所示。

## 7 Conclusion

Generative adversarial networks are a promising class of generative models that has so far been held back by unstable training and by the lack of a proper evaluation metric. This work presents partial solutions to both of these problems. We propose several techniques to stabilize training that allow us to train models that were previously untrainable. Moreover, our proposed evaluation metric (the Inception score) gives us a basis for comparing the quality of these models. We apply our techniques to the problem of semi-supervised learning, achieving state-of-the-art results on a number of different data sets in computer vision. The contributions made in this work are of a practical nature; we hope to develop a more rigorous theoretical understanding in future work.

GANs是一种很有希望的生成式模型，但现在存在训练不稳定的问题，并缺少评估度量标准。本文提出了两个问题的部分解决方案。我们提出了几种技术，以使得训练比较稳定，我们就可以训练之前不能训练的模型了。而且，我们提出的评估标准(Inception score)是比较模型质量的一个基础。我们将技术应用于半监督学习的问题，在几个数据集上得到了目前最好的结果。本文的贡献是比较通用的；未来我们希望给出更严格的理论理解。