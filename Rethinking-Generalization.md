# Understanding Deep Learning Requires Rethinking Generalization

Chiyuan Zhang et. al. MIT, Google Brain etc.

## 0. Abstract

Despite their massive size, successful deep artificial neural networks can exhibit a remarkably small difference between training and test performance. Conventional wisdom attributes small generalization error either to properties of the model family, or to the regularization techniques used during training.

尽管规模巨大，成功的DNN在训练性能和测试性能上的差异可以非常小。传统的智慧将小的泛化误差归因于模型的性质，或在训练过程中的泛化技术。

Through extensive systematic experiments, we show how these traditional approaches fail to explain why large neural networks generalize well in practice. Specifically, our experiments establish that state-of-the-art convolutional networks for image classification trained with stochastic gradient methods easily fit a random labeling of the training data. This phenomenon is qualitatively unaffected by explicit regularization, and occurs even if we replace the true images by completely unstructured random noise. We corroborate these experimental findings with a theoretical construction showing that simple depth two neural networks already have perfect finite sample expressivity as soon as the number of parameters exceeds the number of data points as it usually does in practice.

通过广泛的系统实验，我们证明了，这些传统方法在解释为什么大型神经网络在实际中泛化的非常好中，是失败的。具体的，我们的实验确立了，图像分类的目前最好的CNN，用SGD方法进行训练，很容易适应训练数据的随机标签混洗情况。这种现象通过显式的正则化，在定性上是没有受到影像的，即使我们将真实的图像替换成完全没有结构的随机噪声，也会发生。我们用一个理论构建来证明了这些实验发现，表明，简单的两层神经网络已经有了完美的有限样本表达能力，只要参数数量超过了数据点的数量，这也是在⌚实践中通常做的。

We interpret our experimental findings by comparison with traditional models. 我们通过与传统模型比较，来解释我们的实验发现。

## 1. Introduction

Deep artificial neural networks often have far more trainable model parameters than the number of samples they are trained on. Nonetheless, some of these models exhibit remarkably small generalization error, i.e., difference between “training error” and “test error”. At the same time, it is certainly easy to come up with natural model architectures that generalize poorly. What is it then that distinguishes neural networks that generalize well from those that don’t? A satisfying answer to this question would not only help to make neural networks more interpretable, but it might also lead to more principled and reliable model architecture design.

DNN的可训练模型参数数量，通常比训练样本数量要大的多。尽管如此，一些模型表现出了非常小的泛化误差，即，在训练误差和测试误差之间的差异。同时，也很容易得出很自然的模型架构，其泛化性能很差。那么，泛化性能好的网络，与泛化性能不好的神经网络，其区别在于哪里呢？对这个问题的满意回答，不仅会使得神经网络更具有可解释性，而且会带来更有准则的、更加可靠的模型架构设计。

To answer such a question, statistical learning theory has proposed a number of different complexity measures that are capable of controlling generalization error. These include VC dimension (Vapnik, 1998), Rademacher complexity (Bartlett & Mendelson, 2003), and uniform stability (Mukherjee et al., 2002; Bousquet & Elisseeff, 2002; Poggio et al., 2004). Moreover, when the number of parameters is large, theory suggests that some form of regularization is needed to ensure small generalization error. Regularization may also be implicit as is the case with early stopping.

为回答这样一个问题，统计学习理论已经提出了几个不同的复杂度度量，可以控制泛化误差。这包括VC维，Rademacher复杂度，和统一稳定性。而且，当参数数量很大时，理论表明，需要一些形式的正则化，以确保泛化误差很小。在早停的情况中，正则化也可能是隐式的。

### 1.1 Our Contributions

In this work, we problematize the traditional view of generalization by showing that it is incapable of distinguishing between different neural networks that have radically different generalization performance.

本文中，我们证明泛化的传统视角是有问题的，证明了其不能区分不同的神经网络，这些网络的泛化性能差异很大。

**Randomization tests**. At the heart of our methodology is a variant of the well-known randomization test from non-parametric statistics (Edgington & Onghena, 2007). In a first set of experiments, we train several standard architectures on a copy of the data where the true labels were replaced by random labels. Our central finding can be summarized as:

我们方法论的核心，是著名的非参数统计的随机化测试的一个变体。在第一组实验中，我们在一些复制数据中训练几种标准架构，其中真实的标签替换成了随机标签。我们的核心发现可以总结为：

Deep neural networks easily fit random labels. DNN可以很容易的适应随机标签。

More precisely, when trained on a completely random labeling of the true data, neural networks achieve 0 training error. The test error, of course, is no better than random chance as there is no correlation between the training labels and the test labels. In other words, by randomizing labels alone we can force the generalization error of a model to jump up considerably without changing the model, its size, hyperparameters, or the optimizer. We establish this fact for several different standard architectures trained on the CIFAR10 and ImageNet classification benchmarks. While simple to state, this observation has profound implications from a statistical learning perspective:

更确切的，当在真实数据的随机标签上进行训练时，神经网络可以获得0训练误差。当然，测试误差与随机猜测类似，因为训练标签和测试标签毫无关系。换句话说，通过标签混洗，我们可以迫使一个模型的泛化误差显著增加，而不改变模型，其大小，超参数，或优化器。我们在CIFAR10和ImageNet分类标准测试中，训练几个不同的标准架构，来确立这个事实。说起来简单，这个观察结果从统计学习的角度有很深远的影响：

- The effective capacity of neural networks is sufficient for memorizing the entire data set. 神经网络的有效容量足以记住整个训练数据集。

- Even optimization on random labels remains easy. In fact, training time increases only by a small constant factor compared with training on the true labels. 即使是在随机标签上进行优化也是很容易的。实际上，与在真实标签上进行训练比较起来，训练时间只增加了很少的固定时间。

- Randomizing labels is solely a data transformation, leaving all other properties of the learning problem unchanged. 标签混洗只是一个数据变换，这个学习问题的其他所有性质都没有变化。

Extending on this first set of experiments, we also replace the true images by completely random pixels (e.g., Gaussian noise) and observe that convolutional neural networks continue to fit the data with zero training error. This shows that despite their structure, convolutional neural nets can fit random noise. We furthermore vary the amount of randomization, interpolating smoothly between the case of no noise and complete noise. This leads to a range of intermediate learning problems where there remains some level of signal in the labels. We observe a steady deterioration of the generalization error as we increase the noise level. This shows that neural networks are able to capture the remaining signal in the data, while at the same time fit the noisy part using brute-force.

将第一组实验实验进行拓展，我们还将真实图像替换成完全的随机像素（如，高斯噪声），观察到CNN仍然可以对数据进行很好的拟合，达到0训练误差。这表明，不管什么结构，CNN可以拟合随机噪声。我们进一步变化随机化的程度，在没有噪声和完全噪声之间进行平滑插值。这带来了一些中间的学习问题，在标签中保留了一定程度的信号。我们观察到，在增加噪声水平时，泛化误差会稳定的恶化。这表明，神经网络可以捕获数据中的剩余信号，而同时使用暴力对含噪部分进行拟合。

We discuss in further detail below how these observations rule out all of VC-dimension, Rademacher complexity, and uniform stability as possible explanations for the generalization performance of state-of-the-art neural networks.

我们在下面进一步详细讨论，对于目前最好的神经网络的泛化性能，这些观察怎样排除了VC维度，Rademacher复杂度，统一稳定性作为可能的解释。

**The role of explicit regularization**. If the model architecture itself isn’t a sufficient regularizer, it remains to see how much explicit regularization helps. We show that explicit forms of regularization, such as weight decay, dropout, and data augmentation, do not adequately explain the generalization error of neural networks. Put differently:

**显式正则化的角色**。如果模型架构本身并不是一个充分的正则化器，那么显式正则化可以有多少帮助，值得去看一下。我们证明了，显式形式的正则化，比如权重衰减，dropout，和数据扩增，不足以解释神经网络的泛化误差。换句话说：

Explicit regularization may improve generalization performance, but is neither necessary nor by itself sufficient for controlling generalization error. 显式正则化会改进泛化性能，但其本身对于控制泛化误差来说，既不是必须的，也不是足够的。

In contrast with classical convex empirical risk minimization, where explicit regularization is necessary to rule out trivial solutions, we found that regularization plays a rather different role in deep learning. It appears to be more of a tuning parameter that often helps improve the final test error of a model, but the absence of all regularization does not necessarily imply poor generalization error. As reported by Krizhevsky et al. (2012), l2-regularization (weight decay) sometimes even helps optimization, illustrating its poorly understood nature in deep learning.

在经典的凸经验风险最小化中，显式的正则化足以排除无意义的解，与之相比，我们发现在深度学习中，正则化扮演的角色则非常不一样。其作用似乎更像是一个调节参数，经常可以帮助改进一个模型的最终测试误差，但即使没有任何正则化，也不一定会得到很差的泛化误差。在Krizhevsky等的文章中，l2-正则化（权重衰减）有时候甚至对优化有帮助，说明深度学习的理解本质较差。

**Finite sample expressivity**. We complement our empirical observations with a theoretical construction showing that generically large neural networks can express any labeling of the training data. More formally, we exhibit a very simple two-layer ReLU network with p = 2n + d parameters that can express any labeling of any sample of size n in d dimensions. A previous construction due to Livni et al. (2014) achieved a similar result with far more parameters, namely, O(dn). While our depth 2 network inevitably has large width, we can also come up with a depth k network in which each layer has only O(n/k) parameters.

**有限样本的表达能力**。对于我们的经验观察，我们用一个理论构建来进行补充，证明了一般性的神经网络可以对训练数据的任意标签进行表达。更正式的，我们展示了一个非常简单的2层ReLU网络，参数量为p = 2n + d，可以对规模为n维度为d的任意样本的任意标签进行表达。Livni等之间的构建也获得了类似的结果，其参数量要多的多，即O(dn)。我们的深度为2的网络有很大的宽度，我们也可以得出一个深度为k的网络，其中每层只有O(n/k)个参数。

While prior expressivity results focused on what functions neural nets can represent over the entire domain, we focus instead on the expressivity of neural nets with regards to a finite sample. In contrast to existing depth separations (Delalleau & Bengio, 2011; Eldan & Shamir, 2016; Telgarsky, 2016; Cohen & Shashua, 2016) in function space, our result shows that even depth-2 networks of linear size can already represent any labeling of the training data.

之前的表达能力结果聚焦在，神经网络在整个域中可以表示什么函数，而我们则聚焦在，神经网络对有限样本的表示能力。与已有的在函数空间的深度分割相比，我们的结果表明，即使是深度为2的网络，已经可以训练数据的任意标签。

**The role of implicit regularization**. While explicit regularizers like dropout and weight-decay may not be essential for generalization, it is certainly the case that not all models that fit the training data well generalize well. Indeed, in neural networks, we almost always choose our model as the output of running stochastic gradient descent. Appealing to linear models, we analyze how SGD acts as an implicit regularizer. For linear models, SGD always converges to a solution with small norm. Hence, the algorithm itself is implicitly regularizing the solution. Indeed, we show on small data sets that even Gaussian kernel methods can generalize well with no regularization. Though this doesn’t explain why certain architectures generalize better than other architectures, it does suggest that more investigation is needed to understand exactly what the properties are inherited by models that were trained using SGD.

**隐式正则化的角色**。显式正则化如dropout和权重衰减，对于泛化可能不是必须的，但对训练数据拟合的好的模型，并不是都泛化的都很好，这是显而易见的。确实，在神经网络中，我们的模型几乎都是SGD训练得到的。与线性模型类似，我们分析SGD是怎样作为一个隐式正则化器起作用的。对于线性模型，SGD一直会收敛到一个解，其范数较小。因此，算法本身就在对解有隐式的正则化作用。确实，我们在小型数据集上证明了，即使是高斯核方法也可以在没有正则化的情况下进行很好的泛化。虽然这并没有解释，为什么特定的架构会比其他架构的泛化能力更好，但这确实说明，需要更多的研究来精确理解，SGD训练得到的模型，内在有什么性质。

### 1.2 Related Work

Hardt et al. (2016) give an upper bound on the generalization error of a model trained with stochastic gradient descent in terms of the number of steps gradient descent took. Their analysis goes through the notion of uniform stability (Bousquet & Elisseeff, 2002). As we point out in this work, uniform stability of a learning algorithm is independent of the labeling of the training data. Hence, the concept is not strong enough to distinguish between the models trained on the true labels (small generalization error) and models trained on random labels (high generalization error). This also highlights why the analysis of Hardt et al. (2016) for non-convex optimization was rather pessimistic, allowing only a very few passes over the data. Our results show that even empirically training neural networks is not uniformly stable for many passes over the data. Consequently, a weaker stability notion is necessary to make further progress along this direction.

Hardt等对使用SGD训练得到的模型，以进行的梯度下降的步数为参考，给出了泛化误差的上限。其分析是用统一稳定性的概念进行的。正如我们在本文中所看到的，一个学习算法的统一稳定性是与训练数据的标记是无关的。因此，这个概念不足以区分，在真实标签上训练得到的模型（泛化误差小），和在随机标签上训练得到的模型（泛化误差大）。这也强调了Hardt等对非凸优化的分析为什么是非常悲观的，只允许很小一部分数据。我们的结果表明，即使是通过经验训练的神经网络，也对很多数据并不是一致稳定的。结果是，需要一个更弱的稳定性概念，以在这个方向取得更大的进展。

There has been much work on the representational power of neural networks, starting from universal approximation theorems for multi-layer perceptrons (Cybenko, 1989; Mhaskar, 1993; Delalleau & Bengio, 2011; Mhaskar & Poggio, 2016; Eldan & Shamir, 2016; Telgarsky, 2016; Cohen & Shashua, 2016). All of these results are at the population level characterizing which mathematical functions certain families of neural networks can express over the entire domain. We instead study the representational power of neural networks for a finite sample of size n. This leads to a very simple proof that even O(n)-sized two-layer perceptrons have universal finite-sample expressivity.

在神经网络的表示能力上有过很多工作，从多层感知机的统一近似定理开始。所有这些结果都是在族群这个层次，描述了特定的神经网络族可以表示哪些数学函数。我们研究的是神经网络对一组有限的n个样本的表示能力。这带来了一个非常简单的证明，即使规模是O(n)的2层感知机都有一般性的有限样本表达能力。

Bartlett (1998) proved bounds on the fat shattering dimension of multilayer perceptrons with sigmoid activations in terms of the l1-norm of the weights at each node. This important result gives a generalization bound for neural nets that is independent of the network size. However, for RELU networks the l1-norm is no longer informative. This leads to the question of whether there is a different form of capacity control that bounds generalization error for large neural nets. This question was raised in a thought-provoking work by Neyshabur et al. (2014), who argued through experiments that network size is not the main form of capacity control for neural networks. An analogy to matrix factorization illustrated the importance of implicit regularization.

Bartlett证明了blah blah。这个重要的结果给出了神经网络的泛化界限，与网络的规模无关。但是，对于ReLU网络，l1-范数不在有信息性。这带来了是否有另一种形式的能力控制的问题，对大型神经网络的泛化误差设界。这个问题由Neyshabur等提出，他们通过实验认为，网络大小并不是神经网络的能力控制的主要形式。与矩阵分解的类比，描述了隐式正则化的重要性。

## 2. Effective Capacity of Neural Networks

Our goal is to understand the effective model capacity of feed-forward neural networks. Toward this goal, we choose a methodology inspired by non-parametric randomization tests. Specifically, we take a candidate architecture and train it both on the true data and on a copy of the data in which the true labels were replaced by random labels. In the second case, there is no longer any relationship between the instances and the class labels. As a result, learning is impossible. Intuition suggests that this impossibility should manifest itself clearly during training, e.g., by training not converging or slowing down substantially. To our surprise, several properties of the training process for multiple standard achitectures is largely unaffected by this transformation of the labels. This poses a conceptual challenge. Whatever justification we had for expecting a small generalization error to begin with must no longer apply to the case of random labels.

我们的目标是理解前向神经网络的模型有效容量。为这个目标，我们受到非参数随机测试启发，选择了一种方法。具体的，我们取一个候选架构，在真实数据和将真实标签替换为随机标签的复制数据上对其进行训练。在第二个情况中，样本与类别标签毫无关系。结果是，学习是不可能的。直觉表明，这种不可能性应当在训练过程中清晰的展现出来，如，通过训练不收敛，或极度减缓。但令人惊讶的是，在多个标准架构中，这个训练过程的几种性质，都不受这种标签变换的影响。这就提出了一个概念上的挑战。无论我们对小的泛化误差有什么样的期待，但这对于随机标签的情况来说都不在适用。

To gain further insight into this phenomenon, we experiment with different levels of randomization exploring the continuum between no label noise and completely corrupted labels. We also try out different randomizations of the inputs (rather than labels), arriving at the same general conclusion.

为对这种现象有深入的洞见，我们用不同的标签随机化层次进行实验，探索了从没有标签噪声，到完全破坏的标签。我们尝试不同的随机化输入（而不是标签），得到了同样的一般性结论。

The experiments are run on two image classification datasets, the CIFAR10 dataset (Krizhevsky & Hinton, 2009) and the ImageNet (Russakovsky et al., 2015) ILSVRC 2012 dataset. We test the Inception V3 (Szegedy et al., 2016) architecture on ImageNet and a smaller version of Inception, Alexnet (Krizhevsky et al., 2012), and MLPs on CIFAR10. Please see Section A in the appendix for more details of the experimental setup.

这个试验在两个图像分类数据集上运行，CIFAR10数据集和ImageNet ILSVRC 2012数据集。我们在ImageNet上测试了Inception V3架构，在CIFAR10上测试了更小版本的Inception，Alexnet和MLPs。附录中的A节有试验设置的更多细节。

### 2.1 Fitting Random Labels and Pixels

We run our experiments with the following modifications of the labels and input images: 我们对输入图像和标签进行下面的修改，然后运行我们的试验：

- True labels: the original dataset without modification. 真实标签：原始数据集不进行修改。
- Partially corrupted labels: independently with probability p, the label of each image is corrupted as a uniform random class. 部分破坏的标签：以概率p并独立的，每幅图像的标签都按照统一的随机类别进行破坏。
- Random labels: all the labels are replaced with random ones. 随机标签：所有的标签都用随机标签替换。
- Shuffled pixels: a random permutation of the pixels is chosen and then the same permutation is applied to all the images in both training and test set. 像素混洗：选择像素的随机置换，然后对所有的图像适用同样的置换，包括训练集和测试集。
- Random pixels: a different random permutation is applied to each image independently. 随机像素：对每幅图像独立的应用一种不同的随机置换。
- Gaussian: A Gaussian distribution (with matching mean and variance to the original image dataset) is used to generate random pixels for each image. 高斯：对每幅图像以高斯分布来生成随机像素（均值和方差与原始图像数据集匹配）。

Surprisingly, stochastic gradient descent with unchanged hyperparameter settings can optimize the weights to fit to random labels perfectly, even though the random labels completely destroy the relationship between images and labels. We further break the structure of the images by shuffling the image pixels, and even completely re-sampling random pixels from a Gaussian distribution. But the networks we tested are still able to fit.

令人惊讶的是，不更改SGD的超参数，也可以对权重进行优化，对随机标签进行完美的拟合，即使随机标签完全破坏了图像和标签之间的关系。我们进一步破坏了图像的结构，对图像像素进行混洗，甚至是用高斯分布完全对像素进行随机化，但我们测试的网络仍然可以拟合。

Figure 1a shows the learning curves of the Inception model on the CIFAR10 dataset under various settings. We expect the objective function to take longer to start decreasing on random labels because initially the label assignments for every training sample is uncorrelated. Therefore, large predictions errors are back-propagated to make large gradients for parameter updates. However, since the random labels are fixed and consistent across epochs, the network starts fitting after going through the training set multiple times. We find the following observations for fitting random labels very interesting: a) we do not need to change the learning rate schedule; b) once the fitting starts, it converges quickly; c) it converges to (over)fit the training set perfectly. Also note that “random pixels” and “Gaussian” start converging faster than “random labels”. This might be because with random pixels, the inputs are more separated from each other than natural images that originally belong to the same category, therefore, easier to build a network for arbitrary label assignments.

图1a给出了Inception模型在CIFAR10数据集上、在各种不同的设置下的学习曲线。我们期望目标函数在随机标签上开始下降的要耗费更长时间，因为开始的时候，对每个训练样本的标签指定是不相关的。因此，大的预测误差进行反向传播，得到大的梯度进行参数更新。但是，由于随机标签是固定的，在每轮训练中是一致的，网络在遍历了训练集几次后，开始拟合。在拟合随机标签时，我们发现下面的观察非常有趣：a)我们不需要改变学习速率的计划；b)一旦这个拟合开始了，收敛的速度是很快的；c)收敛时对训练集的拟合是很完美的。注意，随机像素和高斯比随机标签开始收敛的速度要更快。这可能是因为，在随机像素时，输入相互之间的距离比属于同一类的自然图像更远，因此，更容易对任意标签指定构建网络。

On the CIFAR10 dataset, Alexnet and MLPs all converge to zero loss on the training set. The shaded rows in Table 1 show the exact numbers and experimental setup. We also tested random labels on the ImageNet dataset. As shown in the last three rows of Table 2 in the appendix, although it does not reach the perfect 100% top-1 accuracy, 95.20% accuracy is still very surprising for a million random labels from 1000 categories. Note that we did not do any hyperparameter tuning when switching from the true labels to random labels. It is likely that with some modification of the hyperparameters, perfect accuracy could be achieved on random labels. The network also manages to reach ∼90% top-1 accuracy even with explicit regularizers turned on.

在CIFAR10数据集上，AlexNet和MLPs在训练集上都收敛到了0损失。表1中的阴影列给出了确切数值和试验设置。我们还在ImageNet数据集上测试了随机标签。如附录中的表2的最后三行所示，虽然没有达到完美的100%top-1准确率，但对于1000个类别的100万个随机标签来说，有95.20%的准确率，仍然是非常令人惊讶的。注意，在从真实标签切换到随机标签时，我们没有进行任何超参数调节。如果有一定的超参数调节，在随机标签的情况下，也很可能达到完美的准确率。网络在显式正则化开启的情况下，也达到了～90%的top-1准确率。

**Partially corrupted labels**. We further inspect the behavior of neural network training with a varying level of label corruptions from 0 (no corruption) to 1 (complete random labels) on the CIFAR10 dataset. The networks fit the corrupted training set perfectly for all the cases. Figure 1b shows the slowdown of the convergence time with increasing level of label noises. Figure 1c depicts the test errors after convergence. Since the training errors are always zero, the test errors are the same as generalization errors. As the noise level approaches 1, the generalization errors converge to 90% — the performance of random guessing on CIFAR10.

**部分破坏的标签**。我们进一步检视了神经网络在CIFAR10数据集上，用标签的不同层次的破坏训练时的行为，破坏水平从0（没有破坏）到1（完全随机的标签）。网络对于所有情况都对破坏的训练集有完美的拟合。图1b展示了标签噪声水平逐渐增加时，收敛时间的下降。图1c展示了收敛后的测试误差。由于训练误差永远是0，测试误差与泛化误差就一样了。在噪声水平接近1时，泛化误差收敛到90%，也就是在CIFAR10上的随机猜测的性能。

### 2.2 Implications

In light of our randomization experiments, we discuss how our findings pose a challenge for several traditional approaches for reasoning about generalization. 根据我们的随机试验，我们讨论一下我们的发现是怎样对几种对泛化进行推理的传统方法造成挑战的。

**Rademacher complexity and VC-dimension**. Rademacher complexity is commonly used and flexible complexity measure of a hypothesis class. The empirical Rademacher complexity of a hypothesis class H on a dataset {x1, ..., xn} is defined as

Rademacher复杂度经常使用，是假设类别的灵活复杂度度量。一个假设类别H在数据集{x1, ..., xn}上的经验Rademacher复杂度定义为：

$$\hat R_n(H) = E_σ[sup_{h∈H} \frac {1}{n} \sum_{i=1}^n σ_i h(x_i)]$$(1)

where σ1, ..., σn ∈ {±1} are i.i.d. uniform random variables. This definition closely resembles our randomization test. Specifically, $\hat R_n(H)$ measures ability of H to fit random ±1 binary label assignments. While we consider multiclass problems, it is straightforward to consider related binary classification problems for which the same experimental observations hold. Since our randomization tests suggest that many neural networks fit the training set with random labels perfectly, we expect that $\hat R_n(H)≈1$ for the corresponding model class H. This is, of course, a trivial upper bound on the Rademacher complexity that does not lead to useful generalization bounds in realistic settings. A similar reasoning applies to VC-dimension and its continuous analog fat-shattering dimension, unless we further restrict the network. While Bartlett (1998) proves a bound on the fat-shattering dimension in terms of l1 norm bounds on the weights of the network, this bound does not apply to the ReLU networks that we consider here. This result was generalized to other norms by Neyshabur et al. (2015), but even these do not seem to explain the generalization behavior that we observe.

其中σ1, ..., σn ∈ {±1}是独立同分布的均匀随机变量。这个定义与我们的随机化测试很相像。具体的，$\hat R_n(H)$度量的是H拟合随机±1二值标签指定的能力。我们考虑的是多类别问题，但考虑相关的二值分类问题也是很直接的，可以得到同样的试验观察结果。由于我们的随机化测试说明，很多神经网络可以对随机标签的训练集完美拟合，我们期望对于对应的模型类别H，有$\hat R_n(H)≈1$。当然，这对于Rademacher复杂度来说是一个没有意义的上限，在实际的设置中不会带来有用的泛化界限。类似的推理对VC维也进行应用，及其连续类比fat-shattering维度，直到我们对网络进行进一步的约束。Bartlett证明了，对网络的权重，有一个fat-shattering维度的l1范数界限，这个界限对我们这里考虑的ReLU网络并不适用。这个结果由Neyshabur等泛化到其他范数，但即使这些似乎也没有解释我们观察到的泛化行为。

**Uniform stability**. Stepping away from complexity measures of the hypothesis class, we can instead consider properties of the algorithm used for training. This is commonly done with some notion of stability, such as uniform stability (Bousquet & Elisseeff, 2002). Uniform stability of an algorithm A measures how sensitive the algorithm is to the replacement of a single example. However, it is solely a property of the algorithm, which does not take into account specifics of the data or the distribution of the labels. It is possible to define weaker notions of stability (Mukherjee et al., 2002; Poggio et al., 2004; Shalev-Shwartz et al., 2010). The weakest stability measure is directly equivalent to bounding generalization error and does take the data into account. However, it has been difficult to utilize this weaker stability notion effectively.

**一致稳定性**。除了假设类别的复杂度度量，我们考虑一下训练用的算法的性质。这一般是用稳定性的概念来进行的，如一致稳定性。一种算法A的一致稳定性，度量的是算法对单个样本的替换的敏感性。但是，这只是算法的一个性质，并没有将数据或标签的分布纳入考虑。定义稳定性更弱的概念是可能的。最弱的稳定性概念与泛化误差的界限是直接相关的，这将数据纳入了考虑。但是，有效的利用这种更弱的概念，是很困难的。

## 3. The Role of Regularization

Most of our randomization tests are performed with explicit regularization turned off. Regularizers are the standard tool in theory and practice to mitigate overfitting in the regime when there are more parameters than data points (Vapnik, 1998). The basic idea is that although the original hypothesis is too large to generalize well, regularizers help confine learning to a subset of the hypothesis space with manageable complexity. By adding an explicit regularizer, say by penalizing the norm of the optimal solution, the effective Rademacher complexity of the possible solutions is dramatically reduced.

多数随机化测试都是在没有正则化的情况下进行的。正则化器是理论和实践上的标准工具，在参数比数据点多时，可以缓解过拟合的问题。其基本思想是，虽然原始的假设太大，不能泛化的很好，正则化器可以帮助将学习限制在假设空间的一部分，使复杂度可控。通过增加一个显式的正则化器，比如惩罚最优解的范数，可能的解的有效Rademacher复杂性得到了极大的降低。

As we will see, in deep learning, explicit regularization seems to play a rather different role. As the bottom rows of Table 2 in the appendix show, even with dropout and weight decay, InceptionV3 is still able to fit the random training set extremely well if not perfectly. Although not shown explicitly, on CIFAR10, both Inception and MLPs still fit perfectly the random training set with weight decay turned on. However, AlexNet with weight decay turned on fails to converge on random labels. To investigate the role of regularization in deep learning, we explicitly compare behavior of deep nets learning with and without regularizers.

我们将会看到，在深度学习中，显式的正则化的角色似乎非常不同。在附录中的表2的最后一行显示，即使使用了dropout和权重衰减，InceptionV3仍然还是可以对训练集拟合的非常好，只是不是很完美罢了。虽然在CIFAR10上没有显式的展现出来，Inception和MLPs在使用权重衰减时，仍然对随机训练集拟合的非常好。但是，使用权重衰减的AlexNet在随机标签上不能收敛。为研究正则化在深度学习中的作用，我们显式的比较DNN在有或没有正则化器的时候的行为。

Instead of doing a full survey of all kinds of regularization techniques introduced for deep learning, we simply take several commonly used network architectures, and compare the behavior when turning off the equipped regularizers. The following regularizers are covered:

我们没有对深度学习中的正则化技术进行完整的调查，只是使用了几种常用的网络架构，比较了一下在不用正则化器的时候的行为。我们的试验覆盖了下列正则化器：

- Data augmentation: augment the training set via domain-specific transformations. For image data, commonly used transformations include random cropping, random perturbation of brightness, saturation, hue and contrast. 数据扩增：通过特定领域的变换进行训练集的扩增。对图像数据，常用的变换包括，随机剪切，亮度、饱和度、色调和对比度的随机扰动。

- Weight decay: equivalent to a l2 regularizer on the weights; also equivalent to a hard constrain of the weights to an Euclidean ball, with the radius decided by the amount of weight decay. 权重衰减：等价于对权重的l2正则化器；还等价于将权重硬束缚在欧式球上，其半径是由权重衰减的量决定的。

- Dropout (Srivastava et al., 2014): mask out each element of a layer output randomly with a given dropout probability. Only the Inception V3 for ImageNet uses dropout in our experiments. Dropout：以一定的dropout概率随机的将一层的每个元素掩膜掉。在我们的试验中，只有在ImageNet上的InceptionV3使用了dropout。

Table 1 shows the results of Inception, Alexnet and MLPs on CIFAR10, toggling the use of data augmentation and weight decay. Both regularization techniques help to improve the generalization performance, but even with all of the regularizers turned off, all of the models still generalize very well.

表1给出了Inception，AlexNet和MLPs在CIFAR10上的结果，在数据扩增和权重衰减之间切换。两种正则化技术对改进泛化性能有帮助，但即使不使用所有的正则化器，所有的模型仍然泛化的非常好。

Table 2 in the appendix shows a similar experiment on the ImageNet dataset. A 18% top-1 accuracy drop is observed when we turn off all the regularizers. Specifically, the top-1 accuracy without regularization is 59.80%, while random guessing only achieves 0.1% top-1 accuracy on ImageNet. More strikingly, with data-augmentation on but other explicit regularizers off, Inception is able to achieve a top-1 accuracy of 72.95%. Indeed, it seems like the ability to augment the data using known symmetries is significantly more powerful than just tuning weight decay or preventing low training error.

附录中的表2在ImageNet数据集上进行了类似的试验。在不使用所有正则化器时，top-1准确率下降了18%。具体的，不用正则化的top-1准确率为59.80%，而随机猜测只有0.1%的top-1准确率。更令人震撼的是，只使用数据扩增，其他正则化器关掉，Inception可以获得72.95%的top-1准确率。确实，似乎使用已知的对称性对数据进行扩增，比仅仅权重衰减，明显的要有用一些。

Inception achieves 80.38% top-5 accuracy without regularization, while the reported number of the winner of ILSVRC 2012 (Krizhevsky et al., 2012) achieved 83.6%. So while regularization is important, bigger gains can be achieved by simply changing the model architecture. It is difficult to say that the regularizers count as a fundamental phase change in the generalization capability of deep nets.

不使用正则化的情况下，Inception获得了80.38%的top-5准确率，而ILSVRC 2012的获奖者的准确率为83.6%。所以虽然正则化是重要的，使用不同的模型架构就可以获得很大的收获。很难说，正则化器对深度网络的泛化性能具有基本的改变作用。

### 3.1 Implicit Regularizations

Early stopping was shown to implicitly regularize on some convex learning problems (Yao et al., 2007; Lin et al., 2016). In Table 2 in the appendix, we show in parentheses the best test accuracy along the training process. It confirms that early stopping could potentially improve the generalization performance. Figure 2a shows the training and testing accuracy on ImageNet. The shaded area indicate the accumulative best test accuracy, as a reference of potential performance gain for early stopping. However, on the CIFAR10 dataset, we do not observe any potential benefit of early stopping.

早停在一些凸学习问题中表现出了隐式的正则化的作用。在附录中的表2，我们在括号中给出了训练过程中的最好测试准确率。这确认了早停是可能改进泛化性能的。图2a给出了在ImageNet上的训练和测试准确率。阴影区域是累积的最好测试准确率，即早停的可能的性能提升的参考。但是，在CIFAR10数据集，我们没有看到早停的任何潜在的收益。

Batch normalization (Ioffe & Szegedy, 2015) is an operator that normalizes the layer responses within each mini-batch. It has been widely adopted in many modern neural network architectures such as Inception (Szegedy et al., 2016) and Residual Networks (He et al., 2016). Although not explicitly designed for regularization, batch normalization is usually found to improve the generalization performance. The Inception architecture uses a lot of batch normalization layers. To test the impact of batch normalization, we create a “Inception w/o BatchNorm” architecture that is exactly the same as Inception in Figure 3, except with all the batch normalization layers removed. Figure 2b compares the learning curves of the two variants of Inception on CIFAR10, with all the explicit regularizers turned off. The normalization operator helps stablize the learning dynamics, but the impact on the generalization performance is only 3∼4%. The exact accuracy is also listed in the section “Inception w/o BatchNorm” of Table 1.

BN是一个算子，在每个mini-batch中对层的响应进行归一化。在现代神经网络架构中已经被广泛采用，如Inception和ResNet。虽然并不是为正则化设计的，但BN经常发现可以改进泛化性能。Inception架构使用了很多BN层。为测试BN的影响，我们创建了一个Inception没有BN的架构，与图3中的Inception是一样的，除了移除掉了所有的BN层。图2b比较了在CIFAR10上的Inception的两个变体的学习曲线，并把所有的显式正则化器都移除了。归一化算子稳定了学习过程，但泛化性能的影响只有3～4%。精确的准确率在表1中的Inception w/o BatchNorm节中给出。

In summary, our observations on both explicit and implicit regularizers are consistently suggesting that regularizers, when properly tuned, could help to improve the generalization performance. However, it is unlikely that the regularizers are the fundamental reason for generalization, as the networks continue to perform well after all the regularizers removed.

总结起来，我们对显式正则化和隐式正则化的观察一致表明，当正则化器得到合适调整时，对泛化性能是有帮助的。但是，正则化器很可能并不是泛化的根本原因，因为在去掉所有的正则化器时，网络的性能仍然很好。

## 4 Finite-Sample Expressivity

Much effort has gone into characterizing the expressivity of neural networks, e.g, Cybenko (1989); Mhaskar (1993); Delalleau & Bengio (2011); Mhaskar & Poggio (2016); Eldan & Shamir (2016); Telgarsky (2016); Cohen & Shashua (2016). Almost all of these results are at the “population level” showing what functions of the entire domain can and cannot be represented by certain classes of neural networks with the same number of parameters. For example, it is known that at the population level depth k is generically more powerful than depth k − 1.

在神经网络的表示能力的研究有很多。几乎所有结果都是在population层次的，表明用特定类型的神经网络，在相同数量的参数下，是否可以表示整个域上的某种函数。比如，我们已经知道，在population层次上，k层的网络一般比k-1层的网络要更具有表现力。

We argue that what is more relevant in practice is the expressive power of neural networks on a finite sample of size n. It is possible to transfer population level results to finite sample results using uniform convergence theorems. However, such uniform convergence bounds would require the sample size to be polynomially large in the dimension of the input and exponential in the depth of the network, posing a clearly unrealistic requirement in practice.

我们认为，在实践中更相关的是，神经网络对有限数量n的样本的表示能力。使用一致收敛定理，将population层次的结果，迁移到有限数量的样本上，是可能的。但是，这样的一致收敛界会需要样本规模是输入维度的多项式数量的大小，是网络深度的指数级大小，在实践中这很明显是不现实的要求。

We instead directly analyze the finite-sample expressivity of neural networks, noting that this dramatically simplifies the picture. Specifically, as soon as the number of parameters p of a networks is greater than n, even simple two-layer neural networks can represent any function of the input sample. We say that a neural network C can represent any function of a sample of size n in d dimensions if for every sample S ⊆ $R^d$ with |S| = n and every function f : S → R, there exists a setting of the weights of C such that C(x) = f(x) for every x ∈ S.

我们直接分析了神经网络的有限样本表达能力，表明这极大的简化了所有情况。具体的，只要网络的参数数量p大于n，甚至简单的两层NN就能表示输入图像的任意函数。我们可以说，一个神经网络C可以表示n个d维样本的任意函数，如果对于每个样本S ⊆ $R^d$，|S| = n ，对每个函数f : S → R，都存在C的权重的设置，使得对于每个x ∈ S，C(x) = f(x)。

Theorem 1. There exists a two-layer neural network with ReLU activations and 2n + d weights that can represent any function on a sample of size n in d dimensions.

定理1. 对一个d维样本，大小为n，存在一个两层的带有ReLU激活的神经网络，有2n+d个权重，可以表示样本上的任意函数。

The proof is given in Section C in the appendix, where we also discuss how to achieve width O(n/k) with depth k. We remark that it’s a simple exercise to give bounds on the weights of the coefficient vectors in our construction. Lemma 1 gives a bound on the smallest eigenvalue of the matrix A. This can be used to give reasonable bounds on the weight of the solution w.

证明在附录中的C节中给出，其中我们还讨论了在深度k时得到宽度O(n/k)。我们说明，在我们的证明中，对权重的系数向量给出界限，是很简单的操作。推论1给出了矩阵A的最小特征值的界。这可以用于对解w的权重给出合理的界限。

## 5 Implicit Regularization: An Appeal to Linear Models

Although deep neural nets remain mysterious for many reasons, we note in this section that it is not necessarily easy to understand the source of generalization for linear models either. Indeed, it is useful to appeal to the simple case of linear models to see if there are parallel insights that can help us better understand neural networks.

虽然深度神经网络由于很多原因仍然很神秘，我们在本节中说明，对于线性模型系统，要理解泛化的源头，也并不是容易的。确实，可以看看线性模型的简单情况，是否有类似的洞见，可以帮助我们理解神经网络。

Suppose we collect n distinct data points {(xi, yi)} where xi are d-dimensional feature vectors and yi are labels. Letting loss denote a nonnegative loss function with loss(y,y) = 0, consider the empirical risk minimization (ERM) problem

假设我们收集了n个不同的数据点{(xi, yi)}，其中xi是d维特征向量，yi是标签。令损失函数表示非负的损失函数，loss(y,y)=0，考虑经验风险最小化(ERM)问题

$$min_{w∈R^d} \frac{1}{n} \sum_{i=1}^n loss(w^T x_i, y_i)$$(2)

If d ≥ n, then we can fit any labeling. But is it then possible to generalize with such a rich model class and no explicit regularization? 如果d≥n，那么我们可以拟合任意标签。但这样一个丰富的模型类别，在没有显式正则化的情况下，泛化是否可能呢？

Let X denote the n × d data matrix whose i-th row is $x^T_i$. If X has rank n, then the system of equations Xw = y has an infinite number of solutions regardless of the right hand side. We can find a global minimum in the ERM problem (2) by simply solving this linear system.

令X表示n×d数据矩阵，其第i行是$x^T_i$。如果X秩为n，那么系统的方程Xw = y有无数个解，不管右边是什么情况。我们在ERM问题(2)中，通过求解这个线性系统，可以找到一个全局极小值点。

But do all global minima generalize equally well? Is there a way to determine when one global minimum will generalize whereas another will not? One popular way to understand quality of minima is the curvature of the loss function at the solution. But in the linear case, the curvature of all optimal solutions is the same (Choromanska et al., 2015). To see this, note that in the case when yi is a scalar,

但是所有的全局极小值都能泛化的同样好吗？有没有方法可以决定，一个全局极小值可以泛化的好，而另一个不行？理解极小值的质量的一个流行方式是，在解处的损失函数的曲率。但在线性的情况下，所有的最优解的曲率都是一样的。为看到这一点，注意在yi是标量的情况下

$$∇^2 \frac {1}{n} \sum_{i=1}^n loss(w^T x_i, y_i) = \frac {1}{n} X^T diag(β)X, (β_i := \frac {∂^2 loss(z,y_i)}{∂z^2} |_{z=y_i}, ∀i)$$

A similar formula can be found when y is vector valued. In particular, the Hessian is not a function of the choice of w. Moreover, the Hessian is degenerate at all global optimal solutions.

当y是矢量值的时候，有类似的公式。特别是，Hessian不是w的函数。而且，Hessian在所有全局最优解处都是降质的。

If curvature doesn’t distinguish global minima, what does? A promising direction is to consider the workhorse algorithm, stochastic gradient descent (SGD), and inspect which solution SGD converges to. Since the SGD update takes the form $w_{t+1} = w_t - η_t e_t x_{i_t}$ where $η_t$ is the step size and $e_t$ is the prediction error loss. If w0 = 0, we must have that the solution has the form $w = \sum^n_{i=1} α_i x_i$ for some coefficients α. Hence, if we run SGD we have that $w = X^T α$ lies in the span of the data points. If we also perfectly interpolate the labels we have Xw = y. Enforcing both of these identities, this reduces to the single equation

如果曲率不能区分全局极小值，那什么可以呢？一个有希望的方向是，考虑workhorse算法，SGD，检视一下SGD收敛到哪个解。由于SGD的更新的形式为$w_{t+1} = w_t - η_t e_t x_{i_t}$，其中$η_t$为步长，$e_t$是预测误差损失。如果w0 = 0，我们的解的形式则为$w = \sum^n_{i=1} α_i x_i$，有一些系数α。因此，如果我们运行SGD，我们有$w = X^T α$，即是数据点张成的空间中。如果我们还对标签进行了完美的插值，那么有Xw = y。所以我们有：

$$XX^Tα=y$$(3)

which has a unique solution. Note that this equation only depends on the dot-products between the data points x_i. We have thus derived the “kernel trick” (Scho ̈lkopf et al., 2001)—albeit in a roundabout fashion. 这有唯一解的。注意这个解只依赖于数据点xi的点积。因此我们推导了核的技巧。

We can therefore perfectly fit any set of labels by forming the Gram matrix (aka the kernel matrix) on the data $K=XX^T$ and solving the linear system Kα=y for α. This is an n×n linear system that can be solved on standard workstations whenever n is less than a hundred thousand, as is the case for small benchmarks like CIFAR10 and MNIST.

我们能因此完美的拟合任意标签集，只要在数据上形成Gram矩阵$K=XX^T$，对α求解线性系统Kα=y。这是一个n×n的线性系统，可以在标准工作站上进行求解，只要n小于10万，这在较小的基准测试，如CIFAR10和MNIST上是符合的。

Quite surprisingly, fitting the training labels exactly yields excellent performance for convex models. On MNIST with no preprocessing, we are able to achieve a test error of 1.2% by simply solving (3). Note that this is not exactly simple as the kernel matrix requires 30GB to store in memory. Nonetheless, this system can be solved in under 3 minutes in on a commodity workstation with 24 cores and 256 GB of RAM with a conventional LAPACK call. By first applying a Gabor wavelet transform to the data and then solving (3), the error on MNIST drops to 0.6%. Surprisingly, adding regularization does not improve either model’s performance!

令人惊讶的是，拟合训练标签，对凸模型也刚好得到了非常好的性能。在MNIST上，没有预处理，只要求解(3)，我们可以达到测试误差1.2%。注意，这也不是非常简单的，因为核矩阵需要30GB的内存。尽管如此，这个系统在一个商用工作站上，24核，256GB RAM，用传统的LAPACK的调用，只用3分钟就可以进行求解。首先对数据进行Gabor小波变换，然后求解(3)，在MNIST上的误差下降到了0.6%。令人惊讶的是，增加正则化，并不会改进模型的性能。

Similar results follow for CIFAR10. Simply applying a Gaussian kernel on pixels and using no regularization achieves 46% test error. By preprocessing with a random convolutional neural net with 32,000 random filters, this test error drops to 17% error. Adding l2 regularization further reduces this number to 15% error. Note that this is without any data augmentation.

在CIFAR10上也得到的类似的结果。只在像素上应用高斯核，不使用正则化，得到了46%的测试误差。用随机CNN进行预处理，有32000个随机滤波器，测试误差下降到了17%。增加l2正则化进一步将这个数降低到15%。注意这并没有使用任何数据扩增。

Note that this kernel solution has an appealing interpretation in terms of implicit regularization. Simple algebra reveals that it is equivalent to the minimum l2-norm solution of Xw = y. That is, out of all models that exactly fit the data, SGD will often converge to the solution with minimum norm. It is very easy to construct solutions of Xw = y that don’t generalize: for example, one could fit a Gaussian kernel to data and place the centers at random points. Another simple example would be to force the data to fit random labels on the test data. In both cases, the norm of the solution is significantly larger than the minimum norm solution.

注意，这个核的解有一个很吸引人的解释，即隐式正则化。简单的代数会表明，这与Xw = y的最小l2范数解是等价的。即，在严格拟合数据的所有模型中，SGD通常会收敛到具有最小范数的解上。很容易构建不会泛化的Xw=y的解：比如，可以对数据拟合一个高斯核，在随机点中摆放中心。另一个简单的例子是，在测试数据上迫使数据拟合随机标签。在两种情况中，解的范数肯定比最小范数解要大很多。

Unfortunately, this notion of minimum norm is not predictive of generalization performance. For example, returning to the MNIST example, the l2-norm of the minimum norm solution with no preprocessing is approximately 220. With wavelet preprocessing, the norm jumps to 390. Yet the test error drops by a factor of 2. So while this minimum-norm intuition may provide some guidance to new algorithm design, it is only a very small piece of the generalization story.

不幸的是，最小范数的概念并不意味着泛化性能。比如，回到MNIST的例子中，没有预处理的最小范数解的l2范数大约是220。有了小波预处理，范数增加到了390。但测试误差下降了2倍。所以虽然这个最小范数的直觉可能会对新算法的设计有一些指引，这在泛化性能上只会有很小的影响。

## 6. Conclusion

In this work we presented a simple experimental framework for defining and understanding a notion of effective capacity of machine learning models. The experiments we conducted emphasize that the effective capacity of several successful neural network architectures is large enough to shatter the training data. Consequently, these models are in principle rich enough to memorize the training data. This situation poses a conceptual challenge to statistical learning theory as traditional measures of model complexity struggle to explain the generalization ability of large artificial neural networks. We argue that we have yet to discover a precise formal measure under which these enormous models are simple. Another insight resulting from our experiments is that optimization continues to be empirically easy even if the resulting model does not generalize. This shows that the reasons for why optimization is empirically easy must be different from the true cause of generalization.

本文中，我们提出了一个简单的试验框架，定义和理解了机器学习模型的有效容量。我们进行的试验，强调了几种成功的CNN架构的有效容量是足够大的，可以打碎训练数据。结果是，这些模型原则上是足够丰富的，可以记住训练数据。这个情况对统计学习理论提出了概念上的挑战，因为传统的度量模型复杂度的方法，很难解释大型CNN的泛化能力。我们认为，我们尚未发现一个精确的正式的度量，在这个度量下，这些很多模型都是简单的。我们试验得到的另一个洞见是，即使得到的模型不会泛化，优化过程从经验上也是很简单的。这说明，为什么优化从经验上是简单的原因，与泛化的真正原因，肯定是不同的。