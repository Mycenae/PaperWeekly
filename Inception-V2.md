# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift  批归一化：减少internal covariate shift加速深度网络训练

Sergey Ioffe, Christian Szegedy  Google Inc.

## Abstract 摘要

Training Deep Neural Networks is complicated by the fact that the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout. Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. Using an ensemble of batch-normalized networks, we improve upon the best published result on ImageNet classification: reaching 4.9% top-5
validation error (and 4.8% test error), exceeding the accuracy of human raters.

训练深度神经网络是很复杂的，每层输入的分布在训练过程中会不断的变化，因为前一层的参数在不断的变化。这就需要降低学习速率，参数初始化也需要小心选择，从而使训练减速，如果网络存在饱和非线性函数，那么模型训练起来就异常困难。我们称这种现象为internal covariate shift，并准备通过归一化层的输入来解决这个问题。我们的方法将归一化作为模型架构的一部分，在训练的每个mini-batch都进行归一化操作。批归一化使模型可以使用大的多的学习速率，参数初始化也容易很多。批归一化也有正则化的作用，在一些例子中可以看到不需要进行dropout操作了。批归一化应用到最新模型，可以用1/14的训练迭代次数得到相同的准确率。使用批归一化网络的集成模型，我们改进了目前ImageNet分类的最好结果：达到了4.9%的top-5验证错误率（4.8%的测试错误率），超过了人类评估者的准确率。

## 1 Introduction

Deep learning has dramatically advanced the state of the art in vision, speech, and many other areas. Stochastic gradient descent (SGD) has proved to be an effective way of training deep networks, and SGD variants such as momentum (Sutskever et al., 2013) and Adagrad (Duchi et al., 2011) have been used to achieve state of the art performance. SGD optimizes the parameters Θ of the network, so as to minimize the loss

深度学习极大的发展了视觉、语音和很多其他领域。随机梯度下降SGD是一种训练深度网络的有效方法，SGD的变体，如动量法(Sutskever et al., 2013)和Adagrad (Duchi et al., 2011)的应用，也取得了目前最好的表现。SGD优化网络的参数Θ，最小化如下损失函数

$$Θ = argmin_Θ \frac{1}{N} \sum_{i=1}^{N} l(x_i, Θ)$$

where $x_{1...N}$ is the training data set. With SGD, the training proceeds in steps, and at each step we consider a mini-batch $x_{1...m}$ of size *m*. The mini-batch is used to approximate the gradient of the loss function with respect to the parameters, by computing

这里$x_{1...N}$是训练数据集。利用SGD的训练过程是按步进行的，每一步我们选择一个mini-batch，含*m*个样本$x_{1...m}$。我们利用mini-batch样本来近似损失函数对参数的梯度，计算

$$\frac{1}{m} \frac{∂l(x_i,Θ)}{∂Θ}$$

Using mini-batches of examples, as opposed to one example at a time, is helpful in several ways. First, the gradient of the loss over a mini-batch is an estimate of the gradient over the training set, whose quality improves as the batch size increases. Second, computation over a batch can be much more efficient than *m* computations for individual examples, due to the parallelism afforded by the modern computing platforms.

使用mini-batch样本（而不是一次使用一个样本）有几个好处。首先，一个mini-batch上的损失函数的梯度是整个训练集上的梯度的近似估计，batch size越大，估计效果越好；第二，一个batch上的计算比对单个样本的*m*次计算要有效率的多，这是因为现代计算平台提供的并行机制而导致的。

While stochastic gradient is simple and effective, it requires careful tuning of the model hyper-parameters, specifically the learning rate used in optimization, as well as the initial values for the model parameters. The training is complicated by the fact that the inputs to each layer are affected by the parameters of all preceding layers – so that small changes to the network parameters amplify as
the network becomes deeper.

随机梯度下降简单又有效，但模型的超参数需要仔细调整，尤其是优化中用到的学习速率，还有模型参数的初始值。训练过程中每层的输入受到前面所有层的参数影响，所以网络参数的微小变化随着网络加深而得到放大，训练过程非常复杂。

The change in the distributions of layers’ inputs presents a problem because the layers need to continuously adapt to the new distribution. When the input distribution to a learning system changes, it is said to experience covariate shift (Shimodaira, 2000). This is typically handled via domain adaptation (Jiang, 2008). However, the notion of covariate shift can be extended beyond the learning system as a whole, to apply to its parts, such as a sub-network or a layer. Consider a network computing

每层输入的分布不断变化，导致各层需要不断的适应这些新的输入分布。当一个学习系统的输入分布变化时，我们称之为covariate shift (Shimodaira, 2000)，这通常通过domain adaptation解决这个问题。但是，covariate shift的概念可以延展到学习系统之外，应用到学习系统的组件，比如一个子网，或一层。考虑一个网络计算如下式

$$l = F_2 (F_1 (u,Θ_1),Θ_2)$$

where $F_1$ and $F_2$ are arbitrary transformations, and the parameters $Θ_1$ ,$Θ_2$ are to be learned so as to minimize the loss *l*. Learning $Θ_2$ can be viewed as if the inputs $x = F_1 (u,Θ_1)$ are fed into the sub-network $l = F_2 (x,Θ_2)$. For example, a gradient descent step

这里$F_1$和$F_2$是任意变换，参数$Θ_1$ ,$Θ_2$需要进行学习，以最小化损失函数*l*。学习参数$Θ_2$可以看作是，输入$x = F_1 (u,Θ_1)$到子网络$l = F_2 (x,Θ_2)$。比如，梯度下降的步骤

$$Θ_2 ← Θ_2 − \frac{α}{m} \sum_{i=1}^{m} \frac{∂F_2 (x_i ,Θ_2)}{∂Θ_2}$$

(for batch size *m* and learning rate *α*) is exactly equivalent to that for a stand-alone network $F_2$ with input *x*. Therefore, the input distribution properties that make training more efficient – such as having the same distribution between the training and test data – apply to training the sub-network as well. As such it is advantageous for the distribution of *x* to remain fixed over time. Then, $Θ_2$ does not have to readjust to compensate for the change in the distribution of *x*.

（对batch size *m*和学习速率*α*）对于一个单独的网络$F_2$输入为*x*是完全等效的。所以，那些让训练更有效率的输入分布性质，比如训练和测试数据分布相同，对于训练子网络也适用。所以输入*x*的分布保持固定对训练是有好处的。那么，$Θ_2$就不需要因为*x*输入分布的变化而重新调整了。

Fixed distribution of inputs to a sub-network would have positive consequences for the layers outside the subnetwork, as well. Consider a layer with a sigmoid activation function $z = g(Wu + b)$ where *u* is the layer input, the weight matrix *W* and bias vector *b* are the layer parameters to be learned, and $g(x) = \frac{1}{1+exp(−x)}$. As $|x|$ increases, $g′(x)$ tends to zero. This means that for all dimensions of $x = Wu+b$ except those with small absolute values, the gradient flowing down to *u* will vanish and the model will train slowly. However, since *x* is affected by *W,b* and the parameters of all the layers below, changes to those parameters during training will likely move many dimensions of *x* into the saturated regime of the nonlinearity and slow down the convergence. This effect is amplified as the network depth increases. In practice, the saturation problem and the resulting vanishing gradients are usually addressed by using Rectified Linear Units (Nair & Hinton, 2010) *ReLU(x) = max(x,0)*, careful initialization (Bengio & Glorot, 2010; Saxe et al., 2013), and small learning rates. If, however, we could ensure that the distribution of nonlinearity inputs remains more stable as the network trains, then the optimizer would be less likely to get stuck in the saturated regime, and the training would accelerate.

子网络的输入固定分布对于子网以外的层也有正面作用。考虑一层激活函数为sigmoid函数$z = g(Wu + b)$，*u*是输入，权重矩阵*W*和偏置矢量*b*是要学习的参数，$g(x) = \frac{1}{1+exp(−x)}$为sigmoid函数。当$|x|$增加时，$g′(x)$趋向于0。这说明如果$x = Wu+b$绝对值变大，其梯度将会消失，训练就会变慢。但是*x*是受到*W,b*的影响，以及所有前面层的参数影响，训练时这些参数的变化可能会使*x*进入非线性函数的饱和区，进而使收敛变慢。这种效果随着网络深度的增加而放大。在实践中，饱和问题和消失的梯度通常用ReLU函数、仔细选择权值初始化以及小学习速率来解决。如果网络训练的过程中我们可以确保非线性输入的分布保持稳定，那么优化器就很可能不会进入饱和区域，那么训练过程就得到了加速。

We refer to the change in the distributions of internal nodes of a deep network, in the course of training, as Internal Covariate Shift. Eliminating it offers a promise of faster training. We propose a new mechanism, which we call Batch Normalization, that takes a step towards reducing internal covariate shift, and in doing so dramatically accelerates the training of deep neural nets. It accomplishes this via a normalization step that fixes the means and variances of layer inputs. Batch Normalization also has a beneficial effect on the gradient flow through the network, by reducing the dependence of gradients on the scale of the parameters or of their initial values. This allows us to use much higher learning rates without the risk of divergence. Furthermore, batch normalization regularizes the model and reduces the need for Dropout (Srivastava et al., 2014). Finally, Batch Normalization makes it possible to use saturating nonlinearities by preventing the network from getting stuck in the saturated modes.

我们所说的Internal Covariate Shift，是指训练过程中，深度网络中内部节点的分布变化。消除这种变化会使训练加速。我们提出一种新的方法，称为Batch Normalization，可以尽量减少internal covariate shift，通过这种机制，可以极大的加速深度网络的训练过程。其具体做法是，对层的输入增加一个归一化处理，就可以固定其均值和方差。批归一化对于网络中的梯度流也有好处，可以减少梯度对参数的规模或其初始值的依赖程度。这使我们可以使用很高的学习速率，仍然可以得到收敛的结果。进一步，批归一化还有正则化的作用，网络可能不太需要dropout了。最后，批归一化使网络可以使用饱和非线性函数，因为网络不会陷入饱和状态。

In Sec. 4.2, we apply Batch Normalization to the best-performing ImageNet classification network, and show that we can match its performance using only 7% of the training steps, and can further exceed its accuracy by a substantial margin. Using an ensemble of such networks trained with Batch Normalization, we achieve the top-5 error rate that improves upon the best known results on ImageNet classification.

在4.2节中，我们将批归一化应用到ImageNet最好的分类网络中，结果表明只用了7%的训练步数就达到了其原有表现，而且可以更进一步超过其准确率很多。使用这种网络的集成模型，用批归一化的进行训练，我们改进了目前最好的ImageNet分类结果。

## 2 Towards Reducing Internal Covariate Shift

We define Internal Covariate Shift as the change in the distribution of network activations due to the change in network parameters during training. To improve the training, we seek to reduce the internal covariate shift. By fixing the distribution of the layer inputs *x* as the training progresses,we expect to improve the training speed. It has been long known (LeCun et al., 1998b; Wiesler & Ney, 2011) that the network training converges faster if its inputs are whitened – i.e., linearly transformed to have zero means and unit variances, and decorrelated. As each layer observes the inputs produced by the layers below,it would be advantageous to achieve the same whitening of the inputs of each layer. By whitening the inputs to each layer, we would take a step towards achieving the fixed distributions of inputs that would remove the ill effects of the internal covariate shift.

我们将Internal Covariate Shift定义为训练过程中网络参数变化导致的网络激活值的分布变化。为改善训练，我们试图减少internal covariate shift。在训练过程中，我们固定层输入*x*的分布，希望可以改进训练速度。很早以前就知道，如果网络输入经过白化处理，网络训练会收敛更快，白化就是经过线性变换使其零均值单位方差，并解相关。由于每层输入是由前面的层产生的，那么每层都进行同样的白化处理效果会更好。通过对每层的输入进行白化处理，我们可能得到这样的固定输入分布，可以消除internal covariate shift的病态效果。

We could consider whitening activations at every training step or at some interval, either by modifying the network directly or by changing the parameters of the optimization algorithm to depend on the network activation values (Wiesler et al., 2014; Raiko et al., 2012; Povey et al., 2014; Desjardins & Kavukcuoglu). However, if these modifications are interspersed with the optimization steps, then the gradient descent step may attempt to update the parameters in a way that requires the normalization to be updated, which reduces the effect of the gradient step. For example, consider a layer with the input *u* that adds the learned bias *b*, and normalizes the result by subtracting the mean of the activation computed over the training data: $\hat x = x − E[x]$ where $x = u + b$, $X = \{x_{1...N} \}$ is the set of values of *x* over the training set, and $E[x] = \frac{1}{N} \sum_{i=1}^{N} x_i$. If a gradient descent step ignores the dependence of *E[x]* on *b*, then it will update $b ← b + ∆b$, where $∆b ∝ −∂ℓ/∂ \hat x$. Then $u + (b + ∆b) − E[u + (b + ∆b)] = u + b − E[u + b]$. Thus, the combination of the update to *b* and subsequent change in normalization led to no change in the output of the layer nor, consequently, the loss. As the training continues, *b* will grow indefinitely while the loss remains fixed. This problem can get worse if the normalization not only centers but also scales the activations. We have observed this empirically in initial experiments, where the model blows up when the normalization parameters are computed outside the gradient descent step.

我们在每个训练步骤或每个间隔里都进行白化激活，或者通过直接修正网络，或者通过改变优化算法参数，使其与网络激活值有关。如果这些修正散布在优化步骤里，那么梯度下降步骤可能在更新参数时要求这个归一化也同时更新，这会降低梯度的效果。比如，输入为*u*的层，增加偏置*b*，归一化的方法就是减去其均值，$\hat x = x − E[x]$，这里$x = u + b$，$X = \{x_{1...N} \}$是*x* 在训练集中的值的集合，$E[x] = \frac{1}{N} \sum_{i=1}^{N} x_i$。如果梯度下降步骤忽略了*E[x]*对*b*的关系，那么将会更新$b ← b + ∆b$，这里$∆b ∝ −∂ℓ/∂ \hat x$，那么$u + (b + ∆b) − E[u + (b + ∆b)] = u + b − E[u + b]$。所以，更新*b*和后续的变化的组合归一化后，在这层的输出上没有变化，那么在损失函数上也没有变化。当训练过程继续，*b*将无限增长，但损失函数固定不变。如果归一化过程只进行零均值，而没有限制激活的尺度，那么这个问题可能变的更坏。我们通过实验已经观察到了这种情况，当归一化参数在梯度下降过程之外计算时，模型会爆炸。

The issue with the above approach is that the gradient descent optimization does not take into account the fact that the normalization takes place. To address this issue, we would like to ensure that, for any parameter values, the network always produces activations with the desired distribution. Doing so would allow the gradient of the loss with respect to the model parameters to account for the normalization, and for its dependence on the model parameters Θ. Let again *x* be a layer input, treated as a vector, and X be the set of these inputs over the training data set. The normalization can then be written as a transformation

上面方法的问题是，梯度下降优化没有考虑到归一化的过程。为了解决这个问题，我们会确认，对于任何参数值，网络永远会产生期望分布的激活值。这样做可以使损失函数对模型参数的梯度考虑到归一化的因素，还考虑到模型参数Θ的关系。令*x*为层输入矢量，X为在训练数据集上的取值集合。归一化的过程可以写为如下变换

$$\hat x = Norm(x,X)$$

which depends not only on the given training example x but on all examples X – each of which depends on Θ if x is generated by another layer. For backpropagation, we would need to compute the Jacobians

不仅仅依赖于给定的训练样本x，而是依赖于所有样本X，如果x是由其他层产生的，那么就全部与Θ有关。对于反向传播，我们需要计算下面的Jacobian矩阵

$$\frac{∂Norm(x,X)}{∂x} and \frac{∂Norm(x,X)}{∂X}$$

ignoring the latter term would lead to the explosion described above. Within this framework, whitening the layer inputs is expensive, as it requires computing the covariance matrix $Cov[x] = E_{x∈X} [xx^T] − E[x]E[x]^T$ and its inverse square root, to produce the whitened activations $Cov[x]^{−1/2}(x − E[x])$, as well as the derivatives of these transforms for back propagation. This motivates us to seek an alternative that performs input normalization in a way that is differentiable and does not require the analysis of the entire training set after every parameter update.

忽略后面一项将会导致前面所说的模型爆炸。在这个框架中，将层输入进行白化是非常昂贵的，因为这需要计算协方差矩阵$Cov[x] = E_{x∈X} [xx^T] − E[x]E[x]^T$和其平方根逆，为产生白化激活$Cov[x]^{−1/2}(x − E[x])$和反向传播时用到的这些变换的导数。这促使我们寻找一个替代方法，在对输入归一化时要是可微分的，而且在每次参数更新后不需要对整个训练集进行分析。

Some of the previous approaches (e.g. (Lyu & Simoncelli, 2008)) use statistics computed over a single training example, or, in the case of image networks, over different feature maps at a given location. However, this changes the representation ability of a network by discarding the absolute scale of activations. We want to a preserve the information in the network, by normalizing the activations in a training example relative to the statistics of the entire training data.

一些前述的方法，如(Lyu & Simoncelli, 2008)，使用在一次训练样本上的统计量，在图像网络的情况中，是在一个给定位置的不同特征图上计算统计量。但是，这改变了网络的表示能力，因为抛弃了激活的绝对尺度。我们希望在网络中保持这个信息，通过归一化一个训练样本对整个训练数据的统计值的激活值。

## 3 Normalization via Mini-Batch Statistics

Since the full whitening of each layer’s inputs is costly and not everywhere differentiable, we make two necessary simplifications. The first is that instead of whitening the features in layer inputs and outputs jointly, we will normalize each scalar feature independently, by making it have the mean of zero and the variance of 1. For a layer with *d*-dimensional input $x = (x^{(1)} ...x^{(d)})$, we will normalize each dimension

既然每层输入的全白化非常消耗计算量，而且并不是处处可微，我们进行两处必要的简化。第一是，不同时在输入和输出处进行特征白化，我们会单独归一化每个标量特征，使其零均值方差为1。对于一个*d*维输入$x = (x^{(1)} ...x^{(d)})$的层，我们会对每一维进行归一化

$$\hat x^{(k)} = \frac {x^{(k)}-E[x^{(k)}]} {\sqrt{Var[x^{(k)}]}}$$

where the expectation and variance are computed over the training data set. As shown in (LeCun et al., 1998b), such normalization speeds up convergence, even when the features are not decorrelated.

这里期望和方差在整个训练集上进行计算。像(LeCun et al., 1998b)所给出的结果那样，这样的归一化会使收敛加速，即使特征没有解相关。

Note that simply normalizing each input of a layer may change what the layer can represent. For instance, normalizing the inputs of a sigmoid would constrain them to the linear regime of the nonlinearity. To address this, we make sure that the transformation inserted in the network can represent the identity transform. To accomplish this, we introduce, for each activation $x^{(k)}$, a pair of parameters $γ^{(k)} ,β^{(k)}$, which scale and shift the normalized value:

注意仅仅归一化一个层的每个输入可能改变这个层可以表示的内容。比如，对sigmoid的输入进行归一化将会限制其进入非线性处理的线性区域。为解决这个问题，我们确保插入网络的这个变换可以表示恒等变换。为达这个目标，对每个激活$x^{(k)}$，我们引入一对参数$γ^{(k)} ,β^{(k)}$，从幅度和数值上改变归一化的值：

$$y^{(k)} = γ^{(k)} \hat x^{(k)} + β^{(k)}$$

These parameters are learned along with the original model parameters, and restore the representation power
of the network. Indeed, by setting $γ^{(k)} = \sqrt{Var[x^{(k)}]}$ and $β^{(k)} = E[x^{(k)}]$, we could recover the original activations, if that were the optimal thing to do.

这些参数是与原始模型参数一起学习的，可以还原网络的表示能力。如果设$γ^{(k)} = \sqrt{Var[x^{(k)}]}$，$β^{(k)} = E[x^{(k)}]$，我们可以还原原始激活，如果确实需要那么做的话。

In the batch setting where each training step is based on the entire training set, we would use the whole set to normalize activations. However, this is impractical when using stochastic optimization. Therefore, we make the second simplification: since we use mini-batches in stochastic gradient training, each mini-batch produces estimates of the mean and variance of each activation. This way, the statistics used for normalization can fully participate in the gradient back propagation. Note that the use of mini-batches is enabled by computation of per-dimension variances rather than joint covariances; in the joint case, regularization would be required since the mini-batch size is likely to be smaller than the number of activations being whitened, resulting in singular covariance matrices.

在批次设定中，每个训练步骤都是基于整个训练集的，我们要用整个训练集来归一化激活值。但是，这在用SGD优化时是不现实的。所以，我们进行第二个简化：既然我们在SGD训练中用的是mini-batch，每个mini-batch产生每个激活值的均值和方差的估计。这种方法中，归一化用到的统计量可以充分参与梯度的反向传播。注意mini-batch的使用会使每个维度的方差计算起来很方便，而不是计算联合协方差；在协方差的情况里，需要正则化的加入，因为mini-batch样本数很可能比要白化的激活数目要小，这样会得到奇异的协方差矩阵。

Consider a mini-batch B of size *m*. Since the normalization is applied to each activation independently, let us focus on a particular activation $x^{(k)}$ and omit *k* for clarity. We have *m* values of this activation in the mini-batch, $B = \{x_{1...m}\}$.

考虑一个mini-batch B样本数量为*m*。既然每个激活都独立的进行了归一化，我们关注一个特殊的激活$x^{(k)}$，忽略*k*，我们在这个mini-batch中这个激活有*m*个值，$B = \{x_{1...m}\}$。

Let the normalized values be $\hat x_{1...m}$, and their linear transformations be $y_{1...m}$. We refer to the transform

令归一化的值为$\hat x_{1...m}$，其线性变换为$y_{1...m}$。我们称下面的变换

$$BN_{γ,β}: x_{1...m} → y_{1...m}$$

as the Batch Normalizing Transform. We present the BN Transform in Algorithm 1. In the algorithm, $\epsilon$ is a constant added to the mini-batch variance for numerical stability.

为批归一化变换。我们在下面算法1中给出BN变换的具体描述。在算法中，$\epsilon$是常值，作用是数值计算稳定性。

Algorithm 1: Batch Normalizing Transform, applied to activation x over a mini-batch.

算法1：批归一化变换，对一个mini-batch上的激活值x应用

------
**Input**: Values of *x* over a mini-batch: B = { $x_{1...m}$ }; Parameters to be learned: γ, β \
**Output**: {$y_i = BN_{γ, β}(x_i)$} 

$$\text{mini-batch mean:} µ_B ← \frac{1}{m} \sum_{i=1}^{m} x_i, \text{mini-batch variance:} σ_B^2 ← \frac{1}{m} \sum_{i=1}^{m} (x_i - µ_B)^2$$ 

$$\text{normalize:} \hat x_i ← \frac{x_i - µ_B}{\sqrt{σ_B^2 + \epsilon}}, \text{scale and shift:} y_i ← γ \hat x_i + β = BN_{γ,β}(x_i)$$

------

The BN transform can be added to a network to manipulate any activation. In the notation $y = BN_{γ,β}(x)$, we indicate that the parameters γ and β are to be learned, but it should be noted that the BN transform does not independently process the activation in each training example. Rather, $BN_{γ,β}(x)$ depends both on the training example and the other examples in the mini-batch. The scaled and shifted values *y* are passed to other network layers. The normalized activations $\hat x$ are internal to our transformation, but their presence is crucial. The distributions of values of any $\hat x$ has the expected value of 0 and the variance of 1, as long as the elements of each mini-batch are sampled from the same distribution, and if we neglect $\epsilon$. This can be seen by observing that $\sum_{i=1}^{m} \hat x_i = 0$ and $\frac{1}{m} \sum_{i=1}^{m} \hat x_i^2 = 1$, and taking expectations. Each normalized activation $\hat x^{(k)}$ can be viewed as an input to a sub-network composed of the linear transform $y^{(k)} = γ^{(k)} \hat x^{(k)} + β^{(k)}$, followed by the other processing done by the original network. These sub-network inputs all have fixed means and variances, and although the joint distribution of these normalized $\hat x^{(k)}$ can change over the course of training, we expect that the introduction of normalized inputs accelerates the training of the sub-network and, consequently, the network as a whole.

BN变换可以加入网络中去操作任何一个激活值。在式$y = BN_{γ,β}(x)$中，参数γ和β是需要学习得到的，但需要注意的是，对每个训练样本BN变换不单独处理激活，而是$BN_{γ,β}(x)$依赖mini-batch中的所有样本。经过尺度变换和位移变换的值*y*传递给网络其他层。归一化的激活$\hat x$是变换内部的值，但其存在却是非常重要的。任何$\hat x$值的分布，其期望都是0，方差为1，只要mini-batch的元素是从同一分布中取样得到的，并且我们忽略$\epsilon$。这可以通过式$\sum_{i=1}^{m} \hat x_i = 0$和$\frac{1}{m} \sum_{i=1}^{m} \hat x_i^2 = 1$得到验证。每个归一化的激活$\hat x^{(k)}$都可以视为一个子网络的输入，子网络的变换为$y^{(k)} = γ^{(k)} \hat x^{(k)} + β^{(k)}$，后面再接入原网络进行处理。这些子网络的输入都有固定的均值和方差，尽管这些归一化的$\hat x^{(k)}$的联合分布在训练的过程中不断变化，我们期待归一化输入的引入可以加速子网络的训练，进而加速整个网络的训练。

During training we need to backpropagate the gradient of loss ℓ through this transformation, as well as compute the gradients with respect to the parameters of the BN transform. We use chain rule, as follows (before simplification):

在训练过程中，我们需要将损失函数ℓ的梯度反向传播，也要经过这个变换，也要计算对BN变换的参数的导数。我们使用链式法则如下（简化前）：

$$\frac{∂ℓ}{∂\hat x_i} = \frac{∂ℓ}{∂y_i} \cdot  γ$$
$$\frac{∂ℓ}{∂σ_B^2} = \sum_{i=1}^{m} \frac{∂ℓ}{∂\hat x_i} \cdot (x_i-µ_B) \cdot \frac{-1}{2} (σ_B^2+\epsilon)^{-3/2}$$
$$\frac{∂ℓ}{∂µ_B} = (\sum_{i=1}^{m} \frac{∂ℓ}{∂\hat x_i} \cdot \frac{-1}{\sqrt{σ_B^2+\epsilon}}) + \frac{∂ℓ}{∂σ_B^2} \cdot \frac{\sum_{i=1}^{m} -2(x_i - µ_B)}{m}$$
$$\frac{∂ℓ}{∂x_i} = \frac{∂ℓ}{∂\hat x_i} \cdot \frac{1}{\sqrt{σ_B^2+\epsilon}} + \frac{∂ℓ}{∂σ_B^2} \cdot \frac{2x_i-µ_B}{m} + \frac{∂ℓ}{∂µ_B} \cdot \frac{1}{m}$$
$$\frac{∂ℓ}{∂γ} = \sum_{i=1}^{m} \frac{∂ℓ}{∂y} \cdot \hat x_i$$
$$\frac{∂ℓ}{∂β} = \sum_{i=1}^{m} \frac{∂ℓ}{∂y}$$

Thus, BN transform is a differentiable transformation that introduces normalized activations into the network. This ensures that as the model is training, layers can continue learning on input distributions that exhibit less internal covariate shift, thus accelerating the training. Furthermore, the learned affine transform applied to these normalized activations allows the BN transform to represent the identity transformation and preserves the network capacity.

所以，BN变换是可微分变换，向网络里引入了归一化的激活。这确保了模型训练时，每层可以继续从具有很少internal covariate shift的输入分布中学习，所以加速了训练。更进一步，学习到的仿射变换应用在这些归一化的激活中，使BN变换可以表示恒等变换并保护网络的容量。

### 3.1 Training and Inference with Batch-Normalized Networks

To Batch-Normalize a network, we specify a subset of activations and insert the BN transform for each of them, according to Alg. 1. Any layer that previously received x as the input, now receives BN(x). A model employing Batch Normalization can be trained using batch gradient descent, or Stochastic Gradient Descent with a mini-batch size m > 1, or with any of its variants such as Adagrad (Duchi et al., 2011). The normalization of activations that depends on the mini-batch allows efficient training, but is neither necessary nor desirable during inference; we want the output to depend only on the input, deterministically. For this, once the network has been trained, we use the normalization

为批归一化一个网络，我们指定了一个激活值子集，对其中每一个元素根据算法1插入BN变换。任何层以前以x为输入的，现在以BN(x)为输入。一个采用BN变换的模型可以用SGD来进行训练，或采用其变体如Adagrad进行训练。依赖mini-batch的激活值归一化可以有效训练，但在推理过程中既不需要，也不方便；我们希望输出非常确定的只与输入有关。所以，一旦网络训练完成，我们使用归一化

$$\hat x = \frac {x-E[x]} {\sqrt{Var[x]+\epsilon}}$$

using the population, rather than mini-batch, statistics. Neglecting $\epsilon$, these normalized activations have the same mean 0 and variance 1 as during training. We use the unbiased variance estimate $Var[x] = \frac{m}{m-1} \cdot E_B[σ_B^2]$, where the expectation is over training mini-batches of size m and $σ_B^2$ are their sample variances. Using moving averages instead, we can track the accuracy of a model as it trains. Since the means and variances are fixed during inference, the normalization is simply a linear transform applied to each activation. It may further be composed with the scaling by γ and shift by β, to yield a single linear transform that replaces BN(x). Algorithm 2 summarizes the procedure for training batch-normalized networks.

这里使用的不是mini-batch，而是整个集合的统计量。忽略$\epsilon$，在训练过程中，这些归一化的激活值有着同样的均值0和方差1。我们使用无偏方差估计$Var[x] = \frac{m}{m-1} \cdot E_B[σ_B^2]$，这里期望是在样本数为m的mini-batch上计算的，$σ_B^2$是它们的样本方差。使用平滑均值，我们可以跟踪模型训练时的准确度。既然均值和方差在推理过程中是固定的，归一化就是一个简单的线性变换应用在每个激活值上。进一步分析，是由尺度系数γ和位移量β组成的，这样就可以组成一个简单的线性变换可以替换掉BN(x)。算法2总结了这个训练批归一化网络的过程。

Algorithm 2: Training a Batch-Normalized Network 算法2：训练一个批归一化的网络

------
**Input**: Network *N* with trainable parameters Θ; subset of activations $\{x^{(k)} \}_{k=1}^{K}$ \
**Output**: Batch-normalized network for inference, $N^{inf}_{BN}$

1. $N_{BN}^{tr} ← N$, //training BN network
2. **for** *k=1...K* **do**
3.    Add transformation $y^{(k)} = BN_{γ^{(k)},β^{(k)}} (x^{(k)})$ to $N_{BN}^{tr}$ (Alg.1)
4.    Modify each layer in $N_{BN}^{tr}$ with input $x^{(k)}$ to take $y^{(k)}$ instead
5. **end for**
6. Train $N_{BN}^{tr}$ to optimize the parameters $Θ ∪ \{γ^{(k)},β^{(k)} \}_{k=1}^{K}$
7. $N^{inf}_{BN} ← N^{tr}_{BN}$, //Inference BN network with frozen parameters
8. **for** *k=1...K* **do**
9. ///For clarity, $x=x^{(k)}, γ=γ^{(k)}, µ_B=µ_B^{(k)}$, etc
10. Process multiple training mini-batches *B*, each of size *m*, and average over them \
$E[x] ← E_B [µ_B]$ \
$Var[x] ← \frac{m}{m−1} E_B[σ^2_B]$
11. In $N^{inf}_{BN}$,  replace the transform $y = BN_{γ,β}(x)$ with $y = \frac {γ} {\sqrt {Var[x]+ \epsilon}} \cdot x + ( β − \frac {γ E[x]} {\sqrt{Var[x]+\epsilon}} )$
12. **end for**

------

### 3.2 Batch-Normalized Convolutional Networks 批归一化的卷积网络

Batch Normalization can be applied to any set of activations in the network. Here, we focus on transforms that consist of an affine transformation followed by an element-wise nonlinearity:

批归一化可以应用在网络任何类型的激活中。这里，我们重点关注由仿射变换加元素非线性处理的变换：

$$z = g(Wu + b)$$

where W and b are learned parameters of the model, and g(·) is the nonlinearity such as sigmoid or ReLU. This formulation covers both fully-connected and convolutional layers. We add the BN transform immediately before the nonlinearity, by normalizing x = Wu+b. We could have also normalized the layer inputs u, but since u is likely the output of another nonlinearity, the shape of its distribution is likely to change during training, and constraining its first and second moments would not eliminate the covariate shift. In contrast, Wu + b is more likely to have a symmetric, non-sparse distribution, that is “more Gaussian” (Hyvärinen & Oja, 2000); normalizing it is likely to produce activations with a stable distribution.

这里W和b是模型学习到的参数，g(·)是非线性函数，比如sigmoid或ReLU。这个公式既可以代表全连接层，也可以代表卷积层。我们将BN变换放在非线性处理前，对x = Wu+b进行归一化。我们也可以对输入u进行归一化，但u很可能是另一个非线性处理的输出，其分布在训练中很可能变化，对其一时的约束不会消除covariate shift。而作为对比，Wu + b的分布很可能是对称的、非稀疏的，也就是“更加高斯的”(Hyvärinen & Oja, 2000)；对其进行归一化很可能产生具有稳定分布的激活值。

Note that, since we normalize Wu+b,the bias b can be ignored since its effect will be canceled by the subsequent mean subtraction (the role of the bias is subsumed by β in Alg. 1). Thus, z = g(Wu + b) is replaced with

注意，既然我们归一化Wu+b，那么偏置b就可以忽略了，因为其作用会被后面的减去均值所抵消掉（偏置的角色被算法1中的β所替代）。所以，z = g(Wu + b)被下式所代替

$$z = g(BN(Wu))$$

where the BN transform is applied independently to each dimension of x = Wu, with a separate pair of learned parameters $γ^{(k)}, β^{(k)}$ per dimension.

这里BN变换独立作用于x = Wu的每个维度，每个维度都有一对学习好的参数$γ^{(k)}, β^{(k)}$。

For convolutional layers, we additionally want the normalization to obey the convolutional property – so that different elements of the same feature map, at different locations, are normalized in the same way. To achieve this, we jointly normalize all the activations in a mini-batch, over all locations. In Alg. 1, we let B be the set of all values in a feature map across both the elements of a mini-batch and spatial locations – so for a mini-batch of size m and feature maps of size p × q, we use the effective mini-batch of size $m′ = |B| = m · pq$. We learn a pair of parameters $γ^{(k)}$ and $β^{(k)}$ per feature map, rather than per activation. Alg. 2 is modified similarly, so that during inference the BN transform applies the same linear transformation to each activation in a given feature map.

对于卷积层，我们还希望归一化操作遵从卷积的性质，这样不同特征图中的不同元素，在不同的位置，会进行同样的归一化。为达到这个目标，我们在一个mini-batch中同时对所有激活、在所有位置进行归一化。在算法1中，我们令B为特征图中的所有值的集合，既包括mini-batch，也包括所有空间位置，这样对于一个包含m个样本的mini-batch，特征图大小为p × q，我们使用mini-batch的有效大小$m′ = |B| = m · pq$。我们每个特征图学习一对参数$γ^{(k)}$和$β^{(k)}$，而不是每个激活学习两个参数。算法2作类似的修改，这样在推理期间BN变换对给定特征图中每个激活进行同样的线性变换。

### 3.3 Batch Normalization enables higher learning rates

In traditional deep networks, too-high learning rate may result in the gradients that explode or vanish, as well as getting stuck in poor local minima. Batch Normalization helps address these issues. By normalizing activations throughout the network, it prevents small changes to the parameters from amplifying into larger and suboptimal changes in activations in gradients; for instance, it prevents the training from getting stuck in the saturated regimes of nonlinearities.

在传统深度网络里，过高的学习速率会导致梯度爆炸或梯度消失，或者陷入局部极值。批归一化可以帮助解决这个问题。通过将整个网络的激活值进行归一化，使得梯度激活参数的微小变化不会放大成为更大的次优变化；比如，防止训练陷入非线性处理的饱和区域。

Batch Normalization also makes training more resilient to the parameter scale. Normally,large learning rates may increase the scale of layer parameters, which then amplify the gradient during backpropagation and lead to the model explosion. However, with Batch Normalization, back-propagation through a layer is unaffected by the scale of its parameters. Indeed, for a scalar a,

批归一化还使训练对于参数尺度更加具有弹性。正常的，大的学习速率可能会增大层的参数的尺度，然后在反向传播时放大梯度，最后导致梯度爆炸。但在批归一化引入之后，反向传播不受参数尺度的影响。对于标量a

$$BN(Wu) = BN((aW)u)$$

and we can show that 我们可以看到

$$\frac {∂BN((aW)u)} {∂u} = \frac {∂BN(Wu)} {∂u}$$
$$\frac {∂BN((aW)u)} {∂aW} = \frac {1} {a} \frac {∂BN(Wu)} {∂W}$$

The scale does not affect the layer Jacobian nor, consequently, the gradient propagation. Moreover, larger weights lead to smaller gradients, and Batch Normalization will stabilize the parameter growth.

尺度参数不影响层的Jacobian矩阵，所以也不会影响梯度的传播。更大的权值会得到更小的梯度，批归一化会稳定参数增长。

We further conjecture that Batch Normalization may lead the layer Jacobians to have singular values close to 1, which is known to be beneficial for training (Saxe et al., 2013). Consider two consecutive layers with normalized inputs, and the transformation between these normalized vectors: $\hat z = F(\hat x)$. If we assume that $\hat x$ and $\hat z$ are Gaussian and uncorrelated,and that $F(\hat x) ≈ J \hat x$ is a linear transformation for the given model parameters, then both $\hat x$ and $\hat z$ have unit covariances, and $I = Cov[\hat z] = JCov[\hat x]J^T =JJ^T$. Thus, $JJ^T = I$, and so all singular values of J are equal to 1, which preserves the gradient magnitudes during backpropagation. In reality, the transformation is not linear, and the normalized values are not guaranteed to be Gaussian nor independent, but we nevertheless expect Batch Normalization to help make gradient propagation better behaved. The precise effect of Batch Normalization on gradient propagation remains an area of further study.

我们进一步推测，批归一化可能会使层的Jacobian矩阵有接近1的奇异值，这对于训练是有好处的(Saxe et al., 2013)。考虑两个连续的层输入都经过归一化处理，这些归一化的向量之间的变换为$\hat z = F(\hat x)$。如果我们假设$\hat x$和$\hat z$是高斯分布且不相关的，$F(\hat x) ≈ J \hat x$对于给定的模型参数来说是一个线性变换，那么$\hat x$和$\hat z$都有单位协方差阵，$I = Cov[\hat z] = JCov[\hat x]J^T =JJ^T$，所以$JJ^T = I$，所以J的所有奇异值都为1，这就在反向传播的时候保持了梯度幅度。在实际中，变换不是线性的，归一化值也不确定是高斯的或独立的，但我们仍然期望批归一化会帮助梯度传播的更加规范。批归一化对梯度传播的精确效果仍然有待以后进一步研究。

### 3.4 Batch Normalization regularizes the model

When training with Batch Normalization, a training example is seen in conjunction with other examples in the mini-batch, and the training network no longer producing deterministic values for a given training example. In our experiments, we found this effect to be advantageous to the generalization of the network. Whereas Dropout (Srivastava et al., 2014) is typically used to reduce overfitting, in a batch-normalized network we found that it can be either removed or reduced in strength.

当使用批归一化训练时，训练样本同在一个mini-batch中，训练网络对一个给定的训练样本不再产生确定的值。在我们的试验中，我们发现这种效果对网络的泛化有好处。Dropout通常用于减少过拟合效果，在批归一化网络中，我们发现不再需要dropout或其强度可以降低。

## 4 Experiments

### 4.1 Activations over time

To verify the effects of internal covariate shift on training, and the ability of Batch Normalization to combat it, we considered the problem of predicting the digit class on the MNIST dataset (LeCun et al., 1998a). We used a very simple network, with a 28x28 binary image as input, and 3 fully-connected hidden layers with 100 activations each. Each hidden layer computes y = g(Wu+b) with sigmoid nonlinearity, and the weights W initialized to small random Gaussian values. The last hidden layer is followed by a fully-connected layer with 10 activations (one per class) and cross-entropy loss. We trained the network for 50000 steps, with 60 examples per mini-batch. We added Batch Normalization to each hidden layer of the network, as in Sec. 3.1. We were interested in the comparison between the baseline and batch-normalized networks, rather than achieving the state of the art performanceon MNIST (which the described architecture does not).

为验证internal covariate shift在训练中的作用，以及BN对抗的能力，我们在MNIST数据集上进行试验。我们用了很简单一个玩过，输入为28×28二值图像，有3个隐藏全连接层，每层100个激活。在每个隐藏层计算y = g(Wu+b)以及sigmoid非线性处理，权值W初始化为小的随机高斯分布值。最后一个隐藏层是一个10激活值的全连接网络以及交叉熵损失函数。我们训练网络50000步，每个mini-batch 60个样本。如3.1节所述，我们为每个隐藏层增加BN变换。我们感兴趣的是，原始模型和加入BN变换模型的比较，而不是取得多好的识别效果。

Figure 1(a) shows the fraction of correct predictions by the two networks on held-out test data, as training progresses. The batch-normalized network enjoys the higher test accuracy. To investigate why, we studied inputs to the sigmoid, in the original network N and batch-normalized network $N^{tr}_{BN}$ (Alg.2) over the course of training. In Fig.1 (b,c) we show, for one typical activation from the last hidden layer of each network, how its distribution evolves. The distributions in the original network change significantly over time, both in their mean and the variance, which complicates the training of the subsequent layers. In contrast, the distributions in the batch-normalized network are much more stable as training progresses, which aids the training.

图1(a)展示了两个网络在测试数据上的预测结果。BN网络测试准确率高一些。为探明原因，我们研究了在原始网络N和BN网络$N^{tr}_{BN}$（算法2）中，在训练过程中，sigmoid函数的输入。如图1(b,c)所示，每个网络最后一个隐藏层中一个典型的激活，其分布是如何演化的。原网络的分布波动很大，包括均值和方差，这使后续层的训练更加复杂。形成对比的是，BN网络的分布则稳定的多，这对训练有帮助。

Figure 1: (a) The test accuracy of the MNIST network trained with and without Batch Normalization, vs. the number of training steps. Batch Normalization helps the network train faster and achieve higher accuracy. (b,c) The evolution of input distributions to a typical sigmoid, over the course of training, shown as {15,50,85}th percentiles. Batch Normalization makes the distribution more stable and reduces the internal covariate shift.

图1：(a)原网络与BN网络在MNIST数据集上的测试准确率，横轴是训练迭代步数；BN变换帮助网络更快的训练并得到更高的准确度；(b,c)一个典型的sigmoid函数输入分布在训练过程中的演化。BN使分布更稳定，减少了internal covariate shift。

### 4.2 ImageNet classification

We applied Batch Normalization to a new variant of the Inception network (Szegedy et al., 2014), trained on the ImageNet classification task (Russakovsky et al., 2014). The network has a large number of convolutional and pooling layers, with a softmax layer to predict the image class, out of 1000 possibilities. Convolutional layers use ReLU as the nonlinearity. The main difference to the network described in (Szegedy et al., 2014) is that the 5 × 5 convolutional layers are replaced by two consecutive layers of 3 × 3 convolutions with up to 128 filters. The network contains 13.6M parameters, and, other than the top softmax layer, has no fully-connected layers. More details are given in the Appendix. We refer to this model as Inception in the rest of the text. The model was trained using a version of Stochastic Gradient Descent with momentum(Sutskever et al.,2013), using the mini-batch size of 32. The training was performed using a large-scale, distributed architecture (similar to (Dean et al., 2012)). All
networks are evaluated as training progresses by computing the validation accuracy @1, i.e. the probability of predicting the correct label out of 1000 possibilities, on a held-out set, using a single crop per image.

我们将BN变换应用到Inception网络的一种新变体上，在ImageNet分类任务中训练。网络的卷积层和pooling层很多，用softmax层来预测1000个图像类别。卷积层用ReLU非线性处理。与Inception网络的主要区别是5×5卷积层替换成了2个连续的3×3卷积层，128滤波器。网络包括13.6M参数，除了最后的softmax层，没有其他全连接层。更多细节见附录。下文中我们将模型称为Inception模型。模型训练用带有动量的SGD，mini-batch大小为32。训练采用的是一种大型分布式架构。所有网络的评估是通过计算验证正确率。

In our experiments, we evaluated several modifications of Inception with Batch Normalization. In all cases, Batch Normalization was applied to the input of each nonlinearity, in a convolutional way, as described in section 3.2, while keeping the rest of the architecture constant.

在我们的试验中，我们评估Inception的几个BN修正。在所有情况下，BN都是应用在每个非线性处理模块的输入中，以卷积的方式，如3.2节所述，而同时保持架构其他部分为常数。

#### 4.2.1 Accelerating BN Networks

Simply adding Batch Normalization to a network does not take full advantage of our method. To do so, we further changed the network and its training parameters, as follows:

仅仅将BN变换加入到网络中不能充分发挥我们方法的长处。我们进一步修改网络及其训练参数，如下：

Increase learning rate. In a batch-normalized model, we have been able to achieve a training speedup from higher learning rates, with no ill side effects (Sec. 3.3).

增大学习速率。在BN模型中，我们可以采用更高的学习速率进行加速训练，而不会有什么副作用（如3.3节所述）。

Remove Dropout. As described in Sec. 3.4, Batch Normalization fulfills some of the same goals as Dropout. Removing Dropout from Modified BN-Inception speeds up training, without increasing overfitting.

移除Dropout。如3.4节所述，BN变换可以实现Dropout的一些功能，所以可以移除dropout操作来加速训练，不会增加过拟合。

Reduce the L2 weight regularization. While in Inception an L2 loss on the model parameters controls overfitting, in Modified BN-Inception the weight of this loss is reduced by a factor of 5. We find that this improves the accuracy on the held-out validation data.

减少L2权值正则化。在Inception中，模型参数的L2损失控制着过拟合，在修正的BN Inception网络中，权值L2正则化的系数可以减小（除以5）。我们发现这可以提高在验证数据集上的准确度。

Accelerate the learning rate decay. In training Inception, learning rate was decayed exponentially. Because our network trains faster than Inception, we lower the learning rate 6 times faster.

加速学习速率衰减。在训练Inception过程中，学习速率以指数级速度衰减。因为我们的网络比Inception训练的更快，学习速度下降的速度为6倍。

Remove Local Response Normalization. While Inception and other networks (Srivastava et al., 2014) benefit from it, we found that with Batch Normalization it is not necessary.

移除局部响应归一化。带有BN变换的网络不再需要LRN(local response normalization)。

Shuffle training examples more thoroughly. We enabled within-shard shuffling of the training data, which prevents the same examples from always appearing in a mini-batch together. This led to about 1% improvements in the validation accuracy, which is consistent with the view of Batch Normalization as a regularizer (Sec. 3.4): the randomization inherent in our method should be most beneficial when it affects an example differently each time it is seen.

更彻底的打乱训练样本的顺序。我们对训练数据采用了within-shard打乱顺序法，这防止同样的样本一直出现在同一个mini-batch中。这带来了大约1%的验证准确率提升，也说明了BN可以作为正则化使用：我们方法内在的随机性当每次遇到同一个样本时产生不同的影响，这样才最有益处。

Reduce the photometric distortions. Because batch-normalized networks train faster and observe each training example fewer times, we let the trainer focus on more “real” images by distorting them less.

减少光学扭曲变形。因为BN网络训练更快，处理每个样本的次数更少，所以我们对图像的变形也更少一些，使网络尽量处理更真实的图片。

#### 4.2.2 Single-Network Classification

We evaluated the following networks, all trained on the ILSVRC2012 training data, and tested on the validation data:

我们评价以下网络，都在ILSVRC-2012训练集上进行训练，在验证集上进行测试：

Inception : the network described at the beginning of Section 4.2, trained with the initial learning rate of 0.0015.

Inception模型：网络在4.2节介绍，采用学习速率0.0015.

BN-Baseline : Same as Inception with Batch Normalization before each nonlinearity.

BN基准模型：与Inception模型相同，但是在非线性处理前使用了BN变换。

BN-x5 : Inception with Batch Normalization and the modifications in Sec. 4.2.1. The initial learning rate was increased by a factor of 5, to 0.0075. The same learning rate increase with original Inception caused the model parameters to reach machine infinity.

BN-x5模型：带有BN变换的Inception，并进行了4.2.1节的修改。初始学习速率增加了5倍，即0.0075. 原始Inception模型中学习速率进行相同的增加会导致模型参数达到无穷大。

BN-x30 : Like BN-x5 , but with the initial learning rate 0.045 (30 times that of Inception).

BN-x30模型：和BN-x5类似，初始学习速率为30倍，即0.045.

BN-x5-Sigmoid : Like BN-x5 , but with sigmoid nonlinearity $g(x) = \frac {1}{1+exp(−x)}$ instead of ReLU. We also attempted to train the original Inception with sigmoid, but the model remained at the accuracy equivalent to chance.

BN-x5-Sigmoid模型：与BN-x5类似，但采用sigmoid非线性处理，即$g(x) = \frac {1}{1+exp(−x)}$，不用ReLU。我们还用sigmoid函数训练原始Inception模型，但模型的正确率没有改变。

In Figure 2, we show the validation accuracy of the networks, as a function of the number of training steps. Inception reached the accuracy of 72.2% after 31M training steps. The Figure 3 shows, for each network, the number of training steps required to reach the same 72.2% accuracy, as well as the maximum validation accuracy reached by the network and the number of steps to reach it.

在图2中，我们展示了网络在验证集上的正确率，横轴为训练步数。Inception经过了31M迭代达到了72.2%的正确率。如图3所示，对于每个网络，达到72.2%所需的训练步数，以及可以达到的最大验证准确率，和需要的步数。

Figure 2: Single crop validation accuracy of Inception and its batch-normalized variants, vs. the number of training steps.

Figure 3: For Inception and the batch-normalized variants, the number of training steps required to reach the maximum accuracy of Inception (72.2%), and the maximum accuracy achieved by the network.

Model | Steps to 72.2% | Max accuracy
--- | --- | ---
Inception | 31.0M | 72.2%
BN-Baseline | 13.3M | 72.7%
BN-x5 | 2.1M | 73.0%
BN-x30 | 2.7M | 74.8%
BN-x5-Sigmoid | | 69.8%

By only using Batch Normalization ( BN-Baseline ), we match the accuracy of Inception in less than half the number of training steps. By applying the modifications in Sec. 4.2.1, we significantly increase the training speed of the network. BN-x5 needs 14 times fewer steps than Inception to reach the 72.2% accuracy. Interestingly, increasing the learning rate further ( BN-x30 ) causes the model to train somewhat slower initially, but allows it to reach a higher final accuracy. It reaches 74.8% after 6M steps, i.e. 5 times fewer steps than required by Inception to reach 72.2%.

仅仅使用BN变换的模型，达到Inception模型的准确率使用的训练步数不到其一半。使用4.2.1节的修正，极大了增加了网络训练速度。BN-x5模型达到Inception 72.2%的准确率只需要1/14的步数。有趣的是，进一步增大学习速率(BN-30)一开始却减慢了学习了速率，但最后达到了更高的准确率。在6M迭代后达到了74.8%，是Inception达到72.2%迭代数的1/5.

We also verified that the reduction in internal covariate shift allows deep networks with Batch Normalization to be trained when sigmoid is used as the nonlinearity, despite the well-known difficulty of training such networks. Indeed, BN-x5-Sigmoid achieves the accuracy of 69.8%. Without Batch Normalization, Inception with sigmoid never achieves better than 1/1000 accuracy.

我们还验证了internal covariate shift的减少情况，这使BN深度网络可以用sigmoid作为非线性处理，尽管训练这种网络难度很大。确实，BN-x5-Sigmoid取得了69.8%的准确率。没有BN变换，sigmoid Inception准确率不会超过0.1%。

#### 4.2.3 Ensemble Classification

The current reported best results on the ImageNet Large Scale Visual Recognition Competition are reached by the Deep Image ensemble of traditional models (Wu et al., 2015) and the ensemble model of (He et al., 2015). The latter reports the top-5 error of 4.94%, as evaluated by the ILSVRC server. Here we report a top-5 validation error of 4.9%, and test error of 4.82% (according to the ILSVRC server). This improves upon the previous best result, and exceeds the estimated accuracy of human raters according to (Russakovsky et al., 2014).

ILSVRC的最好结果是由(Wu et al., 2015)和(He et al., 2015)的集成方法达到的。后者top-5错误率为4.94%。这里我们得到的top-5验证错误率为4.9%，测试错误率为4.82%。

For our ensemble, we used 6 networks. Each was based on BN-x30, modified via some of the following: increased initial weights in the convolutional layers; using Dropout (with the Dropout probability of 5% or 10%, vs. 40% for the original Inception); and using non-convolutional, per-activation Batch Normalization with last hidden layers of the model. Each network achieved its maximum accuracy after about 6M training steps. The ensemble prediction was based on the arithmetic average of class probabilities predicted by the constituent networks. The details of ensemble and multicrop inference are similar to (Szegedy et al., 2014).

我们的集成模型用了6个网络。每个都是基于BN-x30的，修正方法如下：卷积层增大初始权值；使用dropout（dropout 概率5%或10%，原始Inception算法为40%）；在最后一层隐藏层使用非卷积、预激活BN。每个网络在6M训练步数后达到最佳准确率。集成模型预测是基于构成网络的预测概率的代数平均。集成模型和多剪切块推理的细节与(Szegedy et al., 2014)类似。

We demonstrate in Fig. 4 that batch normalization allows us to set new state-of-the-art by a healthy margin on the ImageNet classification challenge benchmarks.

我们在图4中给出我们的BN模型在ImageNet分类挑战基准测试中得到了最新的最好结果。

## 5 Conclusion

We have presented a novel mechanism for dramatically accelerating the training of deep networks. It is based on the premise that covariate shift, which is known to complicate the training of machine learning systems, also applies to sub-networks and layers, and removing it from internal activations of the network may aid in training. Our proposed method draws its power from normalizing activations, and from incorporating this normalization in the network architecture itself. This ensures that the normalization is appropriately handled by any optimization method that is being used to train the network. To enable stochastic optimization methods commonly used in deep network training, we perform the normalization for each mini-batch, and backpropagate the gradients through the normalization parameters. Batch Normalization adds only two extra parameters per activation, and in doing so preserves the representation ability of the network. We presented an algorithm for constructing,training, and performing inference with batch-normalized networks. The resulting networks can be trained with saturating nonlinearities, are more tolerant to increased training rates, and often do not require Dropout for regularization.

我们提出了一种能极大加速深度网络训练的新方法。这是基于如下假设，即covariate shift会使机器学习系统的学习变得复杂，也使得响应的子网络和层的学习变复杂，如果能从内部激活中去除covariate shift，可以有助于训练。我们提出的方法对激活进行归一化，将这种归一化整合进网络架构中。这确保了优化方法可以合适的处理归一化。为运用训练普通网络中常用的SGD优化方法，我们使用mini-batch归一化，反向传播梯度可以经过归一化参数。BN变换每个激活只增加2个参数，这样还可以保持网络的表示能力。我们提出一种构建、训练网络及推理的BN网络的方法。得到的网络可以使用饱和非线性处理来进行训练，可以增大学习速率，经常不需要dropout这种正则化手段。

Merely adding Batch Normalization to a state-of-the-art image classification model yields a substantial speedup in training. By further increasing the learning rates, removing Dropout, and applying other modifications afforded by Batch Normalization, we reach the previous state of the art with only a small fraction of training steps – and then beat the state of the art in single-network image classification. Furthermore, by combining multiple models trained with Batch Normalization, we perform better than the best known system on ImageNet, by a significant margin.

对现有最好的图像分类模型增加BN结构，可以得到一个加速的网络。进一步加大学习速率，去除dropout，应用其他修正，我们达到现有的最好效果只需要原训练步数的很小一部分，增加训练步骤还可以取得更好的分类效果。更进一步，将多个BN模型组合起来，我们超过了现有ImageNet上最好的结果很多。

Interestingly, our method bears similarity to the standardization layer of (Gülc ¸ehre & Bengio, 2013), though the two methods stem from very different goals, and perform different tasks. The goal of Batch Normalization is to achieve a stable distribution of activation values throughout training, and in our experiments we apply it before the nonlinearity since that is where matching the first and second moments is more likely to result in a stable distribution. On the contrary, (Gülc ¸ehre & Bengio, 2013) apply the standardization layer to the output of the nonlinearity, which results in sparser activations. In our large-scale image classification experiments, we have not observed the nonlinearity inputs to be sparse, neither with nor without Batch Normalization. Other notable differentiating characteristics of Batch Normalization include the learned scale and shift that allow the BN transform to represent identity (the standardization layer did not require this since it was followed by the learned linear transform that, conceptually, absorbs the necessary scale and shift), handling of convolutional layers, deterministic inference that does not depend on the mini-batch, and batch-normalizing each convolutional layer in the network.

有趣的是，我们的方法与(Gülc ¸ehre & Bengio, 2013)的标准化层有类似之处，但这两种方法的目标不一样，进行的任务也不一样。BN的目标是让激活的分布在训练的过程中更加稳定，我们的试验中将其应用在非线性处理之前，因为那是最初匹配进行的时候，更可能得到稳定的分布。相反，(Gülc ¸ehre & Bengio, 2013)将标准化层应用在非线性处理的输出上，这得到了更稀疏的激活。在我们的大规模图像分类试验中，我们没有观察到稀疏的非线性输入，有没有BN都没观察到。BN的其他显著区别包括学习尺度和位移参数，这可以使BN成为恒等变换，BN还可以处理卷积层。

In this work, we have not explored the full range of possibilities that Batch Normalization potentially enables. Our future work includes applications of our method to Recurrent Neural Networks (Pascanu et al., 2013), where the internal covariate shift and the vanishing or exploding gradients may be especially severe, and which would allow us to more thoroughly test the hypothesis that normalization improves gradient propagation(Sec. 3.3). We plan to investigate whether Batch Normalization can help with domain adaptation, in its traditional sense – i.e. whether the normalization performed by the network would allow it to more easily generalize to new data distributions, perhaps with just a recomputation of the population means and variances (Alg. 2). Finally, we believe that further theoretical analysis of the algorithm would allow still more improvements and applications.

本文中，我们没有探讨BN所有的可能性。未来我们将应用到RNN，那里的internal covariate shift和梯度消失/爆炸问题很严重，更进一步测试归一化是不是能改进梯度传播（3.3节）。我们计划研究BN是否可以帮助domain adaptation，即网络的归一化是不是可以更容易的泛化到其他数据分布，或许只是重新计算一下样本的均值和方差（算法2）。最后，我们相信进一步的理论分析可以得到更多改进。

## References

## Appendix