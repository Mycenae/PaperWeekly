# High-frequency Component Helps Explain the Generalization of Convolutional Neural Networks

Haohan Wang et. al. Carnegie Mellon University

## 0. Abstract

We investigate the relationship between the frequency spectrum of image data and the generalization behavior of convolutional neural networks (CNN). We first notice CNN’s ability in capturing the high-frequency components of images. These high-frequency components are almost imperceptible to a human. Thus the observation leads to multiple hypotheses that are related to the generalization behaviors of CNN, including a potential explanation for adversarial examples, a discussion of CNN’s trade-off between robustness and accuracy, and some evidence in understanding training heuristics.

我们研究了图像数据的频率谱分量与CNN的泛化行为之间的关系。我们首先注意到CNN捕获图像的高频分量的能力。这些高频分量对于人类来说几乎是不可感知的。因此这个观察可以带来多个假设，与CNN的泛化行为有关系，包括对对抗样本的可能解释，CNN在稳定性与准确率之间的折中，以及理解训练直觉的一些证据。

## 1. Introduction

Deep learning has achieved many recent advances in predictive modeling in various tasks, but the community has nonetheless become alarmed by the unintuitive generalization behaviors of neural networks, such as the capacity in memorizing label shuffled data [65] and the vulnerability towards adversarial examples [54, 21].

深度学习在各种任务的预测模型中已经取得了很多进展，但深度学习界仍然对神经网络的不同寻常的泛化行为产生了警惕，比如能够记住标签混洗的数据的能力[65]，和对对抗样本的脆弱性[54,21]。

To explain the generalization behaviors of neural networks, many theoretical breakthroughs have been made progressively, including studying the properties of stochastic gradient descent [31], different complexity measures [46], generalization gaps [50], and many more from different model or algorithm perspectives [30, 43, 7, 51].

为解释神经网络的泛化行为，已经逐渐有了很多理论进展，包括研究SGD[31]的属性，研究不同复杂度的度量[46]的属性，研究泛化间隙[50]的属性，以及其他从不同的模型或算法角度研究的工作。

In this paper, inspired by previous understandings that convolutional neural networks (CNN) can learn from confounding signals [59] and superficial signals [29, 19, 58], we investigate the generalization behaviors of CNN from a data perspective. Together with [27], we suggest that the unintuitive generalization behaviors of CNN as a direct outcome of the perceptional disparity between human and models (as argued by Figure 1): CNN can view the data at a much higher granularity than the human can.

本文中，受到之前的研究理解的启发，即CNN可以从混淆的信号和表面的信号学习到东西，我们从数据的角度研究了CNN的泛化行为。与[27]一起，我们建议CNN的不符合直觉的泛化行为，是人类和模型的感知差异的结果（如图1所示）：CNN可以比人类从更高的粒度观察数据。

However, different from [27], we provide an interpretation of this high granularity of the model’s perception: CNN can exploit the high-frequency image components that are not perceivable to human.

但是，与[27]不一样的是，我们提出了对这种模型感知的高度粒度的一种解释：CNN可以利用图像的高频分量，这对于人类是不可感知的。

For example, Figure 2 shows the prediction results of eight testing samples from CIFAR10 data set, together with the prediction results of the high and low-frequency component counterparts. For these examples, the prediction outcomes are almost entirely determined by the high-frequency components of the image, which are barely perceivable to human. On the other hand, the low-frequency components, which almost look identical to the original image to human, are predicted to something distinctly different by the model.

比如，图2给出了CIFAR数据集的8个测试样本的预测结果，以及高频和低频分量的预测结果。对于这些样本，预测结果几乎完全是由图像的高频分量所确定的，这对于人类几乎是不可感知的。另一方面，低频分量的预测结果则完全不同，而这对于人类来说看起来几乎完全一样。

Motivated by the above empirical observations, we further investigate the generalization behaviors of CNN and attempt to explain such behaviors via differential responses to the image frequency spectrum of the inputs (Remark 1). Our main contributions are summarized as follows:

受到上面的经验观察所推动，我们进一步研究了CNN的泛化行为，试图去通过对输入图像频域分量的不同响应来解释这样的行为。我们的主要贡献总结如下：

- We reveal the existing trade-off between CNN’s accuracy and robustness by offering examples of how CNN exploits the high-frequency components of images to trade robustness for accuracy (Corollary 1). 我们给出了CNN怎样利用图像的高频分量来以稳定性换取准确性的例子，揭示了现有的CNN在准确率和稳定性之间的折中关系（推论1）。

- With image frequency spectrum as a tool, we offer hypothesis to explain several generalization behaviors of CNN, especially the capacity in memorizing label-shuffled data. 用图像频谱作为工具，我们给出了假设，解释了几种CNN的泛化行为，尤其是记住标签混洗数据的能力。

- We propose defense methods that can help improving the adversarial robustness of CNN towards simple attacks without training or fine-tuning the model. 我们提出了防卫的方法，可以帮助改进CNN对简单攻击的对抗稳定性，而不需要训练或精调模型。

The remainder of the paper is organized as follows. In Section 2, we first introduce related discussions. In Section 3, we will present our main contributions, including a formal discussion on that CNN can exploit high-frequency components, which naturally leads to the trade-off between adversarial robustness and accuracy. Further, in Section 4-6, we set forth to investigate multiple generalization behaviors of CNN, including the paradox related to capacity of memorizing label-shuffled data (§4), the performance boost introduced by heuristics such as Mixup and BatchNorm (§5), and the adversarial vulnerability (§6). We also attempt to investigate tasks beyond image classification in Section 7. Finally, we will briefly discuss some related topics in Section 8 before we conclude the paper in Section 9.

本文剩下的部分组织如下。在第2部分，我们首先介绍了相关的讨论。第3部分中，我们会提出我们的主要贡献，包括CNN可以利用高频分量的正式讨论，这很自然的带来了对抗稳定性和准确率之间的折中。而且，在第4-6部分中，我们研究了CNN的多种泛化行为，包括与记住标签混洗数据能力的悖论(§4)，由MixUp和BatchNorm带来的性能提升(§5)，以及对抗的脆弱性(§6)。我们还试图在第7部分去研究除了图像分类以外的任务。最后，我们会在第8部分简要的讨论一些相关的话题，并在第9部分总结本文。

## 2. Related Work

The remarkable success of deep learning has attracted a torrent of theoretical work devoted to explaining the generalization mystery of CNN.

深度学习令人印象深刻的成功，已经吸引了很多理论工作致力于解释CNN泛化的谜团。

For example, ever since Zhang et al. [65] demonstrated the effective capacity of several successful neural network architectures is large enough to memorize random labels, the community sees a prosperity of many discussions about this apparent ”paradox” [61, 15, 17, 15, 11]. Arpit et al. [3] demonstrated that effective capacity are unlikely to explain the generalization performance of gradient-based-methods trained deep networks due to the training data largely determine memorization. Kruger et al.[35] empirically argues by showing largest Hessian eigenvalue increased when training on random labels in deep networks.

比如，自从Zhang等[65]证明了几种成功的神经网络架构的有效能力是足够大可以记住随机的标签，研究团体中就有了很多关于这种明显的悖论的很多讨论。Arpit等[3]证明了有效能力不太可能解释基于梯度的方法训练的网络的泛化能力，因为训练数据主要决定了记忆。Kruger等[35]通过经验认为，当在深度网络中训练随机标签时，其最大的Hessian特征值会变大。

The concept of adversarial example [54, 21] has become another intriguing direction relating to the behavior of neural networks. Along this line, researchers invented powerful methods such as FGSM [21], PGD [42], and many others [62, 9, 53, 36, 12] to deceive the models. This is known as attack methods. In order to defend the model against the deception, another group of researchers proposed a wide range of methods (known as defense methods) [1, 38, 44, 45, 24]. These are but a few highlights among a long history of proposed attack and defense methods. One can refer to comprehensive reviews for detailed discussions [2, 10].

对抗样本的概念已经成了另一个令人好奇的方向，与神经网络的行为相关。在这条线上，研究者提出一些很强的方法，如FGSM，PGD和很多其他的方法来欺骗模型。这被称为攻击方法。为保护模型不受欺骗，另一组研究者提出很多方法，称为防御方法。在提出攻击和防御方法的很长的历史中，有很多亮点。可以参考综述进行详细的讨论。

However, while improving robustness, these methods may see a slight drop of prediction accuracy, which leads to another thread of discussion in the trade-off between robustness and accuracy. The empirical results in [49] demonstrated that more accurate model tend to be more robust over generated adversarial examples. While [25] argued that the seemingly increased robustness are mostly due to the increased accuracy, and more accurate models (e.g., VGG, ResNet) are actually less robust than AlexNet. Theoretical discussions have also been offered [56, 67], which also inspires new defense methods [67].

但是，在改进稳健性的同时，这些方法可能会略微降低预测准确率，这带来了另一条讨论线，在稳健性和准确率之间有折中。[49]的经验性结果证明了，更准确的模型在面对生成的对抗样本时会更加稳健。而[25]认为，似乎增加的稳健性，主要是因为增加的准确率，更准确的模型（如VGG，ResNet）没有AlexNet那么稳健。也有一些理论探讨，也启发了新的防御方法。

## 3. High-frequency Components & CNN’s Generalization

We first set up the basic notations used in this paper: ⟨x, y⟩ denotes a data sample (the image and the corresponding label). f(·;θ) denotes a convolutional neural network whose parameters are denoted as θ. We use H to denote a human model, and as a result, f(·;H) denotes how human will classify the data ·. l(·, ·) denotes a generic loss function (e.g., cross entropy loss). α(·, ·) denotes a function evaluating prediction accuracy (for every sample, this function yields 1.0 if the sample is correctly classified, 0.0 otherwise). d(·, ·) denotes a function evaluating the distance between two vectors. F(·) denotes the Fourier transform; thus, $F^{−1}$(·) denotes the inverse Fourier transform. We use z to denote the frequency component of a sample. Therefore, we have z=F(x) and x=$F^{-1}$(z).

我们首先设置好本文中使用的基本表示：⟨x, y⟩表示一个数据样本（图像和对应的标签）。f(·;θ)表示一个CNN，其参数表示为θ。我们使用H来表示一个人类模型，那么f(·;H)表示人类怎样对数据·进行分类。l(·, ·)表示一个通用损失函数（如交叉熵）。α(·, ·)表示一个评估预测准确率的函数（对每个样本，如果正确分类则得出1.0的结果，否则为0.0）。d(·, ·)表示一个评估两个向量之间距离的函数。F(·)表示Fourier变换；因此，$F^{−1}$(·)表示逆Fourier变换。我们使用z来表示一个样本的频率分量。因此，我们有z=F(x)，x=$F^{-1}$(z)。

Notice that Fourier transform or its inverse may introduce complex numbers. In this paper, we simply discard the imaginary part of the results of $F^{-1}$(·) to make sure the resulting image can be fed into CNN as usual.

注意Fourier变换或其逆可能会引入复数。本文中，我们简单的将$F^{-1}$(·)的虚数部分丢弃，以确保结果图像可以正常的送入CNN。

### 3.1. CNN Exploit High-frequency Components

We decompose the raw data x = {$x_l, x_h$}, where $x_l$ and $x_h$ denote the low-frequency component (shortened as LFC) and high-frequency component (shortened as HFC) of x. We have the following four equations:

我们将原始数据分解为x = {$x_l, x_h$}，其中$x_l$和$x_h$表示x的低频分量和高频分量，简写为LFC和HFC。我们有下面4个等式：

$$z = F(x), z_l, z_h = t(z; r)$$
$$x_l = F^{−1}(z_l), x_h = F^{−1}(z_h)$$

where t(·; r) denotes a thresholding function that separates the low and high frequency components from z according to a hyperparameter, radius r.

其中t(·; r)表示一个阈值函数，将z的高频和低频分量根据一个超参数半径r进行分割。

To define t(·; r) formally, we first consider a grayscale (one channel) image of size n × n with N possible pixel values (in other words, $x ∈ N^{n×n}$), then we have $z ∈ C^{n×n}$, where C denotes the complex number. We use z(i, j) to index the value of z at position (i, j), and we use $c_i$, $c_j$ to denote the centroid. We have the equation $z_l, z_h$ = t(z; r) formally defined as:

为正式的定义t(·; r)，我们首先考虑一个灰度图像（单通道），大小为n × n，有N个可能的像素值（换句话说，$x ∈ N^{n×n}$），那么我们有$z ∈ C^{n×n}$，其中C表示复数。我们用z(i, j)表示z在位置(i, j)处的值，我们用$c_i$, $c_j$来表示重心。我们可以将$z_l, z_h$ = t(z; r)正式的定义为：

$$z_l(i,j) = \left\{ \begin{matrix} z(i,j), & if d((i,j),(c_i,c_j)) ≤ r \\ 0, & otherwise \end{matrix} \right.$$
$$z_h(i,j) = \left\{ \begin{matrix} 0, & if d((i,j),(c_i,c_j)) ≤ r \\ z(i,j), & otherwise \end{matrix} \right.$$

We consider d(·, ·) in t(·; r) as the Euclidean distance in this paper. If x has more than one channel, then the procedure operates on every channel of pixels independently.

在本文中我们设t(·; r)中的d(·, ·)为欧式距离。如果x多于一个通道，那么这个过程在每个通道中独立进行。

**Remark 1**. With an assumption (referred to as A1) that presumes “only $x_l$ is perceivable to human, but both $x_l$ and $x_h$ are perceivable to a CNN,” we have:

假设A1人类只能感知到$x_l$，而CNN可以感知到$x_l$和$x_h$，我们有：

$$y := f(x;H) = f(x_l;H)$$

but when a CNN is trained with 但当一个CNN是用下面进行训练的

$$arg min_θ l(f(x; θ), y)$$

which is equivalent to 这等价于

$$arg min_θ l(f({x_l, x_h}; θ), y)$$

CNN may learn to exploit $x_h$ to minimize the loss. As a result, CNN’s generalization behavior appears unintuitive to a human.

CNN可能学习利用$x_h$来最小化损失。结果是，CNN的泛化行为似乎对于人类来说不那么直观。

Notice that “CNN may learn to exploit $x_h$” differs from “CNN overfit” because $x_h$ can contain more information than sample-specific idiosyncrasy, and these more information can be generalizable across training, validation, and testing sets, but are just imperceptible to a human.

注意，CNN可能会学习利用$x_h$，与CNN过拟合，是不一样的，因为$x_h$可以比样本具体的特质包含更多信息，而这些更多的信息在训练、验证和测试集上都可以泛化，但对于人类可能是不可感知的。

As Assumption A1 has been demonstrated to hold in some cases (e.g., in Figure 2), we believe Remark 1 can serve as one of the explanations to CNN’s generalization behavior. For example, the adversarial examples [54, 21] can be generated by perturbing $x_h$; the capacity of CNN in reducing training error to zero over label shuffled data [65] can be seen as a result of exploiting $x_h$ and overfitting sample-specific idiosyncrasy. We will discuss more in the following sections.

因为假设A1在一些情况下是被证明成立的（如，在图2中），我们相信Remark 1可以作为CNN泛化行为的一种解释。比如，对抗样本[54,21]可以通过扰动$x_h$产生；CNN在标签混洗数据上将训练误差缩小为0的能力，可以看作是利用$x_h$的结果，并在样本特定的特质中过拟合。我们在下节中讨论更多。

### 3.2. Trade-off between Robustness and Accuracy

We continue with Remark 1 and discuss CNN’s trade-off between robustness and accuracy given θ from the image frequency perspective. We first formally state the accuracy of θ as:

我们继续Remark 1，在给定θ时，从图像频率的角度，讨论CNN在稳健性和准确率之间的折中。我们首先正式的将θ的准确率表示为：

$$E_{(x,y)} α(f(x; θ), y)$$(1)

and the adversarial robustness of θ as in e.g., [8]: θ的对抗稳健性为，如[8]：

$$E_{(x,y)} min_{x':d(x',x)≤ε} α(f(x'; θ), y)$$(2)

where ε is the upper bound of the perturbation allowed. 其中ε是允许的扰动的上限。

With another assumption (referred to as A2): “for model θ, there exists a sample ⟨x, y⟩ such that: 在另一个假设下（称为A2）：对于模型θ，存在一个样本⟨x, y⟩，使得

$$f(x;θ) \neq f(x_l;θ)$$

we can extend our main argument (Remark 1) to a formal statement: 我们可以将我们的主要观点(Remark 1)拓展成一个正式的表述：

Corollary 1. With assumptions A1 and A2, there exists a sample ⟨x, y⟩ that the model θ cannot predict both accurately (evaluated to be 1.0 by Equation 1) and robustly (evaluated to be 1.0 by Equation 2) under any distance metric d(·,·) and bound ε as long as ε≥d(x,x_l).

推论1. 有了假设A1和A2，存在一个样本⟨x, y⟩，模型θ在任意度量d(·,·)和界限ε下，只要ε≥d(x,x_l)，不可能同时进行准确（式1预测为1）和稳健（式2预测为1）的预测。

The proof is a direct outcome of the previous discussion and thus omitted. The Assumption A2 can also be verified empirically (e.g., in Figure 2), therefore we can safely state that Corollary 1 can serve as one of the explanations to the trade-off between CNN’s robustness and accuracy.

证明是之前的讨论的直接结果，因此忽略。假设A2也可以通过经验进行验证（如图2），因此我们可以安全的表述，推论1可以是CNN的稳健性和准确率的折中的一个解释。

## 4. Rethinking Data before Rethinking Generalization

### 4.1. Hypothesis

Our first aim is to offer some intuitive explanations to the empirical results observed in [65]: neural networks can easily fit label-shuffled data. While we have no doubts that neural networks are capable of memorizing the data due to its capacity, the interesting question arises: “if a neural network can easily memorize the data, why it cares to learn the generalizable patterns out of the data, in contrast to directly memorizing everything to reduce the training loss?”

我们的第一个目标是为[65]中的经验性结果给出一些直观的解释：神经网络可以很容易的适配标签混洗的数据。当我们对神经网络由于其能力可以记住数据毫无疑问，同时出现的有趣的问题出现了：如果神经网络可以很容易的记住数据，为什么还要学习数据的可泛化的模式，而不是直接记住所有事以降低训练损失？

Within the perspective introduced in Remark 1, our hypothesis is as follows: Despite the same outcome as a minimization of the training loss, the model considers different level of features in the two situations:

在Remark 1提出的视角之内，我们的假设如下：尽管和训练损失的最小化是同样的结果，模型在两种情况下考虑不同层次的特征：

- In the original label case, the model will first pick up LFC, then gradually pick up the HFC to achieve higher training accuracy. 在原始标签的情况下，模型首先选择LFC，然后逐渐选择HFC以得到更高的训练准确率；

- In the shuffled label case, as the association between LFC and the label is erased due to shuffling, the model has to memorize the images when the LFC and HFC are treated equally. 在混洗标签的情况下，由于LFC与标签的关系由于混洗被擦除了，在LFC和HFC是等同对待的情况下，模型必须记住图像。

### 4.2. Experiments

We set up the experiment to test our hypothesis. We use ResNet-18 [22] for CIFAR10 dataset [33] as the base experiment. The vanilla set-up, which we will use for the rest of this paper, is to run the experiment with 100 epoches with the ADAM optimizer [32] with learning rate set to be 10^−4 and batch size set to be 100, when weights are initialized with Xavier initialization [20]. Pixels are all normalized to be [0, 1]. All these experiments are repeated in MNIST [14], FashionMNIST [63], and a subset of ImageNet [13]. These efforts are reported in the Appendix. We train two models, with the natural label setup and the shuffled label setup, denote as M_natural and M_shuffle, respectively; the M_shuffle needs 300 epoches to reach a comparative training accuracy. To test which part of the information the model picks up, for any x in the training set, we generate the low-frequency counterparts x_l with r set to 4, 8, 12, 16 respectively. We test the how the training accuracy changes for these low-frequency data collections along the training process.

我们设计实验来测试我们的假设。我们在CIFAR10数据集上使用ResNet-18作为基准实验。我们在本文剩下的篇幅中都使用的经典设置，是实验用ADAM优化器运行100轮训练，学习速率设置为10^−4，批大小设置为100，权重是用Xavier初始化。像素归一化到[0,1]。所有实验都用MNIST，FashionMNIST和ImageNet的子集进行重复。这些工作在附录中进行了叙述。我们训练两个模型，一个是自然的标签设置，一个是混洗的标签设置，分别表示为M_natrual和M_shuffle；M_shuffle需要300轮以达到类似的训练准确率。为测试模型选择了哪部分信息，对于训练集中的任意x，我们分别使用r值为4, 8, 12, 16，生成了低频分量x_l。我们测试了训练准确率对于这些低频数据集合在训练过程中是怎样变化的。

The results are plotted in Figure 3. The first message is the M_shuffle takes a longer training time than M_natural to reach the same training accuracy (300 epoches vs. 100 epoches), which suggests that memorizing the samples as an “unnatural” behavior in contrast to learning the generalizable patterns. By comparing the curves of the low-frequent training samples, we notice that M_natural learns more of the low-frequent patterns (i.e., when r is 4 or 8) than M_shuffle. Also, M_shuffle barely learns any LFC when r = 4, while on the other hand, even at the first epoch, M_natural already learns around 40% of the correct LFC when r = 4. This disparity suggests that when M_natural prefers to pick up the LFC, M_shuffle does not have a preference between LFC vs. HFC.

结果如图3所示。第一个信息是，M_shuffle比M_natural需要更多的训练时间达到相同的训练准确率(300 epoches vs. 100 epoches)，这说明，与学习可泛化的模式相比，记住样本是一个非自然的行为。通过比较低频训练样本的曲线，我们注意到，与M_shuffle相比，M_natural学习的更多是低频模式（即，当r是4或8时）。同时，M_shuffle在r=4时，几乎从LFC中学不到什么，而另一方面，即使在第一轮学习中，在r=4时M_natural已经学习了大约40%的正确的LFC。这种差异说明，M_natural倾向于选择LFC，而M_shuffle在LFC与HFC中并没有什么倾向性选择。

If a model can exploit multiple different sets of signals, then why M_natural prefers to learn LFC that happens to align well with the human perceptual preference? While there are explanations suggesting neural networks’ tendency towards simpler functions [48], we conjecture that this is simply because, since the data sets are organized and annotated by human, the LFC-label association is more “generalizable” than the one of HFC: picking up LFC-label association will lead to the steepest descent of the loss surface, especially at the early stage of the training.

如果一个模型可以利用多个信号集合，那么M_natural为什么倾向于学习LFC呢，这刚好与人类感知倾向很好的贴合。有一些解释说，神经网络倾向于更简单的函数，我们推测这只是因为，由于数据集是由人组织和标注的，LFC-标签之间的关联比HFC泛化能力更加强：选择LFC-标签的关联会带来损失面的最陡下降，尤其是在训练的初级阶段。

To test this conjecture, we repeat the experiment of M_natural, but instead of the original train set, we use the x_l or x_h (normalized to have the standard pixel scale) and test how well the model can perform on original test set. Table 1 suggests that LFC is much more “generalizable” than HFC. Thus, it is not surprising if a model first picks up LFC as it leads to the steepest descent of the loss surface.

为测试这个推测，我们重复了M_natural的实验，但我们没有使用原始的训练集，而是使用了x_l或x_h（进行了归一化，以得到标准的像素尺度），测试模型在原始测试集上表现如何。表1说明，LFC比HFC泛化性能更好。因此，如果模型首先选择LFC，这是不令人惊讶的，因为这带来了损失曲面的最陡下降。

### 4.3. A Remaining Question

Finally, we want to raise a question: The coincidental alignment between networks’ preference in LFC and human perceptual preference might be a simple result of the “survival bias” of the many technologies invented one of the other along the process of climbing the ladder of the state-of-the-art. In other words, the almost-100-year development process of neural networks functions like a “natural selection” of technologies [60]. The survived ideas may happen to match the human preferences, otherwise, the ideas may not even be published due to the incompetence in climbing the ladder.

最后，我们想提出一个问题：网络对LFC的倾向，与人类感知的倾向的恰好重合，可是只是发明的很多技术的幸存者偏差的结果。换句话说，神经网络近乎100年的发展过程，很像技术的自然选择。留存下来的思想可能恰巧与人类倾向相匹配，否则，这些思想甚至不会发表出来，因为不能够爬到这个梯子上。

However, an interesting question will be how well these ladder climbing techniques align with the human visual preference. We offer to evaluate these techniques with our frequency tools.

但是，一个有趣的问题是，这些爬梯子的技术与人类视觉倾向符合度有多少。我们用常用的工具来评估这些技术。

## 5. Training Heuristics

We continue to reevaluate the heuristics that helped in climbing the ladder of state-of-the-art accuracy. We evaluate these heuristics to test the generalization performances towards LFC and HFC. Many renowned techniques in the ladder of accuracy seem to exploit HFC more or less.

我们继续重新评估帮助攀爬目前最好准确率梯子的启发式。我们评估这些启发式，以测试向LFC和HFC的泛化性能。在这个准确率梯子上，很多著名的技术似乎或多或少利用了HFC。

### 5.1. Comparison of Different Heuristics

We test multiple heuristics by inspecting the prediction accuracy over LFC and HFC with multiple choices of r along the training process and plot the training curves.

我们测试了多种启发式，检视了训练过程中多个r对应的LFC和HFC上的预测准确率，画出了训练曲线。

**Batch Size**: We then investigate how the choices of batch size affect the generalization behaviors. We plot the results in Figure 4. As the figure shows, smaller batch size appears to excel in improving training and testing accuracy, while bigger batch size seems to stand out in closing the generalization gap. Also, it seems the generalization gap is closely related to the model’s tendency in capturing HFC: models trained with bigger epoch sizes are more invariant to HFC and introduce smaller differences in training accuracy and testing accuracy. The observed relation is intuitive because the smallest generalization gap will be achieved once the model behaves like a human (because it is the human who annotate the data).

**批次规模**：我们研究了批次规模的选项怎样影响泛化行为。我们在图4中画出了结果。如图所示，小的批次在改进训练和测试准确率上似乎更好，而大的批次似乎在泛化差距的消失上更胜一筹。因此，似乎泛化差距与模型捕获HFC的能力更加紧密相关：用更大的epoch sizes训练的模型，对HFC的不变性更强，在训练准确率与测试准确率的差异上更小。观察到的联系是符合直觉的，因为一旦模型的行为像一个人的时候，就会得到最小的泛化差距（因为是人类标注了数据）。

The observation in Figure 4 also chips in the discussion in the previous section about “generalizable” features. Intuitively, with bigger epoch size, the features that can lead to steepest descent of the loss surface are more likely to be the “generalizable” patterns of the data, which are LFC.

图4中的观察也与上一节中讨论的可泛化的特征相关。直觉上来说，在更大的epoch size下，对损失曲面带来最陡梯度的特征，更可能是数据的可泛化的特征，也就是LFC。

**Heuristics**: We also test how different training methods react to LFC and HFC, including 我们还测试了不同的训练方法对LFC与HFC是怎样反应的，包括：

- Dropout [26]: A heuristic that drops weights randomly during training. We apply dropout on fully-connected layers with p = 0.5. 在训练中随机丢弃权重的启发式。我们在全连接层中以p=0.5来使用dropout。

- Mix-up [66]: A heuristic that linearly integrate samples and their labels during training. We apply it with standard hyperparameter α = 0.5. 将样本及其标签在训练过程中线性叠加的启发式。我们使用标准超参数α = 0.5。

- BatchNorm [28]: A method that perform the normalization for each training mini-batch to accelerate Deep Network training process. It allows us to use a much higher learning rate and reduce overfitting, similar with Dropout. We apply it with setting scale γ to 1 and offset β to 0. 对每个训练的mini-batch进行归一化的方法，以加速深度网络的训练过程。这使我们可以使用更大的学习速率，并减少过拟合的现象，与Dropout类似。我们设尺度γ为1，偏移β为0，并应用这种技术。

- Adversarial Training [42]: A method that augments the data through adversarial examples generated by a threat model during training. It is widely considered as one of the most successful adversarial robustness (defense) method. Following the popular choice, we use PGD with ε = 8/255 (ε = 0.03 ) as the threat model. 使用对抗样本来对数据扩增，对抗样本由威胁模型在训练过程中生成。这是一个最成功的对抗稳健性方法。按照流行的选择，我们在威胁模型中使用PGDε = 8/255 (ε = 0.03)。

We illustrate the results in Figure 5, where the first panel is the vanilla set-up, and then each one of the four heuristics are tested in the following four panels.

我们在图5中描述了结果，其中第一个面板是经典设置，然后下面4个面板分别是4个启发式。

Dropout roughly behaves similarly to the vanilla set-up in our experiments. Mix-up delivers a similar prediction accuracy, however, it catches much more HFC, which is probably not surprising because the mix-up augmentation does not encourage anything about LFC explicitly, and the performance gain is likely due to attention towards HFC.

在我们的实验中，Dropout与经典设置表现类似。Mix-up得到了类似的预测准确率，但是，捕获的更多的是HFC，这可能并不令人惊讶，因为mix-up数据扩增并没有鼓励与LFC有关的分量，性能提升很可能是因为HFC有关的注意力。

Adversarial training mostly behaves as expected: it reports a lower prediction accuracy, which is likely due to the trade-off between robustness and accuracy. It also reports a smaller generalization gap, which is likely as a result of picking up “generalizable” patterns, as verified by its invariance towards HFC (e.g., r = 12 or r = 16). However, adversarial training seems to be sensitive to the HFC when r = 4, which is ignored even by the vanilla set-up.

对抗训练与期望的行为相符：给出了更低的预测准确率，主要是因为稳健性和准确率的折中。还给出了更小的泛化差距，这很可能选择了可泛化的模式的结果，对HFC的不变性（如r=12或r=16）也验证了这个结果。但是，对抗训练似乎在r=4时对HFC很敏感，即使是在经典设置中，这都是可以被忽略的。

The performance of BatchNorm is notable: compared to the vanilla set-up, BatchNorm picks more information in both LFC and HFC, especially when r = 4 and r = 8. This BatchNorm’s tendency in capturing HFC is also related to observations that BatchNorm encourages adversarial vulnerability [18].

BatchNorm的效果是显著的：与经典设置相比，BatchNorm选择了更多的LFC和HFC信息，尤其是在r=4和r=8时。BatchNorm捕获HFC的这种倾向性，与BatchNorm鼓励对抗脆弱性的观察也是相关的。

**Other Tests**: We have also tested other heuristics or methods by only changing along one dimension while the rest is fixed the same as the vanilla set-up in Section 4. 我们还测试了其他的启发式或方法，只改变了一个维度，其他的与第4节中的经典设置一样。

Model architecture: We tested LeNet [37], AlexNet [34], VGG [52], and ResNet [23]. The ResNet architecture seems advantageous toward previous inventions at different levels: it reports better vanilla test accuracy, smaller generalization gap (difference between training and testing accuracy), and a weaker tendency in capturing HFC.

模型架构：我们测试了LeNet, AlexNet, VGG, 和ResNet。与之前的架构相比，ResNet架构似乎更好一些：给出了更好的经典测试准确率，更小的泛化差距（训练和测试准确率之间的差异），和捕获HFC方面更弱的倾向性。

Optimizer: We tested SGD, ADAM [32], AdaGrad [16], AdaDelta [64], and RMSprop. We notice that SGD seems to be the only one suffering from the tendency towards significantly capturing HFC, while the rest are on par within our experiments.

优化器：我们测试了SGD, ADAM, AdaGrad, AdaDelta, 和RMSprop。我们注意到SGD似乎是唯一一个倾向于显著捕获HFC的方法，其余的在我们实验中都比较类似。

### 5.2. A hypothesis on Batch Normalization

Based on the observation, we hypothesized that one of BatchNorm’s advantage is, through normalization, to align the distributional disparities of different predictive signals. For example, HFC usually shows smaller magnitude than LFC, so a model trained without BatchNorm may not easily pick up these HFC. Therefore, the higher convergence speed may also be considered as a direct result of capturing different predictive signals simultaneously.

基于观察，我们假设，BatchNorm的一个优势在于，通过归一化，将不同的预测信号的分布差异对齐了。比如，HFC通常比LFC展现出更小的幅度，所以不用BatchNorm训练出来的模型可能不会很容易选择这些HFC。因此，更快的收敛速度也是同时捕获不同的预测信号的直接结果。

To verify this hypothesis, we compare the performance of models trained with vs. without BatchNorm over LFC data and plot the results in Figure 6.

为验证这个假设，我们比较了使用和不使用BatchNorm训练得到的模型在LFC上的性能，在图6中画出了结果。

As Figure 6 shows, when the model is trained with only LFC, BatchNorm does not always help improve the predictive performance, either tested by original data or by corresponding LFC data. Also, the smaller the radius is, the less the BatchNorm helps. Also, in our setting, BatchNorm does not generalize as well as the vanilla setting, which may raise a question about the benefit of BatchNorm.

如图6所示，当模型只用LFC训练时，BatchNorm并不一定改进预测性能，我们用原始数据或对应的LFC数据都进行了测试。同时，半径越小，BatchNorm的帮助越小。同时，在我们的设置中，BatchNorm并不像经典设置中泛化的那么好，这带来一个关于BatchNorm的优势的问题。

However, BatchNorm still seems to at least boost the convergence of training accuracy. Interestingly, the acceleration is the smallest when r = 4. This observation further aligns with our hypothesis: if one of BatchNorm’s advantage is to encourage the model to capture different predictive signals, the performance gain of BatchNorm is the most limited when the model is trained with LFC when r = 4.

但是，BatchNorm似乎仍然至少加速了训练准确率的收敛。有趣的是，在r=4时加速效果是最小的。这种观察进一步与我们的假设一致：如果BatchNorm的一个优势是鼓励模型捕获不同的预测信号，BatchNorm的性能提升在使用LFC进行训练时是最有限的，即r=4的情况。

## 6. Adversarial Attack & Defense

As one may notice, our observation of HFC can be directly linked to the phenomenon of “adversarial example”: if the prediction relies on HFC, then perturbation of HFC will significantly alter the model’s response, but such perturbation may not be observed to human at all, creating the unintuitive behavior of neural networks.

我们可以注意到，我们对HFC的观察可以与对抗样本的现象直接相关：如果预测依赖于HFC，那么HFC的扰动会显著改变模型的预测，但这样的扰动人类可能观察不到，这就产生了神经网络的反直觉的行为。

This section is devoted to study the relationship between adversarial robustness and model’s tendency in exploiting HFC. We first discuss the linkage between the “smoothness” of convolutional kernels and model’s sensitivity towards HFC (§6.1), which serves the tool for our follow-up analysis. With such tool, we first show that adversarially robust models tend to have “smooth” kernels (§6.2), and then demonstrate that directly smoothing the kernels (without training) can help improve the adversarial robustness towards some attacks (§6.3).

这一节致力于研究对抗稳健性和模型利用HFC的倾向的关系。我们首先讨论卷积核的平滑程度和模型对HFC的敏感度的关系，这是我们后续分析的工具。有了这样的工具，我们首先表明，对抗稳健的模型，其卷积核倾向于很平滑(§6.2)，然后证明了对卷积核的直接平滑（不进行训练）可以帮助改进模型对一些攻击的对抗稳健性(§6.3)。

### 6.1. Kernel Smoothness vs. Image Frequency

As convolutional theorem [6] states, the convolution operation of images is equivalent to the element-wise multiplication of image frequency domain. Therefore, roughly, if a convolutional kernel has negligible weight at the high-end of the frequency domain, it will weigh HFC accordingly. This may only apply to the convolutional kernel at the first layer because the kernels at higher layer do not directly with the data, thus the relationship is not clear.

如卷积定理所述，图像的卷积运算等价于图像频域的逐元素的相乘。因此，大致上，如果一个卷积核在频域的高频部分分量可以忽略，那么就对HFC进行相应的加权。这可能只对第一层的卷积核有用，因为更高层的核与数据并不直接关联，因此其关系是不清楚的。

Therefore, we argue that, to push the model to ignore the HFC, one can consider to force the model to learn the convolutional kernels that have only negligible weights at the high-end of the frequency domain.

因此，我们认为，为迫使模型忽略HFC，我们可以迫使模型学习的卷积核在高频部分只有可忽略的权重。

Intuitively (from signal processing knowledge), if the convolutional kernel is “smooth”, which means that there is no dramatics fluctuations between adjacent weights, the corresponding frequency domain will see a negligible amount of high-frequency signals. The connections have been mathematically proved [47, 55], but these proved exact relationships are out of the scope of this paper.

直觉上来说（从信号处理的知识来说），如果卷积核是平滑的，这意味着在相邻的权重之间没有显著的波动，对应的频域会看到可以忽略的高频信号。这个关系是已经在数学上被证明的，但这些被证明的严格关系不在本文探讨范围之内。

### 6.2. Robust Models Have Smooth Kernels

To understand the connection between “smoothness” and adversarial robustness, we visualize the convolutional kernels at the first layer of the models trained in the vanilla manner (M_natural) and trained with adversarial training (M_adversarial) in Figure 7 (a) and (b).

为理解平滑性和对抗稳健性之间的关系，我们对第一层的卷积核进行可视化，包括经典方式训练的权重(M_natural)，和对抗训练得到的权重(M_adversarial)，如图7a和7b。

Comparing Figure 7(a) and Figure 7(b), we can see that the kernels of M_adversarial tend to show a more smooth pattern, which can be observed by noticing that the adjacent weights of kernels of M_adversarial tend to share the same color. The visualization may not be very clear because the convolutional kernel is only [3 × 3] in ResNet, the message is delivered more clearly in Appendix with other architecture when the first layer has kernel of the size [5 × 5].

比较图7a和图7b，我们可以看到M_adversarial的核倾向于展示更平滑的模式，这可以通过注意到，M_adversarial的核的临近权重倾向于颜色类似，从而得到这个观察。这个可视化可能没那么清楚，因为ResNet中的卷积核只有3 × 3大小，在附录中这个信息可以更清楚的看到，因为其他架构的第一层其核大小为5 × 5。

### 6.3. Smoothing Kernels Improves Adversarial Robustness

The intuitive argument in §6.1 and empirical findings in §6.2 directly lead to a question of whether we can improve the adversarial robustness of models by smoothing the convolutional kernels at the first layer.

在6.1中的直觉观点，和6.2中的经验发现，带来了一个问题，我们是否可以通过平滑第一层的卷积核，来改进对抗稳健性。

Following the discussion, we introduce an extremely simple method that appears to improve the adversarial robustness against FGSM [21] and PGD [36]. For a convolutional kernel w, we use i and j to denote its column and row indices, thus w_i,j denotes the value at i-th row and j-th column. If we use N (i, j) to denote the set of the spatial neighbors of (i, j), our method is simply:

按照讨论，我们介绍了一个极其简单的方法，可以改进对抗FGSM和PGD的对抗稳健性。对于一个卷积核w，我们使用i和j来表示行和列的索引，因此w_i,j表示在第i行第j列的值。如果我们使用N (i, j)来表示(i, j)附近的空间邻域集合，那么我们的方法就是：

$$w_{i,j} = w_{i,j} + \sum_{(h,k)∈N(i,j)} ρw_{h,k}$$(3)

where ρ is a hyperparameter of our method. We fix N (i, j) to have eight neighbors. If (i,j) is at the edge, then we simply generate the out-of-boundary values by duplicating the values on the boundary.

其中ρ是方法的一个超参数。我们固定N (i, j)有8个邻域，如果(i,j)是在边缘，那么我们只是生成out-of-boundary的值，复制边缘的值。

In other words, we try to smooth the kernel through simply reducing the adjacent differences by mixing the adjacent values. The method barely has any computational load, but appears to improve the adversarial robustness of M_natural and M_adversarial towards FGSM and PGD, even when M_adversarial is trained with PGD as the threat model.

换句话说，我们通过混合临近的值，来减小相邻之间的差异，来平滑卷积核。这个方法基本没有计算量，但似乎可以改进M_natural和M_adversarial对FGSM和PGD的对抗稳健性，即使M_adversarial使用PGD作为威胁模型来进行训练。

In Figure 7, we visualize the convolutional kernels with our method applied to M_natural and M_adversarial with ρ = 1.0, denoted as M_natural(ρ = 1.0) and M_adversarial(ρ = 1.0), respectively. As the visualization shows, the resulting kernels tend to show a significantly smoother pattern.

图7中，我们对卷积核进行可视化，用参数ρ = 1.0应用于M_natural和M_adversarial，分别表示为M_natural(ρ = 1.0)和M_adversarial(ρ = 1.0)。如可视化所示，得到的核倾向于展示出明显更平滑的模式。

We test the robustness of the models smoothed by our method against FGSM and PGD with different choices of ε, where the maximum of perturbation is 1.0. As Table 2 shows, when our smoothing method is applied, the performance of clean accuracy directly plunges, but the performance of adversarial robustness improves. In particular, our method helps when the perturbation is allowed to be relatively large. For example, when ε = 0.09 (roughly 23/255), M_natural(ρ = 1.0) even outperforms M_adversarial. In general, our method can easily improve the adversarial robustness of M_natural, but can only improve upon M_adversarial in the case where ε is larger, which is probably because the M_adversarial is trained with PGD(ε = 0.03) as the threat model.

我们测试我们模型对FGSM和PGD的稳健性，选择了不同的ε，其中最大的扰动是1.0。如表2所示，当使用了我们的平滑方法，clean准确率的性能直接下降了，但对抗稳健性的性能改进了。特别是，我们的方法在扰动相对较大时，会有帮助。比如，当ε = 0.09 (大致是23/255)，M_natural(ρ = 1.0)甚至超过了M_adversarial。总体上，我们的方法可以很容易的改进M_natural的对抗稳健性，但只能在ε更大的情况下改进M_adversarial，这可能是因为M_adversarial是用PGD(ε = 0.03)作为威胁模型来训练的。

## 7. Beyond Image Classification

We aim to explore more than image classification tasks. We investigate in the object detection task. We use RetinaNet [40] with ResNet50 [23] + FPN [39] as the backbone. We train the model with COCO detection train set [41] and perform inference in its validation set, which includes 5000 images, and achieve an MAP of 35.6%.

我们的目标不止是探索图像分类任务。我们研究了目标检测任务。我们使用了RetinaNet，用ResNet50+FPN作为骨干。我们使用COCO检测训练集训练模型，在其验证集上进行推理，包括5000幅图像，得到的MAP为35.6%。

Then we choose r = 128 and maps the images into x_l and x_h and test with the same model and get 27.5% MAP with LFC and 10.7% MAP with HFC. The performance drop from 35.6% to 27.5% intrigues us so we further study whether the same drop should be expected from human.

然后我们选择r = 128，将图像映射到x_l和x_h中，测试同样的模型，对LFC得到的MAP为27.5%，对HFC得到的MAP为10.7%。从35.6%到27.5%的性能下降，激发了我们进一步研究，人类是否也会得到相同的性能下降。

### 7.1. Performance Drop on LFC

The performance drop from the x to x_l may be expected because x_l may not have the rich information from the original images when HFC are dropped. In particular, different from image classification, HFC may play a significant role in depicting some objects, especially the smaller ones.

从x到x_l的性能下降是可以预期的，因为在HFC从原始图像中丢弃后，x_l可能不会有那么丰富的信息。特别是，与图像分类不同，HFC在描述一些目标中会起到显著的作用，尤其是一些较小的目标。

Figure 8 illustrates a few examples, where some objects are recognized worse in terms of MAP scores when the input images are replaced by the low-frequent counterparts. This disparity may be expected because the low-frequent images tend to be blurry and some objects may not be clear to a human either (as the left image represents).

图8给出了几个例子，如果输入图像替换为低频分量，其中一些目标识别的MAP更差了。这种差异可能是符合期望的，因为低频图像一般会比较模糊，一些目标对于人类来说可能也不清楚（如同左边图像所表示的）。

### 7.2. Performance Gain on LFC

However, the disparity gets interesting when we inspect the performance gap in the opposite direction. We identified 1684 images that for each of these images, the some objects are recognized better (high MAP scores) in comparison to the original images.

但是，当我们在相反的方向检查性能差异时，出现了有趣的现象。我们发现了1684幅图像，对于每幅图像，与原始图像相比，一些目标识别的更好了(MAP分数更高)。

The results are shown in Figure 9. There seems no apparent reasons why these objects are recognized better in low-frequent images, when inspected by human. These observations strengthen our argument in the perceptual disparity between CNN and human also exist in more advanced computer vision tasks other than image classification.

结果如图9所示。似乎没有明显的原因，为什么这些目标在低频图像中会识别的更好，当由人检查时。这些观察加强了我们关于CNN和人类之间感知差异的观点，不止在图像分类，在更高级的计算机视觉任务中也是存在的。

## 8. Discussion: Are HFC just Noises?

To answer this question, we experiment with another frequently used image denoising method: truncated singular value decomposition (SVD). We decompose the image and separate the image into one reconstructed with dominant singular values and one with trailing singular values. With this set-up, we find much fewer images supporting the story in Figure 2. Our observations suggest the signal CNN exploit is more than just random “noises”.

为回答这个问题，我们用另一种经常使用的图像去噪方法进行实验：截断SVD。我们将图像分解，将图像分成一个用主要奇异值重建的，和一个用trailing奇异值重建的。用这种设置，我们发现更少的图像支持图2中的故事。我们的观察说明，CNN利用的信号不止是随机噪声。

## 9. Conclusion & Outlook

We investigated how image frequency spectrum affects the generalization behavior of CNN, leading to multiple interesting explanations of the generalization behaviors of neural networks from a new perspective: there are multiple signals in the data, and not all of them align with human’s visual preference. As the paper comprehensively covers many topics, we briefly reiterate the main lessons learned:

我们研究了图像频谱怎样影响CNN的泛化行为，带来了CNN泛化行为的新角度的多个有趣的解释：在数据中有多个信号，并不是所有的信号都与人类的视觉偏好一致。由于文章包含了很多主题，我们简要的重复一下学习到的主要观点：

- CNN may capture HFC that are misaligned with human visual preference (§3), resulting in generalization mysteries such as the paradox of learning label-shuffled data (§4) and adversarial vulnerability (§6). CNN可能会捕获到HFC，而这与人类的视觉偏好是不一致的，得到的泛化谜团，比如学习到标签混洗数据的悖论和对抗脆弱性。

- Heuristics that improve accuracy (e.g., Mix-up and BatchNorm) may encourage capturing HFC (§5). Due to the trade-off between accuracy and robustness (§3), we may have to rethink the value of them. 改进准确率（如Mix-up和BatchNorm）的启发式，可能会鼓励捕获HFC。由于准确率与稳健性的折中，我们可能会重新思考它们的价值。

- Adversarially robust models tend to have smooth convolutional kernels, the reverse is not always true (§6). 对抗稳健的模型，其卷积核会很平滑，但反过来并不一定正确。

- Similar phenomena are noticed in the context of object detection (§7), with more conclusions yet to be drawn. 类似的现象在目标检测中也观察到了，但有很多结论尚待得出。

Looking forward, we hope our work serves as a call towards future era of computer vision research, where the state-of-the-art is not as important as we thought. 向前看，我们希望我们的研究为计算机视觉研究的未来时代带来召唤，其中目前最好的结果我们认为并不重要。

- A single numeric on the leaderboard, while can significantly boost the research towards a direction, does not reliably reflect the alignment between models and human, while such an alignment is arguably paramount. 排行榜上的一个数字，可以显著将研究向一个方向推进，但并不可靠的反应模型与人类的一致性，这样一种一致性是非常重要的。

- We hope our work will set forth towards a new testing scenario where the performance of low-frequent counterparts needs to be reported together with the performance of the original images. 我们希望我们的工作可以开启一种新的测试场景，其中低频分量的性能需要与原始图像一起给出。

- Explicit inductive bias considering how a human views the data (e.g., [58, 57]) may play a significant role in the future. In particular, neuroscience literature have shown that human tend to rely on low-frequent signals in recognizing objects [4, 5], which may inspire development of future methods. 人类怎样观看数据有关的显式的引入偏移，可能会在未来扮演重要的角色。特别是，神经科学文献已经证明，人类倾向于依赖于低频信号来识别目标，这可能会启发未来方法的提出。