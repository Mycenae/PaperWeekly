# Identity Mappings in Deep Residual Networks 深度残差网络中的恒等映射

Kaiming He et al. Microsoft Research

## Abstract 摘要

Deep residual networks [1] have emerged as a family of extremely deep architectures showing compelling accuracy and nice convergence behaviors. In this paper, we analyze the propagation formulations behind the residual building blocks, which suggest that the forward and backward signals can be directly propagated from one block to any other block, when using identity mappings as the skip connections and after-addition activation. A series of ablation experiments support the importance of these identity mappings. This motivates us to propose a new residual unit, which makes training easier and improves generalization. We report improved results using a 1001-layer ResNet on CIFAR-10 (4.62% error) and CIFAR-100, and a 200-layer ResNet on ImageNet. Code is available at: https://github.com/KaimingHe/resnet-1k-layers.

深度残差网络[1]是一类极其深的网络架构，有着非常好的准确率和非常好的收敛特性。本文中，我们分析了残差模块背后的传播公式，得出结论，当使用恒等映射作为跳跃连接和加法后的激活时，前向和反向信号可以从一个模块直接传播到另一个模块。一系列分离试验也支持这个结论，即这些恒等映射的重要性。这促使我们提出了一种新的残差模块，这使得训练更容易，改进泛化效果。我们使用1001层的ResNet在CIFAR-10（错误率4.62%）和CIFAR-100上得到了改进的结果，使用200层ResNet在ImageNet上进行了试验。代码已开源。

## 1. Introduction 引言

Deep residual networks (ResNets) [1] consist of many stacked “Residual Units”. Each unit (Fig. 1 (a)) can be expressed in a general form: 深度残差网络(ResNets)[1]包括很多堆叠的残差模块。每个模块（图1a）可以表示为如下一般形式：

$$y_l = h(x_l) + F(x_l, W_l)$$
$$x_{l+1} = f (y_l)$$

where $x_l$ and $x_{l+1}$ are input and output of the l-th unit, and F is a residual function. In [1], $h(x_l) = x_l$ is an identity mapping and f is a ReLU [2] function. 其中$x_l$和$x_{l+1}$是第l个单元的输入和输出，F是残差函数。在[1]中，$h(x_l) = x_l$是一个恒等映射，f是一个ReLU[2]函数。

ResNets that are over 100-layer deep have shown state-of-the-art accuracy for several challenging recognition tasks on ImageNet [3] and MS COCO [4] competitions. The central idea of ResNets is to learn the additive residual function F with respect to $h(x_l)$, with a key choice of using an identity mapping $h(x_l) = x_l$. This is realized by attaching an identity skip connection (“shortcut”).

ResNets超过了100层深度，在几个ImageNet[3]和MS COCO[4]最有挑战的识别任务上取得了目前最好的准确率。ResNets的中心思想是对于$h(x_l)$学习加性的残差函数F，其关键是使用了恒等映射$h(x_l) = x_l$。这是通过增加了一个恒等的跳跃连接（捷径）实现的。

In this paper, we analyze deep residual networks by focusing on creating a “direct” path for propagating information — not only within a residual unit, but through the entire network. Our derivations reveal that if both $h(x_l)$ and $f(y_l)$ are identity mappings, the signal could be directly propagated from one unit to any other units, in both forward and backward passes. Our experiments empirically show that training in general becomes easier when the architecture is closer to the above two conditions.

本文中，我们关注的是为传播信息创建一条直接通道，不止是在残差模块内部，而是在整个网络中，从而分析了深度残差网络。我们的推导说明了，$h(x_l)$和$f(y_l)$都是恒等映射，信号可以从一个单元直接传递到任何其他单元，而且是前向和反向两种方向。我们的试验经验上证明了，当架构与上述两个条件接近时，训练一般都会变得更容易。

To understand the role of skip connections, we analyze and compare various types of $h(x_l)$. We find that the identity mapping $h(x_l) = x_l$ chosen in [1] achieves the fastest error reduction and lowest training loss among all variants we investigated, whereas skip connections of scaling, gating [5,6,7], and 1×1 convolutions all lead to higher training loss and error. These experiments suggest that keeping a “clean” information path (indicated by the grey arrows in Fig. 1, 2, and 4) is helpful for easing optimization.

为理解跳跃连接的角色，我们分析并比较各种类型的$h(x_l)$。我们发现，在所有我们研究的变体中，[1]中选择的恒等映射$h(x_l) = x_l$能得到最快的错误降低速度和最低的训练损失，但是带有变尺度、门[5,6,7]和1×1卷积效果的跳跃连接都会导致更高的训练损失和误差。这些试验说明，保持一条干净的信息通道（如图1,2,4中的灰色箭头）对于降低优化难度是有帮助的。

To construct an identity mapping $f(y_l) = y_l$, we view the activation functions (ReLU and BN [8]) as “pre-activation” of the weight layers, in contrast to conventional wisdom of “post-activation”. This point of view leads to a new residual unit design, shown in (Fig. 1(b)). Based on this unit, we present competitive results on CIFAR-10/100 with a 1001-layer ResNet, which is much easier to train and generalizes better than the original ResNet in [1]. We further report improved results on ImageNet using a 200-layer ResNet, for which the counterpart of [1] starts to overfit. These results suggest that there is much room to exploit the dimension of network depth, a key to the success of modern deep learning.

为构建恒等映射$f(y_l) = y_l$，我们将激活函数(ReLU and BN[8])视为权重层的“预激活”，这是相对传统的“后激活”来说的。这种观点带来了一种新的残差单元设计，如图1(b)。基于这种单元，我们在CIFAR-10/100上用1001层的ResNet给出了很好的结果，训练起来也非常容易，比[1]中的原始ResNet泛化效果要好。我们在ImageNet上使用一个200层的ResNet也得到了改进的结果，而[1]中对应的网络则会开始过拟合。这些结果说明，探索网络深度还有很大的空间，而这是现代深度学习的成功的一个关键因素。

Figure 1. Left: (a) original Residual Unit in [1]; (b) proposed Residual Unit. The grey arrows indicate the easiest paths for the information to propagate, corresponding to the additive term “$x_l$” in Eqn.(4) (forward propagation) and the additive term “1” in Eqn.(5) (backward propagation). Right: training curves on CIFAR-10 of 1001-layer ResNets. Solid lines denote test error (y-axis on the right), and dashed lines denote training loss (y-axis on the left). The proposed unit makes ResNet-1001 easier to train.

图1. 左：(a)[1]中的原始残差单元；(b)提出的残差单元。灰色箭头表示信息传播最容易的路径，对应式(4)中的加法项$x_l$（前向传播），和式(5)中的加法项“1”（反向传播）。右：1001层ResNets在CIFAR-10上的训练曲线。实线表示测试错误率（右边的y轴），虚线表示训练损失（左边的y轴）。我们提出的单元使得ResNet-1001更容易训练。

## 2 Analysis of Deep Residual Networks 深度残差网络的分析

The ResNets developed in [1] are modularized architectures that stack building blocks of the same connecting shape. In this paper we call these blocks “Residual Units”. The original Residual Unit in [1] performs the following computation: [1]中提出的ResNets是模块化的架构，即相同的连接形状的模块的堆叠。本文中，我们称这种模块为“残差单元”。[1]中的原始残差单元进行如下计算：

$$y_l = h(x_l) + F(x_l, W_l)$$(1)
$$x_{l+1} = f (y_l)$$(2)

Here $x_l$ is the input feature to the l-th Residual Unit. $W_l = {W_{l,k} | 1≤k≤K}$ is a set of weights (and biases) associated with the l-th Residual Unit, and K is the number of layers in a Residual Unit (K is 2 or 3 in [1]). F denotes the residual function, e.g., a stack of two 3×3 convolutional layers in [1]. The function f is the operation after element-wise addition, and in [1] f is ReLU. The function h is set as an identity mapping: $h(x_l) = x_l$.(It is noteworthy that there are Residual Units for increasing dimensions and reducing feature map sizes [1] in which h is not identity. In this case the following derivations do not hold strictly. But as there are only a very few such units (two on CIFAR and three on ImageNet, depending on image sizes [1]), we expect that they do not have the exponential impact as we present in Sec. 3. One may also think of our derivations as applied to all Residual Units within the same feature map size.)

这里$x_l$是第l个残差单元的输入特征。$W_l = {W_{l,k} | 1≤k≤K}$是与第l个残差单元相关的权重（偏置）集，K是残差单元的层数（在[1]中K为2或3）。F表示残差函数，如在[1]中是2个3×3的卷积层。函数f是逐元素相加后的运算，在[1]中f是ReLU。函数h设置为恒等映射：$h(x_l) = x_l$。（需要指出，[1]中有增加维度和降低特征图大小的残差单元，其中h不是恒等映射。在这种情况下，下面的推导并不严格成立。但由于这种单元数量很少（在CIFAR中2个，ImageNet中3个，依赖于图像大小[1]），我们希望它们没有指数效应，我们在第3部分中叙述了这种效应。可以认为我们的推导对于同样特征图大小的所有残差单元都适用。）

If f is also an identity mapping: $x_{l+1} ≡ y_l$, we can put Eqn.(2) into Eqn.(1) and obtain: 如果f也是一个恒等映射：$x_{l+1} ≡ y_l$，那么我们可以将式(2)代入式(1)中，得到

$$x_{l+1} = x_l + F(x_l, W_l)$$(3)

Recursively ($x_{l+2} = x_{l+1} + F (x_{l+1}, W_{l+1}) = x_l + F (x_l, W_l) + F (x_{l+1}, W_{l+1})$, etc.) we will have: 迭代起来，我们有：

$$x_L = x_l + \sum_{i=1}^{L-1} F(x_i,W_i)$$(4)

for any deeper unit L and any shallower unit l. Eqn.(4) exhibits some nice properties. (i) The feature $x_L$ of any deeper unit L can be represented as the feature $x_l$ of any shallower unit l plus a residual function in a form of $\sum_{i=1}^{L-1} F$, indicating that the model is in a residual fashion between any units L and l. (ii) The feature $x_L = x_0 + \sum_{i=0}^{L-1} F(x_i, W_i)$, of any deep unit L, is the summation of the outputs of all preceding residual functions (plus $x_0$). This is in contrast to a “plain network” where a feature $x_L$ is a series of matrix-vector products, say, $\prod_{i=0}^{L-1} W_i x_0$ (ignoring BN and ReLU).

对任何更深的单元L和更浅的单元l都成立。式(4)展现了一些很好的性质。(i)任何更深单元L的特征$x_L$，可以表示为任何更浅层单元l的特征$x_l$，与残差函数的和，形式为$\sum_{i=1}^{L-1} F$，说明模型在任何单元L和l之间是残差的形式的；(ii)任何深层单元L的特征，$x_L = x_0 + \sum_{i=0}^{L-1} F(x_i, W_i)$，是之前所有残差函数的输出的和（加上$x_0$）。这与普通网络形成对比，其中特征$x_L$是一系列矩阵-向量的乘积，即$\prod_{i=0}^{L-1} W_i x_0$（忽略BN和ReLU）。

Eqn.(4) also leads to nice backward propagation properties. Denoting the loss function as ϵ, from the chain rule of backpropagation [9] we have: 式(4)会带来很好的反向传播性质。令损失函数为ϵ，从反向传播规则[9]，我们可以得到：

$$\frac {∂ϵ}{∂ x_l} = \frac {∂ϵ}{∂ x_L} \frac {∂ x_L}{∂ x_l} = \frac {∂ϵ}{∂ x_L} (1 + \frac {∂}{∂ x_l} \sum_{i=l}^{L-1} F(x_i, W_i)$$(5)

Eqn.(5) indicates that the gradient $\frac {∂ϵ}{∂ x_l}$ can be decomposed into two additive terms: a term of $\frac {∂ϵ}{∂ x_L}$ that propagates information directly without concerning any weight layers, and another term of $\frac {∂ϵ}{∂ x_L} \frac {∂}{∂ x_l} \sum_{i=l}^{L-1} F$ that propagates through the weight layers. The additive term of $\frac {∂ϵ}{∂ x_L}$ ensures that information is directly propagated back to any shallower unit l. Eqn.(5) also suggests that it is unlikely for the gradient $\frac {∂ϵ}{∂ x_l}$ to be canceled out for a mini-batch, because in general the term $\frac {∂}{∂ x_l} \sum_{i=l}^{L-1} F$ cannot be always -1 for all samples in a mini-batch. This implies that the gradient of a layer does not vanish even when the weights are arbitrarily small.

式(5)表明，梯度$\frac {∂ϵ}{∂ x_l}$可以分解为两个加法项：一个项目是$\frac {∂ϵ}{∂ x_L}$，不考虑权重层直接传播信息，和另一个项$\frac {∂ϵ}{∂ x_L} \frac {∂}{∂ x_l} \sum_{i=l}^{L-1} F$，从权重层传播过来的。加法项$\frac {∂ϵ}{∂ x_L}$确保信息直接传播回任意更浅的层l。式(5)也说明，一个mini-batch的梯度$\frac {∂ϵ}{∂ x_l}$不太可能互相抵消掉，因为总体上项$\frac {∂}{∂ x_l} \sum_{i=l}^{L-1} F$在一个mini-batch中不可能总是-1。这说明，层的梯度即使在权重任意小的时候，也不会消失。

**Discussions**

Eqn.(4) and Eqn.(5) suggest that the signal can be directly propagated from any unit to another, both forward and backward. The foundation of Eqn.(4) is two identity mappings: (i) the identity skip connection $h(x_l) = x_l$, and (ii) the condition that f is an identity mapping. 式(4)和式(5)说明，信号可以直接从一个单元传到另一个单元，包括前向和反向。式(4)的基础是两个恒等映射：(i)恒等跳跃连接$h(x_l) = x_l$，和(ii)f也是一个恒等映射。

These directly propagated information flows are represented by the grey arrows in Fig. 1, 2, and 4. And the above two conditions are true when these grey arrows cover no operations (expect addition) and thus are “clean”. In the following two sections we separately investigate the impacts of the two conditions. 这些直接传播的信息流由图1,2和4中的灰色箭头所示。当这些灰色箭头上没有任何运算（除了加法）时，即为干净的时候，上述两个条件为真。后面两节中，我们分别研究这两个条件的影响。

## 3 On the Importance of Identity Skip Connections 恒等跳跃连接的重要性

Let’s consider a simple modification, $h(x_l) = λ_l x_l$, to break the identity shortcut: 我们考虑一种简单的改变，以打破恒等捷径连接，$h(x_l) = λ_l x_l$：

$$x_{l+1} = λ_l x_l + F(x_l, W_l)$$(6)

where $λ_l$ is a modulating scalar (for simplicity we still assume f is identity). Recursively applying this formulation we obtain an equation similar to Eqn. (4): 其中$λ_l$是调节标量（为简化起见，我们仍然假设f是恒等映射）。将这个公式迭代应用，我们可以得到类似式(4)的等式： $x_L = (\prod_{i=l}^{L-1} λ_i)x_l + \sum_{i=l}^{L-1} (\prod_{j=i+1}^{L-1} λ_j) F(x_i,W_i)$, or simply:

$$x_L = (\prod_{i=l}^{L-1} λ_i)x_l + \sum_{i=l}^{L-1} \hat F(x_i,W_i)$$(7)

where the notation F̂ absorbs the scalars into the residual functions. Similar to Eqn.(5), we have backpropagation of the following form: 其中F̂将这些标量吸收进入残差函数。与式(5)类似，我们有下面的反向传播形式：

$$\frac {∂ϵ}{∂ x_l} = \frac {∂ϵ}{∂ x_L} ((\prod_{i=l}^{L-1} λ_i) + \frac {∂}{∂ x_l} \sum_{i=l}^{L-1} \hat F(x_i, W_i)$$(8)

Unlike Eqn.(5), in Eqn.(8) the first additive term is modulated by a factor $\prod_{i=l}^{L-1} λ_i$. For an extremely deep network (L is large), if $λ_i$ > 1 for all i, this factor can be exponentially large; if $λ_i$ < 1 for all i, this factor can be exponentially small and vanish, which blocks the backpropagated signal from the shortcut and forces it to flow through the weight layers. This results in optimization difficulties as we show by experiments.

与(5)式不同，在式(8)中，第一个加法项有一个调制项$\prod_{i=l}^{L-1} λ_i$。对于极深的深度网络（L很大），如果对所有的i都有$λ_i$ > 1，那么这个调制因子可以以指数级增加；如果对所有i都有$λ_i$ < 1，那么这个因子可以以指数级减小并消失，阻碍反向传播的信号从捷径连接传播，并使其只能通过权重层传播。我们通过试验会证明，这会导致优化困难。

In the above analysis, the original identity skip connection in Eqn.(3) is replaced with a simple scaling $h(x_l) = λ_l x_l$. If the skip connection $h(x_l)$ represents more complicated transforms (such as gating and 1×1 convolutions), in Eqn.(8) the first term becomes $\prod_{i=l}^{L-1} h'_i$ where h' is the derivative of h. This product may also impede information propagation and hamper the training procedure as witnessed in the following experiments.

在上面的分析中，式(3)中的原始的恒等跳跃连接，替换为简单的尺度变化$h(x_l) = λ_l x_l$。如果跳跃连接$h(x_l)$代表更复杂的变换（比如门变换和1×1卷积），在式(8)中第一项会变成$\prod_{i=l}^{L-1} h'_i$，其中h'是h的导数。这种乘积也会阻碍信息传播，下面的试验中还会看到，会妨碍训练过程。

Figure 2. Various types of shortcut connections used in Table 1. The grey arrows indicate the easiest paths for the information to propagate. The shortcut connections in (b-f) are impeded by different components. For simplifying illustrations we do not display the BN layers, which are adopted right after the weight layers for all units here. (a) original (b) constant scaling (c) exclusive gating (d) shortcut-only gating (e) conv shortcut (f) dropout shortcut

图2. 表1中用到的各种类型的捷径连接。灰色箭头表明了信息传播的最容易路径。(b-f)中的捷径连接受到不同的部分阻碍。为简化描述，我们没有显示BN层，在所有单元的权重层后都有BN层。

### 3.1 Experiments on Skip Connections 跳跃连接的试验

We experiment with the 110-layer ResNet as presented in [1] on CIFAR-10 [10]. This extremely deep ResNet-110 has 54 two-layer Residual Units (consisting of 3×3 convolutional layers) and is challenging for optimization. Our implementation details (see appendix) are the same as [1]. Throughout this paper we report the median accuracy of 5 runs for each architecture on CIFAR, reducing the impacts of random variations.

我们在CIFAR-10[10]上用[1]中的110层ResNet进行试验。这种非常深的ResNet-110有54个2层残差单元（包含3×3卷积层），优化起来非常有挑战性。我们的实现细节（见附录）与[1]中相同。本文通篇给出的都是在CIAFR上每种架构运行5次准确率的中值，可以降低随机变化的影响。

Though our above analysis is driven by identity f, the experiments in this section are all based on f = ReLU as in [1]; we address identity f in the next section. Our baseline ResNet-110 has 6.61% error on the test set. The comparisons of other variants (Fig. 2 and Table 1) are summarized as follows:

虽然我们上面的分析中使用的是恒等映射f，本节中的试验使用的是和[1]中一样的f=ReLU；我们在下一节中处理恒等f的问题。我们的基准ResNet-110在测试集上有6.61%错误率。与其他变体的比较（图2和表1）总结如下：

**Constant scaling**. We set λ = 0.5 for all shortcuts (Fig. 2(b)). We further study two cases of scaling F: (i) F is not scaled; or (ii) F is scaled by a constant scalar of 1 − λ = 0.5, which is similar to the highway gating [6,7] but with frozen gates. The former case does not converge well; the latter is able to converge, but the test error (Table 1, 12.35%) is substantially higher than the original ResNet-110. Fig 3(a) shows that the training error is higher than that of the original ResNet-110, suggesting that the optimization has difficulties when the shortcut signal is scaled down.

**常数尺度变换**。我们对所有捷径设λ = 0.5（图2b）。我们进一步研究F尺度变换的两种情况：(i)F不进行尺度变换；(ii)F的尺度变换因子为1 − λ = 0.5，与highway gating[6,7]类似，但gate是冻结的。前一种情况收敛情况不好；后一种可以收敛，但测试误差（表1,12.35%）比原始ResNet-110高了很多。图3(a)表明，训练误差比原始ResNet-110高了很多，说明当捷径信号尺度变小之后，优化存在困难。

Table 1. Classification error on the CIFAR-10 test set using ResNet-110 [1], with different types of shortcut connections applied to all Residual Units. We report “fail” when the test error is higher than 20%.

表1. 使用ResNet-110[1]在CIFAR-10测试集上的分类错误率，在所有残差单元上使用不同类型的捷径连接。当测试错误率高于20%时，我们给出“失败”的结果。

case | Fig. | on shortcut | on F | error(%) | remark
--- | --- | --- | --- | --- | ---
original[1] | Fig.2(a) | 1 | 1 | 6.61 |
constant scaling | Fig.2(b) | 0/0.5/0.5 | 1/1/0.5 | fail/fail/12.35 | This is a plain net / - / frozen gating
exclusive gating | Fig.2(c) | 1-g(x)/1-g(x)/1-g(x) | g(x)/g(x)/g(x) | fail/8.70/9.81 | init bg=0 to -5 / init bg=-6 / init bg=-7
shortcut-only gating | Fig.2(d) | 1-g(x)/1-g(x) | 1/1 | 12.86/6.91 | init bg=0 / init bg=-6
1×1 conv shortcut | Fig.2(e) | 1×1 conv | 1 | 12.22 |
dropout shortcut | Fig.2(f) | dropout 0.5 | 1 | fail |

**Exclusive gating**. Following the Highway Networks [6,7] that adopt a gating mechanism [5], we consider a gating function $g(x) = σ(W_g x + b_g)$ where a transform is represented by weights $W_g$ and biases $b_g$ followed by the sigmoid function $σ(x) = \frac {1}{1+e^{−x}}$. In a convolutional network g(x) is realized by a 1×1 convolutional layer. The gating function modulates the signal by element-wise multiplication.

**Exclusive gating**。使用Highway Networks[6,7]的方法，采用gating机制[5]，我们考虑gating函数$g(x) = σ(W_g x + b_g)$，其中变换由权重$W_g$和偏置$b_g$和sigmoid函数$σ(x) = \frac {1}{1+e^{−x}}$表示。在卷积网络中，g(x)由1×1卷积层实现。门函数通过逐元素的乘积来调制信号。

We investigate the “exclusive” gates as used in [6,7] — the F path is scaled by g(x) and the shortcut path is scaled by 1−g(x). See Fig 2(c). We find that the initialization of the biases $b_g$ is critical for training gated models, and following the guidelines in [6,7], we conduct hyper-parameter search on the initial value of $b_g$ in the range of 0 to -10 with a decrement step of -1 on the training set by cross-validation. The best value (−6 here) is then used for training on the training set, leading to a test result of 8.70% (Table 1), which still lags far behind the ResNet-110 baseline. Fig 3(b) shows the training curves. Table 1 also reports the results of using other initialized values, noting that the exclusive gating network does not converge to a good solution when $b_g$ is not appropriately initialized.

我们研究了在[6,7]中使用的exclusive gates，F路径的尺度由g(x)控制，捷径路径的尺度由1-g(x)控制，见图2(c)。我们发现，对于训练带有门机制的模型，偏置$b_g$的初始化非常关键，使用[6,7]中的指导原则，我们对$b_g$的初始值进行超参数搜索，范围在0到-10之间，步长-1，并进行了交叉验证。最佳值（这里是-6）用于在训练集上进行训练，得到的测试结果为8.70%（表1），远在ResNet-110基准之后。图3(b)为训练曲线。表1还给出了使用其他初始值的结果，注意如果$b_g$没有合理的初始化，那么exclusive gating网络不会收敛到一个很好的结果。

The impact of the exclusive gating mechanism is two-fold. When 1 − g(x) approaches 1, the gated shortcut connections are closer to identity which helps information propagation; but in this case g(x) approaches 0 and suppresses the function F. To isolate the effects of the gating functions on the shortcut path alone, we investigate a non-exclusive gating mechanism in the next.

Exclusive gating机制的影响是两方面的。如果1-g(x)接近1，那么门捷径连接就接近与恒等映射，有助于信息传播；这种情况下g(x)就趋近于0，抑制了函数F。为孤立门函数的对捷径路径的影响，下面我们研究一种non-exclusive门机制。

**Shortcut-only gating**. In this case the function F is not scaled; only the shortcut path is gated by 1 − g(x). See Fig 2(d). The initialized value of $b_g$ is still essential in this case. When the initialized $b_g$ is 0 (so initially the expectation of 1 − g(x) is 0.5), the network converges to a poor result of 12.86% (Table 1). This is also caused by higher training error (Fig 3(c)).

**只有捷径的门机制**。在这种情况下F的尺度没有改变；只有捷径路径用1-g(x)进行了门限，见图2(d)。$b_g$的初始化值在这种情况下仍然是关键的。当$b_g$初始化为0时（这样1-g(x)的期望初始化为0.5），网络收敛到一个很差的结果12.86%（表1）。这也是由于很高的训练错误率导致的（图3c）。

When the initialized $b_g$ is very negatively biased (e.g., −6), the value of 1 − g(x) is closer to 1 and the shortcut connection is nearly an identity mapping. Therefore, the result (6.91%, Table 1) is much closer to the ResNet-110 baseline.

当$b_g$初始值为较大的负值时（如-6），1-g(x)的值接近于1，捷径连接接近于恒等映射。所以，结果（表1,6.91%）非常接近于ResNet-110基准。

**1×1 convolutional shortcut**. Next we experiment with 1×1 convolutional shortcut connections that replace the identity. This option has been investigated in [1] (known as option C) on a 34-layer ResNet (16 Residual Units) and shows good results, suggesting that 1×1 shortcut connections could be useful. But we find that this is not the case when there are many Residual Units. The 110-layer ResNet has a poorer result (12.22%, Table 1) when using 1×1 convolutional shortcuts. Again, the training error becomes higher (Fig 3(d)). When stacking so many Residual Units (54 for ResNet-110), even the shortest path may still impede signal propagation. We witnessed similar phenomena on ImageNet with ResNet-101 when using 1×1 convolutional shortcuts.

**1×1卷积捷径**。下面，我们用1×1卷积捷径连接进行试验，替代恒等连接。这种选项已经在[1]中进行了研究（选项C），但用的是34层的ResNet（16个残差单元），说明1×1捷径连接可能是有用的。但我们发现，在非常多残差单元的情况下，就不是这个结果了。110层ResNet在使用1×1卷积捷径时得到了较差的结果（12.22%，表1）。训练误差也变得很高（图3d）。当叠加了这么多残差单元时（ResNet-110是54个），即使是最近的路径也可能阻碍信号传播。我们使用ResNet-101在ImageNet上，当使用1×1卷积捷径时也观察到了类似的现象。

**Dropout shortcut**. Last we experiment with dropout [11] (at a ratio of 0.5) which we adopt on the output of the identity shortcut (Fig. 2(f)). The network fails to converge to a good solution. Dropout statistically imposes a scale of λ with an expectation of 0.5 on the shortcut, and similar to constant scaling by 0.5, it impedes signal propagation.

**Dropout捷径**。最后我们使用dropout[11]进行试验（比率为0.5），在恒等捷径的输出部分使用（图2f）。网络未能收敛到一个好的结果。Dropout统计上来说，在捷径上增加了一个期望为0.5的尺度因子λ，与常数尺度因子0.5类似，这阻碍了信号传播。

Figure 3. Training curves on CIFAR-10 of various shortcuts. Solid lines denote test error (y-axis on the right), and dashed lines denote training loss (y-axis on the left).

### 3.2 Discussions 讨论

As indicated by the grey arrows in Fig. 2, the shortcut connections are the most direct paths for the information to propagate. Multiplicative manipulations (scaling, gating, 1×1 convolutions, and dropout) on the shortcuts can hamper information propagation and lead to optimization problems.

如图2中的灰色箭头所示，捷径连接是最直接的信息传播路径。捷径上的各种乘性操作（尺度变换，门机制，1×1卷积，和dropout）都会阻碍信息传播，导致优化问题。

It is noteworthy that the gating and 1×1 convolutional shortcuts introduce more parameters, and should have stronger representational abilities than identity shortcuts. In fact, the shortcut-only gating and 1×1 convolution cover the solution space of identity shortcuts (i.e., they could be optimized as identity shortcuts). However, their training error is higher than that of identity shortcuts, indicating that the degradation of these models is caused by optimization issues, instead of representational abilities.

值得提出，门机制和1×1卷积捷径引入了更多参数，应当比恒等捷径有更强的表示能力。实际上，只有捷径的门机制和1×1卷积覆盖了恒等捷径的解决方案空间（即，它们可以优化为恒等连接）。但是，其训练误差比恒等连接更高，说明模型的降质是由优化问题导致的，而不是表示能力。

## 4 On the Usage of Activation Functions 激活函数的使用

Experiments in the above section support the analysis in Eqn.(5) and Eqn.(8), both being derived under the assumption that the after-addition activation f is the identity mapping. But in the above experiments f is ReLU as designed in [1], so Eqn.(5) and (8) are approximate in the above experiments. Next we investigate the impact of f.

上面一节的试验说明式(5)和式(8)的分析是正确的，这都是在加法后的激活函数f是恒等映射的假设下推导出来的。但在上面的试验中，f是ReLU，与[1]中设计的一样，所以式(5)和(8)在上述试验中是近似的。下面我们研究f的影响。

We want to make f an identity mapping, which is done by re-arranging the activation functions (ReLU and/or BN). The original Residual Unit in [1] has a shape in Fig. 4(a) — BN is used after each weight layer, and ReLU is adopted after BN except that the last ReLU in a Residual Unit is after elementwise addition (f = ReLU). Fig. 4(b-e) show the alternatives we investigated, explained as following.

我们希望f为恒等映射，通过重新调整激活函数（ReLU和/或BN）的顺序得到这个效果。[1]中原始的残差单元如图4(a)所示，BN在每个权重层后都有，ReLU在BN之后，除了残差单元中最后一个ReLU是在逐元素相加后进行的(f=ReLU)。图4(b-e)所示的为我们使用的替代品，解释如下。

Figure 4. Various usages of activation in Table 2. All these units consist of the same components — only the orders are different. 表2中使用的各种激活函数。所有这些单元包括的部件是一样的，只是其顺序不一样。(a) original (b) BN after addition (c) ReLU before addition (d) ReLU-only pre-activation (e) full pre-activation

Table 2. Classification error (%) on the CIFAR-10 test set using different activation functions. 在CIFAR-10测试集上使用不同的激活函数的分类错误率(%)

case | Fig. | ResNet-110 | ResNet-164
--- | --- | --- | ---
original Residual Unit [1] | Fig. 4(a) | 6.61 | 5.93
BN after addition | Fig. 4(b) | 8.17 | 6.50
ReLU before addition | Fig. 4(c) | 7.84 | 6.14
ReLU-only pre-activation | Fig. 4(d) | 6.71 | 5.91
full pre-activation | Fig. 4(e) | 6.37 | 5.46

### 4.1 Experiments on Activation 激活的试验

In this section we experiment with ResNet-110 and a 164-layer Bottleneck [1]　architecture (denoted as ResNet-164). A bottleneck Residual Unit consist of a　1×1 layer for reducing dimension, a 3×3 layer, and a 1×1 layer for restoring　dimension. As designed in [1], its computational complexity is similar to the　two-3×3 Residual Unit. More details are in the appendix. The baseline ResNet-164 has a competitive result of 5.93% on CIFAR-10 (Table 2).

本节中，我们用ResNet-110和164层的瓶颈[1]结构（表示为ResNet-164）进行试验。瓶颈残差单元包括一个1×1卷积层以降维，一个3×3卷积层和一个1×1卷积层以恢复维数。如[1]中所设计的，其计算复杂度与两个3×3卷积层的残差单元类似。详见附录。基准ResNet-164在CIFAR-10（表2）的结果为5.93%.

**BN after addition**. Before turning f into an identity mapping, we go the opposite way by adopting BN after addition (Fig. 4(b)). In this case f involves BN and ReLU. The results become considerably worse than the baseline (Table 2). Unlike the original design, now the BN layer alters the signal that passes through the shortcut and impedes information propagation, as reflected by the difficulties on reducing training loss at the beginning of training (Fib. 6 left).

**加法后进行BN**。在将f变换为恒等映射之前，我们采取相反的路径，在加法后进行BN（图4b）。在这种情况下，f就包括了BN和ReLU。其结果比基准要差了不少（表2）。与原始设计不同，现在BN层是对从捷径传播过来的信息进行处理，从而阻碍了信息传播，图6左中，训练开始时的训练损失降低存在困难，就是这种情况的反应。

**ReLU before addition**. A naı̈ve choice of making f into an identity mapping is to move the ReLU before addition (Fig. 4(c)). However, this leads to a non-negative output from the transform F, while intuitively a “residual” function should take values in (−∞, +∞). As a result, the forward propagated signal is monotonically increasing. This may impact the representational ability, and the result is worse (7.84%, Table 2) than the baseline. We expect to have a residual function taking values in (−∞, +∞). This condition is satisfied by other Residual Units including the following ones.

**加法前进行ReLU**。将f变为恒等映射的一种简单选择是，将ReLU移到加法之前（图4c）。但是，这会导致从变换F中的输出都是非负的，而残差函数的输入范围应当是(−∞, +∞)。结果是，前向传播的信号是单调递增的。这可能会影响表示能力，结果也比基准更差了（表2,7.84%）。我们希望残差函数的输入范围为(−∞, +∞)，这个条件其他的残差函数可以满足，包括下面的这些。

**Post-activation or pre-activation?** In the original design (Eqn.(1) and Eqn.(2)), the activation $x_{l+1} = f (y_l)$ affects both paths in the next Residual Unit: $y_{l+1} = f (y_l) + F(f(y_l), W_{l+1})$. Next we develop an asymmetric form where an activation $\hat f$ only affects the F path: $y_{l+1} = y_l + F(\hat f(y_l), W_{l+1})$, for any l (Fig. 5 (a) to (b)). By renaming the notations, we have the following form:

**后激活或预激活？**在原始设计中（式(1)和式(2)），激活$x_{l+1} = f (y_l)$影响下一个残差单元的两条通道：$y_{l+1} = f (y_l) + F(f(y_l), W_{l+1})$。下一步，我们给出一种非对称形式，其中激活$\hat f$只影响F通道：$y_{l+1} = y_l + F(\hat f(y_l), W_{l+1})$，对于任意的l（图5a-b）。对一些表示进行重新命名，我们得到下面的形式：

$$x_{l+1} = x_l + F(\hat f(x_l), W_{l+1})$$(9)

It is easy to see that Eqn.(9) is similar to Eqn.(4), and can enable a backward formulation similar to Eqn.(5). For this new Residual Unit as in Eqn.(9), the new after-addition activation becomes an identity mapping. This design means that if a new after-addition activation $\hat f$ is asymmetrically adopted, it is equivalent to recasting $\hat f$ as the pre-activation of the next Residual Unit. This is illustrated in Fig. 5.

很容易看到式(9)与式(4)类似，也可以得到与式(5)类似的反向传播公式。对于式(9)中新的残差单元，这种新的加法后激活成为了恒等映射。这种设计意味着，如果新的加法后激活$\hat f$采用时是非对称的，那么就与将$\hat f$重新作为下一个残差单元的预激活是等价的。这示于图5。

Figure 5. Using asymmetric after-addition activation is equivalent to constructing a pre-activation Residual Unit.

The distinction between post-activation/pre-activation is caused by the presence of the element-wise addition. For a plain network that has N layers, there are N − 1 activations (BN/ReLU), and it does not matter whether we think of them as post- or pre-activations. But for branched layers merged by addition, the position of activation matters.

后激活与预激活的区别是由逐元素相加的存在导致的。对于一个普通网络，有N层，有N-1个激活(BN/ReLU)，将其视为后激活或预激活都不重要。但对于有分支的层用加法合并，那么激活的位置就重要了。

We experiment with two such designs: (i) ReLU-only pre-activation (Fig. 4(d)), and (ii) full pre-activation (Fig. 4(e)) where BN and ReLU are both adopted before weight layers. Table 2 shows that the ReLU-only pre-activation performs very similar to the baseline on ResNet-110/164. This ReLU layer is not used in conjunction with a BN layer, and may not enjoy the benefits of BN [8].

我们用以下两种设计进行试验：(i)只有ReLU的预激活，图4d，和(ii)完整的预激活（图4e），其中BN和ReLU在权重层之前都有。表2给出了，只有ReLU的预激活与基准的ResNet-110/164表现非常类似。这个ReLU层并不是与BN层一起使用的，所以不一定能享受到BN[8]的好处。

Somehow surprisingly, when BN and ReLU are both used as pre-activation, the results are improved by healthy margins (Table 2 and Table 3). In Table 3 we report results using various architectures: (i) ResNet-110, (ii) ResNet-164, (iii) a 110-layer ResNet architecture in which each shortcut skips only 1 layer (i.e., a Residual Unit has only 1 layer), denoted as “ResNet-110(1layer)”, and (iv) a 1001-layer bottleneck architecture that has 333 Residual Units (111 on each feature map size), denoted as “ResNet-1001”. We also experiment on CIFAR-100. Table 3 shows that our “pre-activation” models are consistently better than the baseline counterparts. We analyze these results in the following.

有些令人吃惊的是，当BN和ReLU都用在预激活上时，结果改进的不错（表2和表3）。在表3中，我们给出了使用几种架构的结果：(i)ResNet-110, (ii)ResNet-164, (iii)110层的ResNet架构，其中每个捷径只跳过了1层（即，每个残差单元只有1层），表示为ResNet-110(1 layer)，(iv)一个1001层的瓶颈架构，有333个残差单元（每种特征图大小有111个），表示为ResNet-1001。我们也在CIFAR-100上试验。表3给出了，我们的预激活模型一直都优于相应的基准。下面我们分析这些结果。

Table 3. Classification error (%) on the CIFAR-10/100 test set using the original Residual Units and our pre-activation Residual Units. 使用原始残差单元和我们的预激活残差单元，在CIFAR-10/100测试集上的分类错误率(%)

dataset | network | baseline unit | pre-activation unit
--- | --- | --- | ---
CIFAR-10 | ResNet-110(1 layer skip) | 9.90 | 8.91
CIFAR-10 | ResNet-110 | 6.61 | 6.37
CIFAR-10 | ResNet-164 | 5.93 | 5.46
CIFAR-10 | ResNet-1001 | 7.61 | 4.92
CIFAR-100 | ResNet-164 | 25.16 | 24.33
CIFAR-100 | ResNet-1001 | 27.82 | 22.71

Figure 6. Training curves on CIFAR-10. Left: BN after addition (Fig. 4(b)) using ResNet-110. Right: pre-activation unit (Fig. 4(e)) on ResNet-164. Solid lines denote test error, and dashed lines denote training loss.

### 4.2 Analysis 分析

We find the impact of pre-activation is twofold. First, the optimization is further eased (comparing with the baseline ResNet) because f is an identity mapping. Second, using BN as pre-activation improves regularization of the models.

我们发现预激活的影响是两方面的。第一，优化更容易了（与基准ResNet相比），因为f是一个恒等映射。第二，使用BN作为预激活改进了模型的正则化性。

**Ease of optimization**. This effect is particularly obvious when training the 1001-layer ResNet. Fig. 1 shows the curves. Using the original design in [1], the training error is reduced very slowly at the beginning of training. For f = ReLU, the signal is impacted if it is negative, and when there are many Residual Units, this effect becomes prominent and Eqn.(3) (so Eqn.(5)) is not a good approximation. On the other hand, when f is an identity mapping, the signal can be propagated directly between any two units. Our 1001-layer network reduces the training loss very quickly (Fig. 1). It also achieves the lowest loss among all models we investigated, suggesting the success of optimization.

**更容易优化**。这种效果在使用1001层的ResNet训练时非常明显。图1是相应的训练曲线。使用[1]中的原始设计，训练误差在开始时下降缓慢。对于f=ReLU，信号如果为负，那么信号就受影响，如果有很多残差单元，这种影响会变得非常明显，式(3)（以及式(5)）不是一个很好的近似。另一方面，当f是一个恒等映射时，信号可以直接在两个单元之前传播。我们的1001层网络在训练时误差下降的非常快（图1）。在所有研究的模型中，也取得了最低损失，说明优化很成功。

We also find that the impact of f = ReLU is not severe when the ResNet has fewer layers (e.g., 164 in Fig. 6(right)). The training curve seems to suffer a little bit at the beginning of training, but goes into a healthy status soon. By monitoring the responses we observe that this is because after some training, the weights are adjusted into a status such that $y_l$ in Eqn.(1) is more frequently above zero and f does not truncate it ($x_l$ is always non-negative due to the previous ReLU, so $y_l$ is below zero only when the magnitude of F is very negative). The truncation, however, is more frequent when there are 1000 layers.

我们还发现，在ResNet层数不多时，f=ReLU的影响不是很严重（如图6右的164层）。训练曲线在训练开始的时候似乎有些下降缓慢，但不久之后就迅速下降。通过监控响应，我们观察到，这是因为在一些训练之后，权重调整到的状态，会使式(1)中的$y_l$经常大于0，f不会对其截断（$x_l$永远是非负的，因为前面的ReLU，所以$y_l$只有在F的幅值非常负的时候，才会低于0）。但这个截断在1001层的时候，就非常频繁了。

**Reducing overfitting**. Another impact of using the proposed pre-activation unit is on regularization, as shown in Fig. 6 (right). The pre-activation version reaches slightly higher training loss at convergence, but produces lower test error. This phenomenon is observed on ResNet-110, ResNet-110(1-layer), and ResNet-164 on both CIFAR-10 and 100. This is presumably caused by BN’s regularization effect [8]. In the original Residual Unit (Fig. 4(a)), although the BN normalizes the signal, this is soon added to the shortcut and thus the merged signal is not normalized. This unnormalized signal is then used as the input of the next weight layer. On the contrary, in our pre-activation version, the inputs to all weight layers have been normalized.

**降低过拟合**。使用提出的预激活单元的另一个影响是在正则化上，如图6右所示。预激活版在收敛时达到的训练损失略微高一点，但是得到的测试损失却更低一些。这种现象在ResNet-110, ResNet-110(1-layer)和ResNet-164在CIFAR-10和100上的试验上都观察到。这很可能是由于BN的正则化效果[8]导致的。在原始的残差单元中（图4a），虽然BN对信号进行了归一化，但马上就与捷径相加在一起，所以合并的信号不是归一化的。这种非归一化的信号然后用于输入到下一个权重层。相反，在我们的预激活版中，对所有权重层的输入都是归一化的。

## 5 Results 结果

**Comparisons on CIFAR-10/100**. Table 4 compares the state-of-the-art methods on CIFAR-10/100, where we achieve competitive results. We note that we do not specially tailor the network width or filter sizes, nor use regularization techniques (such as dropout) which are very effective for these small datasets. We obtain these results via a simple but essential concept — going deeper. These results demonstrate the potential of pushing the limits of depth.

**在CIFAR-10/100上的比较**。表4在CIFAR-10/100上比较了目前最好的方法，其中我们取得了非常好的结果。我们注意，我们没有特意设计网络的宽度或滤波器大小，也没有使用正则化技术（比如dropout），这对于这样的小数据集非常有效。我们通过简单但重要的方法得到这些结果，即更深的网络。这些结果表明了加深网络的潜能。

Table 4. Comparisons with state-of-the-art methods on CIFAR-10 and CIFAR-100 using “moderate data augmentation” (flip/translation), except for ELU [12] with no augmentation. Better results of [13,14] have been reported using stronger data augmentation and ensembling. For the ResNets we also report the number of parameters. Our results are the median of 5 runs with mean±std in the brackets. All ResNets results are obtained with a mini-batch size of 128 except † with a mini-batch size of 64 (code available at https://github.com/KaimingHe/resnet-1k-layers).

表4. 在CIFAR-10和CIFAR-100上使用普通的数据扩充技术（翻转/平移）与目前最好的方法进行比较，除了ELU[12]没有使用扩充。[13,14]使用了更强的数据扩充和模型集成得到更好的结果。对于ResNets我们还给出了参数数量。我们的结果是运行5次的中值，括号中有均值±方差。所有的ResNets结果都是用的mini-batch大小128，除了†为mini-batch 64。

CIFAR-10 | error (%) | CIFAR-100 | error (%)
--- | --- | --- | ---
NIN [15] | 8.81 | NIN [15] | 35.68
DSN [16] | 8.22 | DSN [16] | 34.57
FitNet [17] | 8.39 | FitNet [17] | 35.04
Highway [7] | 7.72 | Highway [7] | 32.39
All-CNN [14] | 7.25 | All-CNN [14] | 33.71
ELU [12] | 6.55 | ELU [12] | 24.28
FitResNet, LSUV [18] | 5.84 | FitNet, LSUV [18] | 27.66
ResNet-110 [1] (1.7M) | 6.61 | ResNet-164 [1] (1.7M) | 25.16
ResNet-1202 [1] (19.4M) | 7.93 | ResNet-1001 [1] (10.2M) | 27.82
ResNet-164 [ours] (1.7M) | 5.46 | ResNet-164 [ours] (1.7M) | 24.33
ResNet-1001 [ours] (10.2M) | 4.92(4.89±0.14) | ResNet-1001 [ours] (10.2M) | 22.71(22.68±0.22)
ResNet-1001 [ours] (10.2M) † | 4.62(4.69±0.20)

**Comparisons on ImageNet**. Next we report experimental results on the 1000-class ImageNet dataset [3]. We have done preliminary experiments using the skip connections studied in Fig. 2 & 3 on ImageNet with ResNet-101 [1], and observed similar optimization difficulties. The training error of these non-identity shortcut networks is obviously higher than the original ResNet at the first learning rate (similar to Fig. 3), and we decided to halt training due to limited resources. But we did finish a “BN after addition” version (Fig. 4(b)) of ResNet-101 on ImageNet and observed higher training loss and validation error. This model’s single-crop (224×224) validation error is 24.6%/7.5%, vs. the original ResNet-101’s 23.6%/7.1%. This is in line with the results on CIFAR in Fig. 6 (left).

**在ImageNet上的比较**。下一步，我们在1000类的ImageNet数据集[3]上给出试验结果。我们用ResNet-101[1]在ImageNet上使用图2和3中的跳跃连接进行了初步的试验，观察到了类似的优化困难。这些非恒等捷径网络的训练误差比原始的ResNet在第一种学习速率下明显要高很多（与图3类似），由于资源有限，我们决定停止训练。但是我们完成了一个“加法后BN”版本（图4b）ResNet-101在ImageNet上的训练，观察到更高的训练损失和验证误差。这个模型的单剪切(224×224)验证错误率为24.6%/7.5%，原始的ResNet-101为23.6%/7.1%。这与图6左中在CIFAR上的结果结论一致。

Table 5 shows the results of ResNet-152 [1] and ResNet-200(The ResNet-200 has 16 more 3-layer bottleneck Residual Units than ResNet-152, which are added on the feature map of 28×28), all trained from scratch. We notice that the original ResNet paper [1] trained the models using scale jittering with shorter side s ∈ [256, 480], and so the test of a 224×224 crop on s = 256 (as did in [1]) is negatively biased. Instead, we test a single 320×320 crop from s = 320, for all original and our ResNets. Even though the ResNets are trained on smaller crops, they can be easily tested on larger crops because the ResNets are fully convolutional by design. This size is also close to 299×299 used by Inception v3 [19], allowing a fairer comparison.

表5给出了ResNet-152[1]和ResNet-200（这个ResNet-200比ResNet-152多了16个3层瓶颈残差单元，在特征图大小28×28时加入）的结果，都是从头开始训练的。我们注意到，原始ResNet论文[1]训练模型时，使用了尺度抖动，短边范围s ∈ [256, 480]，所以大小为224×224的剪切块在s=256上的测试是负偏置的。所以我们测试320×320大小的剪切块，对于所有的原始ResNets和我们的ResNets都这样进行试验。即使ResNets在更小的剪切块上进行训练的，在更大的剪切块上也可以很容易测试，因为ResNets在设计上是全卷积的。大小也与Inception v3[19]使用的299×299非常接近，使得比较更为公平。

The original ResNet-152 [1] has top-1 error of 21.3% on a 320×320 crop, and our pre-activation counterpart has 21.1%. The gain is not big on ResNet-152 because this model has not shown severe generalization difficulties. However, the original ResNet-200 has an error rate of 21.8%, higher than the baseline ResNet-152. But we find that the original ResNet-200 has lower training error than ResNet-152, suggesting that it suffers from overfitting.

原始的ResNet-152[1]在320×320剪切块大小下的top-1错误率为21.3%，我们的预激活版本为21.1%。在ResNet-152上的这个提升并不大，因为这个模型没有表现出很严重的泛化困难。但是，原始的ResNet-200错误率为21.8%，比基准ResNet-152还要高。但我们发现原始ResNet-200的训练误差比ResNet-152要低，说明这有过拟合问题。

Our pre-activation ResNet-200 has an error rate of 20.7%, which is 1.1% lower than the baseline ResNet-200 and also lower than the two versions of ResNet-152. When using the scale and aspect ratio augmentation of [20,19], our ResNet-200 has a result better than Inception v3 [19] (Table 5). Concurrent with our work, an Inception-ResNet-v2 model [21] achieves a single-crop result of 19.9%/4.9%. We expect our observations and the proposed Residual Unit will help this type and generally other types of ResNets.

我们的预激活ResNet-200的错误率为20.7%，这比基准ResNet-200低了1.1%，也比ResNet-152的两个版本要低。当使用[20,19]中的尺度和纵横比数据扩充技术时，我们的ResNet-200比Inception v3[19]的效果要好（见表5）。与我们的工作同时，一种Inception-ResNet-v2模型[21]取得了单剪切块的结果19.9%/4.9%。我们期待我们的观察结果和提出的残差单元可以帮助这种类型的其他一般类型的ResNets。

Table 5. Comparisons of single-crop error on the ILSVRC 2012 validation set. All ResNets are trained using the same hyper-parameters and implementations as [1]). Our Residual Units are the full pre-activation version (Fig. 4(e)). † : code/model available at https://github.com/facebook/fb.resnet.torch/tree/master/pretrained, using scale and aspect ratio augmentation in [20].

method | augmentation | train crop | test crop | top-1 | top-5
--- | --- | --- | --- | --- | ---
ResNet-152, original Residual Unit [1] | scale | 224×224 | 224×224 | 23.0 | 6.7
ResNet-152, original Residual Unit [1] | scale | 224×224 | 320×320 | 21.3 | 5.5
ResNet-152, pre-act Residual Unit | scale | 224×224 | 320×320 | 21.1 | 5.5
ResNet-200, original Residual Unit [1] | scale | 224×224 | 320×320 | 21.8 | 6.0
ResNet-200, pre-act Residual Unit | scale | 224×224 | 320×320 | 20.7 | 5.3
ResNet-200, pre-act Residual Unit | scale+asp ratio | 224×224 | 320×320 | 20.1† | 4.8†
Inception v3 [19] | scale+asp ratio | 299×299 | 299×299 | 21.2 | 5.6

**Computational Cost**. Our models’ computational complexity is linear on depth (so a 1001-layer net is ∼10× complex of a 100-layer net). On CIFAR, ResNet-1001 takes about 27 hours to train on 2 GPUs; on ImageNet, ResNet-200 takes about 3 weeks to train on 8 GPUs (on par with VGG nets [22]).

**计算代价**。我们模型的计算复杂度是与深度成线性关系的（所以1001层的网络是100层网络复杂度的10倍）。在CIFAR上，ResNet-1001在2块GPU上训练大约需要27小时；在ImageNet上，ResNet-200在8块GPU上大约需要3个星期来训练（与VGG网络[22]类似）。

## 6 Conclusions 结论

This paper investigates the propagation formulations behind the connection mechanisms of deep residual networks. Our derivations imply that identity shortcut connections and identity after-addition activation are essential for making information propagation smooth. Ablation experiments demonstrate phenomena that are consistent with our derivations. We also present 1000-layer deep networks that can be easily trained and achieve improved accuracy.

本文研究了深度残差网络连接机制后的传播公式。我们的推导说明，恒等捷径连接和恒等加法后激活对于信息平稳传播非常关键。分离试验证明了与我们推导一致的现象。我们还给出了可以很轻松训练的1001层深度网络，可以得到改进的准确率。