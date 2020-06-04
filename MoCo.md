# Momentum Contrast for Unsupervised Visual Representation Learning

Kaiming He et. al. Facebook AI Research (FAIR)

## 0. Abstract

We present Momentum Contrast (MoCo) for unsupervised visual representation learning. From a perspective on contrastive learning [29] as dictionary look-up, we build a dynamic dictionary with a queue and a moving-averaged encoder. This enables building a large and consistent dictionary on-the-fly that facilitates contrastive unsupervised learning. MoCo provides competitive results under the common linear protocol on ImageNet classification. More importantly, the representations learned by MoCo transfer well to downstream tasks. MoCo can outperform its supervised pre-training counterpart in 7 detection/segmentation tasks on PASCAL VOC, COCO, and other datasets, sometimes surpassing it by large margins. This suggests that the gap between unsupervised and supervised representation learning has been largely closed in many vision tasks.

我们提出了动量对比度(MoCo)算法进行无监督的视觉表示学习。通过将对比学习视为字典查找的角度，我们构建了一种动态字典，带有一个队列和一个滑动平均的编码器。这可以构建一个大型的连贯的运行中的字典，促进了对比无监督学习。MoCo在ImageNet分类中，在常见的线性协议下，得到了很有竞争力的结果。更重要的是，MoCo学习到的表示，可以很好的迁移到下游的任务中。MoCo在PASCAL VOC, COCO和其他数据集的7个检测/分割任务中，可以超过其有监督的预训练对应模型，有时候超过的很多。这说明，无监督表示学习和有监督表示学习的差距，在很多视觉任务中已经基本不存在了。

## 1. Introduction

Unsupervised representation learning is highly successful in natural language processing, e.g., as shown by GPT [50, 51] and BERT [12]. But supervised pre-training is still dominant in computer vision, where unsupervised methods generally lag behind. The reason may stem from differences in their respective signal spaces. Language tasks have discrete signal spaces (words, sub-word units, etc.) for building tokenized dictionaries, on which unsupervised learning can be based. Computer vision, in contrast, further concerns dictionary building [54, 9, 5], as the raw signal is in a continuous, high-dimensional space and is not structured for human communication (e.g., unlike words).

无监督表示学习在自然语言处理中非常成功，如GPT模型[50,51]和BERT[12]模型。但有监督的预训练在计算机视觉中仍然是占主导地位的，其中的无监督方法一般还差的很远。其原因可能源于其各自的信号空间的差异。语言任务有着离散的信号空间（语句，子语句单元，等）来构建标志化的字典，这可以基于无监督的学习。比较起来，计算机视觉要进一步考虑字典构建[54,9,5]，因为原始信号是在一个连续的高维度空间中的，对于人类的沟通来说并不是结构化的（如，与语句并不相象）。

Several recent studies [61, 46, 36, 66, 35, 56, 2] present promising results on unsupervised visual representation learning using approaches related to the contrastive loss [29]. Though driven by various motivations, these methods can be thought of as building dynamic dictionaries. The “keys” (tokens) in the dictionary are sampled from data (e.g., images or patches) and are represented by an encoder network. Unsupervised learning trains encoders to perform dictionary look-up: an encoded “query” should be similar to its matching key and dissimilar to others. Learning is formulated as minimizing a contrastive loss [29].

几个最近的研究在无监督视觉表示学习中给出了很有希望的结果，使用的方法与对比损失相关。虽然是有各种不同的推动原因，这些方法都可以视作构建动态字典。字典中的“关键字”（标志）是从数据（如图像或图像块）中采样得到的，是用一个编码器网络表示得到的。无监督学习训练编码器以进行字典查找：一种编码的查询应当与其匹配的关键字是类似的，与其他是不类似的。学习因此就可以表述为对对比损失的最小化。

From this perspective, we hypothesize that it is desirable to build dictionaries that are: (i) large and (ii) consistent as they evolve during training. Intuitively, a larger dictionary may better sample the underlying continuous, high-dimensional visual space, while the keys in the dictionary should be represented by the same or similar encoder so that their comparisons to the query are consistent. However, existing methods that use contrastive losses can be limited in one of these two aspects (discussed later in context).

从这个角度来说，我们假设可以构建这样的字典：(i)大型的，(ii)在训练的过程中变化时，是具有连续性和一致性的。直觉上来说，更大的字典可以更好的对潜在的连续的、高维的视觉空间进行采样，而字典中的关键字应当由相同的或类似的编码器表示，这样其与查询的比较才可以一致连续。但是，现有使用对比损失的方法在这两方面中的一个都是受限的。

We present Momentum Contrast (MoCo) as a way of building large and consistent dictionaries for unsupervised learning with a contrastive loss (Figure 1). We maintain the dictionary as a queue of data samples: the encoded representations of the current mini-batch are enqueued, and the oldest are dequeued. The queue decouples the dictionary size from the mini-batch size, allowing it to be large. Moreover, as the dictionary keys come from the preceding several mini-batches, a slowly progressing key encoder, implemented as a momentum-based moving average of the query encoder, is proposed to maintain consistency.

我们提出动量对比度(MoCo)，这是一种构建大型连续字典的方法，可以用对比损失进行无监督学习（见图1）。我们将字典作为数据采样的队列来进行维护：当前mini-batch的编码的表示编入队列，最老的从队列中移除。这个队列将字典的大小与mini-batch大小解除耦合，使其可以很大。而且，由于字典关键字是来自于之前的几个mini-batches，即一个缓慢过程的关键字编码器，实现成了一个基于动量的查询编码器的滑动平均，可以维持一致性。

Figure 1. Momentum Contrast (MoCo) trains a visual representation encoder by matching an encoded query q to a dictionary of encoded keys using a contrastive loss. The dictionary keys {k0,k1,k2,...} are defined on-the-fly by a set of data samples. The dictionary is built as a queue, with the current mini-batch enqueued and the oldest mini-batch dequeued, decoupling it from the mini-batch size. The keys are encoded by a slowly progressing encoder, driven by a momentum update with the query encoder. This method enables a large and consistent dictionary for learning visual representations.

MoCo is a mechanism for building dynamic dictionaries for contrastive learning, and can be used with various pretext tasks. In this paper, we follow a simple instance discrimination task [61, 63, 2]: a query matches a key if they are encoded views (e.g., different crops) of the same image. Using this pretext task, MoCo shows competitive results under the common protocol of linear classification in the ImageNet dataset [11].

MoCo是构建动态字典进行对比学习的机制，可以在很多pretext任务中使用。本文中，我们进行了一个简单的实例区分任务：一个查询如果其是同样图像的不同视角的编码（如不同的剪切块），则与一个关键字匹配。使用这种pretext任务，MoCo在ImageNet数据集上，在线性分类的常见协议下，展现出了很有竞争力的结果。

A main purpose of unsupervised learning is to pre-train representations (i.e., features) that can be transferred to downstream tasks by fine-tuning. We show that in 7 downstream tasks related to detection or segmentation, MoCo unsupervised pre-training can surpass its ImageNet supervised counterpart, in some cases by nontrivial margins. In these experiments, we explore MoCo pre-trained on ImageNet or on a one-billion Instagram image set, demonstrating that MoCo can work well in a more real-world, billion-image scale, and relatively uncurated scenario. These results show that MoCo largely closes the gap between unsupervised and supervised representation learning in many computer vision tasks, and can serve as an alternative to ImageNet supervised pre-training in several applications.

无监督学习的一个主要目标是预训练表示（即，特征），这个表示可以通过精调来迁移到下游的任务。我们在与检测或分割有关的7个下游任务中，展示了MoCo无监督预训练可以超过其ImageNet有监督训练的对应模型，在一些情况中，超过的还比较多。在这些试验中，我们将MoCo在ImageNet或在一个有十亿Instagram图像的数据集上进行预训练，证明了MoCo在一个更加接近真实世界的、十亿级图像的规模上可以很好的工作，而且是一种相对不经维护的场景。这些结果表明，MoCo在很多计算机视觉任务中基本上消灭了无监督和有监督表示学习的差距，可以在几种应用中用作ImageNet上有监督预训练的一种替代。

## 2. Related Work

Unsupervised/self-supervised learning methods generally involve two aspects: pretext tasks and loss functions. The term “pretext” implies that the task being solved is not of genuine interest, but is solved only for the true purpose of learning a good data representation. Loss functions can often be investigated independently of pretext tasks. MoCo focuses on the loss function aspect. Next we discuss related studies with respect to these two aspects.

无/自监督学习方法一般涉及两方面：pretext任务和损失函数。术语“pretext”意思是，要解决这个任务并不是真的对其感兴趣，而是解决这个任务的真正目的是，学习一个好的数据表示。损失函数通常可以独立于pretext任务进行研究。MoCo关注在损失函数这个方面。下面我们讨论与这两个方面相关的研究。

**Loss functions**. A common way of defining a loss function is to measure the difference between a model’s prediction and a fixed target, such as reconstructing the input pixels (e.g., auto-encoders) by L1 or L2 losses, or classifying the input into pre-defined categories (e.g., eight positions [13], color bins [64]) by cross-entropy or margin-based losses. Other alternatives, as described next, are also possible.

**损失函数**。定义损失函数的一种常见方法是，衡量一个模型的预测和固定目标之间的差异，比如用L1损失或L2损失衡量重建输入的像素（如，自动编码器），或将输入分类到预定义的类别（如，8个位置[13]，色彩bins[64]），常用交叉熵损失或基于边缘差距的损失。其他的替代下面会讨论，这也是可能的。

Contrastive losses [29] measure the similarities of sample pairs in a representation space. Instead of matching an input to a fixed target, in contrastive loss formulations the target can vary on-the-fly during training and can be defined in terms of the data representation computed by a network [29]. Contrastive learning is at the core of several recent works on unsupervised learning [61, 46, 36, 66, 35, 56, 2], which we elaborate on later in context (Sec. 3.1).

对比损失[29]，衡量的是样本对在一个表示空间的相似度。对比损失的表述不是将输入与固定目标匹配，而是与在训练过程中可以随时变化的目标相匹配，可以用网络计算的数据表示来定义[29]。对比学习是最近几篇无监督学习的核心，我们在后续的上下文中会详述。

Adversarial losses [24] measure the difference between probability distributions. It is a widely successful technique for unsupervised data generation. Adversarial methods for representation learning are explored in [15, 16]. There are relations (see [24]) between generative adversarial networks and noise-contrastive estimation (NCE) [28].

对抗损失[24]衡量的是概率分布之间的差异。在无监督数据生成中，这是一个在很广范围内都很成功的技术。表示学习的对抗方法[15,16]中进行了探索。在生成式对抗网络和噪声对比估计(NCE)之间有一定的关系。

**Pretext tasks**. A wide range of pretext tasks have been proposed. Examples include recovering the input under some corruption, e.g., denoising auto-encoders [58], context auto-encoders [48], or cross-channel auto-encoders (colorization) [64, 65]. Some pretext tasks form pseudo-labels by, e.g., transformations of a single (“exemplar”) image [17], patch orderings [13, 45], tracking [59] or segmenting objects [47] in videos, or clustering features [3, 4].

**Pretext任务**。已经提出了很多种pretext任务。例子包括，在一定的腐蚀下恢复输入，如去噪声的自动编码器[58]，上下文的自动编码器[48]，或跨通道的自动编码器（上色）[64,65]。一些pretext任务会以一些方法形成伪标签，如单幅图像（样板）的变换[17]，图像块的排序[13,45]，跟踪[59]或在视频中分割目标[47]，或特征聚类[3,4]。

**Contrastive learning vs. pretext tasks**. Various pretext tasks can be based on some form of contrastive loss functions. The instance discrimination method [61] is related to the exemplar-based task [17] and NCE [28]. The pretext task in contrastive predictive coding (CPC) [46] is a form of context auto-encoding [48], and in contrastive multiview coding (CMC) [56] it is related to colorization [64].

**对比学习vs pretext任务**。各种pretext任务都可以基于一些形式的对比损失函数。实例区分方法[61]与基于样板的任务[17]和NCE[28]有关系。在对比预测编码(CPC)[46]中的pretext任务，是上下文自动编码[48]的一种形式，而对比多视编码(CMC)[56]与上色[64]有关。

## 3. Method

### 3.1. Contrastive Learning as Dictionary Look-up

Contrastive learning [29], and its recent developments, can be thought of as training an encoder for a dictionary look-up task, as described next. 对比学习[29]及其最近的发展，可以视为为一个字典查找任务训练一个编码器，如下所述。

Consider an encoded query q and a set of encoded samples {k0, k1, k2, ...} that are the keys of a dictionary. Assume that there is a single key (denoted as k+) in the dictionary that q matches. A contrastive loss [29] is a function whose value is low when q is similar to its positive key k+ and dissimilar to all other keys (considered negative keys for q). With similarity measured by dot product, a form of a contrastive loss function, called InfoNCE [46], is considered in this paper:

考虑一个编码的查询q，和一个编码的样本集{k0, k1, k2, ...}，是一个字典的关键字。假设字典中有一个关键字（表示为k+）与q匹配。对比损失[29]是一个函数，参数是q与一个关键字，使q与其匹配的关键字k+函数值低，与其不匹配的其他关键字其函数值高。在相似度用点积度量的情况下，本文采用的一种对比损失函数，称为InfoNCE[46]，如下：

$$L_q = −log \frac{exp(q·k_+/τ)}{\sum_{i=0}^K exp(q·k_i/τ)}$$(1)

where τ is a temperature hyper-parameter[61]. The sum is over one positive and K negative samples. Intuitively, this loss is the log loss of a (K+1)-way softmax-based classifier that tries to classify q as k+. Contrastive loss functions can also be based on other forms [29, 59, 61, 36], such as margin-based losses and variants of NCE losses.

其中τ是一个温度超参数[61]。求和是对一个正样本和K个负样本进行的。直觉上来说，这个损失是一个K+1路的基于softmax的分类器的损失的log值，分类器试图将q分类为k+。对比损失函数也可以基于其他形式，如基于边缘距离的损失和NCE损失的变体。

The contrastive loss serves as an unsupervised objective function for training the encoder networks that represent the queries and keys [29]. In general, the query representation is $q = f_q(x^q)$ where $f_q$ is an encoder network and $x^q$ is a query sample (likewise, $k = f_k(x^k)$). Their instantiations depend on the specific pretext task. The input $x^q$ and $x^k$ can be images [29, 61, 63], patches [46], or context consisting a set of patches [46]. The networks $f_q$ and $f_k$ can be identical [29, 59, 63], partially shared [46, 36, 2], or different [56].

对比损失可以作为无监督目标函数，用于训练表示查询和关键字的编码器网络。一般来说，查询表示是$q = f_q(x^q)$，其中$f_q$是一个编码器网络，$x^q$是一个查询样本（类似的，$k = f_k(x^k)$）。其实例化依赖于具体的pretext任务。输入$x^q$和$x^k$可以是图像，图像块，或包含一系列图像块的上下文。网络$f_q$和$f_k$可以是完全一样的，部分共享的，或不同的。

### 3.2. Momentum Contrast

From the above perspective, contrastive learning is a way of building a discrete dictionary on high-dimensional continuous inputs such as images. The dictionary is dynamic in the sense that the keys are randomly sampled, and that the key encoder evolves during training. Our hypothesis is that good features can be learned by a large dictionary that covers a rich set of negative samples, while the encoder for the dictionary keys is kept as consistent as possible despite its evolution. Based on this motivation, we present Momentum Contrast as described next.

从上述观点出发，对比学习是一种在高维连续输入上构建离散字典的方法，比如图像。关键字是随机取样的，关键字编码器在训练的过程中在演变，在这个意义上，字典是动态的。我们的假设是，好的特征可以由一个大的字典学习到，这个字典覆盖了负样本的丰富的集合，而字典关键字的编码器尽管是在演化中，也保持尽可能的一致。基于上述动力，我们提出了动量对比，如下所述。

**Dictionary as a queue**. At the core of our approach is maintaining the dictionary as a queue of data samples. This allows us to reuse the encoded keys from the immediate preceding mini-batches. The introduction of a queue decouples the dictionary size from the mini-batch size. Our dictionary size can be much larger than a typical mini-batch size, and can be flexibly and independently set as a hyper-parameter.

**字典形成一个队列**。在我们方法的核心维护着一个字典，是数据样本的一个队列。这使我们可以从前一个mini-batch重用编码的关键字。这个队列的引入，使字典大小与mini-batch大小成为无关的。我们的字典大小可以比典型的mini-batch大小要大很多，可以很灵活并独立的设置为一个超参数。

The samples in the dictionary are progressively replaced. The current mini-batch is enqueued to the dictionary, and the oldest mini-batch in the queue is removed. The dictionary always represents a sampled subset of all data, while the extra computation of maintaining this dictionary is manageable. Moreover, removing the oldest mini-batch can be beneficial, because its encoded keys are the most outdated and thus the least consistent with the newest ones.

字典中的样本逐步被替换掉。目前的mini-batch编码进字典，队列中最老的mini-batch被去除掉。字典永远代表了所有数据的一个采样的子集，而维护这个字典的额外计算是可以处理的。而且，去除掉最老的mini-batch是有好处的，因为其编码的关键字是最陈旧的，因此与最新的关键字是最不一致的。

**Momentum update**. Using a queue can make the dictionary large, but it also makes it intractable to update the key encoder by back-propagation (the gradient should propagate to all samples in the queue). A naive solution is to copy the key encoder $f_k$ from the query encoder $f_q$, ignoring this gradient. But this solution yields poor results in experiments (Sec. 4.1). We hypothesize that such failure is caused by the rapidly changing encoder that reduces the key representations’ consistency. We propose a momentum update to address this issue.

**动量更新**。使用一个队列，可以使得字典很大，但很难通过反向传播来更新关键字的编码器（梯度应当传播到队列中的所有样本中）。一种天然的解决方案是从查询编码器$f_q$中复制关键字编码器$f_k$，忽略梯度。但这种方法在试验中得到了很差的结果（4.1节）。我们假设，这种失败是由于迅速变化的编码器导致的，降低了关键字表示的一致性。我们提出动量更新来解决这个问题。

Formally, denoting the parameters of $f_k$ as $θ_k$ and those of $f_q$ as $θ_q$, we update $θ_k$ by: 形式上，我们将$f_k$的参数表示为$θ_k$，$f_q$的参数表示为$θ_q$，我们通过下式更新$θ_k$：

$$θ_k ← mθ_k + (1 − m)θ_q$$(2)

Here m ∈ [0, 1) is a momentum coefficient. Only the parameters $θ_q$ are updated by back-propagation. The momentum update in Eqn.(2) makes $θ_k$ evolve more smoothly than $θ_q$. As a result, though the keys in the queue are encoded by different encoders (in different mini-batches), the difference among these encoders can be made small. In experiments, a relatively large momentum (e.g., m = 0.999, our default) works much better than a smaller value (e.g., m = 0.9), suggesting that a slowly evolving key encoder is a core to making use of a queue.

这里m ∈ [0, 1)是一个动量系数。只有参数$θ_q$是通过反向传播更新的。式(2)中的动量更新使$θ_k$比$θ_q$变化的更加平滑。结果是，虽然队列中的关键字是通过不同的编码器编码的（在不同的mini-batches），这些编码器之间的差异可以是很小的。在试验中，一个相对较大的动量（如，m = 0.999，默认是这个值）比较小的值（如，m = 0.9）的效果要好的多，说明缓慢变化的关键字编码器是使用队列的关键核心。

**Relations to previous mechanisms**. MoCo is a general mechanism for using contrastive losses. We compare it with two existing general mechanisms in Figure 2. They exhibit different properties on the dictionary size and consistency.

**与之前的机制间的关系**。MoCo是使用对比损失的通用机制。我们与两种已有的通用机制在图2中进行了对比。它们在字典大小和一致性上表现出了不同的性质。

Figure 2. Conceptual comparison of three contrastive loss mechanisms (empirical comparisons are in Figure 3 and Table 3). Here we illustrate one pair of query and key. The three mechanisms differ in how the keys are maintained and how the key encoder is updated. (a): The encoders for computing the query and key representations are updated end-to-end by back-propagation (the two encoders can be different). (b): The key representations are sampled from a memory bank [61]. (c): MoCo encodes the new keys on-the-fly by a momentum-updated encoder, and maintains a queue (not illustrated in this figure) of keys.

The **end-to-end** update by back-propagation is a natural mechanism (e.g., [29, 46, 36, 63, 2, 35], Figure 2a). It uses samples in the current mini-batch as the dictionary, so the keys are consistently encoded (by the same set of encoder parameters). But the dictionary size is coupled with the mini-batch size, limited by the GPU memory size. It is also challenged by large mini-batch optimization [25]. Some recent methods [46, 36, 2] are based on pretext tasks driven by local positions, where the dictionary size can be made larger by multiple positions. But these pretext tasks may require special network designs such as patchifying the input [46] or customizing the receptive field size [2], which may complicate the transfer of these networks to downstream tasks.

通过反向传播的**端到端**的更新是一种很自然的机制。它使用目前mini-batch中的样本作为字典，所以关键字是一致的（由相同的编码器参数的集合）编码的。但字典大小是与mini-batch大小是紧密联系的，受限于GPU内存的大小，同时还受到大的mini-batch优化的挑战[25]。一些最近的方法是基于pretext任务的，是受局部位置驱动的，其中字典大小可以通过多个位置变得更大。但这些pretext任务需要特殊的网络设计，比如将输入图像分成很多图像块，或定制感受野，这使得这些网络迁移到下游任务会更加复杂。

Another mechanism is the **memory bank** approach proposed by [61] (Figure 2b). A memory bank consists of the representations of all samples in the dataset. The dictionary for each mini-batch is randomly sampled from the memory bank with no back-propagation, so it can support a large dictionary size. However, the representation of a sample in the memory bank was updated when it was last seen, so the sampled keys are essentially about the encoders at multiple different steps all over the past epoch and thus are less consistent. A momentum update is adopted on the memory bank in [61]. Its momentum update is on the representations of the same sample, not the encoder. This momentum update is irrelevant to our method, because MoCo does not keep track of every sample. Moreover, our method is more memory-efficient and can be trained on billion-scale data, which can be intractable for a memory bank.

另一种机制是[61]提出的**存储组**方法。一个存储组包含数据集中所有样本的表示。每个mini-batch的字典是从存储组中随机取样得到的，不需要反向传播，所以可以支持很大的字典大小。但是，存储组中一个样本的表示，在最后一个遇到时进行更新，所以取样的关键字是基本上是关于，在最后一轮数据处理中，编码器在多个不同的步骤的输出值，因此是不那么一致的。在[61]在存储组上采用了一个动量更新。其动量更新是在同样样本的表示上的，而不是在编码器上的。这种动量更新与我们的方法是无关的，因此MoCo并不跟踪每个样本。而且，我们的方法更加节约存储，可以在十亿级的数据上进行训练，对于存储组来说是很难进行的。

Sec. 4 empirically compares these three mechanisms. 第4部分从经验上比较了这三种机制。

### 3.3. Pretext Task

Contrastive learning can drive a variety of pretext tasks. As the focus of this paper is not on designing a new pretext task, we use a simple one mainly following the instance discrimination task in [61], to which some recent works [63, 2] are related.

对比学习可以驱动很多pretext任务。由于本文的焦点并不是设计一个新的pretext任务，我们使用一个简单的任务，主要是[61]中的实例区分的任务，最近的[63,2]也是相关的。

Following [61], we consider a query and a key as a positive pair if they originate from the same image, and otherwise as a negative sample pair. Following [63, 2], we take two random “views” of the same image under random data augmentation to form a positive pair. The queries and keys are respectively encoded by their encoders, $f_q$ and $f_k$. The encoder can be any convolutional neural network [39].

按照[61]，我们考虑一个查询和一个关键字，如果其源于同样的图像，那么就是一个正样本对，否则就是一个负样本对。按照[63,2]，我们对图像的图像进行一些随机数据扩增，即两个随机“视角”，以形成一个正对。查询和关键字分别由其编码器$f_q$和$f_k$进行编码。编码器可以是任意的卷积神经网络[39]。

Algorithm 1 provides the pseudo-code of MoCo for this pretext task. For the current mini-batch, we encode the queries and their corresponding keys, which form the positive sample pairs. The negative samples are from the queue.

算法1给出了在这个pretext任务中的MoCo的伪代码。对于当前的mini-batch，我们对查询及其对应的关键字进行编码，这形成了正样本对。负样本是从队列中得到的。

**Technical details**. We adopt a ResNet [33] as the encoder, whose last fully-connected layer (after global average pooling) has a fixed-dimensional output (128-D [61]). This output vector is normalized by its L2-norm [61]. This is the representation of the query or key. The temperature τ in Eqn.(1) is set as 0.07 [61]. The data augmentation setting follows [61]: a 224×224-pixel crop is taken from a randomly resized image, and then undergoes random color jittering, random horizontal flip, and random grayscale conversion, all available in PyTorch’s torchvision package.

**技术细节**。我们采用一种ResNet作为编码器，其最后的全连接层（在全局平均池化之后）有固定维度的输出(128-D[61])。这个输出向量由其L2范数进行归一化[61]。这是查询或关键字的表示。式(1)中的温度τ设置为0.07[61]。数据扩充的设置按照[61]：对一个随机变换大小的图像取一个224×224的剪切块，然后进行随机的色彩抖动，随机的水平翻转和随机的灰度变换，这在PyTorch的torchvision模块中都是可用的。

**Shuffling BN**. Our encoders $f_q$ and $f_k$ both have Batch Normalization (BN) [37] as in the standard ResNet [33]. In experiments, we found that using BN prevents the model from learning good representations, as similarly reported in [35] (which avoids using BN). The model appears to “cheat” the pretext task and easily finds a low-loss solution. This is possibly because the intra-batch communication among samples (caused by BN) leaks information.

**混洗BN**。我们的编码器$f_q$和$f_k$都有BN层，和标准的ResNet一样。在试验中，我们发现使用BN会阻碍模型学习到好的表示，[35]也有类似的发现（所以没有使用BN）。模型似乎会欺骗pretext任务，很轻松的找到一个低损失的函数。这可能是因为批次内样本的通信（由BN导致的）会泄漏信息。

We resolve this problem by shuffling BN. We train with multiple GPUs and perform BN on the samples independently for each GPU (as done in common practice). For the key encoder $f_k$, we shuffle the sample order in the current mini-batch before distributing it among GPUs (and shuffle back after encoding); the sample order of the mini-batch for the query encoder $f_q$ is not altered. This ensures the batch statistics used to compute a query and its positive key come from two different subsets. This effectively tackles the cheating issue and allows training to benefit from BN.

我们通过混洗BN来解决这个问题。我们用多个GPUs进行训练，在样本上独立的对每个GPU进行BN（这是常规操作）。对于关键字编码器$f_k$，我们在当前mini-batch中对样本顺序进行混洗，然后再在各个GPUs中分发（在编码后再混洗回来）；对于查询的编码器$f_q$的mini-batch样本顺序并没有进行改变。这确保了用于计算一个查询和其正关键字的批次的统计来自两个不同的子集。这有效的解决了作弊的问题，使训练可以从BN中受益。

We use shuffled BN in both our method and its end-to-end ablation counterpart (Figure 2a). It is irrelevant to the memory bank counterpart (Figure 2b), which does not suffer from this issue because the positive keys are from different mini-batches in the past.

我们在我们的方法中及其端到端的分离试验的对应模型中使用混洗BN（图2a）。这与存储组的对应模型无关（图2b），所以不会受这个问题困扰，因为正的关键字是在过去的不同的mini-batch中来的。

## 4. Experiments

We study unsupervised training performed in: 我们在下列数据集中研究无监督训练：

**ImageNet-1M (IN-1M)**: This is the ImageNet [11] training set that has ∼1.28 million images in 1000 classes (often called ImageNet-1K; we count the image number instead, as classes are not exploited by unsupervised learning). This dataset is well-balanced in its class distribution, and its images generally contain iconic view of objects. 这是ImageNet训练集，有大约128万图像，1000类别。这个数据集在类别分布上很均衡，其图像一般都是目标的标志性视角。

**Instagram-1B (IG-1B)**: Following [44], this is a dataset of ∼1 billion (940M) public images from Instagram. The images are from ∼1500 hashtags [44] that are related to the ImageNet categories. This dataset is relatively uncurated comparing to IN-1M, and has a long-tailed, unbalanced distribution of real-world data. This dataset contains both iconic objects and scene-level images. 这个数据集有大约十亿幅公开图像，是从Instagram中收集的。图像大约包含1500个hashtags，与ImageNet的类别有关。这个数据集与上面相比，没有经过整理，所以是真实世界的数据，不均衡分布，有很长的拖尾。这个数据集包含标志性视角的目标和场景级的图像。

**Training**. We use SGD as our optimizer. The SGD weight decay is 0.0001 and the SGD momentum is 0.9. For IN-1M, we use a mini-batch size of 256 (N in Algorithm 1) in 8 GPUs, and an initial learning rate of 0.03. We train for 200 epochs with the learning rate multiplied by 0.1 at 120 and 160 epochs [61], taking ∼53 hours training ResNet-50. For IG-1B, we use a mini-batch size of 1024 in 64 GPUs, and a learning rate of 0.12 which is exponentially decayed by 0.9× after every 62.5k iterations (64M images). We train for 1.25M iterations (∼1.4 epochs of IG-1B), taking ∼6 days for ResNet-50.

**训练**。我们使用SGD作为优化器。SGD权重衰减是0.0001，SGD动量为0.9。对于IN-1M，我们使用的mini-batch大小为256，8个GPUs，初始学习速率为0.03。我们训练200轮，在第120和160轮时，学习速率乘以0.1，训练ResNet-50耗费了53个小时。对于IG-1B，我们使用的mini-batch大小为1024，60个GPUs，学习速率为0.12，每62.5k次迭代指数衰减0.9。我们训练1.25M次迭代，大约是1.4轮，对于ResNet-50大约耗费6天时间。

### 4.1. Linear Classification Protocol

We first verify our method by linear classification on frozen features, following a common protocol. In this sub-section we perform unsupervised pre-training on IN-1M. Then we freeze the features and train a supervised linear classifier (a fully-connected layer followed by softmax). We train this classifier on the global average pooling features of a ResNet, for 100 epochs. We report 1-crop, top-1 classification accuracy on the ImageNet validation set.

我们首先通过在冻结的特征上的线性分类，遵循常见协议，来验证我们的方法。在这个小节中，我们在IN-1M上进行无监督预训练。然后我们将特征冻结，训练一个有监督的线性分类器（一个全连接层跟着softmax）。我们在ResNet的全局平均池化特征上训练这个分类器，训练100个循环。我们在ImageNet验证集上给出了一个剪切块的top-1分类准确率。

For this classifier, we perform a grid search and find the optimal initial learning rate is 30 and weight decay is 0 (similarly reported in [56]). These hyper-parameters perform consistently well for all ablation entries presented in this subsection. These hyper-parameter values imply that the feature distributions (e.g., magnitudes) can be substantially different from those of ImageNet supervised training, an issue we will revisit in Sec. 4.2.

对这个分类器，我们进行一个网格搜索，找到最优的初始学习速率30，权重衰减为0（[56]中给出了类似的结果）。这些超参数在本小节中的所有分离试验中都表现的非常好。这些超参数值说明，特征分布（如，幅度）与在ImageNet上监督训练得到的特征有显著不同，我们在4.2节中会重新讨论。

**Ablation: contrastive loss mechanisms**. We compare the three mechanisms that are illustrated in Figure 2. To focus on the effect of contrastive loss mechanisms, we implement all of them in the same pretext task as described in Sec. 3.3. We also use the same form of InfoNCE as the contrastive loss function, Eqn.(1). As such, the comparison is solely on the three mechanisms.

**分离研究：对比损失机制**。我们在图2中比较了三种机制。为聚焦在对比损失机制的效果上，我们在3.3节所述的同一个pretext任务上进行实现。我们还使用同样的InfoNCE作为对比损失函数，即式(1)。这样，对比的就只是这三种机制。

The results are in Figure 3. Overall, all three mechanisms benefit from a larger K. A similar trend has been observed in [61, 56] under the memory bank mechanism, while here we show that this trend is more general and can be seen in all mechanisms. These results support our motivation of building a large dictionary.

结果如图3所示。总体上，所有三种机制都会受益于较大的K。在[61,56]中的存储组的机制中，也观察到了类似的趋势，而这里我们展示了，这个趋势更一般化，可以在所有机制中看到。这些结果支持我们构建一个大型字典的动机。

The **end-to-end** mechanism performs similarly to MoCo when K is small. However, the dictionary size is limited by the mini-batch size due to the end-to-end requirement. Here the largest mini-batch a high-end machine (8 Volta 32GB GPUs) can afford is 1024. More essentially, large mini-batch training is an open problem [25]: we found it necessary to use the linear learning rate scaling rule [25] here, without which the accuracy drops (by ∼2% with a 1024 mini-batch). But optimizing with a larger mini-batch is harder [25], and it is questionable whether the trend can be extrapolated into a larger K even if memory is sufficient.

在K很小时，**端到端**的机制与MoCo表现类似。但是，字典大小在端到端的需要下，受限于mini-batch大小。这里最大的mini-batch，一个高端机器（8个Volta 32GB GPUs）可以达到1024。更重要的，大的mini-batch训练是一个开放的问题：我们发现这里很有必要使用线性学习速率缩放的规则，如果不采用，则准确率就会下降（在1024的mini-batch中下降大约2%）。但使用更大的mini-batch进行优化更加困难，在内存足够的情况下，这个趋势是否可以外插到一个更大的K值，这是存有疑问的。

The **memory bank** [61] mechanism can support a larger dictionary size. But it is 2.6% worse than MoCo. This is inline with our hypothesis: the keys in the memory bank are from very different encoders all over the past epoch and they are not consistent. Note the memory bank result of 58.0% reflects our improved implementation of [61].

**存储组**机制可以支持更大的字典大小。但比MoCo性能低了2.6%。这与我们的假设是一致的：存储组中的关键字是在过去的训练轮中从不同的编码器中得到的，并不一致。注意存储组的结果58.0%反映了我们对[61]的实现的改进。

**Ablation: momentum**. The table below shows ResNet-50 accuracy with different MoCo momentum values (m in Eqn.(2)) used in pre-training (K = 4096 here) :

**分离试验：动量**。下表给出了在预训练中使用不同的MoCo动量值（即式2中的m值）的ResNet-50准确率（这里K=4096）：

momentum m | 0 | 0.9 | 0.99 | 0.999 | 0.9999 
--- | --- | --- | --- | --- | ---
accuracy (%) | fail | 55.2 | 57.8 | 59.0 | 58.9

It performs reasonably well when m is in 0.99 ∼ 0.9999, showing that a slowly progressing (i.e., relatively large momentum) key encoder is beneficial. When m is too small (e.g., 0.9), the accuracy drops considerably; at the extreme of no momentum (m is 0), the training loss oscillates and fails to converge. These results support our motivation of building a consistent dictionary.

当m在0.99到0.9999的范围内时，表现很好，这说明缓慢变化的关键字编码器是有好处的（即，相对较大的动量）。当m很小时（如，0.9），准确率显著下降；在没有动量的极限情况时，训练损失在一直震荡，没有收敛。这些结果支持我们构建一个一致的字典的动机。

**Comparison with previous results**. Previous unsupervised learning methods can differ substantially in model sizes. For a fair and comprehensive comparison, we report **accuracy vs. #parameters** trade-offs. Besides ResNet-50 (R50) [33], we also report its variants that are 2× and 4× wider (more channels), following [38]. We set K=65536 and m = 0.999. Table 1 is the comparison.

**与之前的结果的对比**。之前的无监督学习的方法在模型大小上差异很大。为进行公平和综合的比较，我们给出准确率vs参数量的折中。除了ResNet-50 (R50)[33]，我们还给出了其两个变体，一个宽了2倍，一个宽了4倍（更多的通道数）。我们设K=65536，m=0.999。表1是比较结果。

Table 1. Comparison under the linear classification protocol on ImageNet. The figure visualizes the table. All are reported as unsupervised pre-training on the ImageNet-1M training set, followed by supervised linear classification trained on frozen features, evaluated on the validation set. The parameter counts are those of the feature extractors. We compare with improved reimplementations if available (referenced after the numbers).

MoCo with R50 performs competitively and achieves 60.6% accuracy, better than all competitors of similar model sizes (∼24M). MoCo benefits from larger models and achieves 68.6% accuracy with R50w4×.

使用R50的MoCo的性能很有竞争力，得到了60.6%的准确率，比所有大小类似(∼24M)的模型都要好。MoCo受益于更大的模型，在使用R50w4x时，准确率达到了68.6%。

Notably, we achieve competitive results using a standard ResNet-50 and require no specific architecture designs, e.g., patchified inputs [46, 35], carefully tailored receptive fields [2], or combining two networks [56]. By using an architecture that is not customized for the pretext task, it is easier to transfer features to a variety of visual tasks and make comparisons, studied in the next subsection.

值得注意的是，我们使用标准的ResNet-50获得了很有竞争力的结果，而不需要特定的架构设计，如，图像块化的输入，仔细定制的感受野，或两个网络的结合。使用的架构并不是为pretext任务定制的，更容易将特征迁移到更多视觉任务中，以进行比较，这在下一小节进行研究。

This paper’s focus is on a mechanism for general contrastive learning; we do not explore orthogonal factors (such as specific pretext tasks) that may further improve accuracy. As an example, “MoCo v2” [8], an extension of a preliminary version of this manuscript, achieves 71.1% accuracy with R50 (up from 60.6%), given small changes on the data augmentation and output projection head [7]. We believe that this additional result shows the generality and robustness of the MoCo framework.

本文的焦点是通用对比学习的机制；我们并没有探索其他正交的因素（如特定的pretext任务），这些因素是可以改进准确率的。如MoCo v2[8]，是目前初步版本的延伸，使用R50获得了71.1%的准确率，在数据扩增和输出投影头上做了较小的改进。我们相信，这个额外的结果可以说明我们MoCo框架的泛化性和稳健性。

### 4.2. Transferring Features

A main goal of unsupervised learning is to learn features that are transferrable. ImageNet supervised pre-training is most influential when serving as the initialization for fine-tuning in downstream tasks (e.g., [21, 20, 43, 52]). Next we compare MoCo with ImageNet supervised pre-training, transferred to various tasks including PASCAL VOC [18], COCO [42], etc. As prerequisites, we discuss two important issues involved [31]: normalization and schedules.

无监督学习的一个主要目标是学习可以迁移的目标。ImageNet监督预训练影响力很大，在下游任务中可以用作精调的初始化权重。下面我们将MoCo与ImageNet监督预训练比较，迁移到各种任务中，包括PASCAL VOC，COCO等。作为先决条件，我们讨论两个相关的重要问题：归一化和方案。

**Normalization**. As noted in Sec. 4.1, features produced by unsupervised pre-training can have different distributions compared with ImageNet supervised pre-training. But a system for a downstream task often has hyper-parameters (e.g., learning rates) selected for supervised pre-training. To relieve this problem, we adopt feature normalization during fine-tuning: we fine-tune with BN that is trained (and synchronized across GPUs [49]), instead of freezing it by an affine layer [33]. We also use BN in the newly initialized layers (e.g., FPN [41]), which helps calibrate magnitudes.

**归一化**。如4.1节所述，无监督预训练得到的特征，与ImageNet监督预训练得到的特征有非常不同的分布。但下游任务的系统通常都为监督预训练选择的有超参数（如，学习速率）。为缓解这个问题，我们在精调时采用了特征归一化：我们用训练好的BN进行精调（在不同GPU之间进行同步），而不是将其用一个仿射层进行冻结。我们还在新初始化的层中使用BN（如FPN），这帮助校准幅度。

We perform normalization when fine-tuning supervised and unsupervised pre-training models. MoCo uses the same hyper-parameters as the ImageNet supervised counterpart.

我们在精调监督和无监督预训练模型时进行归一化。MoCo与ImageNet对应的监督模型使用相同的超参数。

**Schedules**. If the fine-tuning schedule is long enough, training detectors from random initialization can be strong baselines, and can match the ImageNet supervised counter-part on COCO [31]. Our goal is to investigate transferability of features, so our experiments are on controlled schedules, e.g., the 1× (∼12 epochs) or 2× schedules [22] for COCO, in contrast to 6×∼9× in [31]. On smaller datasets like VOC, training longer may not catch up [31].

**方案**。如果精调方案足够长，从随机初始化训练的检测器可以作为很强的基准，可以在COCO上与ImageNet监督模型匹配。我们的目标是研究特征的可迁移性，所以我们的试验是在受控的方案中，如，对于COCO是1x或2x的方案，而在[31]中是6x-9x。我们较小的数据集，如VOC，训练的时间更长，可能赶不上[31]。

Nonetheless, in our fine-tuning, MoCo uses the same schedule as the ImageNet supervised counterpart, and random initialization results are provided as references.

尽管如此，在我们的精调中，MoCo与ImageNet监督模型使用相同的方案，随机初始化的结果只是作为参考。

Put together, our fine-tuning uses the same setting as the supervised pre-training counterpart. This may place MoCo at a disadvantage. Even so, MoCo is competitive. Doing so also makes it feasible to present comparisons on multiple datasets/tasks, without extra hyper-parameter search.

放在一起，我们的精调使用了与监督预训练算法相同的设置。这可能会将MoCo放到一个较为不利的位置。即使这样，MoCo仍然是很有竞争力的。这样做，也使得在多个数据集/任务中可以进行比较，而不需要额外的超参数搜索。

#### 4.2.1 PASCAL VOC Object Detection

**Setup**. The detector is Faster R-CNN[52] with a backbone of R50-dilated-C5 or R50-C4 [32] (details in appendix), with BN tuned, implemented in [60]. We fine-tune all layers end-to-end. The image scale is [480, 800] pixels during training and 800 at inference. The same setup is used for all entries, including the supervised pre-training baseline. We evaluate the default VOC metric of AP50 (i.e., IoU threshold is 50%) and the more stringent metrics of COCO-style AP and AP75. Evaluation is on the VOC test2007 set.

**设置**。检测器是使用R50-dilated-C5或R50-C4作为骨干的Faster R-CNN，进行了BN调节，在[60]中的实现。我们端到端的精调了所有层。图像大小是在训练时是[480, 800]，推理时是800。所有方法都使用的一样的设置，包括监督预训练基准。我们评估的是默认VOC度量AP50（即，IoU的阈值为50%），和更严格的COCO类型的度量AP和AP75。评估是在VOC test2007集上进行的。

**Ablation: backbones**. Table 2 shows the results fine-tuned on trainval07+12 (∼16.5k images). For R50-dilated-C5 (Table 2a), MoCo pre-trained on IN-1M is comparable to the supervised pre-training counterpart, and MoCo pretrained on IG-1B surpasses it. For R50-C4 (Table 2b), MoCo with IN-1M or IG-1B is better than the supervised counterpart: up to +0.9 AP50, +3.7 AP, and +4.9 AP75.

**分离试验：骨干**。表2给出了在trainval07+12上精调的结果。对于R50-dilated-C5（图2a），MoCo在IN-1M上的预训练结果与监督预训练模型结果是类似的，MoCo在IG-1B上的预训练则超过了这个结果。对于R50-C4（表2b），在IN-1M或IG-1B上的MoCo都比对应的监督训练模型更好：+0.9 AP50, +3.7 AP, +4.9 AP75。

Table 2. Object detection fine-tuned on PASCAL VOC trainval07+12. Evaluation is on test2007: AP50 (default VOC metric), AP (COCO-style), and AP75, averaged over 5 trials. All are fine-tuned for 24k iterations (∼23 epochs). In the brackets are the gaps to the ImageNet supervised pre-training counterpart. In green are the gaps of at least +0.5 point.

Interestingly, the transferring accuracy depends on the detector structure. For the C4 backbone, by default used in existing ResNet-based results [14, 61, 26, 66], the advantage of unsupervised pre-training is larger. The relation between pre-training vs. detector structures has been veiled in the past, and should be a factor under consideration.

有趣的是，迁移准确率与检测器结构有关。对于C4骨干网络，也是在现有的基于ResNet的默认结构，无监督预训练的优势更大一些。预训练与检测器结构之间的关系过去也曾经研究过，也是应当要考虑的一个因素。

**Ablation: contrastive loss mechanisms**. We point out that these results are partially because we establish solid detection baselines for contrastive learning. To pin-point the gain that is solely contributed by using the MoCo mechanism in contrastive learning, we fine-tune the models pre-trained with the end-to-end or memory bank mechanism, both implemented by us (i.e., the best ones in Figure 3), using the same fine-tuning setting as MoCo.

**分离试验：对比损失机制**。我们指出，这些结果部分是因为，我们在对比学习的检测基准上确定的。为确定只是由使用对比学习的MoCo得到的改进，我们精调了用端到端机制或存储组机制预训练的模型，都是由我们实现的（即，图3中最好的那些），使用的精调设置与MoCo一样。

These competitors perform decently (Table 3). Their AP and AP75 with the C4 backbone are also higher than the ImageNet supervised counterpart’s, c.f . Table 2b, but other metrics are lower. They are worse than MoCo in all metrics. This shows the benefits of MoCo. In addition, how to train these competitors in larger-scale data is an open question, and they may not benefit from IG-1B.

这些模型表现都还不错。使用C4骨干的AP和AP75比ImageNet监督训练的要高，参考表2b，但其他标准要低一些。他们在所有度量中都要比MoCo要差。这说明MoCo的好处。另外，怎样在大型数据集中训练这些模型还是一个公开的问题，它们可能无法从IG-1B中受益。

Table 3. Comparison of three contrastive loss mechanisms on PASCAL VOC object detection, fine-tuned on trainval07+12 and evaluated on test2007 (averages over 5 trials). All models are implemented by us (Figure 3), pre-trained on IN-1M, and fine-tuned using the same settings as in Table 2.

**Comparison with previous results**. Following the competitors, we fine-tune on trainval2007 (∼5k images) using the C4 backbone. The comparison is in Table 4.

**与之前的结果的比较**。按照竞争者模型，我们在trainval2007上使用C4骨干精调。比较结果如表4。

Table 4. Comparison with previous methods on object detection fine-tuned on PASCAL VOC trainval2007. Evaluation is on test2007. The ImageNet supervised counterparts are from the respective papers, and are reported as having the same structure as the respective unsupervised pre-training counterparts. All entries are based on the C4 backbone. The models in [14] are R101 v2 [34], and others are R50. The RelPos (relative position) [13] result is the best single-task case in the Multi-task paper [14]. The Jigsaw [45] result is from the ResNet-based implementation in [26]. Our results are with 9k-iteration fine-tuning, averaged over 5 trials. In the brackets are the gaps to the ImageNet supervised pre-training counterpart. In green are the gaps of at least +0.5 point.

For the AP50 metric, no previous method can catch up with its respective supervised pre-training counterpart. MoCo pre-trained on any of IN-1M, IN-14M (full ImageNet), YFCC-100M [55], and IG-1B can outperform the supervised baseline. Large gains are seen in the more stringent metrics: up to +5.2 AP and +9.0 AP75. These gains are larger than the gains seen in trainval07+12 (Table 2b).

对于AP50度量，之前的方法没有能和对应的监督预训练模型比较的。在IN-1M，IN-14M，YFCC-100M，IG-1B任何一个上的MoCo预训练结果都超过了监督基准。在更严格的度量中，可以看到很大的提升：+5.2 AP和+9.0 AP75。这些提升比在trainval07+12上看到的提升还要大（表2b）。

#### 4.2.2 COCO Object Detection and Segmentation

**Setup**. The model is Mask R-CNN [32] with the FPN [41] or C4 backbone, with BN tuned, implemented in [60]. The image scale is in [640, 800] pixels during training and is 800 at inference. We fine-tune all layers end-to-end. We fine-tune on the train2017 set (∼118k images) and evaluate on val2017. The schedule is the default 1× or 2× in [22].

**设置**。模型是使用了FPN或C4骨干的Mask R-CNN，经过了BN调整，在[60]中实现的。训练图像大小是[640, 800]像素，推理时是800。我们对所有的层进行端到端的精调。我们在train2017集上进行精调，在val2017上进行评估。方案是默认的1x或2x。

**Results**. Table 5 shows the results on COCO with the FPN (Table 5a, b) and C4 (Table 5c, d) backbones. With the 1× schedule, all models (including the ImageNet supervised counterparts) are heavily under-trained, as indicated by the ∼2 points gaps to the 2× schedule cases. With the 2× schedule, MoCo is better than its ImageNet supervised counterpart in all metrics in both backbones.

**结果**。表5给出了在COCO上带有FPN和使用C4的结果。在使用1x方案时，所有模型（包括ImageNet监督训练模型）都是未训练好的，因为与2x方案情况的训练有大约2个点的差距。在2x方案下，MoCo比其ImageNet监督训练模型，在所有度量中，在两种骨干下，都要好。

Table 5. Object detection and instance segmentation fine-tuned on COCO: bounding-box AP (APbb) and mask AP (APmk) evaluated on val2017. In the brackets are the gaps to the ImageNet supervised pre-training counterpart. In green are the gaps of at least +0.5 point.

#### 4.2.3 More Downstream Tasks

Table 6 shows more downstream tasks (implementation details in appendix). Overall, MoCo performs competitively with ImageNet supervised pre-training: 表6给出了更多的下游任务。总体上，MoCo与ImageNet监督预训练相比性能类似。

Table 6. MoCo vs. ImageNet supervised pre-training, fine-tuned on various tasks. For each task, the same architecture and schedule are used for all entries (see appendix). In the brackets are the gaps to the ImageNet supervised pre-training counterpart. In green are the gaps of at least +0.5 point.

**COCO keypoint detection**: supervised pre-training has no clear advantage over random initialization, whereas MoCo outperforms in all metrics. 监督预训练比随机初始化并没有明显的优势，而MoCo在所有度量中都有提高。

**COCO dense pose estimation** [1]: MoCo substantially outperforms supervised pre-training, e.g., by 3.7 points in $AP^{dp}_{75}$, in this highly localization-sensitive task. MoCo超过了监督预训练很多，如，在这个高度对位置敏感的任务中，在$AP^{dp}_{75}$上超过了3.7个点。

**LVIS v0.5 instance segmentation** [27]: this task has ∼1000 long-tailed distributed categories. Specifically in LVIS for the ImageNet supervised baseline, we find fine-tuning with frozen BN (24.4 APmk) is better than tunable BN (details in appendix). So we compare MoCo with the better supervised pre-training variant in this task. MoCo with IG-1B surpasses it in all metrics. 这个任务大约有1000个长尾分布的类别。具体的在LVIS中，对于ImageNet监督基准，我们发现，使用冻结BN进行精调比可调节的BN要好。所以我们在这个任务中，将MoCo与更好的监督的预训练变体进行比较。用IG-1B训练的MoCo在所有度量中都胜出了。

**Cityscapes instance segmentation** [10]: MoCo with IG-1B is on par with its supervised pre-training counterpart in APmk, and is higher in $AP^{mk}_{50}$. 在IG-1B上训练的MoCo，与其监督预训练模型在APmk上类似，在$AP^{mk}_{50}$上则胜出。

**Semantic segmentation**: On Cityscapes [10], MoCo outperforms its supervised pre-training counterpart by up to 0.9 point. But on VOC semantic segmentation, MoCo is worse by at least 0.8 point, a negative case we have observed. 在CityScapes中，MoCo超过了其监督的预训练模型0.9点。但在VOC语义分割中，MoCo则差了至少0.8个点，这是我们观察到的一个负面情况。

**Summary**. In sum, MoCo can outperform its ImageNet supervised pre-training counterpart in 7 detection or segmentation tasks. Besides, MoCo is on par on CityScapes instance segmentation, and lags behind on VOC semantic segmentation; we show another comparable case on iNaturalist [57] in appendix. Overall, MoCo has largely closed the gap between unsupervised and supervised representation learning in multiple vision tasks. 总结起来，MoCo在7个检测或分割任务中都超过了ImageNet监预训练对应模型。除此以外，MoCo在CityScapes实例分割上性能类似，在VOC语义分割上性能落后了一点；我们在附录中给出了在iNaturallist的另一个可比的例子。总体来说，MoCo基本上在多个视觉任务中，终结了无监督和监督表示学习的差距。

Remarkably, in all these tasks, MoCo pre-trained on IG-1B is consistently better than MoCo pre-trained on IN-1M. This shows that MoCo can perform well on this large-scale, relatively uncurated dataset. This represents a scenario towards real-world unsupervised learning.

令人印象深刻的是，在所有这些任务中，MoCo在IG-1B上的预训练模型，比在IN-1M上预训练的MoCo一直都要好。这说明，MoCo在这个大规模，而相对又是不能维护的数据集中表现很好。这代表了真实世界的无监督学习的一个场景。

## 5. Discussion and Conclusion

Our method has shown positive results of unsupervised learning in a variety of computer vision tasks and datasets. A few open questions are worth discussing. MoCo’s improvement from IN-1M to IG-1B is consistently noticeable but relatively small, suggesting that the larger-scale data may not be fully exploited. We hope an advanced pretext task will improve this. Beyond the simple instance discrimination task [61], it is possible to adopt MoCo for pretext tasks like masked auto-encoding, e.g., in language [12] and in vision [46]. We hope MoCo will be useful with other pretext tasks that involve contrastive learning.

我们的方法在很多计算机视觉任务和数据集中，得到了无监督学习的很好的结果。一些开放的问题是值得探讨的。MoCo从IN-1M到IG-1B的改进是持续可注意到的，但相对较小，说明大规模数据可能没有得到充分利用。我们希望有一个高级的pretext任务会改进这种情况。除了简单的实例区分任务，采用MoCo进行掩膜自动编码的pretext任务是可能的，如在视觉和在语言中，我们希望MoCo会在其他使用了对比学习的pretext任务中会有用。