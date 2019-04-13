# Auto-Keras: An Efficient Neural Architecture Search System 一种有效的神经架构搜索系统

Haifeng Jin et al. Texas A&M University

## Abstract 摘要

Neural architecture search (NAS) has been proposed to automatically tune deep neural networks, but existing search algorithms, e.g., NASNet [41], PNAS [22], usually suffer from expensive computational cost. Network morphism, which keeps the functionality of a neural network while changing its neural architecture, could be helpful for NAS by enabling more efficient training during the search. In this paper, we propose a novel framework enabling Bayesian optimization to guide the network morphism for efficient neural architecture search. The framework develops a neural network kernel and a tree-structured acquisition function optimization algorithm to efficiently explores the search space. Intensive experiments on real-world benchmark datasets have been done to demonstrate the superior performance of the developed framework over the state-of-the-art methods. Moreover, we build an opensource AutoML system based on our method, namely Auto-Keras. The system runs in parallel on CPU and GPU, with an adaptive search strategy for different GPU memory limits.

神经架构搜索(NAS)的提出是用于自动调节深度神经网络，但现有的搜索算法，如NASNet[41]，PNAS[22]，通常计算代价非常难以忍受。网络态射，可以在保持神经网络功能的同时，改变其神经架构，这对NAS是有用的，在搜索架构的过程中可以更有效的训练。本文中，我们提出了一种新的框架，使用贝叶斯优化来对网络态射进行指引，进行有效的网络架构搜索。这个框架发展出了一个神经网络核心和一个树结构的获得函数优化算法，可以有效的探索搜索空间。我们在真实世界基准测试数据集上进行了广泛的试验，证明了提出的框架是目前最好的方法。而且，基于我们的方法构建了一个开源的AutoML系统，称为Auto-Keras。这个系统在CPU和GPU上并行运行，对不同的GPU内存限制，有自适应的搜索策略。

**Keywords** Automated Machine Learning, AutoML, Neural Architecture Search, Bayesian Optimization, Network Morphism 自动机器学习，神经架构搜索，贝叶斯优化，网络态射

## 1 Introduction 引言

Automated Machine Learning (AutoML) has become a very important research topic with wide applications of machine learning techniques. The goal of AutoML is to enable people with limited machine learning background knowledge to use the machine learning models easily. Work has been done on automated model selection, automated hyperparameter tunning, and etc. In the context of deep learning, neural architecture search (NAS), which aims to search for the best neural network architecture for the given learning task and dataset, has become an effective computational tool in AutoML. Unfortunately, existing NAS algorithms are usually computationally expensive. The time complexity of NAS is $O(n \bar t)$, where n is the number of neural architectures evaluated during the search, and $\bar t$ is the average time consumption for evaluating each of the n neural networks. Many NAS approaches, such as deep reinforcement learning [2, 30, 40, 41], gradient-based methods [26] and evolutionary algorithms [10, 23, 31, 32, 34], require a large n to reach a good performance. Also, each of the n neural networks is trained from scratch which is very slow.

自动机器学习(AutoML)已经成为了一个非常重要的研究课题，广泛的应用了很多机器学习技术。AutoML的目标是使机器学习背景知识有限的人也可以很容易的使用机器学习模型。在自动模型选择、自动超参数调节等工作上都有了一些工作。在深度学习的领域中，神经架构搜索(NAS)，其目标是，对给定的学习任务和数据集，搜索最好的神经网络架构，已经在AutoML中成为了一个有效的计算工具。不幸的是，现有的NAS算法计算量都非常大。NAS算法的时间复杂度为$O(n \bar t)$，其中n是搜索过程中评估的神经架构数量，$\bar t$是评估这n种神经网络中一个的平均耗时。很多NAS方法，如深度强化学习[2,30,40,41]，基于梯度的方法[26]和演化算法[10,23,31,32,34]，需要很大的n来得到很好的性能。还有，这n个神经网络的每一个都是从头开始训练，速度非常慢。

Initial efforts have been devoted to making use of network morphism in neural architecture search [6, 11]. It is a technique to morph the architecture of a neural network but keep its functionality [8, 36]. Therefore, we are able to modify a trained neural network into a new architecture using the network morphism operations, e.g., inserting a layer or adding a skip-connection. Only a few more epochs are required to further train the new architecture towards better performance. Using network morphism would reduce the average training time $\bar t$ in neural architecture search. The most important problem to solve for network morphism-based NAS methods is the selection of operations, which is to select an operation from the network morphism operation set to morph an existing architecture to a new one. The network morphism-based NAS methods are not efficient enough. They either require a large number of training examples [6], or inefficient in exploring the large search space [11]. How to perform efficient neural architecture search with network morphism remains a challenging problem.

在神经架构搜索中使用网络态射已经有了一些努力[6,11]。保持网络的功能，同时改变网络的架构，这是一项技术[8,36]。因此，我们可以使用网络态射的操作，修改一个训练好的神经网络，成为一个新的架构，如插入一层，或增加一个跳跃连接。只需要将新架构多训练几轮，就可以得到更好的性能。使用网络态射会在神经架构搜索中降低平均训练时间$\bar t$。对于基于网络态射的NAS方法来说，需要解决的最重要问题是操作的选择，即从网络态射操作集中选择一个操作来改变现有的网络架构，成为新的架构。基于网络态射的NAS方法还不够高效。要么需要大量的训练样本[6]，要么在探索大的搜索空间时不够高效[11]。怎样使用网络态射来进行高效的网络架构搜索是一个很有挑战性的问题。

As we know, Bayesian optimization [33] has been widely adopted to efficiently explore black-box functions for global optimization, whose observations are expensive to obtain. For example, it has been used in hyperparameter tuning for machine learning models [13, 15, 17, 35], in which Bayesian optimization searches among different combinations of hyperparameters. During the search, each evaluation of a combination of hyperparameters involves an expensive process of training and testing the machine learning model, which is very similar to the NAS problem. The unique properties of Bayesian optimization motivate us to explore its capability in guiding the network morphism to reduce the number of trained neural networks n to make the search more efficient.

据我们所知，贝叶斯优化[33]已经广泛用于高效的探索黑盒函数以进行全局优化，要得到全局优化的结果非常昂贵。比如，对于机器学习模型可以用于超参数调节[13,15,17,35]，其中贝叶斯优化用于搜索超参数的不同组合。在搜索过程中，每个超参数的组合的评估都是训练和测试机器学习模型的昂贵过程，这与NAS问题非常类似。贝叶斯搜索的唯一性质促使我们探索其指引网络态射的能力，以减少训练的神经网络n的数量，使搜索过程更加高效。

It is non-trivial to design a Bayesian optimization method for network morphism-based NAS due to the following challenges. First, the underlying Gaussian process (GP) is traditionally used for learning probability distribution of functions in Euclidean space. To update the Bayesian optimization model with observations, the underlying GP is to be trained with the searched architectures and their performances. However, the neural network architectures are not in Euclidean space and hard to parameterize into a fixed-length vector. Second, an acquisition function needs to be optimized for Bayesian optimization to generate the next architecture to observe. However, in the context of network morphism, it is not to maximize a function in Euclidean space, but finding a node in a tree-structured search space, where each node represents a neural architecture and each edge is a morph operation. Thus traditional gradient-based methods cannot be simply applied. Third, the changes caused by a network morphism operation is complicated. A network morphism operation on one layer may change the shapes of some intermediate output tensors, which no longer match input shape requirements of the layers taking them as input. How to maintain such consistency is a challenging problem.

对基于网络态射的NAS设计一个贝叶斯优化方法有些难度，因为有如下的挑战。第一，潜在的高斯过程(GP)传统上用于学习欧几里得空间函数的概率分布。为用观测来更新贝叶斯优化的模型，用搜索到的架构及其性能来训练潜在的GP。但是，神经网络架构不是欧几里得空间的，很难参数化为固定长度的向量。第二，需要优化一个获得函数，贝叶斯优化才能生成下一个待观测的架构。但是，在网络态射的领域，并不是要最大化一个欧几里得空间的函数，而是在一个树状搜索空间中找到一个节点，其中每个节点代表一个神经架构，每个边都是一个变形操作。所以传统的基于梯度的方法不能简单的得到应用。第三，网络态射操作造成的变化是复杂的。对一个层进行一个网络态射操作，可能会改变某些中间输出张量的形状，这会使以其作为输入的层造成形状不匹配的问题。怎样保持这种连续性是一个有挑战的问题。

In this paper, an efficient neural architecture search with network morphism is proposed, which utilizes Bayesian optimization to guide through the search space by selecting the most promising operations each time. To tackle the aforementioned challenges, an edit-distance neural network kernel is constructed. Being consistent with the key idea of network morphism, it measures how many operations are needed to change one neural network to another. Besides, a novel acquisition function optimizer, which is capable of balancing between the exploration and exploitation, is designed specially for the tree-structure search space to enable Bayesian optimization to select from the operations. In addition, a graph-level network morphism is defined to address the changes in the neural architectures based on layer-level network morphism. The proposed approach is compared with the state-of-the-art NAS methods [11, 16] on benchmark datasets of MNIST, CIFAR10, and FASHION-MNIST. Within a limited search time, the architectures found by our method achieves the lowest error rates on all of the datasets.

本文中，提出了一种使用网络态射的高效的神经架构搜索方法，使用了贝叶斯优化来指引通过搜索空间，在每次都选择最有希望的操作。为处理前面所述的挑战，构建了一种编辑距离的神经网络核，这与网络态射的核心思想一致，它衡量了从一种神经网络转换成另一种需要多少操作。另外，设计了一种新的获得函数优化器，可以在探索和利用之间进行平衡，这是特意为树状搜索空间设计的，使得贝叶斯优化可以从操作集中选择。另外，定义了图级的网络态射，基于层级的网络态射来处理神经架构的变化。提出的方法与目前最好的NAS方法[11,16]在基准测试数据集MNIST、CIFAR10和FASHION-MNIST上进行了比较。在有限的搜索时间内，我们的方法发现的架构在所有数据集上都得到了最低的错误率。

In addition, we have developed a widely adopted open-source AutoML system based on our proposed method, namely Auto-Keras. It is an open-source AutoML system, which can be download and installed locally. The system is carefully designed with a concise interface for people not specialized in computer programming and data science to use. To speed up the search, the workload on CPU and GPU can run in parallel. To address the issue of different GPU memory, which limits the size of the neural architectures, a memory adaption strategy is designed for deployment.

另外，我们还基于我们提出的方法开发了广泛采用的开源AutoML系统，名为Auto-Keras。这是一种开源的AutoML系统，可以下载并在本地安装。这个系统经过了仔细设计，界面简洁，可供非计算机编程和数据科学人士使用。为加速搜索，可以使用多CPU和GPU并行运行。为处理不同GPU内存的问题（这个问题限制了神经架构的大小），设计了一种内存自适应策略在部署中使用。

The main contributions of the paper are as follows: 本文的主要贡献如下：

- Propose an algorithm for efficient neural architecture search based on network morphism guided by Bayesian optimization. 提出了一种贝叶斯优化指引的基于网络态射的算法进行高效的神经架构搜索。

- Conduct intensive experiments on benchmark datasets to demonstrate the superior performance of the proposed method over the baseline methods. 在基准测试数据集上进行了广泛的试验，证明了提出的方法优于基准方法。

- Develop an open-source system, namely Auto-Keras, which is one of the most widely used AutoML systems. 开发了一个开源系统，称为Auto-Keras，得到了广泛的使用。

## 2. Problem statement 问题描述

The general neural architecture search problem we studied in this paper is defined as: Given a neural architecture search space F, the input data D divided into $D_{train}$ and $D_{val}$, and the cost function $Cost(·)$, we aim at finding an optimal neural network $f^∗ ∈ F$, which could achieve the lowest cost on dataset D. The definition is equivalent to finding $f^∗$ satisfying:

本文中我们研究的一般性的神经架构搜索问题定义如下：给定一个神经架构搜索空间F，输入数据D分为$D_{train}$和$D_{val}$，和代价函数$Cost(·)$，我们的目标是找到最佳神经网络$f^∗ ∈ F$，在数据集D上得到最低的代价函数。这个定义等价于寻找满足下式的$f^∗$：

$$f^∗ = argmin_{f∈F} Cost(f(θ^∗), D_{val})$$(1)
$$θ^∗ = argmin_θ L(f(θ), D_{train})$$(2)

where $Cost(·, ·)$ is the evaluation metric function, e.g., accuracy, mean sqaured error, $θ^∗$ is the learned parameter of f. 其中$Cost(·, ·)$是评估标准函数，如准确率，均方误差，$θ^∗$是f学习到的参数。

The search space F covers all the neural architectures, which can be morphed from the initial architectures. The details of the morph operations are introduced in 3.3. Notably, the operations can change the number of filters in a convolutional layer, which makes F larger than methods with fixed layer width [24].

搜索空间F覆盖了所有的神经网络架构，可以从初始架构变形得到。变形操作的细节在3.3节中介绍。值得注意的是，这些操作可以改变卷积层中滤波器的数量，这使得F比那些使用固定层深度的方法要大[24]。

## 3. Network morphism guided by Bayesian optimization 贝叶斯优化指引的网络态射

The key idea of the proposed method is to explore the search space via morphing the neural architectures guided by Bayesian optimization (BO) algorithm. Traditional Bayesian optimization consists of a loop of three steps: update, generation, and observation. In the context of NAS, our proposed Bayesian optimization algorithm iteratively conducts: (1) Update: train the underlying Gaussian process model with the existing architectures and their performance; (2) Generation: generate the next architecture to observe by optimizing a delicately defined acquisition function; (3) Observation: obtain the actual performance by training the generated neural architecture. There are three main challenges in designing a method for morphing the neural architectures with Bayesian optimization. We introduce three key components separately in the subsequent sections coping with the three challenges.

我们提出的方法的关键思想是，通过贝叶斯优化(BO)指引的神经架构变形算法来探索搜索空间。传统的贝叶斯优化包括三个步骤的循环：更新，生成和观察。在NAS的领域中，我们提出的贝叶斯优化算法迭代着进行：(1)更新：用现有的架构和其性能，训练潜在的高斯过程模型；(2)生成：通过优化一个仔细定义的获得函数，来生成下一个架构进行观察；(3)观察：训练生成的神经架构，获得实际的性能。设计一种贝叶斯优化的神经架构变形方法有三个主要挑战。我们在后续的小节中介绍三个关键的部件，处理这三个挑战。

### 3.1 Edit-Distance Neural Network Kernel for Gaussian Process 高斯过程中的编辑距离神经网络核

The first challenge we need to address is that the NAS space is not a Euclidean space, which does not satisfy the assumption of traditional Gaussian process (GP). Directly vectorizing the neural architecture is impractical due to the uncertain number of layers and parameters it may contain. Since the Gaussian process is a kernel method, instead of vectorizing a neural architecture, we propose to tackle the challenge by designing a neural network kernel function. The intuition behind the kernel function is the edit-distance for morphing one neural architecture to another. More edits needed from one architecture to another means the further distance between them, thus less similar they are. The proof of validity of the kernel function is presented in Appendix E.

我们需要处理的第一个挑战是，NAS空间不是一个欧几里得空间，这不符合传统高斯过程(GP)的假设。将神经架构直接向量化是不实际的，因为可能包含的层数和参数数量不确定。由于高斯过程是一个核方法，我们不去对神经架构向量化，而是通过设计一个神经网络核函数来处理这个问题。核函数背后的直觉是，将一个神经架构变形到另外一个的编辑距离。从一种架构到另一种架构需要更多的编辑，这意味着之间的距离越远，所以相似度更低。核函数的有效性的证明在附录E中给出。

**Kernel Definition**: Suppose $f_a$ and $f_b$ are two neural networks. Inspired by Deep Graph Kernels [38], we propose an edit-distance kernel for neural networks. Edit-distance here means how many operations are needed to morph one neural network to another. The concrete kernel function is defined as:

**核的定义**：假设$f_a$和$f_b$是两个神经网络。受Deep Graph Kernels[38]启发，我们提出了一种神经网络的编辑距离核。编辑距离这里的意思是从一个神经网络变形到另外一个需要多少操作。具体的核函数定义为：

$$κ(f_a, f_b) = e^{−ρ^2 (d(f_a, f_b))}$$(3)

where function d(·, ·) denotes the edit-distance of two neural networks, whose range is [0, +∞), ρ is a mapping function, which maps the distance in the original metric space to the corresponding distance in the new space. The new space is constructed by embedding the original metric space into a new one using Bourgain Theorem [3], which ensures the validity of the kernel.

其中函数d(·, ·)表示两个神经网络的编辑距离，范围是[0, +∞)，ρ是一个映射函数，将原度量空间中的距离映射到新空间中对应的距离。新空间是用Bourgain定理[3]将原度量空间嵌入到新空间中构建的，这确保了核的有效性。

Calculating the edit-distance of two neural networks can be mapped to calculating the edit-distance of two graphs, which is an NP-hard problem [39]. Based on the search space F defined in Section 2, we tackle the problem by proposing an approximated solution as follows:

计算两个神经网络的编辑距离可以映射为计算两个图的编辑距离，这是一个NP难题[39]。基于第2部分定义的搜索空间F，我们提出一种近似的解来解决这个问题，如下所示：

$$d(f_a, f_b) = D_l (L_a, L_b ) + λD_s (S_a, S_b)$$(4)

where $D_l$ denotes the edit-distance for morphing the layers, i.e., the minimum edits needed to morph $f_a$ to $f_b$ if the skip-connections are ignored, $L_a = \{l_a^{(1)}, l_a^{(2)}, . . .\}$ and $L_b = \{l_b^{(1)}, l_b^{(2)}, . . .\}$ are the layer sets of neural networks $f_a$ and $f_b$, $D_s$ is the approximated edit-distance for morphing skip-connections between two neural networks, $S_a = \{s_a^{(1)}, s_a^{(2)}, . . .\}$ and $S_b = \{s_b^{(1)}, s_b^{(2)}, . . .\}$ are the skip-connection sets of neural network $f_a$ and $f_b$, and λ is the balancing factor between the distance of the layers and the skip-connections.

其中$D_l$表示层的变形的编辑距离，即，将$f_a$变形到$f_b$所需的最少编辑，忽略掉跳跃连接，$L_a = \{l_a^{(1)}, l_a^{(2)}, . . .\}$和$L_b = \{l_b^{(1)}, l_b^{(2)}, . . .\}$是神经网络$f_a$和$f_b$的层的集合，$D_s$是将两个神经网络之间跳跃连接变形的近似编辑距离，$S_a = \{s_a^{(1)}, s_a^{(2)}, . . .\}$和$S_b = \{s_b^{(1)}, s_b^{(2)}, . . .\}$是神经网络$f_a$和$f_b$的跳跃连接集合，λ是层的距离和跳跃连接距离的平衡因子。

**Calculating $D_l$**: We assume $|L_a| < |L_b|$, the edit-distance for morphing the layers of two neural architectures $f_a$ and $f_b$ is calculated by minimizing the follow equation: **计算$D_l$**： 我们假设$|L_a| < |L_b|$，从神经架构$f_a$通过层变形到$f_b$的编辑距离的计算，可以通过最小化下式得到：

$$D_l(L_a, L_b) = min \sum_{i=1}^{|L_a|} d_l(l_a^{(i)}, φ_l(l_a^{(i)})) + ||L_b| - |L_a||$$(5)

where $φ_l : L_a → L_b$ is an injective matching function of layers satisfying: $∀i < j, φ_l (l_a^{(i)} ) ≺ φ_l (l_a^{(j)} )$ if layers in $L_a$ and $L_b$ are all sorted in topological order. $d_l (·, ·)$ denotes the edit-distance of widening a layer into another defined in Equation (6):

其中$φ_l : L_a → L_b$是一个层的单射匹配函数，满足：$∀i < j, φ_l (l_a^{(i)} ) ≺ φ_l (l_a^{(j)} )$，如果层$L_a$和$L_b$都按照拓扑顺序排列好。$d_l (·, ·)$表示将一个层加宽成另一个层的边界距离，如式(6)所示：

$$d_l (l_a, l_b) = \frac {|w(l_a)-w(l_b)|}{max[w(l_a),w(l_b)]}$$(6)

where w(l) is the width of layer l. 其中w(l)是层l的宽度。

The intuition of Equation (5) is consistent with the idea of network morphism shown in Figure 1. Suppose a matching is provided between the nodes in two neural networks. The sizes of the tensors are indicators of the width of the previous layers (e.g., the output vector length of a fully-connected layer or the number of filters of a convolutional layer). The matchings between the nodes are marked by light blue. So a matching between the nodes can be seen as matching between the layers. To morph $f_a$ to $f_b$ with the given matching, we need to first widen the three nodes in $f_a$ to the same width as their matched nodes in $f_b$, and then insert a new node of width 20 after the first node in $f_a$. Based on this morphing scheme, the edit-distance of the layers is defined as $D_l$ in Equation (5).

式(5)中的直觉与图1所示的网络变形是一致的。假设已经有了两个神经网络间的节点匹配关系。张量的大小指示的是上一层的宽度（如，一个全连接层的输出矢量长度，或一个卷积层的滤波器的数量）。节点间的匹配标记为浅蓝色。所以节点间的匹配可以视为层间的匹配。给定匹配后，为$f_a$变形到到$f_b$，我们首先需要将$f_a$中的三个节点加宽，与$f_b$中匹配的节点宽度相同，然后在$f_a$中的第一个节点后插入一个宽度20的新节点。基于这种变形方案，层间的编辑距离定义为式(5)中的$D_l$项。

Figure 1: Neural Network Kernel. Given two neural networks $f_a, f_b$, and matchings between the similar layers, the figure shows how the layers of $f_a$ can be changed to the same as $f_b$. Similarly, the skip-connections in $f_a$ also need to be changed to the same as $f_b$ according to a given matching.

图1：神经网络核。给定两个神经网络$f_a, f_b$，以及类似层之间的匹配，上图给出了$f_a$中的层怎样变得与$f_b$一样。类似的，$f_a$中的跳跃连接也需要根据给定的匹配变得与$f_b$中一样。

Since there are many ways to morph $f_a$ to $f_b$, to find the best matching between the nodes that minimizes $D_l$, we propose a dynamic programming approach by defining a matrix $A_{|L_a|×|L_b|}$, which is recursively calculated as follows: 从$f_a$变形到$f_b$有很多种方法，为找到使$D_l$最小化的节点间最佳匹配，我们提出了一种动态规划方法，定义了一个矩阵$A_{|L_a|×|L_b|}$，迭代计算如下：

$$A_{i, j} = max[A_{i−1, j} + 1, A_{i, j−1} + 1, A_{i−1, j−1} + d_l (l_a, l_b)]$$(7)

where $A_{i, j}$ is the minimum value of $D_l (L_a^{(i)}, L_b^{(j)})$, where $L_a = \{l_a^{(1)}, l_a^{(2)}, . . . , l_a^{(i)}\}$ and $L_b = \{l_b^{(1)}, l_b^{(2)}, . . . , l_b^{(j)} \}$.

**Calculating $D_s$**: The intuition of $D_s$ is the sum of the the edit-distances of the matched skip-connections in two neural networks into pairs. As shown in Figure 1, the skip-connections with the same color are matched pairs. Similar to $D_l (·, ·)$, $D_s (·, ·)$ is defined as follows: **计算$D_s$**：直觉上来说，$D_s$是两个神经网络中匹配成对的跳跃连接的编辑距离之和。如图1所示，相同颜色的跳跃连接是匹配好的对。与$D_l (·, ·)$类似，$D_s (·, ·)$定义如下：

$$D_s(S_a,S_b)=min \sum_{i=1}^{|S_a|} d_s(s_a^{(i)}, φ_s(s_a^{(i)}) + ||S_a|-|S_b||$$(8)

where we assume $|S_a| < |S_b|$. $(|S_b| − |S_a|)$ measures the total edit-distance for non-matched skip-connections since each of the non-matched skip-connections in $S_b$ calls for an edit of inserting a new skip connection into $f_a$. The mapping function $φ_s : S_a → S_b$ is an injective function. $d_s (·, ·)$ is the edit-distance for two matched skip-connections defined as:

这里我们假设$|S_a| < |S_b|$。$(|S_b| − |S_a|)$度量的是未匹配的跳跃连接的总编辑距离，因为$S_b$中每个未匹配的跳跃连接都需要一次编辑，即往$f_a$中插入一个新的跳跃连接。映射函数$φ_s : S_a → S_b$是一个单射函数，$d_s (·, ·)$是两个匹配的跳跃连接间的编辑距离，定义如下：

$$d_s(s_a,s_b) = \frac {|u(s_a)-u(s_b)| + |δ(s_a)-δ(s_b)|} {max[u(s_a), u(s_b)] + max[δ(s_a),δ(s_b)]}$$(9)

where u(s) is the topological rank of the layer the skip-connection s started from, δ(s) is the number of layers between the start and end point of the skip-connection s. 其中u(s)是跳跃连接s开始的层的拓扑层级，δ(s)是跳跃连接s开始和结束点间的层数。

This minimization problem in Equation (8) can be mapped to a bipartite graph matching problem, where $f_a$ and $f_b$ are the two disjoint sets of the graph, each skip-connection is a node in its corresponding set. The edit-distance between two skip-connections is the weight of the edge between them. The weighted bipartite graph matching problem is solved by the Hungarian algorithm (Kuhn-Munkres algorithm) [19].

式(8)的最小化问题可以映射为一个二分图匹配问题，其中$f_a$和$f_b$是两个不相交的图集，每个跳跃连接是其对应集合中的一个节点。两个跳跃连接之间的编辑距离就是其之间的边的加权。加权二分图匹配问题可以用Hugarian算法求解(Kuhn-Munkres算法)。

### 3.2 Optimization for Tree Structured Space 树状结构空间中的优化

The second challenge of using Bayesian optimization to guide network morphism is the optimization of the acquisition function. The traditional acquisition functions are defined on Euclidean space. The optimization methods are not applicable to the tree-structured search via network morphism. To optimize our acquisition function, we need a method to efficiently optimize the acquisition function in the tree-structured space. To deal with this problem, we propose a novel method to optimize the acquisition function on tree-structured space.

使用贝叶斯优化指引网络变形的第二个挑战是获得函数的优化。传统的获得函数是定义在欧几里得空间的，其优化方法不适用于网络变形中的树状搜索空间。为优化我们的获得函数，我们需要一种能高效的在树状空间中优化获得函数的方法。为处理这个问题，我们提出了一种在树状空间中优化获得函数的新方法。

Upper-confidence bound (UCB) [1] is selected as our acquisition function, which is defined as: 我们选取置信度上限(Upper-confidence bound, UCB)作为我们的获得函数，其定义为

$$α(f) = μ(y_f) − βσ(y_f)$$(10)

where $y_f = Cost(f, D)$, β is the balancing factor, $μ(y_f)$ and $σ(y_f)$ are the posterior mean and standard deviation of variable $y_f$. It has two important properties, which fit our problem. First, it has an explicit balance factor β for exploration and exploitation. Second, α(f) is directly comparable with the cost function value $c^{(i)}$ in search history $H = \{(f^{(i)} , θ^{(i)} , c^{(i)})\}$. It estimates the lowest possible cost given the neural network f. $\hat f = argmin_f α(f)$ is the generated neural architecture for next observation.

其中$y_f = Cost(f, D)$，β是平衡参数，$μ(y_f)$和$σ(y_f)$是变量$y_f$的后验均值和标准差。函数有两个重要的性质，适合我们的问题。第一，有显式的平衡参数β可以探索并利用。第二，α(f)与搜索历史$H = \{(f^{(i)} , θ^{(i)} , c^{(i)})\}$中的代价函数值$c^{(i)}$可以直接比较。函数在给定神经网络f的情况下估计最低的可能代价。下一步观察所用的神经架构由$\hat f = argmin_f α(f)$生成。

The tree-structured space is defined as follows. During the optimization of α(f), $\hat f$ should be obtained from f(i) and O, where f(i) is an observed architecture in the search history H, O is a sequence of operations to morph the architecture into a new one. Morph f to $\hat f$ with O is denoted as $\hat f ← M(f, O)$, where M(·, ·) is the function to morph f with the operations in O. Therefore, the search can be viewed as a tree-structured search, where each node is a neural architecture, whose children are morphed from it by network morphism operations.

树状结构可以定义如下。在优化α(f)的过程中，$\hat f$应当从f(i)和O得到，其中f(i)是在搜索历史H中的一个观察结构，O是一个操作序列，将神经架构变形到一个新的架构。将f用O变形到$\hat f$表示为$\hat f ← M(f, O)$，其中M(·, ·)是用O中的操作对f进行变形的函数。所以，搜索可以看作一个树状的搜索，其中每个节点是一个神经架构，其子节点就是网络通过变形操作可以成为的架构。

The most common defect of network morphism is it only grows the size of the architecture instead of shrinking them. Using network morphism for NAS may end up with a very large architecture without enough exploration on the smaller architectures. However, our tree-structure search, we not only expand the leaves but also the inner nodes, which means the smaller architectures found in the early stage can be selected multiple times to morph to more comparatively small architectures.

网络变形的最常见缺陷是，网络架构的规模只增加，而不缩减。使用网络变形进行NAS会导致结果成为很大的架构，而没有对更小的架构进行足够的探索。但是，我们的树状搜索不仅扩展叶节点，也包括内部节点，这意味着较早期发现的更小的架构可以多次被选中，变形成为相对较小的架构。

Inspired by various heuristic search algorithms for exploring the tree-structured search space and optimization methods balancing between exploration and exploitation, a new method based on A* search and simulated annealing is proposed. A* algorithm is widely used for tree-structure search. It maintains a priority queue of nodes and keeps expanding the best node in the queue. Since A* always exploits the best node, simulated annealing is introduced to balance the exploration and exploitation by not selecting the estimated best architecture with a probability.

很多启发式搜索方法探索树状搜索空间，很多优化方法在探索和利用上进行了均衡，受其启发，我们提出了一种基于A*搜索和模拟退火的新方法。A*搜索广泛的用于树状空间搜索，维护了节点的优先队列，不断的拓展队列中的最佳节点。由于A*一直利用最佳节点，我们引入模拟退火算法以在探索和利用上进行均衡，即以一定概率不选择估计的最优架构。

As shown in Algorithm 1, the algorithm takes minimum temperature $T_{low}$, temperature decreasing rate r for simulated annealing, and search history H described in Section 2 as the input. It outputs a neural architecture f ∈ H and a sequence of operations O to morph f into the new architecture. From line 2 to 6, the searched architectures are pushed into the priority queue, which sorts the elements according to the cost function value or the acquisition function value. Since UCB is chosen as the acquisiton function, α(f) is directly comparable with the history observation values $c^{(i)}$. From line 7 to 18, it is the loop optimizing the acquisition function. Following the setting in A* search, in each iteration, the architecture with the lowest acquisition function value is popped out to be expanded on line 8 to 10, where Ω(f) is all the possible operations to morph the architecture f, M(f, o) is the function to morph the architecture f with the operation sequence o. However, not all the children are pushed into the priority queue for exploration purpose. The decision of whether it is pushed into the queue is made by simulated annealing on line 11, where $e^{\frac {c_{min} - α(f')} {T}}$ is a typical acceptance function in simulated annealing. $c_{min}$ and $f_{min}$ are updated from line 14 to 16, which record the minimum acquisition function value and the corresponding architecture.

如算法1所示，算法的输入为，最低温度$T_{low}$，模拟退火用的温度下降速度r，第2部分所述的搜索历史H；输出为一个神经架构f∈H，和将f变形到新的架构的操作序列O。从第2行到第6行，搜索到的架构压入优先队列，这个队列根据代价函数值或获得函数值对元素进行排序。由于采用了UCB作为获得函数，α(f)直接与历史观测值$c^{(i)}$进行比较。从第7行到18行，是优化获得函数的循环。遵循A*搜索的设置，在每次迭代中，最低获得函数值的架构被弹出并拓展，这是8到10行，其中Ω(f)是架构f所有可能的变形操作，M(f, o)是用操作序列o对架构f进行变形的函数。但是，并不是所有的子节点都压入优先队列进行探索。决定是否将其压入队列的决定是第11行的模拟退火，其中$e^{\frac {c_{min} - α(f')} {T}}$是模拟退火的典型的接受函数。$c_{min}$和$f_{min}$在14行到16行得到更新，记录了获得函数的最低值和对应的架构。

### 3.3 Graph-Level Network Morphism 图级网络变形

The third challenge is to maintain the intermediate output tensor shape consistency when morphing the architectures. Previous work showed how to preserve the functionality of the layers the operators applied on, namely layer-level morphism. However, from a graph-level view, any change of a single layer could have a butterfly effect on the entire network. Otherwise, it would break the input and output tensor shape consistency. To tackle the challenge, a graph-level morphism is proposed to find and morph the layers influenced by a layer-level operation in the entire network.

第三个挑战是在架构变形的同时，保持中间的输出张量形状的连续性。之前的工作都是怎样保持操作所应用的层的功能，即层级别的变形。但是，从图级的视角来看，任何一层的变化都可能对整个网络有蝴蝶效应。否则，就会破坏输入输出张量的形状连续性。为应对这个挑战，提出了一种图级的变形，来找到并变形相应的层，该层是整个网络中层级别的操作影响到的。

Follow the four network morphism operations on a neural network f ∈ F defined in [11], which can all be reflected in the change of the computational graph G. The first operation is inserting a layer to f to make it deeper denoted as deep(G, u), where u is the node marking the place to insert the layer. The second one is widening a node in f denoted as wide(G, u), where u is the node representing the intermediate output tensor to be widened. Widen here could be either making the output vector of the previous fully-connected layer of u longer, or adding more filters to the previous convolutional layer of u, depending on the type of the previous layer. The third is adding an additive connection from node u to node v denoted as add(G, u, v). The fourth is adding an concatenative connection from node u to node v denoted as concat(G, u, v). For deep(G, u), no other operation is needed except for initializing the weights of the newly added layer. However, for all other three operations, more changes are required to G.

[11]中定义了神经网络f∈F的4种网络变形的操作，都可以反映到计算图G的变化上，我们也采用这种方法。第一种操作是向f中插入一层，使其变得更深，记作deep(G,u)，其中u是标记插入层的节点。第二个是使f中的一个节点变宽，记作wide(G,u)，其中u代表了要加宽的中间输出张量。这里的加宽可以是使前面的全连接层的输出向量u更长，或者前面的计算层u增加更多的滤波器，取决于前面这一层的类型。第三个是从节点u到节点v增加一个加性连接，记作add(G,u,v)。第四个从节点u到节点v增加了一个拼接连接，表示为concat(G,u,v)。对于deep(G,u)，需要初始化新增加的层的权重，别的操作就不需要了。但是，对于其他三种操作，需要对G进行更多改变。

First, we define an effective area of wide($G, u_0$) as γ to better describe where to change in the network. The effective area is a set of nodes in the computational graph, which can be recursively defined by the following rules: 1. $u_0 ∈ γ$. 2. v ∈ γ , if $∃e_{u→v} !\in L_s$, u ∈ γ. 3. v ∈ γ , if $∃e_{v→u} !\in L_s$, u ∈ γ. $L_s$ is the set of fully-connected layers and convolutional layers. Operation wide($G, u_0$) needs to change two set of layers, the previous layer set $L_p = \{e_{u→v} ∈ L_s |v ∈ γ \}$, which needs to output a wider tensor, and next layer set $L_n = \{e_{u→v} ∈ L_s |u ∈ γ \}$, which needs to input a wider tensor. Second, for operator add(G, $u_0, v_0$), additional pooling layers may be needed on the skip-connection. $u_0$ and $v_0$ have the same number of channels, but their shape may differ because of the pooling layers between them. So we need a set of pooling layers whose effect is the same as the combination of all the pooling layers between $u_0$ and $v_0$, which is defined as $L_o = \{e ∈ L_{pool} |e ∈ p_{u_0 →v_0} \}$, where $p_{u_0 →v_0}$ could be any path between $u_0$ and $v_0$, $L_{pool}$ is the pooling layer set. Another layer $L_c$ is used after to pooling layers to process $u_0$ to the same width as $v_0$. Third, in concat(G, $u_0, v_0$), the concatenated tensor is wider than the original tensor $v_0$. The concatenated tensor is input to a new layer $L_c$ to reduce the width back to the same width as $v_0$. Additional pooling layers are also needed for the concatenative connection.

首先，我们定义wide($G,u_0$)的一个有效区域，记为γ，以更好的描述要改变网络中的哪个地方。这个有效区域是计算图中的节点集合，通过如下规则递归定义：1.$u_0 ∈ γ$. 2. v ∈ γ , if $∃e_{u→v} !\in L_s$, u ∈ γ. 3. v ∈ γ , if $∃e_{v→u} !\in L_s$, u ∈ γ. $L_s$是全连接层和卷积层的集合。操作wide($G, u_0$)需要改变两种层的集合，前一层的集合$L_p = \{e_{u→v} ∈ L_s |v ∈ γ \}$，需要输出一个更宽的张量，和下一层的集合，$L_n = \{e_{u→v} ∈ L_s |u ∈ γ \}$，需要输入一个更宽的张量。第二，对于算子add(G, $u_0, v_0$)，这个跳跃连接可能需要额外的pooling层。$u_0$和$v_0$有着相同的通道数，但其形状可能不同，因为它们中间的pooling层。所以我们需要一个pooling层的集合，其效果与$u_0$和$v_0$中间所有pooling层的组合相同，定义为$L_o = \{e ∈ L_{pool} |e ∈ p_{u_0 →v_0} \}$，其中$p_{u_0 →v_0}$可以是$u_0$和$v_0$间的任何路径，$L_{pool}$是pooling层的集合。在这些pooling层之后，还需要另一层$L_c$来处理$u_0$，使其宽度与$v_0$相同。第三，拼接操作concat(G, $u_0, v_0$)，拼接的张量比原来的张量$v_0$要宽。拼接的张量输入到一个新的层$L_c$中，以将宽度降低到$v_0$的宽度。拼接连接也需要另外的pooling层。

### 3.4 Time Complexity Analysis 时间复杂度分析

As described at the start of Section 3, Bayesian optimization can be roughly divided into three steps: update, generation, and observation. The bottleneck of the algorithm efficiency is observation, which involves the training of the generated neural architecture. Let n be the number of architectures in the search history. The time complexity of the update is $O(n^2 log_2 n)$. In each generation, the kernel is computed between the new architectures during optimizing acquisition function and the ones in the search history, the number of values in which is O(nm), where m is the number of architectures computed during the optimization of the acquisition function. The time complexity for computing d(·, ·) once is $O(l^2 + s^3)$, where l and s are the number of layers and skip-connections. So the overall time complexity is $O(nm(l^2 + s^3) + n^2 log_2 n)$. The magnitude of these factors is within the scope of tens. So the time consumption of update and generation is trivial comparing to the observation.

如同第3部分开始所述，贝叶斯优化可以大致分为三个步骤：更新，生成，和观察。算法效率的瓶颈是观察，这涉及到生成的神经架构的训练。令n为搜索历史中的架构数量。更新的时间复杂度为$O(n^2 log_2 n)$。在每次生成中，核心的计算，是在优化获得函数时的新架构，和搜索历史中的那些架构，其中值的数量是O(nm)，m为优化获得函数时计算的架构的数量。计算d(·, ·)的时间复杂度是$O(l^2 + s^3)$，其中l和s是层及跳跃连接的数量。所以总计的时间复杂度为$O(nm(l^2 + s^3) + n^2 log_2 n)$。这些因素的数量级为几十以内，所以更新和生成所耗费的时间与观察的时间相比是不重要的。

## 4. AUTO-KERAS

Based on the proposed neural architecture search method, we developed an open-source AutoML system, namely Auto-Keras. It is named after Keras [9], which is known for its simplicity in creating neural networks. Similar to SMAC [15], TPOT [28], Auto-WEKA [35], and Auto-Sklearn [13], the goal is to enable domain experts who are not familiar with machine learning technologies to use machine learning techniques easily. However, Auto-Keras is focusing on the deep learning tasks, which is different from the systems focusing on the shallow models mentioned above.

基于提出的神经架构搜索方法，我们开发了一个开源的自动机器学习系统，即Auto-Keras。以Keras[9]命名，Keras以创建神经网络的简洁性著名。与SMAC[15], TPOT[28], Auto-WEKA[35]和Auto-Sklearn[13]类似，其目标是使对机器学习技术不熟悉的领域专家也可以使用机器学习技术。但是，Auto-Keras聚焦于深度学习任务，与前面这些聚焦于浅层模型的系统不同。

Although, there are several AutoML services available on large cloud computing platforms, three things are prohibiting the users from using them. First, the cloud services are not free to use, which may not be affordable for everyone who wants to use AutoML techniques. Second, the cloud-based AutoML usually requires complicated configurations of Docker containers and Kubernetes, which is not easy for people without a rich computer science background. Third, the AutoML service providers are honest-but-curious [7], which cannot guarantee the security and privacy of the data. An open-source software, which is easily downloadable and runs locally, would solve these problems and make the AutoML accessible to everyone. To bridge the gap, we developed Auto-Keras.

虽然现在在大型云计算平台有一些AutoML服务可用，但有三件事阻碍了用户的使用。首先，云服务的使用不是免费的，想要使用AutoML技术的人，并不是人人都负担的起。第二，基于云的AutoML一般需要Docker容器和Kubernetes的复杂配置，这对于计算机科学背景不丰富的人来说并不容易。第三，AutoML服务供应者是诚实的，但是也是好奇的[7]，这不能保证数据的安全性和隐私。一个可以很容易下载并运行的开源软件，可以解决这些问题，使AutoML可以对所有人都可用。为填补这样的空白，我们开发了Auto-Keras。

It is challenging, to design an easy-to-use and locally deployable system. First, we need a concise and configurable application programming interface (API). For the users who don’t have rich experience in programming, they could easily learn how to use the API. For the advanced users, they can still configure the details of the system to meet their requirements. Second, the local computation resources may be limited. We need to make full use of the local computation resources to speed up the search. Third, the available GPU memory may be of different sizes in different environments. We need to adapt the neural architecture sizes to the GPU memory during the search.

设计一个容易使用并可本地部署的系统是很有挑战性的。首先，我们需要简洁可配置的应用程序接口(API)。对于没有太多编程经验的用户来说，也可以很容易的学习怎样使用这些API。对于高级用户，他们还可以很详细的配置系统，以满足其需求。第二，本地的计算资源可能很有限。我们希望充分使用本地的计算资源，以加速搜索。第三，可用的GPU内存在不同的环境中可能不同。我们需要在搜索的过程中根据GPU内存量调整神经架构的大小。

### 4.1 System Overview 系统概览

The system architecture of Auto-Keras is shown in Figure 2. We design this architecture to fully make use of the computational resource of both CPU and GPU, and utilize the memory efficiently by only placing the currently useful information on the RAM, and save the rest on the storage devices, e.g., hard drives. The top part is the API, which is directly called by the users. It is responsible for calling corresponding middle-level modules to complete certain functionalities. The Searcher is the module of the neural architecture search algorithm containing Bayesian Optimizer and Gaussian Process. These search algorithms run on CPU. The Model Trainer is a module responsible for the computation on GPUs. It trains given neural networks with the training data in a separate process for parallelism. The Graph is the module processing the computational graphs of neural networks, which is controlled by the Searcher for the network morphism operations. The current neural architecture in the Graph is placed on RAM for faster access. The Model Storage is a pool of trained models. Since the size of the neural networks are large and cannot be stored all in memory, the model storage saves all the trained models on the storage devices.

Auto-Keras的系统架构如图2所示。我们设计的这种架构可以充分使用CPU和GPU的计算资源，可以有效的利用内存资源，因为只将目前有用的信息放在RAM里，将其他的保存在存储设备上，如硬盘。最上面的部分是API，由用户直接调用。API部分负责调用对应的中间层模块，以完成某些功能。Searcher是神经架构搜索算法模块，包括贝叶斯优化和高斯过程。这些搜索算法运行在CPU上。Model Trainer是负责GPU运算的模块，它用训练数据训练给定的神经网络，是一个单独的进程，可以并行化。Graph是模型处理神经网络计算图的部分，由于Searcher控制进行网络变形操作。Graph中当前的神经架构是放在RAM中的，可以更快的获取使用。Model Storage是训练好的模型的集合。由于神经网络的规模很大，不能全部存储在内存中，因此Model Storage将所有训练好的模型保存在存储设备中。

A typical workflow for the Auto-Keras system is as follows. The user initiated a search for the best neural architecture for the dataset. The API received the call, preprocess the dataset, and pass it to the Searcher to start the search. The Bayesian Optimizer in the Searcher would generate a new architecture using CPU. It calls the Graph module to build the generated neural architecture into a real neural network in the RAM. The new neural architecture is copied the GPU for the Model Trainer to train with the dataset. The trained model is saved in the Model Storage. The performance of the model is feedback to the Searcher to update the Gaussian Process.

Auto-Keras的典型工作流是这样的。用户初始化一个搜索，搜索对于某数据集最好的神经架构。API收到调用，对数据集进行预处理，将其传入Searcher，开始搜索。Searcher中的贝叶斯优化器使用CPU生成了一个新的架构，然后调用Graph模块构建生成的神经架构，成为一个真正的RAM中的神经网络。新的神经架构复制到GPU中，由Model Trainer训练这个数据集。训练好的模型保存到Model Storage中。模型的性能反馈到Searcher中，以更新高斯过程。

Figure 2: Auto-Keras System Overview. (1) User calls the API. (2) The Searcher generates neural architectures on CPU. (3) Graph builds real neural networks with parameters on RAM from the neural architectures. (4) The neural network is copied to GPU for training. (5) Trained neural networks are saved on storage devices.

### 4.2 Application Programming Interface

The design of the API follows the classic design of the Scikit-Learn API [5, 29], which is concise and configurable. The training of a neural network requires as few as three lines of code calling the constructor, the fit and predict function respectively. To accommodate the needs of different users, we designed two levels of APIs. The first level is named as task-level. The users only need to know their task, e.g., Image Classification, Text Regression, to use the API. The second level is named search-level, which is for advanced users. The user can search for a specific type of neural network architectures, e.g., multi-layer perceptron, convolutional neural network. To use this API, they need to preprocess the dataset by themselves and know which type of neural network, e.g., CNN or MLP, is the best for their task.

API的设计遵循的是Scikit-Learn API的经典设计[5,29]，简洁可配置。神经网络的训练需要至少3行代码，分别调用constructor函数、fit函数和predict函数。为适应不同用户的需求，我们设计了两个层次的API。第一个层次命名为任务级的。用户只需要知道其任务，如图像分类，文本回归，以使用API。第二个层次命名为搜索级的，这是为高级用户设计的。用户可以搜索特定类型的神经网络架构，如，MLP，CNN等。为使用这个API，他们需要自己预处理数据集，知道哪种类型的神经网络最适合其任务，如CNN或MLP。

Several accommodations have been implemented to enhance the user experience with the Auto-Keras package. First, the user can restore and continue a previous search which might be accidentally killed. From the users’ perspective, the main difference of using Auto-Keras comparing with the AutoML systems aiming at shallow models is the much longer time consumption, since a number of deep neural networks are trained during the neural architecture search. It is possible for some accident to happen to kill the process before the search finishes. Therefore, the search outputs all the searched neural network architectures with their trained parameters into a specific directory on the disk. As long as the path to the directory is provided, the previous search can be restored. Second, the user can export the search results, which are neural architectures, as saved Keras models for other usages. Third, for advanced users, they can specify all kinds of hyperparameters of the search process and neural network optimization process by the default parameters in the interface.

已经实现了一些便利性，以增强Auto-Keras包的用户体验。首先，用户可以恢复并继续之前意外中断的一个搜索。从用户的角度来说，使用Auto-Keras，与使用其他浅层模型的AutoML系统相比，主要的区别是耗费的时间更长，因为在搜索神经架构的过程中要训练一些深度神经网络。在搜索结束之前，可能有一些因素导致过程意外中止。所以，搜索过程会将搜索到的神经网络架构，及其训练好的参数，保存到硬盘上的指定文件夹中。只要提供这个文件夹的路径，就可以恢复之前的搜索。第二，用户可以导出搜索结果，即神经架构，导出成Keras模型作他用。第三，对于高级用户，他们可以使用接口中的默认参数，来指定搜索过程和神经网络优化过程的各种超参数。

### 4.3 CPU and GPU Parallelism

To make full use of the limited local computation resources, the program can run in parallel on the GPU and the CPU at the same time. If we do the observation (training of the current neural network), update, and generation of Bayesian optimization in sequential order. The GPUs will be idle during the update and generation. The CPUs will be idle during the observation. To improve the efficiency, the observation is run in parallel with the generation in separated processes. A training queue is maintained as a buffer for the Model Trainer. Figure 3 shows the Sequence diagram of the parallelism between the CPU and the GPU. First, the Searcher requests the queue to pop out a new graph and pass it to GPU to start training. Second, while the GPU is busy, the searcher requests the CPU to generate a new graph. At this time period, the GPU and the CPU work in parallel. Third, the CPU returns the generated graph to the searcher, who pushes the graph into the queue. Finally, the Model Trainer finished training the graph on the GPU and returns it to the Searcher to update the Gaussian process. In this way, the idle time of GPU and CPU are dramatically reduced to improve the efficiency of the search process.

为使用有限的本地计算资源，程序可以在GPU和CPU上同时并行运行。如果我们将贝叶斯优化以顺序进行，即观测（训练目前的神经网络），更新和生成，在更新和生成的时候GPU将会闲置，在观测的时候CPU将会闲置。为改进效率，观测与生成在分离的过程中并行运行。维护一个训练队列，作为Model Trainer的缓冲区。图3展示了CPU和GPU并行化的序列图。首先，Searcher请求队列弹出一个新的图，将其传给GPU以开始训练。第二，在GPU忙的时候，Searcher要求CPU生成一个新图。在这个时间段里，GPU和CPU并行工作。第三，CPU将生成的图返回Searcher，将这个图压入队列。最后，Model Trainer在GPU上完成图的训练，将其返回给Searcher，以更新高斯过程。以这种过程运行，CPU和GPU的空闲时间会急剧减少，提高了搜索过程的效率。

### 4.4 GPU Memory Adaption

Since different deployment environments have different limitations on the GPU memory usage, the size of the neural networks needs to be limited according to the GPU memory. Otherwise, the system would crash because of running out of GPU memory. To tackle this challenge, we implement a memory estimation function on our own data structure for the neural architectures. An integer value is used to mark the upper bound of the neural architecture size. Any new computational graph whose estimated size exceeds the upper bound is discarded. However, the system may still crash because the management of the GPU memory is very complicated, which cannot be precisely estimated. So whenever it runs out of GPU memory, the upper bound is lowered down to further limit the size of the generated neural networks.

由于不同的部署环境在GPU内存使用上有着不同的限制，神经网络的规模要根据可用的GPU内存进行一定的限制。否则，系统会因为用完GPU内存而崩溃。为处理这个问题，我们在我们的神经架构数据结构上，实现了一个内存估算函数。使用了一个整数值来标记神经架构规模的上限。新计算图的估计规模，如果超过了上限，就会被抛弃。但是，由于GPU内存的管理非常复杂，系统仍然可能崩溃，这不能准确估计。所以不管什么时候GPU内存使用光了，上限都会被进一步降低，以限制生成的神经网络的规模。

## 5 Experiments 试验

In the experiments, we aim at answering the following questions. 1) How effective is the search algorithm with limited running time? 2) How much efficiency is gained from Bayesian optimization and network morphism? 3) What are the influences of the important hyperparameters of the search algorithm? 4) Does the proposed kernel function correctly measure the similarity among neural networks in terms of their actual performance?

我们试验的目标是回答以下几个问题。1)搜索算法在有限的运行时间中的效率；2)从贝叶斯优化和网络变形中得到的效率；3)搜索算法中重要的超参数的影响；4)提出的核函数，有没有在其实际性能的意义上，正确的衡量神经网络之间的相似性。

**Datasets**. Three benchmark datasets, MNIST [20], CIFAR10 [18], and FASHION [37] are used in the experiments to evaluate our method. They prefer very different neural architectures to achieve good performance. 数据集。我们的试验使用了三个基准测试数据集，MNIST [20], CIFAR10 [18]和FASHION [37]，它们各自倾向于在不同的神经架构上得到很好的性能。

**Baselines**. Four categories of baseline methods are used for comparison, which are elaborated as follows: 基准。使用了四种基准方法进行比较，列举如下：

- Straightforward Methods: random search (RAND) and grid search (GRID). They search the number of convolutional layers and the width of those layers. 直接法：随即搜索和网格搜索。它们搜索卷积层的数量和这些层的宽度。

- Conventional Methods: SPMT [33] and SMAC [15]. Both SPMT and SMAC are designed for general hyperparameters tuning tasks of machine learning models instead of focusing on the deep neural networks. They tune the 16 hyperparameters of a three-layer convolutional neural network, including the width, dropout rate, and regularization rate of each layer. 传统方法：SPMT[33]和SMAC[15]。SPMT和SMAC都是设计用于机器学习模型一般性的超参数调节任务，而不是专注于深度神经网络。它们调节三层卷积神经网络的16个超参数，包括每层的宽度、dropout率和正则化率。

- State-of-the-art Methods: SEAS [11], NASBOT [16]. We carefully implemented the SEAS as described in their paper. For NASBOT, since the experimental settings are very similar, we directly trained their searched neural architecture in the paper. They did not search architectures for MNIST and FASHION dataset, so the results are omitted in our experiments. 目前最好的方法：SEAS[11], NASBOT[16]。我们仔细的实现了文章中描述的SEAS。对于NASBOT，由于试验设置非常类似，我们直接训练他们文章中搜索得到的神经架构。他们没有针对MNIST和FASHION数据集搜索神经架构，所以在试验中忽略了这个结果。

- Variants of the proposed method: BFS and BO. Our proposed method is denoted as AK. BFS replaces the Bayesian optimization in AK with the breadth-first search. BO is another variant, which does not employ network morphism to speed up the training. For AK, β is set to 2.5, while λ is set to 1 according to the parameter sensitivity analysis. 提出的方法的变体：BFS和BO。我们提出的方法表示为AK。BFS将AK中的贝叶斯优化方法替换为广度优先搜索。BO是另一个变体，没有使用网络变形来对训练提速。对于AK，根据参数敏感性分析，β设为2.5, 而λ设为1 。

In addition, the performance of the deployed system of Auto-Keras (AK-DP) is also evaluated in the experiments. The difference from the AK above is that AK-DP uses various advanced techniques to improve the performance including learning rate scheduling, multiple manually defined initial architectures.

另外，Auto-Keras的部署系统(AK-DP)的性能也在试验中进行了评估。与上面的AK的区别在于，AK-DP使用了各种高级技术来改进性能，包括学习速率安排，多个手动定义的初始架构。

**Experimental Setting**. The general experimental setting for evaluation is described as follows: First, the original training data of each dataset is further divided into training and validation sets by 80-20. Second, the testing data of each dataset is used as the testing set. Third, the initial architecture for SEAS, BO, BFS, and AK is a three-layer convolutional neural network with 64 filters in each layer. Fourth, each method is run for 12 hours on a single GPU (NVIDIA GeForce GTX 1080 Ti) on the training and validation set with batch size of 64. Fifth, the output architecture is trained with both the training and validation set. Sixth, the testing set is used to evaluate the trained architecture. Error rate is selected as the evaluation metric since all the datasets are for classification. For a fair comparison, the same data processing and training procedures are used for all the methods. The neural networks are trained for 200 epochs in all the experiments. Notably, AK-DP uses a real deployed system setting, whose result is not directly comparable with the rest of the methods. Except for AK-DP, all other methods are fairly compared using the same initial architecture to start the search.

**试验设置**。用于评估的一般试验设置描述如下：首先，每个数据集的原始训练集进一步分割成训练集和验证集，比例为80-20。第二，每个数据集的测试数据都用作测试集。第三，SEAS、BO、BFS和AK的初始架构是三层卷积网络，每层64个滤波器。第四，每种方法都在单个GPU(NVidia GeForce GTX 1080 Ti)上运行12个小时，进行训练和验证，批规模为64。第五，输出架构用训练集和验证集一起训练。第六，测试集用于评估训练好的架构。我们选择用错误率作为评估度量标准，因为所有数据集都是用作分类的。为公平比较，所有方法都使用相同的数据处理和训练过程。在试验中，神经网络训练200轮。值得注意的是，AK-DP使用真实的部署系统设置，其结果不能与其他方法直接比较。除了AK-DP，所有其他方法都使用相同的架构开始搜索，是公平比较的。

### 5.1 Evaluation of Effectiveness 有效性评估

We first evaluate the effectiveness of the proposed method. The results are shown in Table 1. The following conclusions can be drawn based on the results. 我们首先评估提出的方法的有效性，结果如表1所示，从中可以得出如下结论：

Table 1: Classification Error Rate

Methods | MNIST(%) | CIFAR10(%) | FASHION(%)
--- | --- | --- | ---
RANDOM | 1.79 | 16.86 | 11.36
GRID | 1.68 | 17.17 | 10.28
SPMT | 1.36 | 14.68 | 9.62
SMAC | 1.43 | 15.04 | 10.87
SEAS | 1.07 | 12.43 | 8.05
NASBOT | - | 12.30 | -
BFS | 1.56 | 13.84 | 9.13
BO | 1.83 | 12.90 | 7.99
AK | 0.55 | 11.44 | 7.42
AK-DP | 0.60 | 3.60 | 6.72

(1) AK-DP is evaluated to show the final performance of our system, which shows deployed system (AK-DP) achieved state-of-the-art performance on all three datasets. AK-DP的评估结果是我们系统的最终结果，这表明部署系统(AK-DP)在所有三个数据集上都取得了目前最好的性能。

(2) The proposed method AK achieves the lowest error rate on all the three datasets, which demonstrates that AK is able to find simple but effective architectures on small datasets (MNIST) and can explore more complicated structures on larger datasets (CIFAR10). 我们提出的方法AK在所有三个数据集上都得到了最低的错误率，这表明AK可以在小型数据集(MNIST)上找到简单但有效的架构，在较大的数据集上(CIFAR10)也可以探索更复杂的结构。

(3) The straightforward approaches and traditional approaches perform well on the MNIST dataset, but poorly on the CIFAR10 dataset. This may come from the fact that: naive approaches like random search and grid search only try a limited number of architectures blindly while the two conventional approaches are unable to change the depth and skip-connections of the architectures. 直接方法和传统方法在MNIST数据集上表现还不错，但在CIFAR10上表现不好。这可能是由于以下事实：单纯的方法，像随机搜索和网格搜索只盲目尝试了有限数量的架构，而两种传统的方法不能改变网络架构的深度和跳跃连接。

(4) Though the two state-of-the-art approaches achieve acceptable performance, SEAS could not beat our proposed model due to its subpar search strategy. The hill-climbing strategy it adopts only takes one step at each time in morphing the current best architecture, and the search tree structure is constrained to be unidirectionally extending. Comparatively speaking, NASBOT possesses stronger search expandability and also uses Bayesian optimization as our proposed method. However, the low efficiency in training the neural architectures constrains its power in achieving comparable performance within a short time period. By contrast, the network morphism scheme along with the novel searching strategy ensures our model to achieve desirable performance with limited hardware resources and time budges.

虽然两种最好的方法取得了可以接受的结果，SEAS没有达到我们提出的方法的表现，是因为其搜索策略不够好。它采用的爬坡策略在对目前的最佳架构变形时，只前进一步，而搜索树结构被约束为只能单向拓展。比较来说，NASBOT的搜索拓展性更好，也使用了贝叶斯优化，像我们的方法一样。但是，训练神经架构的低效性限制了其能力，使其不能在短期内达到可比较的性能。比较起来，我们提出方法的网络变形方案，以及新的搜索策略，确保了能够在有限的硬件资源上和时间预算内得到理想的性能。

(5) For the two variants of AK, BFS preferentially considers searching a vast number of neighbors surrounding the initial architecture, which constrains its power in reaching the better architectures away from the initialization. By comparison, BO can jump far from the initial architecture. But without network morphism, it needs to train each neural architecture with much longer time, which limits the number of architectures it can search within a given time.

对于AK的两个变体，BFS优先考虑搜索其初始架构周边的大量架构，这约束了其达到远离初始架构的更好架构的能力。比较起来，BO可以从初始架构跳的更远。但没有网络变形，它需要花长的多的时间来训练每个神经架构，这限制了其在给定时间内可以搜索的架构的数量。

### 5.2 Evaluation of Efficiency 效率评估

In this experiment, we try to evaluate the efficiency gain of the proposed method in two aspects. First, we evaluate whether Bayesian optimization can really find better solutions with a limited number of observations. Second, we evaluated whether network morphism can enhance the training efficiency.

这个试验中，我们从两方面评估提出的方法的效率提升。首先，我们评估在有限数量的观察内，贝叶斯优化是否可以真的找到更好的解。第二，我们评估网络变形是否可以提高训练效率。

We compare the proposed method AK with its two variants, BFS and BO, to show the efficiency gains from Bayesian optimization and network morphism, respectively. BFS does not adopt Bayesian optimization but only network morphism, and use breadth-first search to select the network morphism operations. BO does not employ network morphism but only Bayesian optimization. Each of the three methods is run on CIFAR10 for twelve hours. The left part of Figure 4 shows the relation between the lowest error rate achieved and the number of neural networks searched. The right part of Figure 4 shows the relation between the lowest error rate achieved and the searching time.

我们将提出的方法AK，与其两种变体BFS和BO进行了比较，以分别表明贝叶斯优化和网络变形带来的效率提升。BFS没有采用贝叶斯优化，而只采用了网络变形，使用的是广度优先搜索来选择网络变形操作。BO没有使用网络变形，只采用了贝叶斯优化。这三种方法都在CIFAR10数据集上运行12个小时。图4的左边部分表明了，得到的最低错误率与搜索的神经网络数量的关系，右边部分表明了，最低错误率与搜索时间的关系。

Two conclusions can be drawn by comparing BFS and AK. First, Bayesian optimization can efficiently find better architectures with a limited number of observations. When searched the same number of neural architectures, AK could achieve a much lower error rate than BFS. It demonstrates that Bayesian optimization could effectively guide the search in the right direction, which is much more efficient in finding good architectures than the naive BFS approach. Second, the overhead created by Bayesian optimization during the search is low. In the left part of Figure 4, it shows BFS and AK searched similar numbers of neural networks within twelve hours. BFS is a naive search strategy, which does not consume much time during the search besides training the neural networks. AK searched slightly less neural architectures than BFS because of higher time complexity.

从与BFS和AK的比较中可以得出两个结论。首先，贝叶斯优化可以在有限数量的观察中有效的发现更好的架构。搜索的神经架构数量相同的时候，AK比BFS可以得到低很多的错误率。这表明贝叶斯优化可以有效的指引正确的搜索方向，这在发现好的架构上比单纯的BFS方法高效的多。第二，贝叶斯优化在搜索时产生的耗费是很低的。图4的左边表明，BFS和AK在12个小时内搜索的神经网络数量是差不多的。BFS是一个很单纯的搜索策略，除了训练神经网络，搜索过程并没有消耗很多时间。AK搜索的神经架构数量比BFS略少，因为其时间复杂度略高。

Two conclusions can be drawn by comparing BO and AK. First, network morphism does not negatively impact the search performance. In the left part of Figure 4, when BO and AK search a similar number of neural architectures, they achieve similar lowest error rates. Second, network morphism increases the training efficiency, thus improve the performance. As shown in left part of Figure 4, AK could search much more architectures than BO within the same amount of time due to the adoption of network morphism. Since network morphism does not degrade the search performance, searching more architectures results in finding better architectures. This could also be confirmed in the right part of Figure 4. At the end of the searching time, AK achieves lower error rate than BO.

从BO与AK的比较中可以得出两个结论。第一，网络变形对搜索性能没有负面影响。在图4左边，当BO和AK搜索了类似数量的神经架构时，它们得到的最低错误率是类似的。第二，网络变形提高了训练效率，所以提高了性能。在图4左边可以看到，AK在同样时间内可以搜索的网络架构数量比BO要多，因为采用了网络变形方法。由于网络变形不会降低搜索性能，搜索更多的架构会得到更好的架构。这在图4的右边也可以得到确认。在搜索时间结束的时候，AK的错误率比BO要低。

Figure 4: Evaluation of Efficiency. The two figures plot the same result with different X-axis. BFS uses network morphism. BO uses Bayesian optimization. AK uses both.

### 5.3 Parameter Sensitivity Analysis 参数敏感性分析

We now analyze the impacts of the two most important hyperparameters in our proposed method, i.e., β in Equation (10) balancing the exploration and exploitation of the search strategy, and λ in Equation (4) balancing the distance of layers and skip connections. For other hyperparameters, since r and $T_{low}$ in Algorithm 1 are just normal hyperparameters of simulated annealing instead of important parameters directly related to neural architecture search, we do not delve into them here. In this experiment, we use the CIFAR10 dataset as an example. The rest of the experimental setting follows the setting of Section 5.1.

我们现在分析提出方法的两个最重要的超参数的影响，即，式(10)中的参数β，在搜索策略的探索和利用上进行了均衡，式(4)中的λ，在层的距离和跳跃连接距离之间的均衡。对于其他的超参数，由于算法1中的r和$T_{low}$只是模拟退火的常规超参数，而不是与神经架构搜索直接相关的重要参数，我们没有太深入研究其影响。在本试验中，我们使用CIFAR10数据集作为例子。剩下的试验设置采用5.1所述的设置。

From Figure 5, we can observe that the influences of β and λ to the performance of our method are similar. As shown in the left part of Figure 5, with the increase of β from $10^{−2}$ to $10^2$, the error rate decreases first and then increases. If β is too small, the search process is not explorative enough to search the architectures far from the initial architecture. If it is too large, the search process would keep exploring the far points instead of trying the most promising architectures. Similarly, as shown in the right part of Figure 5, the increase of λ would downgrade the error rate at first and then upgrade it. This is because if λ is too small, the differences in the skip-connections of two neural architectures are ignored; conversely, if it is too large, the differences in the convolutional or fully-connected layers are ignored. The differences in layers and skip-connections should be balanced in the kernel function for the entire framework to achieve a good performance.

从图5中我们可以观察到，β和λ对我们的方法的影响是类似的。如图5左边部分所示，随着β从$10^{-2}$增加到$10^2$，错误率先下降然后上升。如果β太小，搜索过程的探索性不够强，所探索的架构与初始架构距离不够远。如果太大，搜索过程会持续探索很远的点，而不是尝试最有希望的架构。类似的，如图5右边部分所示，λ的增加首先使错误率下降，然后增加。这是因为如果λ太小，两个网络架构中跳跃连接的差异会被忽略；相反的，如果λ太大，卷积层或全连接层的差异会被忽略。层和跳跃连接的差异应当在核函数中得到均衡，以使整个框架得到一个很好的性能。

Figure 5: Parameter Sensitivity Analysis. β balances the exploration and exploitation of the search strategy. λ balances the distance of layers and skip connections.

### 5.4 Evaluation of Kernel Quality 核质量的评估

To show the quality of the edit-distance neural network kernel, we investigate the difference between the two matrices K and P. $K_{n×n}$ is the kernel matrix, where $K_{i, j} = κ(f^{(i)}, f^{(j)})$. $P_{n×n}$ describes the similarity of the actual performance between neural networks, where $P_{i, j} = −|c^{(i)} − c^{(j)}|$, $c^{(i)}$ is the cost function value in the search history H described in Section 3. We use CIFAR10 as an example here, and adopt error rate as the cost metric. Since the values in K and P are in different scales, both matrices are normalized to the range [−1,1]. We quantitatively measure the difference between K and P with mean square error, which is $1.12 × 10^{−1}$.

为验证编辑距离神经网络核的质量，我们研究了两个矩阵K和P的区别。$K_{n×n}$为核矩阵，$K_{i, j} = κ(f^{(i)}, f^{(j)})$。$P_{n×n}$描述了描述的是不同神经网络之间的实际性能之间的相似性，其中$P_{i, j} = −|c^{(i)} − c^{(j)}|$，$c^{(i)}$是第3部分描述的搜索历史H中的代价函数值。我们使用CIFAR10作为例子，采用错误率作为代价度量标准。由于K和P的值的数量级不一样，两个矩阵都归一化到[-1,1]的范围。我们用均方误差定量测量K和P的差异，其值为0.112。

K and P are visualized in Figure 6a and 6b. Lighter color means larger values. There are two patterns can be observed in the figures. 图6a和6b为K和P的可视化结果，浅色为更大的值，图中可以看出两种模式。

First, the white diagonal of Figure 6a and 6b. According to the definiteness property of the kernel, $κ(f_x, f_x)$ = 1, ∀$f_x ∈ F$, thus the diagonal of K is always 1. It is the same for P since no difference exists in the performance of the same neural network.

第一，图6a和6b的白色对角线。根据核的确定性性质，$κ(f_x, f_x)$ = 1, ∀$f_x ∈ F$, 所以K的对角线永远是1。对于P也是一样，因为相同的神经网络之间没有性能差异。

Second, there is a small light square area on the upper left of Figure 6a. These are the initial neural architectures to train the Bayesian optimizer, which are neighbors to each other in terms of network morphism operations. A similar pattern is reflected in Figure 6b, which indicates that when the kernel measures two architectures as similar, they tend to have similar performance.

第二，在图6a的左上角有一个小的浅色方块区域。这是训练贝叶斯优化器的初始神经架构，从网络变形操作来说，其之间互相是近邻。图6b中也有类似的模式，这表明，当核度量的两个架构是类似的，它们的性能也类似。

## 6 Conclusion and Future Work

In this paper, a novel method for efficient neural architecture search with network morphism is proposed. It enables Bayesian optimization to guide the search by designing a neural network kernel, and an algorithm for optimizing acquisition function in tree-structured space. The proposed method is wrapped into an open-source AutoML system, namely Auto-Keras, which can be easily downloaded and used with an extremely simple interface. The method has shown good performance in the experiments and outperformed several traditional hyperparameter-tuning methods and state-of-the-art neural architecture search methods. We plan to study the following open questions in future work. (1) The search space may be expanded to the recurrent neural networks. (2) Tune the neural architecture and the hyperparameters of the training process jointly. (3) Design task-oriented NAS to solve specific machine learning problems, e.g., image segmentation [21] and object detection [25].

本文中，我们提出了一种新方法，可以用网络变形高效的搜索神经架构。我们的方法设计了一个神经网络核，和一个算法优化树状空间的获得函数，以使用贝叶斯优化指引搜索。提出的方法在开源AutoML系统中得到了应用，即Auto-Keras，容易下载，接口极其简单易用。我们的方法在试验中取得了很好的结果，超过了几个传统的超参数调节方法，和目前最好的神经架构搜索方法。我们计划将来研究以下开放问题。(1)搜索空间可能拓展到循环神经网络；(2)同时调节网络架构和训练过程的超参数；(3)设计与任务相关的NAS，求解特定的机器学习问题，如图像分割[21]和目标检测[25]。

## Appendix: Reproducibility

In this section, we provide the details of our implementation and proofs for reproducibility. 这个部分中，我们给出了实现的细节和重现的证据。

- The default architectures used to initialized are introduced. 介绍了初始化的默认架构。

- The details of the implementation of the four network morphism operations are provided. 给出了四种网络变形操作的实现的细节。

- The details of preprocessing the datasets are shown. 给出了预处理数据集的细节。

- The details of the training process are described. 描述了训练过程的细节。

- The proof of the validity of the kernel function is provided. 核函数的正确性的证明。

- The process of using ρ(·) to distort the approximated edit-distance of the neural architectures d(·, ·) is introduced. 介绍了使用ρ(·)来使神经网络架构的近似编辑距离d(·, ·)变形的过程。

Notably, the code and detailed documentation are available at Auto-Keras official website (https://autokeras.com).

### A Default Architechure 默认架构

As we introduced in the experiment section, for all other methods except AK-DP, are using the same three-layer convolutional neural network as the default architecture. The AK-DP is initialized with ResNet, DenseNet and the three-layer CNN. In the current implementation, ResNet18 and DenseNet121 specifically are chosen as the among all the ResNet and DenseNet architectures.

如同我们在试验部分介绍的，除了AK-DP，其他所有方法都使用相同的三层卷积网络作为默认架构。AK-DP是使用的ResNet、DenseNet和三层CNN初始化。在目前的实现中，从所有的ResNet和DenseNet架构中选择了ResNet18和DenseNet121。

The three-layer CNN is constructed as follows. Each convolutional layer is actually a convolutional block of a ReLU layer, a batch-normalization layer, the convolutional layer, and a pooling layer. All the convolutional layers are with kernel size equal to three, stride equal to one, and number of filters equal to 64.

三层CNN构建如下。每个卷积层实际上是ReLU层、BN层、卷积层和pooling层的卷积块。所有卷积层的核大小为3，步长为1，滤波器数量为64。

All the default architectures share the same fully-connected layers design. After all the convolutional layers, the output tensor passes through a global average pooling layer followed by a dropout layer, a fully-connected layer of 64 neurons, a ReLU layer, another fully-connected layer, and a softmax layer.

所有默认架构共享相同的全卷积层设计。在所有卷积层之后，输出张量都会经过全局平均pooling层，然后是一个dropout层，64个神经元的全连接层，另一个全连接层和softmax层。

### B Network Morphism Implementation 网络变形的实现

The implementation of the network morphism is introduced from two aspects. First, we describe how the new weights are initialized. Second, we introduce a pool of possible operations which the Bayesian optimizer can select from, e.g. the possible start and end points of a skip connection.

网络变形的实现从两个方面介绍。第一，我们介绍一下新的权重是怎么初始化的。第二，我们介绍了一个可能操作的池，贝叶斯优化器可以从中进行选择，如，跳跃连接的可能起始点和终点。

The four network morphism operations all involve adding new weights during inserting new layers and expanding existing layers. We initialize the newly added weights with zeros. However, it would create a symmetry prohibiting the newly added weights to learn different values during backpropagation. We follow the Net2Net [8] to add noise to break the symmetry. The amount of noise added is the largest noise possible not changing the output.

四个网络变形操作在插入新的层和扩展现有的层时，都涉及到增加新的权重。我们初始化新增加的权重为0。但是，这会产生一种对称性，在反向传播的时候，阻止新加入的权重学习不同的值。我们遵循Net2Net[8]中的方法，添加了噪声，以打破对称性。增加的噪声的幅度是不改变输出的可能的最大噪声值。

There are a large amount of possible network morphism operations we can choose. Although there are only four types of operations we can choose, a parameter of the operation can be set to a large number of different values. For example, when we use the deep(G, u) operation, we need to choose the location u to insert the layer. In the tree-structured search, we actually cannot exhaust all the operations to get all the children. We will keep sampling from the possible operations until we reach eight children for a node. For the sampling, we randomly sample an operation from deep, wide and skip (add and concat), with equally likely probability. The parameters of the corresponding operation are sampled accordingly. If it is the deep operation, we need to decide the location to insert the layer. In our implementation, any location except right after a skip-connection. Moreover, we support inserting not only convolutional layers, but activation layers, batch-normalization layers, dropout layer, and fully-connected layers as well. They are randomly sampled with equally likely probability. If it is the wide operation, we need to choose the layer to be widened. It can be any convolutional layer or fully-connected layer, which are randomly sampled with equally likely probability. If it is the skip operations, we need to decide if it is add or concat. The start point and end point of a skip-connection can be the output of any layer except the already-exist skip-connection layers. So all the possible skip-connections are generated in the form of tuples of the start point, end point and type (add or concat), among which we randomly sample a skip-connection with equally likely probability.

可以选择的网络变形操作非常多。虽然只有四种类型的操作可以选择，但操作的参数可以设置为很多不同值。比如，当我们使用deep(G,u)操作时，我们需要选择插入层的位置u。在树状结构的搜索中，我们实际上不能穷尽所有操作以得到所有子代。我们会从可能的操作中持续筛选采样，直到我们达到一个节点的8个子代。对于采样，我们从deep, wide和skip(add and concat)中随机选择一个操作，选择概率相同。对应操作的参数也相应的进行取样。如果是deep操作，我们需要确定插入层的位置。在我们的实现中，除了跳跃连接的下一层的任何层都可以。而且，我们支持插入的不仅仅是卷积层，还支持激活层、BN层、dropout层和全连接层。它们都是以同样的概率进行随机取样。如果是wide操作，我们需要选择加宽的层。可以是任何卷积层或全连接层，也是以相同的概率随机进行取样。如果是skip操作，我们需要确定是add还是concat。跳跃连接的开始点和终点可以是任何层的输出，除了已经存在的跳跃连接层。所以会生成所有可能的跳跃连接，生成的形式是开始点，终点和类型(add或concat)的元组，从中我们以等概率随机选择一个跳跃连接。

### C Preprocessing the Datasets 预处理数据集

The benchmark datasets, e.g., MNIST, CIFAR10, FASHION, are preprocessed before the neural architecture search. It involves normalization and data augmentation. We normalize the data to the standard normal distribution. For each channel, a mean and a standard deviation are calculated since the values in different channels may have different distributions. The mean and standard deviation are calculated using the training and validation set together. The testing set is normalized using the same values. The data augmentation includes random crop, random horizontal flip, and cutout, which can improve the robustness of the trained model.

基准测试数据集，如MNIST，CIFAR10，FASHION，在神经网络搜索之前要预处理，这指的是归一化和数据扩充。我们归一化数据为标准正态分布。对每个通道，计算出一个均值和标准差，因为不同通道的数据可能有不同的分布。使用训练集和验证集的数据一起计算均值和标准差。测试集用相同的值进行归一化。数据扩充包括随机剪切，随机水平翻转，cutout，这可以改进训练模型的稳健性。

### D Performance Estimation 性能估计

During the observation phase, we need to estimate the performance of a neural architecture to update the Gaussian process model in Bayesian optimization. Since the quality of the observed performances of the neural architectures is essential to the neural architecture search algorithm, we propose to train the neural architectures instead of using the performance estimation strategies used in literatures [4, 12, 30]. The quality of the observations is essential to the neural architecture search algorithm. So the neural architectures are trained during the search in our proposed method.

在观察阶段，我们需要估计一个神经架构的性能，以更新贝叶斯优化中的高斯过程模型。由于观察到的神经网络的性能的质量，对于神经架构搜索算法非常关键，我们提出训练神经网络架构，而不是使用性能估计的策略，性能估计在文献[4,12,30]中进行了应用。观测的质量对于神经架构搜索算法非常关键。所以在我们的方法搜索的过程中，对神经架构进行训练。

There two important requirements for the training process. First, it needs to be adaptive to different architectures. Different neural networks require different numbers of epochs in training to converge. Second, it should not be affected by the noise in the performance curve. The final metric value, e.g., mean squared error or accuracy, on the validation set is not the best performance estimation since there is random noise in it.

对训练过程，有两个重要的需求。第一，需要对不同的架构更具有适应性。不同的神经网络需要不同数量的训练轮数才能收敛。第二，不应该被性能曲线中的噪声所影响。在验证集上的最终度量值，如，均方误差或准确率，不是最好的性能估计，因为其中有噪声。

To be adaptive to architectures of different sizes, we use the same strategy as the early stop criterion in the multi-layer perceptron algorithm in Scikit-Learn [29]. It sets a maximum threshold τ . If the loss of the validation set does not decrease in τ epochs, the training stops. Comparing with the methods using a fixed number of training epochs, it is more adaptive to different neural architectures.

为能适应各种不同大小的框架，我们使用相同的策略，即Scikit-learn[29]中MLP的早停准则。它设置了一个最大阈值τ，如果验证集的损失在τ轮中没有下降，那么训练停止。与使用固定数量训练轮数的方法比，对不同神经架构的适应性更强。

To avoid being affected by the noise in the performance, the mean of metric values of the last τ epochs on the validation set is used as the estimated performance for the given neural architecture. It is more accurate than the final metric value on the validation set.

为防止受到性能中的噪声影响，使用最后τ轮验证集上的度量标准值的均值，作为给定神经架构的估计性能。这比在验证集上的最终度量标准值要更加准确。

### E Validity of the Kernel

Theorem 1. d($f_a, f_b$) is a metric space distance.

Theorem 2. κ($f_a, f_b$) is a valid kernel.

### F Distance Distortion

In this section, we introduce how Bourgain theorem is used to distort the learned calculated edit-distance into an isometrically embeddable distance for Euclidean space in the Bayesian optimization process.

在本节中，我们介绍一下Bourgain定理怎样用于，在贝叶斯优化过程中，将学习到的编辑距离，在欧几里得空间中，变形为isometrically embeddable distance。

From Bourgain theorem, a Bourgain embedding algorithm is designed. The input for the algorithm is a metric distance matrix. Here we use the edit-distance matrix of neural architectures. The outputs of the algorithm are some vectors in Euclidean space corresponding to the instances. In our case, the instances are neural architectures. From these vectors, we can calculate a new distance matrix using Euclidean distance. The objective of calculating these vectors is to minimize the difference between the new distance matrix and the input distance matrix, i.e., minimize the distortions on the distances.

We apply this Bourgain algorithm during the update process of the Bayesian optimization. The edit-distance matrix of previous training examples, i.e., the neural architectures, is stored in memory. Whenever new examples are used to train the Bayesian optimization, the edit-distance is expanded to include the new distances. The distorted distance matrix is computed using Bourgain algorithm from the expanded edit-distance matrix. It is isometrically embeddable to the Euclidean space. The kernel matrix computed using the distorted distance matrix is a valid kernel.