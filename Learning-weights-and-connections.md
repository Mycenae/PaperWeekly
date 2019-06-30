# Learning both Weights and Connections for Efficient Neural Networks

Song Han et al. Stanford University NVIDIA

## Abstract 摘要

Neural networks are both computationally intensive and memory intensive, making them difficult to deploy on embedded systems. Also, conventional networks fix the architecture before training starts; as a result, training cannot improve the architecture. To address these limitations, we describe a method to reduce the storage and computation required by neural networks by an order of magnitude without affecting their accuracy by learning only the important connections. Our method prunes redundant connections using a three-step method. First, we train the network to learn which connections are important. Next, we prune the unimportant connections. Finally, we retrain the network to fine tune the weights of the remaining connections. On the ImageNet dataset, our method reduced the number of parameters of AlexNet by a factor of 9×, from 61 million to 6.7 million, without incurring accuracy loss. Similar experiments with VGG-16 found that the total number of parameters can be reduced by 13×, from 138 million to 10.3 million, again with no loss of accuracy.

神经网络计算量大，对内存需求大，使其很难在嵌入式系统中部署。同时，传统网络在训练之前就固定了网络架构；结果是，训练不能改进架构。为解决这个局限，我们描述了一种方法，在不损失准确率的情况下，通过只学习重要的连接，将神经网络所需的内存和计算量降低了一个数量级。我们的方法使用三个步骤修剪掉冗余的连接。第一，我们训练网络学习重要的连接。然后，我们修剪掉不重要的连接。最后，我们重新训练网络，以精调剩余连接的权重。在ImageNet数据集上，我们的方法将AlexNet的参数降低了9倍，从61 million到6.7 million，没有带来准确率损失。在VGG-16上类似的试验发现，参数数量可以降低13x，从138 million到10.3 million，也没有损失准确率。

## 1 Introduction 引言

Neural networks have become ubiquitous in applications ranging from computer vision [1] to speech recognition [2] and natural language processing [3]. We consider convolutional neural networks used for computer vision tasks which have grown over time. In 1998 Lecun et al. designed a CNN model LeNet-5 with less than 1M parameters to classify handwritten digits [4], while in 2012, Krizhevsky et al. [1] won the ImageNet competition with 60M parameters. Deepface classified human faces with 120M parameters [5], and Coates et al. [6] scaled up a network to 10B parameters.

神经网络在计算机视觉、语音识别和自然语言处理上得到了广泛的应用。我们考虑的是在计算机视觉中应用的卷积神经网络，这已经得到了一段时间的发展。在1998年，Lecun等设计了一个CNN模型LeNet-5，参数数量少于1M，以对手写数字进行分类；2012年，Krizhevsky等用60M参数赢得了ImageNet比赛。Deepface用120M参数对人脸进行了分类，Coates等将一个网络放大到10B参数。

While these large neural networks are very powerful, their size consumes considerable storage, memory bandwidth, and computational resources. For embedded mobile applications, these resource demands become prohibitive. Figure 1 shows the energy cost of basic arithmetic and memory operations in a 45nm CMOS process. From this data we see the energy per connection is dominated by memory access and ranges from 5pJ for 32 bit coefficients in on-chip SRAM to 640pJ for 32bit coefficients in off-chip DRAM [7]. Large networks do not fit in on-chip storage and hence require the more costly DRAM accesses. Running a 1 billion connection neural network, for example, at 20Hz would require (20Hz)(1G)(640pJ) = 12.8W just for DRAM access - well beyond the power envelope of a typical mobile device. Our goal in pruning networks is to reduce the energy required to run such large networks so they can run in real time on mobile devices. The model size reduction from pruning also facilitates storage and transmission of mobile applications incorporating DNNs.

这些大型神经网络能力很强，其规模占用了很多存储空间、内存带宽和计算资源。对于嵌入式移动应用，这些资源需求变得不可行。图1展示了在45nm CMOS过程中，基本的代数运算和内存操作所消耗的能量。从这些数据中，我们看到每个连接的能量主要都是内存访问，从片上SRAM 32 bit系数的5pJ，到片下DRAM 32 bit系数的640pJ。大型网络不能只用片上存储运行，所以需要更昂贵的DRAM访问。比如，以20Hz运行一个1 billion连接的神经网络，需要(20Hz)(1G)(640pJ) = 12.8W的能量进行DRAM访问，远远超过了典型嵌入式设备的能耗。我们修剪网络的目标是降低运行这样大型网络所需的能耗，使其可以在移动设备上实时运行。修剪带来的模型大小降低也使得使用DNNs的移动应用的存储和传输变得很便利。

To achieve this goal, we present a method to prune network connections in a manner that preserves the original accuracy. After an initial training phase, we remove all connections whose weight is lower than a threshold. This pruning converts a dense, fully-connected layer to a sparse layer. This first phase learns the topology of the networks — learning which connections are important and removing the unimportant connections. We then retrain the sparse network so the remaining connections can compensate for the connections that have been removed. The phases of pruning and retraining may be repeated iteratively to further reduce network complexity. In effect, this training process learns the network connectivity in addition to the weights - much as in the mammalian brain [8,9], where synapses are created in the first few months of a child’s development, followed by gradual pruning of little-used connections, falling to typical adult values.

为达到这个目标，我们提出了一种修剪网络连接的方法，可以保持原始准确率不降低。在初始训练阶段之后，我们去除掉权重低于一定阈值的所有连接。这种修剪将一个密集的全连接层转化成一个稀疏层。这个第一阶段学到了网络的拓扑，学习到了哪些连接是重要的，去掉不重要的连接。然后我们重新训练这个稀疏网络，这样剩下的连接可以弥补那些修剪掉的连接。修剪的阶段和重新训练的阶段可以重复迭代进行，以进一步降低网络复杂度。实际上，这种训练过程除了学习权重，还学习了网络连接，就像哺乳动物大脑那样，在孩子发育的最开始几个月会生成突触，然后逐渐修剪很少使用的连接，形成典型的成人值。

Figure 1: Energy table for 45nm CMOS process [7]. Memory access is 3 orders of magnitude more energy expensive than simple arithmetic.

Operation | Energy [pJ] | Relative Cost
--- | --- | ---
32 bit int ADD | 0.1 | 1
32 bit float ADD | 0.9 | 9
32 bit Register File | 1 | 10
32 bit int MULT | 3.1 | 31
32 bit float MULT | 3.7 | 37
32 bit SRAM Cache | 5 | 50
32 bit DRAM Memory | 640 | 6400

## 2 Related Work 相关工作

Neural networks are typically over-parameterized, and there is significant redundancy for deep learning models [10]. This results in a waste of both computation and memory. There have been various proposals to remove the redundancy: Vanhoucke et al. [11] explored a fixed-point implementation with 8-bit integer (vs 32-bit floating point) activations. Denton et al. [12] exploited the linear structure of the neural network by finding an appropriate low-rank approximation of the parameters and keeping the accuracy within 1% of the original model. With similar accuracy loss, Gong et al. [13] compressed deep convnets using vector quantization. These approximation and quantization techniques are orthogonal to network pruning, and they can be used together to obtain further gains [14].

神经网络一般都是参数过量的，在深度学习模型中都有明显的冗余。这导致了计算量和内存的浪费。有很多去除冗余性的尝试：Vanhoucke等[11]尝试了8-bit整型的定点实现（vs 32-bit浮点型）。Denton等[12]利用了神经网络的线性结构，找到了参数的低秩合理近似，保持准确率与原始模型相差不到1%。在类似的准确率损失下，Gong等[13]使用矢量量化压缩了深度神经网络。这些近似和量化技巧与网络剪枝是无关的，它们可以一起使用，以得到更多收益[14]。

There have been other attempts to reduce the number of parameters of neural networks by replacing the fully connected layer with global average pooling. The Network in Network architecture [15] and GoogLenet [16] achieves state-of-the-art results on several benchmarks by adopting this idea. However, transfer learning, i.e. reusing features learned on the ImageNet dataset and applying them to new tasks by only fine-tuning the fully connected layers, is more difficult with this approach. This problem is noted by Szegedy et al. [16] and motivates them to add a linear layer on the top of their networks to enable transfer learning.

也有其他降低参数数量的尝试，如将全连接网络替换为全局平均池化。Network in Network架构[15]和GoogLenet[16]利用这种思想在几个基准测试上得到了目前最好的结果。但是，迁移学习，即重用在ImageNet上学习到的特征，将其应用到新的任务上，只对全连接层进行精调，使用这种方法就非常困难。Szegedy等[16]也提到了这个问题，在网络之上也加入了一个线性层，以使迁移学习可以进行。

Network pruning has been used both to reduce network complexity and to reduce over-fitting. An early approach to pruning was biased weight decay [17]. Optimal Brain Damage [18] and Optimal Brain Surgeon [19] prune networks to reduce the number of connections based on the Hessian of the loss function and suggest that such pruning is more accurate than magnitude-based pruning such as weight decay. However, second order derivative needs additional computation.

网络剪枝可以用于降低网络复杂度，也可以降低过拟合。剪枝的早期方法是权重衰减的偏置[17]。Optimal Brain Damage[18]和Optimal Brain Surgeon[19]对网络进行剪枝，以降低连接数，是基于损失函数的Hessian矩阵，表示这样的剪枝比基于幅度的剪枝（如权重衰减）更准确。但是，两阶导数需要额外的计算。

HashedNets [20] is a recent technique to reduce model sizes by using a hash function to randomly group connection weights into hash buckets, so that all connections within the same hash bucket share a single parameter value. This technique may benefit from pruning. As pointed out in Shi et al. [21] and Weinberger et al. [22], sparsity will minimize hash collision making feature hashing even more effective. HashedNets may be used together with pruning to give even better parameter savings.

HashedNets[20]是一种最近的技术，可以降低模型大小，使用了hash函数以将连接权重随机分组到hash buckets中，这样在相同hash bucket中的所有连接都共享单个参数值。这种技术可以从剪枝中受益。如Shi等[21]和Weinberger等[22]指出，稀疏性会最小化hash冲突，使特征hash更有效。HashedNets可以与剪枝一起使用，以得到更好的降低参数数量的效果。

## 3 Learning Connections in Addition to Weights 学习权重以及连接

Our pruning method employs a three-step process, as illustrated in Figure 2, which begins by learning the connectivity via normal network training. Unlike conventional training, however, we are not learning the final values of the weights, but rather we are learning which connections are important. The second step is to prune the low-weight connections. All connections with weights below a threshold are removed from the network — converting a dense network into a sparse network, as shown in Figure 3. The final step retrains the network to learn the final weights for the remaining sparse connections. This step is critical. If the pruned network is used without retraining, accuracy is significantly impacted.

我们的剪枝方法采用三步过程，如图2所示，开始于通过正常网络训练学习网络的连接性。但是，与传统卷积不同的是，我们没有学习权重的最终值，而是学习哪些连接是重要的。第二步是剪枝低权重的连接。权重低于某一阈值的连接从网络中移除掉，也就将密集连接的网络转化成了一个稀疏网络，如图3所示。最后的一步是重新训练网络，为剩余的稀疏连接学习最终的权重。最后一步是很关键的。如果修剪后的网络没有进行重新训练，会明显影响准确率。

Figure 2: Three-Step Training Pipeline. Train Connectivity -> | Prune Connections -> Train Weights | |iterates|

Figure 3: Synapses and neurons before and after pruning.

### 3.1 Regularization 正则化

Choosing the correct regularization impacts the performance of pruning and retraining. L1 regularization penalizes non-zero parameters resulting in more parameters near zero. This gives better accuracy after pruning, but before retraining. However, the remaining connections are not as good as with L2 regularization, resulting in lower accuracy after retraining. Overall, L2 regularization gives the best pruning results. This is further discussed in experiment section.

选择正确的正则化方法影响剪枝和重新训练的性能。L1正则化惩罚的是非零参数，会得到更多在零附近的参数值。这在剪枝之后、重新训练之前会得到更好的准确率。但是，剩下的连接使用L2正则化更好一些，如果使用L1正则化重新训练，会得到更低的准确率。总体上来说，使用L2正则化会得到最好的剪枝效果。在试验部分会进一步讨论这个效果。

### 3.2 Dropout Ratio Adjustment 

Dropout [23] is widely used to prevent over-fitting, and this also applies to retraining. During retraining, however, the dropout ratio must be adjusted to account for the change in model capacity. In dropout, each parameter is probabilistically dropped during training, but will come back during inference. In pruning, parameters are dropped forever after pruning and have no chance to come back during both training and inference. As the parameters get sparse, the classifier will select the most informative predictors and thus have much less prediction variance, which reduces over-fitting. As pruning already reduced model capacity, the retraining dropout ratio should be smaller.

Dropout[23]广泛用于防止过拟合，这也用于重新训练的过程。但在重新的过程中，dropout率必须考虑到模型容量的变化。在dropout中，每个参数都在训练过程中以一定概率弃置不用，但在推理的时候是会使用的。在剪枝过后，参数是会被永久丢弃的，在训练和推理过程中都不会在出现。由于参数变得稀疏，分类器会选择最有信息量的预测器，所以会有更小的预测变化，这会减少过拟合。由于剪枝已经降低了模型容量/能力，重新训练的dropout率应当更小一些。

Quantitatively, let $C_i$ be the number of connections in layer i, $C_{io}$ for the original network, $C_{ir}$ for the network after retraining, $N_i$ be the number of neurons in layer i. Since dropout works on neurons, and $C_i$ varies quadratically with $N_i$, according to Equation 1 thus the dropout ratio after pruning the parameters should follow Equation 2, where $D_o$ represent the original dropout rate, $D_r$ represent the dropout rate during retraining.

定量的说，令$C_i$为层i中的连接数量，$C_{io}$是原始网络的连接数量，$C_{ir}$是重新训练后网络的连接数量，$N_i$是层i中的神经元数量。由于dropout是作用于神经元的，$C_i$随着$N_i$数量变化以平方级变化，根据式1，修剪参数后，dropout率要遵循式2，其中$D_o$代表原始的dropout率，$D_r$代表重新训练时的dropout率。


$$C_i = N_i N_{i−1}$$(1)
$$D_r = D_o \sqrt{\frac{C_{ir}}{C_{io}}}$$(2)

### 3.3 Local Pruning and Parameter Co-adaptation 局部剪枝和参数的相互适应

During retraining, it is better to retain the weights from the initial training phase for the connections that survived pruning than it is to re-initialize the pruned layers. CNNs contain fragile co-adapted features [24]: gradient descent is able to find a good solution when the network is initially trained, but not after re-initializing some layers and retraining them. So when we retrain the pruned layers, we should keep the surviving parameters instead of re-initializing them.

在重新训练的过程中，对于经过修剪剩下的连接，最好保持初始训练阶段的权重，这比重新初始化剪枝过的层要好。CNNs包含脆弱的共同适应的特征[24]：在网络经过初始化训练后，梯度下降可以找到好的解决方案，但重新初始化一些层并重新训练后，反而可能找不到。所以当重新训练剪枝过的层时，我们应当保持其参数，而不是去重新初始化。

Retraining the pruned layers starting with retained weights requires less computation because we don’t have to back propagate through the entire network. Also, neural networks are prone to suffer the vanishing gradient problem [25] as the networks get deeper, which makes pruning errors harder to recover for deep networks. To prevent this, we fix the parameters for CONV layers and only retrain the FC layers after pruning the FC layers, and vice versa.

保持权重以重新训练剪枝过的层，需要更少的计算量，因为我们不需要对整个网络进行反向传播。同时，在网络变得更深时，神经网络容易受梯度消失问题的困扰，这使剪枝的错误在深度网络中很难恢复。为防止这个问题，我们在对FC层剪枝过后，固定CONV层的参数，只重新训练FC层；反之亦然。

### 3.4 Iterative Pruning 迭代剪枝

Learning the right connections is an iterative process. Pruning followed by a retraining is one iteration, after many such iterations the minimum number connections could be found. Without loss of accuracy, this method can boost pruning rate from 5× to 9× on AlexNet compared with single-step aggressive pruning. Each iteration is a greedy search in that we find the best connections. We also experimented with probabilistically pruning parameters based on their absolute value, but this gave worse results.

学习正确的连接是一个迭代的过程。剪枝过后进行重新训练，这是一次迭代，在多次这样的迭代后，连接数量达到了最少。在不损失准确率的情况下，与一步激进剪枝相比，这种方法可以将在AlexNet上的剪枝率将5x提高到9x。每次迭代都是一次贪婪搜索，可以找到最好的连接。我们还试验了基于参数绝对值的概率剪枝，但得到的结果更差一些。

### 3.5 Pruning Neurons 修剪神经元

After pruning connections, neurons with zero input connections or zero output connections may be safely pruned. This pruning is furthered by removing all connections to or from a pruned neuron. The retraining phase automatically arrives at the result where dead neurons will have both zero input connections and zero output connections. This occurs due to gradient descent and regularization. A neuron that has zero input connections (or zero output connections) will have no contribution to the final loss, leading the gradient to be zero for its output connection (or input connection), respectively. Only the regularization term will push the weights to zero. Thus, the dead neurons will be automatically removed during retraining.

在修剪连接后，没有输入连接或没有输出连接的神经元可以安全的剪枝掉。通过移除剪枝神经元的输入输出连接，这种剪枝可以进一步进行下去。重新训练的阶段，在没有输入也没有输出连接的死亡神经元上，会得到这样的结果。由于梯度下降和正则化的存在，这就会发生。零输入（或零输出）的神经元对于最终损失函数没有任何贡献，分别会导致其输出连接（或输入连接）的梯度为零。只有正则化项会将权重推向零。所以，死亡的神经元会在重新训练时自动从网络中移除。

## 4 Experiments 试验

We implemented network pruning in Caffe [26]. Caffe was modified to add a mask which disregards pruned parameters during network operation for each weight tensor. The pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer’s weights. We carried out the experiments on Nvidia TitanX and GTX980 GPUs.

我们在Caffe中实现网络剪枝。Caffe经过修改，增加了一个掩膜，在网络运算中，对每个权重张量，会忽视掉剪枝的参数。剪枝的阈值，是一个质量参数乘以一层权重的标准偏差。我们在NVidia TitanX和GTX980 GPU上进行试验。

We pruned four representative networks: Lenet-300-100 and Lenet-5 on MNIST, together with AlexNet and VGG-16 on ImageNet. The network parameters and accuracy before and after pruning are shown in Table 1(Reference model is from Caffe model zoo, accuracy is measured without data augmentation).

我们对四种有代表性的网络进行剪枝：在MNIST上的LeNet-300-100和LeNet-5，在ImageNet上的AlexNet和VGG-16。剪枝前后的网络参数和准确率如表1所示（参考模型是Caffe模型库中的，准确率是在没有数据扩充的时候度量的）。

Table 1: Network pruning can save 9× to 13× parameters with no drop in predictive performance.

Network | Top-1 Error | Top-5 Error | Parameters | Compress Rate
--- | --- | --- | --- | ---
LeNet-300-100 Ref | 1.64% | - | 267K | -
LeNet-300-100 Pruned | 1.59% | - | 22K | 12x
LeNet-5 Ref | 0.80% | - | 431K | -
LeNet-5 Pruned | 0.77% | - | 36K | 12x
AlexNet Ref | 42.78% | 19.73% | 61M | -
AlexNet Pruned | 42.77% | 19.67% | 6.7M | 9x
VGG-16 Ref | 31.50% | 11.32% | 138M | -
VGG-16 Pruned | 31.34% | 10.88% | 10.3M | 13x

### 4.1 LeNet on MNIST

We first experimented on MNIST dataset with the LeNet-300-100 and LeNet-5 networks [4]. LeNet-300-100 is a fully connected network with two hidden layers, with 300 and 100 neurons each, which achieves 1.6% error rate on MNIST. LeNet-5 is a convolutional network that has two convolutional layers and two fully connected layers, which achieves 0.8% error rate on MNIST. After pruning, the network is retrained with 1/10 of the original network’s original learning rate. Table 1 shows pruning saves 12× parameters on these networks. For each layer of the network the table shows (left to right) the original number of weights, the number of floating point operations to compute that layer’s activations, the average percentage of activations that are non-zero, the percentage of non-zero weights after pruning, and the percentage of actually required floating point operations.

我们首先在MNIST上对LeNet-300-100和LeNet-5网络进行试验。LeNet-300-100是一个全连接网络，有2个隐藏层，分别有300和100个神经元，在MNIST上得到了1.6%的错误率。LeNet-5是一个卷积网络，有2个卷积层，2个全连接层，在MNIST上得到了0.8%的错误率。经过修剪后，网络进行重新训练，使用原网络1/10的学习速率。表1的结果表明，剪枝使网络参数降低了12倍。对网络中的每个层，表2,3给出了（从左到右），原始权重的数量，计算这一层的激活的浮点运算数量，非零激活的平均比率，修剪后非零权重的比率，修剪后需要的浮点运算数量。

Table 2: For Lenet-300-100, pruning reduces the number of weights by 12× and computation by 12×.

Layer | Weights | FLOP | Act% | Weights% | FLOP%
--- | --- | --- | --- | --- | ---
fc1 | 235K | 470K | 38% | 8% | 8%
fc2 | 30K | 60K | 65% | 9% | 4%
fc3 | 1K | 2K | 100% | 26% | 17%
Total | 266K | 532K | 46% | 8% | 8%

Table 3: For Lenet-5, pruning reduces the number of weights by 12× and computation by 6×.

Layer | Weights | FLOP | Act% | Weights% | FLOP%
--- | --- | --- | --- | --- | ---
conv1 | 0.5K | 576K | 82% | 66% | 66%
conv2 | 25K | 3200K | 72% | 12% | 10%
fc1 | 400K | 800K | 55% | 8% | 6%
fc2 | 5K | 10K | 100% | 19% | 10%
Total | 431K | 4586K | 77% | 8% | 16%

An interesting byproduct is that network pruning detects visual attention regions. Figure 4 shows the sparsity pattern of the first fully connected layer of LeNet-300-100, the matrix size is 784 ∗ 300. It has 28 bands, each band’s width 28, corresponding to the 28 × 28 input pixels. The colored regions of the figure, indicating non-zero parameters, correspond to the center of the image. Because digits are written in the center of the image, these are the important parameters. The graph is sparse on the left and right, corresponding to the less important regions on the top and bottom of the image. After pruning, the neural network finds the center of the image more important, and the connections to the peripheral regions are more heavily pruned.

一个有趣的副产品是，网络剪枝可以检测到视觉上的注意力区域。图4展示了，LeNet-300-100网络的第一个全连接层的稀疏模式，矩阵大小为784×300。它有28个带状区域，每个带宽度28，对应着28×28的输入像素。图中有颜色的区域，表示非零参数，对应图像的中间区域。因为数字是在图像中间部分书写的，这些是重要的参数。这个图在左边和右边都是稀疏的，对应着图像上部分和下部分区域的不重要部分。在剪枝后，神经网络发现图像中间区域更加重要，外围区域的连接很多都被修剪了。

Figure 4: Visualization of the first FC layer’s sparsity pattern of Lenet-300-100. It has a banded structure repeated 28 times, which correspond to the un-pruned parameters in the center of the images, since the digits are written in the center.

### 4.2 AlexNet on ImageNet

We further examine the performance of pruning on the ImageNet ILSVRC-2012 dataset, which has 1.2M training examples and 50k validation examples. We use the AlexNet Caffe model as the reference model, which has 61 million parameters across 5 convolutional layers and 3 fully connected layers. The AlexNet Caffe model achieved a top-1 accuracy of 57.2% and a top-5 accuracy of 80.3%. The original AlexNet took 75 hours to train on NVIDIA Titan X GPU. After pruning, the whole network is retrained with 1/100 of the original network’s initial learning rate. It took 173 hours to retrain the pruned AlexNet. Pruning is not used when iteratively prototyping the model, but rather used for model reduction when the model is ready for deployment. Thus, the retraining time is less a concern. Table 1 shows that AlexNet can be pruned to 1/9 of its original size without impacting accuracy, and the amount of computation can be reduced by 3×.

我们进一步在ImageNet ILSVRC-2012数据集上进行了剪枝，数据集包含1.2M训练样本，50K验证样本。我们使用AlexNet Caffe模型作为参考模型，它有61 million参数，5个卷积层，3个全连接层。AlexNet Caffe模型得到的top-1准确率为57.2%，top-5准确率为80.3%。原始AlexNet模型在NVidia Titan X GPU上耗费75小时进行训练。在剪枝后，整个网络进行重新训练，使用原始网络初始学习速率的1/100。重新训练修剪后的AlexNet耗费了173小时。在迭代准备模型原型的时候没有进行剪枝，而是在模型准备好进行部署的时候，用于减少模型大小。因此，重新训练的时间并不是首要考虑的问题。表4的结果表明，AlexNet可以修剪到原始大小的1/9，而且准确率基本没有影响，计算时间可以降低3x。

Table 4: For AlexNet, pruning reduces the number of weights by 9× and computation by 3×.

Layer | Weights | FLOP | Act% | Weights% | FLOP%
--- | --- | --- | --- | --- | ---
conv1 | 35K | 211M | 88% | 84% | 84%
conv2 | 307K | 448M | 52% | 38% | 33%
conv3 | 885K | 299M | 37% | 35% | 18%
conv4 | 663K | 224M | 40% | 37% | 14%
conv5 | 442K | 150M | 34% | 37% | 14%
fc1 | 38M | 75M | 36% | 9% | 3%
fc2 | 17M | 34M | 40% | 9% | 3%
fc3 | 4M | 8M | 100% | 25% | 10%
Total | 61M | 1.5B | 54% | 11% | 30%

### 4.3 VGG-16 on ImageNet

With promising results on AlexNet, we also looked at a larger, more recent network, VGG-16 [27], on the same ILSVRC-2012 dataset. VGG-16 has far more convolutional layers but still only three fully-connected layers. Following a similar methodology, we aggressively pruned both convolutional and fully-connected layers to realize a significant reduction in the number of weights, shown in Table 5. We used five iterations of pruning an retraining.

在AlexNet上得到了很好的结果，我们还对最近的一个更大的网络进行了试验，VGG-16[27]，也是在ILSVRC-2012数据集上。VGG-16的卷积层多了很多，但仍然只有3个全连接层。按照相同的方法，我们对卷积层和全连接层都进行了修剪，实现了权重数量的明显降低，如表5所示。我们进行了剪枝和重新训练的5次迭代。

The VGG-16 results are, like those for AlexNet, very promising. The network as a whole has been reduced to 7.5% of its original size (13× smaller). In particular, note that the two largest fully-connected layers can each be pruned to less than 4% of their original size. This reduction is critical for real time image processing, where there is little reuse of fully connected layers across images (unlike batch processing during training).

VGG-16的结果与AlexNet上的结果类似，非常有希望。网络整体大小降低到了原大小的7.5%（小了13x）。特别是，注意两个最大的全连接层可以修剪到原始大小的4%。这种程度的降低对于实时图像处理非常重要，因为图像之间全连接层的重复使用非常少（在训练时的批处理则不一样）。

Table 5: For VGG-16, pruning reduces the number of weights by 12× and computation by 5×.

Layer | Weights | FLOP | Act% | Weights% | FLOP%
--- | --- | --- | --- | --- | ---
conv1_1 | 2K | 0.2B | 53% | 58% | 58%
conv1_2 | 37K | 3.7B | 89% | 22% | 12%
conv2_1 | 74K | 1.8B | 80% | 34% | 30%
conv2_2 | 148K | 3.7B | 81% | 36% | 29%
conv3_1 | 295K | 1.8B | 68% | 53% | 43%
conv3_2 | 590K | 3.7B | 70% | 24% | 16%
conv3_3 | 590K | 3.7B | 64% | 42% | 29%
conv4_1 | 1M | 1.8B | 51% | 32% | 21%
conv4_2 | 2M | 3.7B | 45% | 27% | 14%
conv4_3 | 2M | 3.7B | 34% | 34% | 15%
conv5_1 | 2M | 925M | 32% | 35% | 12%
conv5_2 | 2M | 925M | 29% | 29% | 9%
conv5_3 | 2M | 925M | 19% | 36% | 11%
fc6 | 103M | 206M | 38% | 4% | 1%
fc7 | 17M | 34M | 42% | 4% | 2%
fc8 | 4M | 8M | 100% | 23% | 9%
Total | 138M | 30.9B | 64% | 7.5% | 21%

## 5 Discussion 讨论

The trade-off curve between accuracy and number of parameters is shown in Figure 5. The more parameters pruned away, the less the accuracy. We experimented with L1 and L2 regularization, with and without retraining, together with iterative pruning to give five trade off lines. Comparing solid and dashed lines, the importance of retraining is clear: without retraining, accuracy begins dropping much sooner — with 1/3 of the original connections, rather than with 1/10 of the original connections. It’s interesting to see that we have the “free lunch” of reducing 2× the connections without losing accuracy even without retraining; while with retraining we are ably to reduce connections by 9×.

准确率与参数数量的折中曲线如图5所示。修剪掉更多的参数，准确率就会越低。我们对L1和L2正则化，有/没有重新训练进行了试验，还有迭代剪枝，共计给出了5条折中线。比较实线和虚线，可以发现重新训练的重要性是明显的：没有重新训练，准确率很快就开始下降，即在原始连接数量的1/3的时候就开始，而重新训练后可以达到1/10数量后才开始下降。很有趣的是，我们发现参数减少一半，在没有重新训练的情况下也不会造成准确率降低；而在有重新训练的情况下，我们可以降低连接数9x。

Figure 5: Trade-off curve for parameter reduction and loss in top-5 accuracy. L1 regularization performs better than L2 at learning the connections without retraining, while L2 regularization performs better than L1 at retraining. Iterative pruning gives the best result.

L1 regularization gives better accuracy than L2 directly after pruning (dotted blue and purple lines) since it pushes more parameters closer to zero. However, comparing the yellow and green lines shows that L2 outperforms L1 after retraining, since there is no benefit to further pushing values towards zero. One extension is to use L1 regularization for pruning and then L2 for retraining, but this did not beat simply using L2 for both phases. Parameters from one mode do not adapt well to the other.

剪枝后，L1正则化比L2正则化可以得到更好的准确率（点状蓝色和紫色线），因为L1正则化比L2正则化将跟多参数推向零值。但是，黄线和绿线的比较说明，重新训练后，L2比L1要好，因为将权重值推向零已经没有任何好处了。一个拓展情况是，使用L1正则化进行剪枝，然后使用L2进行重新训练，但这样得到的结果也没有比在两个阶段都使用L2更好。一种模式下得到的参数，在另一种模式下适应情况不太好。

The biggest gain comes from iterative pruning (solid red line with solid circles). Here we take the pruned and retrained network (solid green line with circles) and prune and retrain it again. The leftmost dot on this curve corresponds to the point on the green line at 80% (5× pruning) pruned to 8×. There’s no accuracy loss at 9×. Not until 10× does the accuracy begin to drop sharply.

最大的收益来自迭代剪枝（带有实心圆的红色实线）。这里我们选用剪枝后重新训练的网络（有圆圈的绿色实线），对其进行再次的剪枝和重新训练。这条曲线上左上的点对应着绿线上修剪了80%的参数的点（5x剪枝），进一步剪枝到了8x。在9x剪枝下，没有任何准确率损失。在10x剪枝的情况下，准确率开始急剧下降。

Two green points achieve slightly better accuracy than the original model. We believe this accuracy improvement is due to pruning finding the right capacity of the network and hence reducing overfitting.

两个绿色的点比原始模型得到了略好一点的准确率。我们相信这种改进是因为，剪枝发现了网络的正确容量/能力，所以降低了过拟合。

Both CONV and FC layers can be pruned, but with different sensitivity. Figure 6 shows the sensitivity of each layer to network pruning. The figure shows how accuracy drops as parameters are pruned on a layer-by-layer basis. The CONV layers (on the left) are more sensitive to pruning than the fully connected layers (on the right). The first convolutional layer, which interacts with the input image directly, is most sensitive to pruning. We suspect this sensitivity is due to the input layer having only 3 channels and thus less redundancy than the other convolutional layers. We used the sensitivity results to find each layer’s threshold: for example, the smallest threshold was applied to the most sensitive layer, which is the first convolutional layer.

卷积层和全连接层都可以进行剪枝，但敏感度不同。图6展示了，每层对网络剪枝的敏感性。图中给出了，在参数逐层修剪的情况下，准确率是如何下降的。与全连接层（左边）相比，卷积层（左边）对剪枝更敏感。第一个卷积层，直接对输入图像进行计算，对剪枝最敏感。我们认为这种敏感性是因为输入层只有3个通道，所以冗余度比其他卷积层更小。我们使用敏感度结果来寻找每一层的阈值：比如，最小的阈值对应最敏感的层，也就是第一个卷积层。

Figure 6: Pruning sensitivity for CONV layer (left) and FC layer (right) of AlexNet.

Storing the pruned layers as sparse matrices has a storage overhead of only 15.6%. Storing relative rather than absolute indices reduces the space taken by the FC layer indices to 5 bits. Similarly, CONV layer indices can be represented with only 8 bits.

将修剪过的层存储为稀疏矩阵，存储占用空间只有15.6%。FC层的索引，使用相对索引，而不是绝对索引，可以采用5 bit数据表示。类似的，卷积层的索引可以使用8 bit数据表示。

After pruning, the storage requirements of AlexNet and VGGNet are are small enough that all weights can be stored on chip, instead of off-chip DRAM which takes orders of magnitude more energy to access (Table 1). We are targeting our pruning method for fixed-function hardware specialized for sparse DNN, given the limitation of general purpose hardware on sparse computation.

在剪枝过后，AlexNet和VGGNet的存储需求非常的小，所有权重都可以在片上进行存储，而不需要存储在DRAM上，访问能耗要大一个量级。鉴于通用硬件在稀疏计算上的局限，我们会对专用于稀疏DNN的固定作用硬件进行剪枝方法改进。

Figure 7 shows histograms of weight distribution before (left) and after (right) pruning. The weight is from the first fully connected layer of AlexNet. The two panels have different y-axis scales. The original distribution of weights is centered on zero with tails dropping off quickly. Almost all parameters are between [−0.015, 0.015]. After pruning the large center region is removed. The network parameters adjust themselves during the retraining phase. The result is that the parameters form a bimodal distribution and become more spread across the x-axis, between [−0.025, 0.025].

图7给出了剪枝前后权重分布的直方图。这里的权重是AlexNet的第一个全连接层的。两幅图的y轴尺度是不同的。权重的原始分布是以零为中心的，尾部分布迅速下降。几乎所有参数都是在[-0.015, 0.015]的范围内。在剪枝后，大的中心区域没有了。网络参数在重新训练阶段进行了自我调整。得到的结果是，所有参数形成了一个双峰分布，在x轴上分布更广了，范围为[-0.025, 0.025]。

Figure 7: Weight distribution before and after parameter pruning. The right figure has 10× smaller scale.

## 6 Conclusion 结论

We have presented a method to improve the energy efficiency and storage of neural networks without affecting accuracy by finding the right connections. Our method, motivated in part by how learning works in the mammalian brain, operates by learning which connections are important, pruning the unimportant connections, and then retraining the remaining sparse network. We highlight our experiments on AlexNet and VGGNet on ImageNet, showing that both fully connected layer and convolutional layer can be pruned, reducing the number of connections by 9× to 13× without loss of accuracy. This leads to smaller memory capacity and bandwidth requirements for real-time image processing, making it easier to be deployed on mobile systems.

我们提出了一种方法，通过找到正确的连接，可以改进能耗和存储神经网络的效率，而不影响准确率。我们的方法部分是受到哺乳动物的大脑的学习工作机理启发的，学习了哪些连接是重要的，修剪掉不重要的连接，然后重新训练剩余的稀疏网络。我们在AlexNet和VGGNet上进行了试验，表明全连接层和卷积层都可以修剪，连接数量可以降低9x到13x，而且不影响准确率。得到的模型需要更小的内存和带宽，可以进行实时图像处理，更容易在移动系统上部署。

Table 6: Comparison with other model reduction methods on AlexNet. Data-free pruning [28] saved only 1.5× parameters with much loss of accuracy. Deep Fried Convnets [29] worked on fully connected layers only and reduced the parameters by less than 4×. [30] reduced the parameters by 4× with inferior accuracy. Naively cutting the layer size saves parameters but suffers from 4% loss of accuracy. [12] exploited the linear structure of convnets and compressed each layer individually, where model compression on a single layer incurred 0.9% accuracy penalty with biclustering + SVD.

Network | Top-1 Error | Top-5 Error | Parameters | Compress Rate
--- | --- | --- | --- | ---
Baseline Caffemodel [26] | 42.78% | 19.73% | 61.0M | 1x
Data-free pruning [28] | 44.40% | - | 39.6M | 1.5x
Fastfood-32-AD [29] | 41.93% | - | 32.8M | 2x
Fastfood-16-AD [29] | 42.90% | - | 16.4M | 3.7x
Collins & Kohli [30] | 44.40% | - | 15.2M | 4x
Naive Cut | 47.18% | 23.23% | 13.8M | 4.4x
SVD [12] | 44.02% | 20.56% | 11.9M | 5x
Network Pruning | 42.77% | 19.67% | 6.7x | 9x