# Deep Compression: Compressing Deep Neural Networks with Pruning, Training Quantization and Huffman Coding

Song Han et al. Standford University Tsinghua University

## Abstract 摘要

Neural networks are both computationally intensive and memory intensive, making them difficult to deploy on embedded systems with limited hardware resources. To address this limitation, we introduce “deep compression”, a three stage pipeline: pruning, trained quantization and Huffman coding, that work together to reduce the storage requirement of neural networks by 35× to 49× without affecting their accuracy. Our method first prunes the network by learning only the important connections. Next, we quantize the weights to enforce weight sharing, finally, we apply Huffman coding. After the first two steps we retrain the network to fine tune the remaining connections and the quantized centroids. Pruning, reduces the number of connections by 9× to 13×; Quantization then reduces the number of bits that represent each connection from 32 to 5. On the ImageNet dataset, our method reduced the storage required by AlexNet by 35×, from 240MB to 6.9MB, without loss of accuracy. Our method reduced the size of VGG-16 by 49× from 552MB to 11.3MB, again with no loss of accuracy. This allows fitting the model into on-chip SRAM cache rather than off-chip DRAM memory. Our compression method also facilitates the use of complex neural networks in mobile applications where application size and download bandwidth are constrained. Benchmarked on CPU, GPU and mobile GPU, compressed network has 3× to 4× layerwise speedup and 3× to 7× better energy efficiency.

神经网络计算量和内存消耗都非常大，使其很难在嵌入式系统中部署。为解决这种局限性，我们提出深度压缩，这是三个步骤的过程：剪枝，量化训练和Huffman编码，三者一起降低了神经网络的大小达35x到49x，而且不影响准确率。我们的方法，首先学习重要的连接，来对网络进行修剪。下一步，我们对权重进行量化，以进行权重共享。最后，我们进行Huffman编码。在前两个步骤后，我们重新训练网络，以精调剩余的连接和量化的质心。剪枝降低了连接数量9x到13x；量化将表示每个连接的bit数从32降低到5。在ImageNet数据集上，我们的方法将AlexNet需要的存储空间降低了35x，从240MB降到了6.9MB，没有降低准确率。我们的方法将VGG-16的大小降低了49x，从552MB到11.3MB，准确率也没有损失。这使得模型可以放入片上SRAM缓存中，而不是片下的DRAM内存。我们的压缩方法方便了复杂神经网络的移动应用，因为移动应用对大小和下载带宽都有很多限制。在CPU、GPU和移动GPU上的基准测试，压缩的网络每层都加速了3x到4x，能耗降低了3x到7x。

## 1 Introduction 引言

Deep neural networks have evolved to the state-of-the-art technique for computer vision tasks (Krizhevsky et al., 2012)(Simonyan & Zisserman, 2014). Though these neural networks are very powerful, the large number of weights consumes considerable storage and memory bandwidth. For example, the AlexNet Caffemodel is over 200MB, and the VGG-16 Caffemodel is over 500MB (BVLC). This makes it difficult to deploy deep neural networks on mobile system.

DNN已经成为了计算机视觉中目前最好的技术。虽然这些神经网络能力很强，但大量权重消耗了相当多的存储空间和内存带宽。比如，AlexNet Caffemodel超过了200MB，VGG-16 Caffemodel超过了500MB(BVLC)。这使得在移动系统中部署DNN非常困难。

First, for many mobile-first companies such as Baidu and Facebook, various apps are updated via different app stores, and they are very sensitive to the size of the binary files. For example, App Store has the restriction “apps above 100 MB will not download until you connect to Wi-Fi”. As a result, a feature that increases the binary size by 100MB will receive much more scrutiny than one that increases it by 10MB. Although having deep neural networks running on mobile has many great features such as better privacy, less network bandwidth and real time processing, the large storage overhead prevents deep neural networks from being incorporated into mobile apps.

首先，对于很多移动优先的公司来说，如Baidu和Facebook，各种app都是通过不同的app store更新的，这对二进制文件的大小很敏感。比如，App Store有一个限制，超过100MB的app必须在连接Wi-Fi的情况下才能下载。结果是，超过100MB的这样的特征肯定会比小于10MB的受到更多审查。虽然DNN在移动端运行有很多很好的特征，如更好的隐私，更少的网络带宽和实时处理，但占用空间很大，会阻碍DNN在移动情况下的应用。

The second issue is energy consumption. Running large neural networks require a lot of memory bandwidth to fetch the weights and a lot of computation to do dot products— which in turn consumes considerable energy. Mobile devices are battery constrained, making power hungry applications such as deep neural networks hard to deploy.

第二个问题是能耗。运行大型神经网络需要大量内存带宽，以获取权重，也需要大量计算量进行点积运算，这也消耗了很多能量。移动设备受到电池容量限制，使得消耗很多电量的DNN难以部署。

Energy consumption is dominated by memory access. Under 45nm CMOS technology, a 32 bit floating point add consumes 0.9pJ, a 32bit SRAM cache access takes 5pJ, while a 32bit DRAM memory access takes 640pJ, which is 3 orders of magnitude of an add operation. Large networks do not fit in on-chip storage and hence require the more costly DRAM accesses. Running a 1 billion connection neural network, for example, at 20fps would require (20Hz)(1G)(640pJ) = 12.8W just for DRAM access - well beyond the power envelope of a typical mobile device.

能耗主要反应在内存访问上。在45nm CMOS技术下，32bit浮点加法消耗能力0.9pJ，32bit SRAM缓存访问消耗5pJ，而32bit DRAM内存访问消耗640pJ，这比加法运算高了三个量级。大型网络不能容纳于片上存储，因此需要能耗更多的DRAM访问。比如，以20Hz运行1 billion连接的神经网络，需要(20Hz)(1G)(640pJ) = 12.8W，这仅仅是DRAM访问，这远超过典型移动设备的能耗。

Our goal is to reduce the storage and energy required to run inference on such large networks so they can be deployed on mobile devices. To achieve this goal, we present “deep compression”: a three-stage pipeline (Figure 1) to reduce the storage required by neural network in a manner that preserves the original accuracy. First, we prune the networking by removing the redundant connections, keeping only the most informative connections. Next, the weights are quantized so that multiple connections share the same weight, thus only the codebook (effective weights) and the indices need to be stored. Finally, we apply Huffman coding to take advantage of the biased distribution of effective weights.

我们的目标是降低大型网络推理时的存储空间和能耗，使其可以在移动设备上部署。为达到这个目标，我们提出了“深度压缩”：包含三个步骤（图1），可以降低神经网络需要的存储空间，而且保持原始准确率。第一，我们通过移除冗余的连接，只保留信息量最多的连接，以对网络进行剪枝。下一步，权重进行量化，这样多个连接可以共享相同的权重，只需要保存codebook（有效权重）和索引。最后，由于有效权重的分布，我们可以对其使用Huffman编码。

Our main insight is that, pruning and trained quantization are able to compress the network without interfering each other, thus lead to surprisingly high compression rate. It makes the required storage so small (a few megabytes) that all weights can be cached on chip instead of going to off-chip DRAM which is energy consuming. Based on “deep compression”, the EIE hardware accelerator Han et al. (2016) was later proposed that works on the compressed model, achieving significant speedup and energy efficiency improvement.

我们的主要思想是，剪枝和量化训练可以压缩网络，而互不影响，所以可以得到很高的压缩率。压缩后所需的存储非常小（只有几MB），这样所有权重都可以进行片上存储，而不需要使用片下DRAM存储，片下存储消耗了很多能量。基于“深度压缩”，EIE硬件加速器也提出在压缩模型上工作，得到了明显了加速和能耗效率提高。

Figure 1: The three stage compression pipeline: pruning, quantization and Huffman coding. Pruning reduces the number of weights by 10×, while quantization further improves the compression rate: between 27× and 31×. Huffman coding gives more compression: between 35× and 49×. The compression rate already included the meta-data for sparse representation. The compression scheme doesn’t incur any accuracy loss.

## 2 Network Pruning 网络剪枝

Network pruning has been widely studied to compress CNN models. In early work, network pruning proved to be a valid way to reduce the network complexity and over-fitting (LeCun et al., 1989; Hanson & Pratt, 1989; Hassibi et al., 1993; Ström, 1997). Recently Han et al. (2015) pruned state-of-the-art CNN models with no loss of accuracy. We build on top of that approach. As shown on the left side of Figure 1, we start by learning the connectivity via normal network training. Next, we prune the small-weight connections: all connections with weights below a threshold are removed from the network. Finally, we retrain the network to learn the final weights for the remaining sparse connections. Pruning reduced the number of parameters by 9× and 13× for AlexNet and VGG-16 model.

网络剪枝以压缩CNN模型，已经得到了广泛的研究。在早期的工作中，网络剪枝已经被证明是一种有效的降低网络复杂度和过拟合的方法。最近，Han等对目前最好的CNN模型进行了修剪，而且没有损失准确率。我们的方法构建于这种方法之上。如图1的左边所示，我们开始通过正常的网络训练，学习连接性。下一步，我们修剪小权重的连接；所有权重低于某一阈值的连接都被从网络中移除了。最后，我们重新训练网络，为剩余的稀疏连接学习最终的权重。通过剪枝，可以降低AlexNet和VGG-16模型参数的9x和13x。

We store the sparse structure that results from pruning using compressed sparse row (CSR) or compressed sparse column (CSC) format, which requires 2a + n + 1 numbers, where a is the number of non-zero elements and n is the number of rows or columns.

我们将剪枝得到的结果，使用压缩稀疏行(Compressed Sparse Row, CSR)或压缩稀疏列(Compressed Sparse Column, CSC)进行存储，这需要2a+n+1个数，其中a是非零元素的数量，n是行或列的数量。

To compress further, we store the index difference instead of the absolute position, and encode this difference in 8 bits for conv layer and 5 bits for fc layer. When we need an index difference larger than the bound, we the zero padding solution shown in Figure 2: in case when the difference exceeds 8, the largest 3-bit (as an example) unsigned number, we add a filler zero.

为进一步压缩，我们存储索引之差，而不是其绝对位置，在卷积层中以8 bits编码其差值，在全连接层中，以5 bit编码其差值。当索引差值大于这个界限时，我们使用图2中的补零解决方案；在差值超过8（最大的3 bit无符号数值）的情况下，我们增加一个填充的0。

Figure 2: Representing the matrix sparsity with relative index. Padding filler zero to prevent overflow.

## 3 Trained Quantization and Weight Sharing 训练量化和权重共享

Network quantization and weight sharing further compresses the pruned network by reducing the number of bits required to represent each weight. We limit the number of effective weights we need to store by having multiple connections share the same weight, and then fine-tune those shared weights.

网络量化和权重共享，通过降低表示每个权重所需的bit数，进一步压缩了剪枝的网络。我们使多个连接共享同样的权重，从而限制了需要存储的有效权重的数量，然后对这些共享的权重进行精调。

Weight sharing is illustrated in Figure 3. Suppose we have a layer that has 4 input neurons and 4 output neurons, the weight is a 4 × 4 matrix. On the top left is the 4 × 4 weight matrix, and on the bottom left is the 4 × 4 gradient matrix. The weights are quantized to 4 bins (denoted with 4 colors), all the weights in the same bin share the same value, thus for each weight, we then need to store only a small index into a table of shared weights. During update, all the gradients are grouped by the color and summed together, multiplied by the learning rate and subtracted from the shared centroids from last iteration. For pruned AlexNet, we are able to quantize to 8-bits (256 shared weights) for each CONV layers, and 5-bits (32 shared weights) for each FC layer without any loss of accuracy.

权重共享如图3所示。假设我们有一个层，有4个输入神经元，4个输出神经元，权重为4×4的矩阵。左上是4×4的权重矩阵，左下是4×4的梯度矩阵。权重量化成4级（用4种颜色表示），在相同级上的所有权重共享相同的值，所以对于每个权重，我们只需要在这个共享权重的表中存储一个小的索引。在更新时，所有梯度也用颜色进行归类，然后相加，乘以学习速率，减去上一次迭代的共享重心。对于剪枝的AlexNet，我们可以对每个CONV层量化到8-bits（256个共享权重），对每个FC层量化到5-bit（32个共享权重），而不损失准确率。

Figure 3: Weight sharing by scalar quantization (top) and centroids fine-tuning (bottom).

To calculate the compression rate, given k clusters, we only need $log_2 (k)$ bits to encode the index. In general, for a network with n connections and each connection is represented with b bits, constraining the connections to have only k shared weights will result in a compression rate of: 为计算压缩率，给定k个聚类，我们只需要$log_2 (k)$ bits来对索引进行编码。一般来说，对于有n个连接的网络，每个连接用b bit表示，限制连接只有k个共享连接，得到的压缩率为：

$$r = \frac {nb} {nlog_2 (k) + kb}$$(1)

For example, Figure 3 shows the weights of a single layer neural network with four input units and four output units. There are 4 × 4 = 16 weights originally but there are only 4 shared weights: similar weights are grouped together to share the same value. Originally we need to store 16 weights each has 32 bits, now we need to store only 4 effective weights (blue, green, red and orange), each has 32 bits, together with 16 2-bit indices giving a compression rate of 16 ∗ 32/(4 ∗ 32 + 2 ∗ 16) = 3.2.

例如，图3展示了一个神经网络层的权重，层有4个输入单元，4个输出单元。原来有16个权重，但现在4个共享权重：类似的权重被分组到一起，共享相同的值。原来我们需要存储16个权重，每个32 bits，现在我们只需要存储4个有效权重（蓝色，绿色，红色和橙色），每个32 bits，与16个2 bit的索引一起，压缩率为16 ∗ 32/(4 ∗ 32 + 2 ∗ 16) = 3.2。

### 3.1 Weight Sharing

We use k-means clustering to identify the shared weights for each layer of a trained network, so that all the weights that fall into the same cluster will share the same weight. Weights are not shared across layers. We partition n original weights W = {$w_1, w_2, ..., w_n$} into k clusters C = {$c_1, c_2, ..., c_k$}, n >> k, so as to minimize the within-cluster sum of squares (WCSS):

我们使用k均值聚类法，对每一层训练好的网络找出共享权重，所有落入同一聚类的权重会共享相同的权重。不同层的权重不共享。我们将n个原始权重W = {$w_1, w_2, ..., w_n$}聚成k类C = {$c_1, c_2, ..., c_k$}, n >> k, 这样可以最小化类内平方和(WCSS)：

$$argmin_C \sum_{i=1}^k \sum_{w∈c_i} |w-c_i|^2$$(2)

Different from HashNet (Chen et al., 2015) where weight sharing is determined by a hash function before the networks sees any training data, our method determines weight sharing after a network is fully trained, so that the shared weights approximate the original network.

HashNet中，权重共享是由hash函数在网络尚未进行训练时确定的，我们与之不同，是在网络经过训练之后确定的共享权重，这样共享的权重可以近似原网络。

### 3.2 Initialization of Shared Weights

Centroid initialization impacts the quality of clustering and thus affects the network’s prediction accuracy. We examine three initialization methods: Forgy(random), density-based, and linear initialization. In Figure 4 we plotted the original weights’ distribution of conv3 layer in AlexNet (CDF in blue, PDF in red). The weights forms a bimodal distribution after network pruning. On the bottom it plots the effective weights (centroids) with 3 different initialization methods (shown in blue, red and yellow). In this example, there are 13 clusters.

重心初始化影响聚类的质量，所以影响网络的预测准确率。我们检验三种初始化方法：Forgy（随机），基于密度的，和线性初始化。在图4中，我们画出AlexNet的conv3原始权重的分布(CDF in blue, PDF in red)。权重在剪枝过后，形成了双峰分布。在底部，画出了三种不同初始化方法的有效权重（重心），分别以蓝红黄表示。本例子中，有13个聚类。 CDF: Cumulative Distribution Function; PDF: Probability Density Function.

Figure 4: Left: Three different methods for centroids initialization. Right: Distribution of weights (blue) and distribution of codebook before (green cross) and after fine-tuning (red dot).

Forgy (random) initialization randomly chooses k observations from the data set and uses these as the initial centroids. The initialized centroids are shown in yellow. Since there are two peaks in the bimodal distribution, Forgy method tend to concentrate around those two peaks.

随机初始化，从数据集中随机选取k个观测，将其用作初始重心。初始化的重心用黄色表示。由于双峰分布中有两个峰值，随机方法倾向于在这两个峰值附近聚积。

Density-based initialization linearly spaces the CDF of the weights in the y-axis, then finds the horizontal intersection with the CDF, and finally finds the vertical intersection on the x-axis, which becomes a centroid, as shown in blue dots. This method makes the centroids denser around the two peaks, but more scatted than the Forgy method.

基于密度的初始化对权重的CDF沿着y轴进行线性分隔，然后找到CDF的水平分隔，最后找到在x轴上的垂直分隔，成为重心，以蓝色的点显式出来。这种方法使得重心在双峰处更密集，但比随机初始化方法更分散。

Linear initialization linearly spaces the centroids between the [min, max] of the original weights. This initialization method is invariant to the distribution of the weights and is the most scattered compared with the former two methods.

线性初始化，在原始权重的取值范围[min, max]中进行重心线性分隔。这种初始化方法对权重的分布是不变的，与前两种方法相比，是最分散的。

Larger weights play a more important role than smaller weights (Han et al., 2015), but there are fewer of these large weights. Thus for both Forgy initialization and density-based initialization, very few centroids have large absolute value which results in poor representation of these few large weights. Linear initialization does not suffer from this problem. The experiment section compares the accuracy of different initialization methods after clustering and fine-tuning, showing that linear initialization works best.

大的权重比小的权重更重要，但大的权重很少。因此，对于随机初始化和基于密度的初始化来说，很少重心的绝对值是大的，这会导致很少的大权重的很差的表示。线性初始化就没有这个问题。试验部分在聚类和精调后，比较了不同初始化方法的准确率，表明线性初始化效果最好。

### 3.3 Feed-Forward and Back-Propogation 前向和反向

The centroids of the one-dimensional k-means clustering are the shared weights. There is one level of indirection during feed forward phase and back-propagation phase looking up the weight table. An index into the shared weight table is stored for each connection. During back-propagation, the gradient for each shared weight is calculated and used to update the shared weight. This procedure is shown in Figure 3.

一维k均值聚类得到的重心就是共享的权重。在前向阶段和反向传播阶段，查询这些权重表时，有一些迂回。每个连接都会存储一个共享权重表的索引。在反向传播时，每个共享权重的梯度都会得到计算并用于更新共享权重。这个过程如图3所示。

We denote the loss by L, the weight in the ith column and jth row by $W_{ij}$, the centroid index of element $W_{i,j}$ by $I_{ij}$, the kth centroid of the layer by $C_k$. By using the indicator function 1(.), the gradient of the centroids is calculated as:

我们设损失函数为L，第i行第j列的权重为by $W_{ij}$，元素$W_{i,j}$的重心索引为$I_{ij}$，层的第k个重心为$C_k$。使用指示器函数1(.)，重心的梯度计算为：

$$\frac {∂L}{∂C_k} = \sum_{i,j} \frac {∂L}{∂W_{ij}} \frac {∂W_{ij}}{∂C_k} = \sum_{i,j} \frac {∂L} {∂W_{ij}} 1(I_{ij} = k)$$(3)

## 4 Huffman Coding

A Huffman code is an optimal prefix code commonly used for lossless data compression(Van Leeuwen, 1976). It uses variable-length codewords to encode source symbols. The table is derived from the occurrence probability for each symbol. More common symbols are represented with fewer bits.

Huffman编码是一种最有的前缀编码，常用于无损数据压缩。它使用变长码词来编码源符号。通过每个符号的出现概率推理得到表格。更常用的符号，使用更少的bits来表示。

Figure 5 shows the probability distribution of quantized weights and the sparse matrix index of the last fully connected layer in AlexNet. Both distributions are biased: most of the quantized weights are distributed around the two peaks; the sparse matrix index difference are rarely above 20. Experiments show that Huffman coding these non-uniformly distributed values saves 20% − 30% of network storage.

表5给出了，AlexNet最后一个全连接层中，量化后权重的概率分布，和对应的稀疏矩阵索引。两种分布都是有偏的：多数量化后的权重都在两个峰值附近分布；稀疏矩阵的索引差值很少大于20。试验表明，对这些非均匀分布的值进行Huffman编码会减少20%-30%的网络存储。

## 5 Experiments

We pruned, quantized, and Huffman encoded four networks: two on MNIST and two on ImageNet data-sets. The network parameters and accuracy - (Reference model is from Caffe model zoo, accuracy is measured without data augmentation) before and after pruning are shown in Table 1. The compression pipeline saves network storage by 35× to 49× across different networks without loss of accuracy. The total size of AlexNet decreased from 240MB to 6.9MB, which is small enough to be put into on-chip SRAM, eliminating the need to store the model in energy-consuming DRAM memory.

我们对四个网络进行剪枝、量化和Huffman编码：两个在MNIST上，两个在ImageNet数据集上。剪枝前后的网络参数和准确率如表1所示（参考模型是在Caffe模型库中的，准确率是在没有数据扩充的情况下得到的）。压缩过程使网络存储降低了35x到49x，准确率则没有损失。AlexNet的总大小从240MB降到了9.6MB，这种大小已经可以放出片上SRAM中，不需要将模型存储在能耗很大的DRAM存储中。

Training is performed with the Caffe framework (Jia et al., 2014). Pruning is implemented by adding a mask to the blobs to mask out the update of the pruned connections. Quantization and weight sharing are implemented by maintaining a codebook structure that stores the shared weight, and group-by-index after calculating the gradient of each layer. Each shared weight is updated with all the gradients that fall into that bucket. Huffman coding doesn’t require training and is implemented offline after all the fine-tuning is finished.

训练是在Caffe框架下进行的。剪枝的实现是对blobs增加了一个掩膜，将修剪掉的连接遮挡掉。量化和权重共享是通过一个代码本结构实现的，保存着共享的权重，在计算好每层的梯度后，以索引进行分组。每个共享的权重用落入这个组的梯度进行更新。Huffman编码不需要训练，是在所有精调完成后离线实现的。

Table 1: The compression pipeline can save 35× to 49× parameter storage with no loss of accuracy.

Network | Top-1 Error | Top-5 Error | Parameters | Compress. Rate
--- | --- | --- | --- | ---
LeNet-300-100 Ref | 1.64% | - | 1070KB | -
LeNet-300-100 Compressed | 1.58% | - | 27KB | 40x
LeNet-5 Ref | 0.80% | - | 1720KB | -
LeNet-5 Compressed | 0.74% | - | 44KB | 39x
AlexNet Ref | 42.78% | 19.73% | 240MB | -
AlexNet Compressed | 42.78% | 19.70% | 6.9MB | 35x
VGG-16 Ref | 31.50% | 11.32% | 552MB | -
VGG-16 Compressed | 31.17% | 10.91% | 11.3MB | 49x

### 5.1 LeNet-300-100 and LeNet-5 on MNIST

We first experimented on MNIST dataset with LeNet-300-100 and LeNet-5 network (LeCun et al., 1998). LeNet-300-100 is a fully connected network with two hidden layers, with 300 and 100 neurons each, which achieves 1.6% error rate on Mnist. LeNet-5 is a convolutional network that has two convolutional layers and two fully connected layers, which achieves 0.8% error rate on Mnist. Table 2 and table 3 show the statistics of the compression pipeline. The compression rate includes the overhead of the codebook and sparse indexes. Most of the saving comes from pruning and quantization (compressed 32×), while Huffman coding gives a marginal gain (compressed 40×).

我们首先在MNIST上对LeNet-300-100和LeNet-5网络进行试验。LeNet-300-100是一个全连接网络，有2个隐藏层，各有300和100个神经元，在MNIST上分类错误率1.6%。LeNet-5是一个卷积网络，有2个卷积层和2个全连接层，在MNIST上得到了0.8%的分类错误率。表2和表3给出了压缩过程的统计数据。压缩率包括了代码本和稀疏索引的开销。压缩的绝大部分来自于剪枝和量化（压缩了32x），而Huffman编码得到了一些收益（压缩了40x）。

Table 2: Compression statistics for LeNet-300-100. P: pruning, Q:quantization, H:Huffman coding.

Table 3: Compression statistics for LeNet-5. P: pruning, Q:quantization, H:Huffman coding.

### 5.2 AlexNet on ImageNet

We further examine the performance of Deep Compression on the ImageNet ILSVRC-2012 dataset, which has 1.2M training examples and 50k validation examples. We use the AlexNet Caffe model as the reference model, which has 61 million parameters and achieved a top-1 accuracy of 57.2% and a top-5 accuracy of 80.3%. Table 4 shows that AlexNet can be compressed to 2.88% of its original size without impacting accuracy. There are 256 shared weights in each CONV layer, which are encoded with 8 bits, and 32 shared weights in each FC layer, which are encoded with only 5 bits. The relative sparse index is encoded with 4 bits. Huffman coding compressed additional 22%, resulting in 35× compression in total.

我们进一步研究在ImageNet ILSVRC-2012数据集上深度压缩的效果，这个数据集包含1.2M训练样本，和50k验证样本。我们使用AlexNet Caffe模型作为参考模型，有61 million参数，其top-1准确率为57.2%，top-5准确率为80.3%。表4说明，AlexNet可以压缩到其原始大小的2.88%，而且不损失准确率。在每个CONV层中，有256个共享的权重，编码为8 bits，在每个FC层中有32个共享的权重，编码为5 bits。相关的稀疏矩阵索引编码为4 bits。Huffman编码另外压缩了22%的空间，得到了总计35x的压缩率。

Table 4: Compression statistics for AlexNet. P: pruning, Q: quantization, H:Huffman coding.

### 5.3 VGG-16 on ImageNet

With promising results on AlexNet, we also looked at a larger, more recent network, VGG-16 (Simonyan & Zisserman, 2014), on the same ILSVRC-2012 dataset. VGG-16 has far more convolutional layers but still only three fully-connected layers. Following a similar methodology, we aggressively compressed both convolutional and fully-connected layers to realize a significant reduction in the number of effective weights, shown in Table5.

在AlexNet上得到了很好的结果后，我们还在最近更大的VGG-16网络上进行了试验。VGG-16有更多的卷积层，但还是三个全连接层。采用类似的方法，我们压缩了卷积层和全连接层，在有效权重上进行了明显的压缩，如表5所示。

The VGG16 network as a whole has been compressed by 49×. Weights in the CONV layers are represented with 8 bits, and FC layers use 5 bits, which does not impact the accuracy. The two largest fully-connected layers can each be pruned to less than 1.6% of their original size. This reduction is critical for real time image processing, where there is little reuse of these layers across images (unlike batch processing). This is also critical for fast object detection algorithms where one CONV pass is used by many FC passes. The reduced layers will fit in an on-chip SRAM and have modest bandwidth requirements. Without the reduction, the bandwidth requirements are prohibitive.

VGG-16网络整体上被压缩了49x。CONV层的权重表示为8 bits，FC层表示为5 bits，而且不影响准确率。最大的两个全连接层可以修剪到原始大小的不到1.6%。这种压缩对于实时图像处理是很关键的，其中在图像之间对这些层的重用非常之少（与批处理不一样）。这对于快速目标检测算法也很关键，其中一个卷积过程用于几个FC过程。压缩的层可以放入片上SRAM执行，带宽需求也很小。在没有压缩的情况下，带宽需求就非常大了。

Table 5: Compression statistics for VGG-16. P: pruning, Q:quantization, H:Huffman coding.

## 6 Discussion

### 6.1 Pruning and Quantization Working together

Figure 6 shows the accuracy at different compression rates for pruning and quantization together or individually. When working individually, as shown in the purple and yellow lines, accuracy of pruned network begins to drop significantly when compressed below 8% of its original size; accuracy of quantized network also begins to drop significantly when compressed below 8% of its original size. But when combined, as shown in the red line, the network can be compressed to 3% of original size with no loss of accuracy. On the far right side compared the result of SVD, which is inexpensive but has a poor compression rate.

图6给出了剪枝和量化一起/单独进行时，不同压缩率下的准确率。当单独进行时，如紫色和黄色线条所示，剪枝网络的准确率在压缩到原始网络的8%以下时，准确率就会显著下降；量化网络的准确率，在压缩到原始大小的8%以后，就会显著下降。但结合到一起后，如红线所示，网络可以压缩到原始大小的3%，而准确率也不下降。在图中最右边，与SVD也进行了比较，这是一种简单的压缩方法，压缩率很低。

Figure 6: Accuracy v.s. compression rate under different compression methods. Pruning and quantization works best when combined.

The three plots in Figure 7 show how accuracy drops with fewer bits per connection for CONV layers (left), FC layers (middle) and all layers (right). Each plot reports both top-1 and top-5 accuracy. Dashed lines only applied quantization but without pruning; solid lines did both quantization and pruning. There is very little difference between the two. This shows that pruning works well with quantization.

图7中的三幅图，展示了在每个连接使用更少的bits表示时，准确率下降有多少，包含卷积层（左），全连接层（中）和所有层（右）。每幅图都给出top-1和top-5准确率的情况。虚线只进行了量化，没有剪枝；实线同时进行量化和剪枝。在这两者之间几乎没有区别。这说明，剪枝与量化一起使用也很好。

Figure 7: Pruning doesn’t hurt quantization. Dashed: quantization on unpruned network. Solid: quantization on pruned network; Accuracy begins to drop at the same number of quantization bits whether or not the network has been pruned. Although pruning made the number of parameters less, quantization still works well, or even better(3 bits case on the left figure) as in the unpruned network.

Quantization works well on pruned network because unpruned AlexNet has 60 million weights to quantize, while pruned AlexNet has only 6.7 million weights to quantize. Given the same amount of centroids, the latter has less error.

量化在剪枝后的网络中也可以很好工作，这是因为未剪枝的AlexNet有60 million权重要量化，而剪枝后的AlexNet只有6.7 million权重需要量化。在同样数量的重心下，后者错误率更小。

The first two plots in Figure 7 show that CONV layers require more bits of precision than FC layers. For CONV layers, accuracy drops significantly below 4 bits, while FC layer is more robust: not until 2 bits did the accuracy drop significantly.

图7中的前两幅图说明，CONV层比FC层需要更多bits的精度。对于CONV层，准确率在4 bits以下准确率显著下降，而FC层则更稳健：直到2 bits以下，准确率才开始显著下降。

### 6.2 Centroid Initalization 重心初始化

Figure 8 compares the accuracy of the three different initialization methods with respect to top-1 accuracy (Left) and top-5 accuracy (Right). The network is quantized to 2 ∼ 8 bits as shown on x-axis. Linear initialization outperforms the density initialization and random initialization in all cases except at 3 bits.

图8比较了三种不同初始化方法下的准确率，包括top-1（左）和top-5（右）准确率。网络量化成2-8 bits，如x轴所示。在所有情况下（除了3 bits），线性初始化超过了基于密度的初始化和随机初始化。

Figure 8: Accuracy of different initialization methods. Left: top-1 accuracy. Right: top-5 accuracy. Linear initialization gives best result.

The initial centroids of linear initialization spread equally across the x-axis, from the min value to the max value. That helps to maintain the large weights as the large weights play a more important role than smaller ones, which is also shown in network pruning Han et al. (2015). Neither random nor density-based initialization retains large centroids. With these initialization methods, large weights are clustered to the small centroids because there are few large weights. In contrast, linear initialization allows large weights a better chance to form a large centroid.

线性初始化的初始重心在x轴上从最小值到最大值分布很均匀。这有助于保持值较大的权重，因为大的权重比更小的作用更大，这在网络剪枝中也有体现。随机初始化和基于密度的初始化都没有得到很大的重心。在这些初始化方法下，大的权重分类到了小的重心处，因为大的权重太少了。比较之下，线性初始化的情况下，大的权重更容易形成大的重心。

### 6.3 Speedup and Energy Efficiency

Deep Compression is targeting extremely latency-focused applications running on mobile, which requires real-time inference, such as pedestrian detection on an embedded processor inside an autonomous vehicle. Waiting for a batch to assemble significantly adds latency. So when benchmarking the performance and energy efficiency, we consider the case when batch size = 1. The cases of batching are given in Appendix A.

深度压缩的目标是极度关注延迟的应用，运行在移动平台，需要实时推理，比如自动驾驶交通工具中的嵌入式设备上的行人检测。等待一个批次一起输入，会明显增加延迟。所以在进行性能和能耗的基准测试时，我们考虑batch size=1的情况。在附录A中给出了其他batching的情况。

Fully connected layer dominates the model size (more than 90%) and got compressed the most by Deep Compression (96% weights pruned in VGG-16). In state-of-the-art object detection algorithms such as fast R-CNN (Girshick, 2015), upto 38% computation time is consumed on FC layers on uncompressed model. So it’s interesting to benchmark on FC layers, to see the effect of Deep Compression on performance and energy. Thus we setup our benchmark on FC6, FC7, FC8 layers of AlexNet and VGG-16. In the non-batched case, the activation matrix is a vector with just one column, so the computation boils down to dense / sparse matrix-vector multiplication for original / pruned model, respectively. Since current BLAS library on CPU and GPU doesn’t support indirect look-up and relative indexing, we didn’t benchmark the quantized model.

全连接层是模型规模的主要部分（超过90%），通过深度压缩可以得到最好的效果（在VGG-16中96%的权重都修剪掉了）。在目前最好的目标检测算法中，如R-CNN，超过38%的时间是在未压缩模型的FC层进行的。所以在FC层上进行基准测试是非常有趣的，观察一下深度压缩在性能和能耗上的效果。所以我们在AlexNet和VGG-16中的FC6、FC7和FC8层中进行基准测试。在非批次的情况下，激活矩阵是一个向量，只有一列，所以原始/剪枝后的模型的计算分别归结为稠密/稀疏矩阵-向量乘积。由于目前在CPU和GPU上的BLAS库不支持间接查询和相对索引，所以我们没有对量化模型进行基准测试。

We compare three different off-the-shelf hardware: the NVIDIA GeForce GTX Titan X and the Intel Core i7 5930K as desktop processors (same package as NVIDIA Digits Dev Box) and NVIDIA Tegra K1 as mobile processor. To run the benchmark on GPU, we used cuBLAS GEMV for the original dense layer. For the pruned sparse layer, we stored the sparse matrix in in CSR format, and used cuSPARSE CSRMV kernel, which is optimized for sparse matrix-vector multiplication on GPU. To run the benchmark on CPU, we used MKL CBLAS GEMV for the original dense model and MKL SPBLAS CSRMV for the pruned sparse model.

我们比较了三种不同的硬件平台：NVidia GeForce GTX Titan X和Intel Core i7 5930K作为桌面处理器（与NVidia Digits Dev Box相同），NVidia Tegra K1作为移动处理器。为在GPU上运行基准测试，我们为原始的密集层使用cuBLAS GEMV。对于修剪过的稀疏层，我们将稀疏矩阵存储为CSR格式，并使用cuSPARSE CSRMV核，这个库为GPU上的稀疏矩阵-向量乘积进行了优化。为在CPU上运行基准测试，我们为原始的密集模型使用MKL CBLAS GEMV，为修剪过的稀疏模型使用MKL SPBLAS CSRMV。

To compare power consumption between different systems, it is important to measure power at a consistent manner (NVIDIA, b). For our analysis, we are comparing pre-regulation power of the entire application processor (AP) / SOC and DRAM combined. On CPU, the benchmark is running on single socket with a single Haswell-E class Core i7-5930K processor. CPU socket and DRAM power are as reported by the pcm-power utility provided by Intel. For GPU, we used nvidia-smi utility to report the power of Titan X. For mobile GPU, we use a Jetson TK1 development board and measured the total power consumption with a power-meter. We assume 15% AC to DC conversion loss, 85% regulator efficiency and 15% power consumed by peripheral components (NVIDIA, a) to report the AP+DRAM power for Tegra K1.

为计算不同系统下的能耗，以统一的方式测量能耗是非常重要的。对于我们的分析，我们比较的是规范化之前的整个应用处理器/SOC的能耗和DRAM的和。在CPU上，基准测试是用单socket单Haswell-E类的i7-5930核的处理器。CPU socket和DRAM能耗由于Intel给的pcm-power utility。对于GPU，我们使用nvidia-smi utility给出Titan X的能耗。对于移动GPU，我们使用一个Jetson TK1开发板，使用一个能量计测量总计能耗。我们假设对于Tegra K1，AC到DC转换损失效率15%, regulator效率85%，15%的能耗是在周边器件上。

The ratio of memory access over computation characteristic with and without batching is different. When the input activations are batched to a matrix the computation becomes matrix-matrix multiplication, where locality can be improved by blocking. Matrix could be blocked to fit in caches and reused efficiently. In this case, the amount of memory access is O($n^2$), and that of computation is O($n^3$), the ratio between memory access and computation is in the order of 1/n.

内存访问与计算的比率，在有没有batching的情况下是不一样的。当输入激活是成批的，形成一个矩阵时，计算变成了矩阵-矩阵的乘积，其中局部性可以通过blocking改进。矩阵可以通过blocked以放入缓存中，进行高效的重用。在这种情况下，内存访问的数量为O($n^2$)，计算量则为O($n^3$)，内存访问与计算之间的比率为1/n。

In real time processing when batching is not allowed, the input activation is a single vector and the computation is matrix-vector multiplication. In this case, the amount of memory access is O($n^2$), and the computation is O($n^2$), memory access and computation are of the same magnitude (as opposed to 1/n). That indicates MV is more memory-bounded than MM. So reducing the memory footprint is critical for the non-batching case.

在实时处理的时候，则没有了batching，输入激活是单个向量，计算变成了矩阵-向量乘积。在这种情况下，内存访问的数量是O($n^2$)，计算量也是O($n^2$)，内存访问与计算是同样数量级的（与前面的1/n相比）。这说明MV与MM相比，更受到内存限制。所以对于非批次的情况，降低内存空间是非常关键的。

Figure 9 illustrates the speedup of pruning on different hardware. There are 6 columns for each benchmark, showing the computation time of CPU / GPU / TK1 on dense / pruned network. Time is normalized to CPU. When batch size = 1, pruned network layer obtained 3× to 4× speedup over the dense network on average because it has smaller memory footprint and alleviates the data transferring overhead, especially for large matrices that are unable to fit into the caches. For example VGG16’s FC6 layer, the largest layer in our experiment, contains 25088 × 4096 × 4 Bytes ≈ 400M B data, which is far from the capacity of L3 cache.

图9所示的是，在不同硬件上剪枝的加速效果。每个基准测试中有6列，表示密集/剪枝网络在CPU/GPU/TK1上的计算时间。batch size = 1， 时间以CPU上的计算时间进行归一化，剪枝的网络比密集网络平均取得了3x到4x的加速，因为占用内存更少，缓解了数据迁移的消耗，尤其是对于大型矩阵，无法在缓存中保存的情况。比如VGG16的FC6层，这是我们试验中最大的层，包含25088×4096×4 Bytes，约为400MB数据，这对于L3缓存来说太大了。

Figure 9: Compared with the original network, pruned network layer achieved 3× speedup on CPU, 3.5× on GPU and 4.2× on mobile GPU on average. Batch size = 1 targeting real time processing. Performance number normalized to CPU.

In those latency-tolerating applications, batching improves memory locality, where weights could be blocked and reused in matrix-matrix multiplication. In this scenario, pruned network no longer shows its advantage. We give detailed timing results in Appendix A.

在对于延迟有容忍度的应用中，batching改进了内存本地性(locality)，其中权重可以分块(blocked)，在矩阵-矩阵乘法中重用。在这种情况下，剪枝的网络显示不出其优势。我们在附录A中给出详细的计时结果。

Figure 10 illustrates the energy efficiency of pruning on different hardware. We multiply power consumption with computation time to get energy consumption, then normalized to CPU to get energy efficiency. When batch size = 1, pruned network layer consumes 3× to 7× less energy over the dense network on average. Reported by nvidia-smi, GPU utilization is 99% for both dense and sparse cases.

图10描述的是在不同硬件下，剪枝带来的功耗效率。我们将功耗与计算时间相乘，得到消耗的能量，然后对CPU的情况进行归一化，得到能耗效率。在batch size = 1时，剪枝的网络消耗的能量比密集网络平均少3x到7x。根据nvidia-smi给出的结果，GPU利用率在密集和稀疏的情况下都是99%。

Figure 10: Compared with the original network, pruned network layer takes 7× less energy on CPU, 3.3× less on GPU and 4.2× less on mobile GPU on average. Batch size = 1 targeting real time processing. Energy number normalized to CPU.

### 6.4 Ratio of Weights, Index and Codebook

Pruning makes the weight matrix sparse, so extra space is needed to store the indexes of non-zero elements. Quantization adds storage for a codebook. The experiment section has already included these two factors. Figure 11 shows the breakdown of three different components when quantizing four networks. Since on average both the weights and the sparse indexes are encoded with 5 bits, their storage is roughly half and half. The overhead of codebook is very small and often negligible.

剪枝使得权重矩阵稀疏，所以需要额外的空间来存储非零元素的索引。量化增加了代码本的存储。试验部分对这两部分进行了试验。图11给出了在量化四种网络的时候，这三个不同的部分的分解图。由于平均起来，权重和稀疏索引都采用5 bits进行编码，所以其存储大概是一半对一半。代码本的存储非常小，可以忽略。

Figure 11: Storage ratio of weight, index and codebook.

## 7 Related Work

Neural networks are typically over-parametrized, and there is significant redundancy for deep learning models(Denil et al., 2013). This results in a waste of both computation and memory usage. There have been various proposals to remove the redundancy: Vanhoucke et al. (2011) explored a fixed-point implementation with 8-bit integer (vs 32-bit floating point) activations. Hwang & Sung (2014) proposed an optimization method for the fixed-point network with ternary weights and 3-bit activations. Anwar et al. (2015) quantized the neural network using L2 error minimization and achieved better accuracy on MNIST and CIFAR-10 datasets. Denton et al. (2014) exploited the linear structure of the neural network by finding an appropriate low-rank approximation of the parameters and keeping the accuracy within 1% of the original model.

神经网络通常参数都过多，有明显的冗余。这浪费了计算量和存储空间。有很多去除冗余的工作：Vanhoucke等探索了激活的8-bit定点实现（vs 32-bit浮点实现）。Hwang等提出定点网络的优化实现。Anwar等使用L2误差最小化量化神经网络，在MNIST和CIFAR-10上得到了更好的准确率。Denton等探索了神经网络的线性结构，找到了参数的合适低秩近似，准确率比原模型的降低不超过1%。

The empirical success in this paper is consistent with the theoretical study of random-like sparse networks with +1/0/-1 weights (Arora et al., 2014), which have been proved to enjoy nice properties (e.g. reversibility), and to allow a provably polynomial time algorithm for training.

本文的成功在经验上是与理论研究是一致的，即Arora等人的随机稀疏网络，权重为+1/0/-1，已经证明有很优美的性质（如，reversibility），训练的时间是可证明的多项式时间。

Much work has been focused on binning the network parameters into buckets, and only the values in the buckets need to be stored. HashedNets(Chen et al., 2015) reduce model sizes by using a hash function to randomly group connection weights, so that all connections within the same hash bucket share a single parameter value. In their method, the weight binning is pre-determined by the hash function, instead of being learned through training, which doesn’t capture the nature of images. Gong et al. (2014) compressed deep convnets using vector quantization, which resulted in 1% accuracy loss. Both methods studied only the fully connected layer, ignoring the convolutional layers.

很多工作关注的是将网络参数聚类到不同的bin中，只有在bucket中的值需要存储。HashedNets使用一个hash函数来对连接权重随机分组，从而降低模型大小，所以在同一hash bucket中的所有连接共享一个参数值。在他们的方法中，权重binning是由hash函数预先确定的，而不是通过训练学习到的，所以没有捕获到图像的本质。Gong等使用矢量量化压缩了深度卷积网络，准确率损失不超过1%。这两种方法都只研究了全连接层，忽略了卷积层。

There have been other attempts to reduce the number of parameters of neural networks by replacing the fully connected layer with global average pooling. The Network in Network architecture(Lin et al., 2013) and GoogLenet(Szegedy et al., 2014) achieves state-of-the-art results on several benchmarks by adopting this idea. However, transfer learning, i.e. reusing features learned on the ImageNet dataset and applying them to new tasks by only fine-tuning the fully connected layers, is more difficult with this approach. This problem is noted by Szegedy et al. (2014) and motivates them to add a linear layer on the top of their networks to enable transfer learning.

还有其他降低参数数量的尝试，如将全连接层替换成全局平均池化。Network in Network架构和GoogLenet采用这种思想在几个基准测试中取得了目前最好的结果。但是，迁移学习，即重用在ImageNet上学习到的特征，用于其他新的任务，只精调全连接层，采用这种方法则非常困难。Szegedy等指出了这个问题，并在他们的网络上增加了一个线性层，以进行迁移学习。

Network pruning has been used both to reduce network complexity and to reduce over-fitting. An early approach to pruning was biased weight decay (Hanson & Pratt, 1989). Optimal Brain Damage (LeCun et al., 1989) and Optimal Brain Surgeon (Hassibi et al., 1993) prune networks to reduce the number of connections based on the Hessian of the loss function and suggest that such pruning is more accurate than magnitude-based pruning such as weight decay. A recent work (Han et al., 2015) successfully pruned several state of the art large scale networks and showed that the number of parameters could be reduce by an order of magnitude. There are also attempts to reduce the number of activations for both compression and acceleration Van Nguyen et al. (2015).

网络剪枝曾被用于降低网络复杂度，和降低过拟合。剪枝的一个早期方法是biased weight decay。Optimal Brain Damage和Optimal Brain Surgeon，基于损失函数的Hessian矩阵，来降低连接的数量，以修剪网络，得出的结论是这样的修剪比基于幅度的修剪如weight decay更加准确。Han等最近的工作成功的修剪了几个目前最好的大型网络，显示参数数量可以降低一个数量级。也有降低激活数量的尝试，以进行压缩和加速。

## 8 Future Work

While the pruned network has been benchmarked on various hardware, the quantized network with weight sharing has not, because off-the-shelf cuSPARSE or MKL SPBLAS library does not support indirect matrix entry lookup, nor is the relative index in CSC or CSR format supported. So the full advantage of Deep Compression that fit the model in cache is not fully unveiled. A software solution is to write customized GPU kernels that support this. A hardware solution is to build custom ASIC architecture specialized to traverse the sparse and quantized network structure, which also supports customized quantization bit width. We expect this architecture to have energy dominated by on-chip SRAM access instead of off-chip DRAM access.

修剪的网络在几种硬件上进行了基准测试，共享权重的量化网络则没有，因为现有的cuSPARSE或MKL SPBLAS库不支持间接矩阵索引查询，也不支持CSC或CSR格式的相对索引。所以深度压缩的优势并没有完全显现出来。一种软件的解决方案是写出定制的GPU核以支持。一种硬件的解决方案是构建定制的ASIC架构，专门处理稀疏量化网络结构，也支持定制的量化bit宽度。我们希望这种架构支持更多的片上SRAM访问。

## 9 Conclusion

We have presented “Deep Compression” that compressed neural networks without affecting accuracy. Our method operates by pruning the unimportant connections, quantizing the network using weight sharing, and then applying Huffman coding. We highlight our experiments on AlexNet which reduced the weight storage by 35× without loss of accuracy. We show similar results for VGG-16 and LeNet networks compressed by 49× and 39× without loss of accuracy. This leads to smaller storage requirement of putting convnets into mobile app. After Deep Compression the size of these networks fit into on-chip SRAM cache (5pJ/access) rather than requiring off-chip DRAM memory (640pJ/access). This potentially makes deep neural networks more energy efficient to run on mobile. Our compression method also facilitates the use of complex neural networks in mobile applications where application size and download bandwidth are constrained.

我们提出了深度压缩方法，压缩神经网络的同时，不影响其准确率。我们的方法首先修剪不重要的连接，使用共享权重对网络进行量化，然后进行Huffman编码。我们重点对AlexNet进行了试验，权重存储降低了35x，没有损失准确率。对VGG-16和LeNet网络，压缩了49x和39x，也没有损失准确率。存储降低了，可以将卷积网络放入移动app中。在深度压缩后，这些网络可以放入片上SRAM缓存中，而不需要进行片下DRAM内存昂文。这使得DNN更节约能耗，可以在移动设备上运行。我们的压缩方法可以使得在移动应用中可以更方便的使用复杂的神经网络。

Table 6: Accuracy of AlexNet with different aggressiveness of weight sharing and quantization. 8/5 bit quantization has no loss of accuracy; 8/4 bit quantization, which is more hardware friendly, has negligible loss of accuracy of 0.01%; To be really aggressive, 4/2 bit quantization resulted in 1.99% and 2.60% loss of accuracy.

Table 7: Comparison with other compression methods on AlexNet. (Collins & Kohli, 2014) reduced the parameters by 4× and with inferior accuracy. Deep Fried Convnets(Yang et al., 2014) worked on fully connected layers and reduced the parameters by less than 4×. SVD save parameters but suffers from large accuracy loss as much as 2%. Network pruning (Han et al., 2015) reduced the parameters by 9×, not including index overhead. On other networks similar to AlexNet, (Denton et al., 2014) exploited linear structure of convnets and compressed the network by 2.4× to 13.4× layer wise, with 0.9% accuracy loss on compressing a single layer. (Gong et al., 2014) experimented with vector quantization and compressed the network by 16× to 24×, incurring 1% accuracy loss.

## A Appendix: Detailed Timing / Power Reports of Dense & Sparse Network Layers

