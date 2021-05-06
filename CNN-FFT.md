# Fast Training of Convolutional Networks through FFTs

Michael Mathieu, Mikael Henaff, Yann LeCun New York University

## 0. Abstract

Convolutional networks are one of the most widely employed architectures in computer vision and machine learning. In order to leverage their ability to learn complex functions, large amounts of data are required for training. Training a large convolutional network to produce state-of-the-art results can take weeks, even when using modern GPUs. Producing labels using a trained network can also be costly when dealing with web-scale datasets. In this work, we present a simple algorithm which accelerates training and inference by a significant factor, and can yield improvements of over an order of magnitude compared to existing state-of-the-art implementations. This is done by computing convolutions as pointwise products in the Fourier domain while reusing the same transformed feature map many times. The algorithm is implemented on a GPU architecture and addresses a number of related challenges.

CNN是计算机视觉和机器学习中广泛采用的架构之一。为利用其学习复杂函数的能力，需要大量数据进行训练。训练一个大型CNN，以得到目前最好的效果，即使是使用现代GPUs，也可以消耗数星期。使用训练好的网络以产生标签，在处理网络规模的数据集时，也是消耗非常大的。在本文中，我们给出了一个简单的算法，极大加速了训练和推理，与现有最好的实现相比，产生了超过一个数量级的改进。这是将卷积作为Fourier域的点积来进行计算，多次重用相同的变换特征图。算法在GPU架构上进行了实现，处理了几个相关的挑战。

## 1. Introduction

As computer vision and machine learning aim to solve increasingly challenging tasks, models of greater complexity are required. This in turn requires orders of magnitude more data to take advantage of these powerful models while avoiding overfitting. While early benchmark datasets in machine learning contained thousands or tens of thousands of samples [7, 3, 10], current datasets are of the order of millions [6, 2]. This brings about new challenges as to how to train networks in a feasible amount of time. Even using parallel computing environments, training a network on ImageNet can take weeks [8]. In addition, although inference of labels using a trained network is comparatively fast, real-world applications such as producing labels for all images on the internet can represent a significant cost in terms of time and resources. Therefore, there is an important need to develop fast algorithms for training and inference.

计算机视觉和机器学习的目标是解决日益复杂的任务，需要更加复杂的模型。这又需要多出几个数量级的数据，以利用这些强大的模型，同时避免过拟合。早期机器学习中的基准测试数据集包括数千或数万个样本，目前的数据集是数百万量级的规模。这带来了新的挑战，即怎样在有限的时间内训练网络。即使使用并行计算环境，在ImageNet上训练网络也要消耗数星期。另外，虽然使用训练好的网络推理标签相对来说较快，真实世界的应用，比如对互联网上的所有图像产生标签，在时间和资源上也会消耗非常多。因此，开发快速算法进行训练和推理的需求非常高。

In this work, we present a simple algorithm which accelerates training and inference using convolutional networks. The idea is based on performing convolutions as products in the Fourier domain, and reusing transformed feature maps many times. The significant operations in training convolutional networks can all be viewed as convolutions between pairs of 2-D matrices, which can represent input and output feature maps, gradients of the loss with respect to feature maps, or weight kernels. Typically, convolutions are performed for all pairings between two sets of 2-D matrices. By computing the Fourier transforms of the matrices in each set once, we can efficiently perform all convolutions as pairwise products.

本文中，我们提出一种简单算法加速使用CNN的训练和推理。其思想是基于将卷积作为Fourier域的点积进行计算，对变换的特征图重用很多次。训练CNN时的主要运算都可以视为成对的2D矩阵之间的卷积，表示的是输入和输出的特征图，损失函数对特征图的梯度，或权重核。典型的，对2D矩阵的两个集合之间的所有对，都要进行卷积运算。通过对每个集合中的矩阵计算一次其Fourier变换，我们可以将所有的卷积作为成对的点积进行计算。

Although it has long been known that convolutions can be computed as products in the Fourier domain, until recently the number of feature maps used in convolutional networks has been too small to make a method like ours effective. Previous work in the 90's [1] explored the possibility of using FFTs to accelerate inference at the first layer of a trained network, where the Fourier transforms of the filters could be precomputed offline. However, this was not used during training, possibly because the number of feature maps used at the time was too small to make the overhead of computing FFTs at every iteration worthwhile. When the number of feature maps is large, as is the case for modern convolutional networks, using FFTs accelerates training and inference by a significant factor and can lead to a speedup of over an order of magnitude.

卷积可以在Fourier域作为点积来计算，这已经是广为所知的，但直到最近，在卷积网络中使用的特征图都太小了，不能用类似我们的方法高效的计算。之前的工作[1]探索了使用FFTs来加速训练好的网络的第一层的推理的可能性，其中滤波器的Fourier变换可以离线进行预计算。但是，这并不是在训练时使用的，可能是因为那时候使用的特征图数量较小，在每次迭代时计算FFTs还不是特别值得。当特征图的数量很大，就像在现代CNN中的情况，使用FFTs可以极大的加速训练和推理，达到数量级的加速效果。

## 2. Theory

### 2.1. Backpropagation

The backpropagation algorithm [9] is the standard method to compute the gradient when training a convolutional network. During training, each layer performs three tasks, which we now describe. First we fix some notation: for a given layer, we have a set of input feature maps $x_f$ indexed by f, each one being a 2-D image of dimensions n×n. The output is a set of feature maps $y_{f'}$ indexed by f', which are also 2-D images whose dimension depends on the convolutional kernel and its stride. The layer's trainable parameters consist of a set of weights $w_{f'f}$, each of which is a small kernel of dimensions k × k.

反向传播算法，是训练CNN计算梯度时的标准方法。在训练时，每一层进行三个任务，我们现在进行叙述。首先，我们固定一些表示：对一个给定的层，我们有输入特征图的集合$x_f$，由f进行索引，每个都是一个2D图像，维度为nxn。输出特征图集合$y_{f'}$，索引是f'，这也是2D图像，其维度依赖于卷积核和其步长。层中的可训练参数构成了权重集合$w_{f'f}$，每个都是一个小的核，维度为kxk。

In the forward pass, each output feature map is computed as a sum of the input feature maps convolved with the corresponding trainable weight kernel:

在前向过程中，每个输出特征图都计算为，输入特征图与对应的可训练的权重核的卷积的和。

$$y_{f'} = \sum_f x_f * w_{f'f}$$(1)

During the backward pass, the gradients with respect to the inputs are computed by convolving the transposed weight kernel with the gradients with respect to the outputs:

在反向过程中，对输入的梯度，计算为权重核的转置，与对输出的梯度的卷积：

$$\frac {∂L}{∂x_f} = \frac {∂L} {∂y_{f'}} * w_{f'f}^T$$(2)

This step is necessary for computing the gradients in (3) for the previous layer. Finally, the gradients of the loss with respect to the weight are computed by convolving each input feature map with the gradients with respect to the outputs:

对计算(3)式中前一层的梯度，这一步骤是必须的。最后，损失对权重的梯度，是将每个输入特征图与对输出的梯度相卷积：

$$\frac {∂L}{w_{f'f}} = \frac {∂L}{∂y_{f'}} * x_f$$(3)

Note that $∂L/∂y_{f'}$ is a 2-D matrix with the same dimensions as the output feature map $y_{f'}$, and that all operations consist of convolutions between various sets of 2-D matrices.

注意，$∂L/∂y_{f'}$是一个2D矩阵，与输出特征图$y_{f'}$的维度相同，所有运算都是由各种2D矩阵的集合之间的卷积构成的。

### 2.2 Algorithm

The well-known Convolution Theorem states that circular convolutions in the spatial domain are equivalent to pointwise products in the Fourier domain. Letting F denote the Fourier transform and $F^{−1}$ its inverse, we can compute convolutions between functions f and g as follows:

著名的卷积定理说，空域中的圆形卷积，与Fourier域的点积等价。令F表示Fourier变换，$F^{−1}$为其逆，我们可以计算f和g之间的卷积如下：

$$f*g =F^{-1}(F(f)⋅F(g))$$

Typically, this method is used when the size of the convolution kernel is close to that of the input image. Note that a convolution of an image of size n × n with a kernel of size k × k using the direct method requires $(n − k + 1)^2k^2$ operations. The complexity of the FFT-based method requires $6Cn^2 log (n + 4n^2)$ operations: each FFT requires $O(n^2 log n^2) = O(2n^2 log n) = 2Cn^2 log n$, and the pointwise product in the frequency domain requires $4n^2$ (note that the products are between two complex numbers). Here C represents the hidden constant in the O notation.

典型的，这个方法是在卷积核的大小与输入图像的大小接近的时候使用的。注意，图像大小nxn，与核的大小kxk，其使用直接方法的卷积，需要$(n − k + 1)^2k^2$次运算。基于FFT的方法需要$6Cn^2 log (n + 4n^2)$次运算：每个FFT需要$O(n^2 log n^2) = O(2n^2 log n) = 2Cn^2 log n$次运算，频域的点积需要$4n^2$次运算（注意，点积是两个复数之间的）。这里C表示O表示中的隐藏常数。

Our algorithm is based on the observation that in all of the operations (1), (2) and (3), each of the matrices indexed by f is convolved with each of the matrices indexed by f'. We can therefore compute the FFT of each matrix once, and all pairwise convolutions can be performed as products in the frequency domain. Even though using the FFT-based method may be less efficient for a given convolution, we can effectively reuse our FFTs many times which more than compensates for the overhead.

我们的算法是基于下面的观察，在(1)(2)(3)中的所有运算中，由f索引的每个矩阵，与由f'索引的每个矩阵相卷积。我们因此可以一次性计算每个矩阵的FFT，所有的成对卷积可以在频域中由点积来计算。即使是使用基于FFT的方法，可能对给定的卷积也不是那么高效，但我们可以高效的重新利用我们的FFTs很多次，这样就补偿了开销。

The following analysis makes this idea precise. Assume we have f input feature maps, f' output feature maps, images consisting of n × n pixels and kernels of k × k pixels. Also assume we are performing updates over minibatches of size S, and that C represents the hidden constant in the FFT complexity. As an example, using the direct approach (1) will take a total of $S·f'·f·(n−k+1)^2·k^2$ operations. Our approach requires $(2C·n^2 log n)(S·f + f'·f)$ operations to transform the input feature maps and kernels to the Fourier domain, a total of $4S·f'·f·n^2$ additions and multiplications in the Fourier domain, and $S·f'·(2C·n^2 log n)$ operations to transform the output feature maps back to the spatial domain. The same analysis yields similar complexity estimates for the other operations:

下面的分析使这个思想更加准确。假设我们有f个输入特征图，f'个输出特征图，图像是由nxn个像素组成，核kxk个像素。假设我们在minibatch大小S上进行更新，C表示FFT复杂度内的隐藏常数。举例来说，使用直接方法(1)共计需要$S·f'·f·(n−k+1)^2·k^2$次运算，我们的方法需要$(2C·n^2 log n)(S·f + f'·f)$次运算来将输入特征图和核变换到Fourier域，在Fourier域共需要$4S·f'·f·n^2$次加法和乘法，将输出特征图转换到空域中又需要$S·f'·(2C·n^2 log n)$次运算。对其他运算，用相同的分析可以得到类似的复杂度估计：

Here n' = (n − k + 1) represents the size of the output feature map. Note that the high complexity of the direct method for convolution comes from the product of five terms, whereas our method has a sum of products with at most four terms. Figure 2 shows the theoretical number of operations for direct convolution and our FFT method for various input sizes.

这里n' = (n − k + 1)表示输出特征图的大小。注意直接方法对卷积的高复杂度，来自于前5项的积，而我们的方法最多只有前4项的和。图2展示了，对各种输入大小，直接卷积和FFT法的理论运算数量。

### 2.3. Implementation and Memory Considerations

Although conceptually straighforward, a number of challenges relating to GPU implementation needed to be addressed. First, current GPU implementations of the FFT such as cuFFT are designed to parallelize over individual transforms. This can be useful for computing a limited number of transforms on large inputs, but is not suitable for our task since we are performing many FFTs over relatively small inputs. Therefore, we developed a custom CUDA implementation of the Cooley-Tukey FFT algorithm [5] which enabled us to parallelize over feature maps, minibatches and within each 2-D transform. Note that 2-D FFTs lend themselves naturally to parallelization since they can be decomposed into two sets of 1-D FFTs (one over rows and the other over columns), and each set can be done in parallel.

虽然在概念上很直接，但还是要处理一些与GPU实现相关的挑战。第一，FFT目前的GPU实现，比如cuFFT，其设计是对单个的变换进行并行化。这对计算大输入的有限数量的变换是有用的，但并不适用于我们的任务，因为我们是在相对较小的输入上进行很多FFTs。因此，我们开发了Cooley-Tukey FFT算法的定制CUDA实现，使我们可以对特征图、minibatches和每个2D变换中进行并行化。注意2D FFTs可以很自然的进行并行化，因为可以分解成两个1-D FFTs的集合（一个在行上，一个在列上），每个集合都可以并行进行。

Second, additional memory is required to store the feature maps in the Fourier domain. Note that by keeping the Fourier representations in memory for all layers after the forward pass, we could avoid recomputing several of the FFTs during the backward pass. However, this might become prohibitively expensive in terms of memory for large networks. Therefore we reuse the same memory for all the different convolutions in the network, so that the necessary amount of memory is determined only by the largest convolution layer. All of the analysis in the previous section and all experiments in the remainder of the paper assume we are using this memory-efficient approach.

第二，需要额外的内存来在Fourier域存储特征图。注意，将所有层在前向过程后的Fourier表示存储在内存中，我们可以在反向过程中避免重新计算几个FFTs。但是，对于大型网络，这可能需要很大量的内存。因此我们对网络中所有不同的卷积重用了相同的内存，这样内存的必须量，只由最大的卷积层来决定。前节中的所有分析，和本文剩余篇幅的所有试验，我们都使用这种内存高效的方法。

For a convolution layer taking an input of size n × n, with f input features, f' output features and a minibatch of size S, we need to store a total of S·f + S·f' + f·f' frequency representations of size n × n. As another means to save memory, we can use symmetry properties of FFTs of real inputs to store only half the data, i.e. n(n+1)/2 complex numbers. Assuming float representations, the necessary memory in bytes is:

对于一个卷积层，输入大小为nxn，f个输入特征，f'个输出特征，minibatch大小为S，我们需要存储S·f + S·f' + f·f'个频域表示，大小nxn。作为另一种节约内存的方法，我们可以使用实属输入的FFTs的对称性质，只存储半数数据，即，n(n+1)/2个复数值。假设是浮点表示，需要存储的bytes为：

$$4n(n + 1)(S·f + S·f' + f·f')$$

The following table shows the amount of RAM used for typical sizes of convolutions: 下表给出了典型大小的卷积所需的RAM量：

S | n | f | f' | RAM used
--- | --- | --- | --- | ---
128 | 16 | 96 | 256 | 76MB
128 | 32 | 96 | 256 | 294MB
64 | 64 | 96 | 256 | 784MB
128 | 64 | 96 | 256 | 1159MB
128 | 16 | 256 | 384 | 151MB
128 | 32 | 256 | 384 | 588MB
128 | 16 | 384 | 384 | 214MB
128 | 32 | 384 | 384 | 830MB

Note that this is a relatively small additional memory requirement compared to the total amount of memory used by large networks.

注意，与大型网络所使用的总计内存量相比，这是相对很少的内存需求。

## 3. Experiments

To test our analysis, we ran a series of experiments comparing our method to the CudaConv GPU implementation of [8] and a custom implementation using the Torch 7 machine learning environment [4]. Both of these implementations compute convolutions using the direct method in the spatial domain. All experiments were performed on the same GeForce GTX Titan GPU. We began by performing unit tests comparing the results of convolutions computed by our method to those computed by the Torch implementation for each of the three operations. We found that the differences in results for operations (1) and (2) to be of the order of 10^−5 and for operation (3) to be of the order 10^−4. The differences are likely due to rounding errors in floating-point operations and are within an acceptable range.

为测试我们的分析，我们运行了一系列试验，比较了我们的方法，和[8]的CudaConv GPU实现，和Torch 7机器学习环境的定制实现[4]。这些实现计算卷积使用的是空域中的直接方法。所有的试验都是在相同的GeForce GTX Titan GPU上进行的。我们首先进行单元测试，比较我们的方法和Torch实现的方法，对这三种运算得到的卷积结果。我们发现运算(1)和(2)的差异是10^-5级的，(3)的差异是10^-4级的。这些差异很可能是由于浮点运算中的四舍五入的效果，是在可接受的范围内的。

We then compared how each method performed in terms of speed with varying kernel sizes, input sizes and minibatch sizes. The results are shown in Figure 3. For all experiments, we chose 96 input feature maps and 256 output feature maps, which represents a typical configuration of a deep network's second layer. The functions updateOutput, updateGradInput and accGradParameters correspond to the operations in (1), (2) and (3) respectively. All times are measured in seconds.

我们然后比较了，每种方法在不同的核的大小、输入大小和minibatch大小下，不同的速度表现。结果如图3所示。对所有试验，我们选择了96个输入特征和256个输出特征图，这是DNN第二层的典型配置。函数updateOutput, updateGradInput和accGradParameters分别对应着(1)(2)(3)。所有的时间都是以秒为单位的。

We see that our method significantly outperforms the other two in nearly all cases. The improvement is especially pronounced for the accGradParameters operation, which is the most computationally expensive. This is likely due to the fact that the convolution we are computing has a large kernel, for which FFTs are better suited in any case. Also note that our method performs the same regardless of kernel size, since we pad the kernel to be the same size as the input image before applying the FFT. This enables the use of much larger kernels, which we intend to explore in future work.

可以看出，我们的方法在几乎所有情况中，都超过了其他两种方法。对accGradParameters运算的改进尤其明显，其计算量非常的大。这很可能是因为，我们要计算的卷积核比较大，FFTs比较适合这种情况。我们还注意到，不管核的大小多大，我们的方法表现相同，因为在应用FFT之前，我们将核的大小补零成输入图像的大小。这使得可以利用大的多的核，我们在未来的工作中会进一步进行探索。

We next ran experiments with parameter configurations typical of those used in different layers of a large convolutional network. The time taken by the different methods are given in milliseconds. The top row is a 4-tuple (k, n, f, f') indicating the width of the kernel, width of the input image, number of input feature maps and number of output feature maps. All kernels and input images are square, of size k × k and n × n respectively. All configurations have minibatches of size 128. The first configuration represents the first layer, which is why we did not report times for the updateGradInput operation. For each configuration, the best-performing method is highlighted in bold.

下一步，我们进行的试验，参数配置与大型CNN中的不同层使用的类似。不同方法所需的时间以ms为单位给出。顶行的四元组(k, n, f, f')表明了核的宽度，输入图像的宽度，输入特征图的数量，和输出特征图的数量。所有的核和输入图像都是方形的，大小分别为kxk和nxn。所有的配置的minibatch大小都是128。第一个配置代表第一层，所以我们不需要给出updateGradInput运算的时间。对每个配置，最佳表现的方法以粗体表示。

We see that our FFT-based method performs faster in total for all configurations, sometimes to a substantial degree. The improvement is very significant on the forward pass, which makes the method especially well suited for inference on very large datasets using a trained network.

我们看到，我们的基于FFT的方法在所有配置中运行的都更快，有时候会快很多。在前向过程中改进非常显著，这使得这个方法更加适合于使用训练好的网络在非常大的数据集上进行推理。

Finally, we tested times taken to perform a training iteration for a network obtained by composing the above layers, inserting max-pooling and rectified linear units between them, and adding a fully connected layer for prediction with 1000 outputs. This was to account for possible changes in performance due to implementation details such as padding, accessing memory and so on. The following table shows the results in milliseconds:

最后，我们测试了由上述层组成的网络（在其中插入了最大池化和ReLU，加入了全连接层进行预测1000个输出）进行训练迭代时所需的时间。这要计入由于一些实现细节导致的可能的性能变化，比如padding，访问内存等。下表给出了以ms为单位的结果：

Our FFT-based method still significantly outperforms the other two implementations. 我们的基于FFT的方法仍然明显超过了另外两种实现。

## 4. Discussion and Future Work

We have presented a simple and fast algorithm for training and inference using convolutional networks. It outperforms known state-of-the-art implementations in terms of speed, as verified by numerical experiments. In the future we plan to explore the possibility of learning kernels directly in the Fourier domain. Another interesting direction would be to investigate the use of non-linearities in the Fourier domain rather than in the spatial domain, since this would remove the need for inverse transforms and accelerate training and inference further.

我们给出了使用CNN进行训练和推理的一种简单快速的实现。比目前最好的实现，在速度上有显著的提升，在几个试验中得到了验证。在未来，我们计划探索在Fourier域直接学习核的可能性。另一个有趣的方向是，研究在Fourier域中使用非线性，而不是在空域中，因为这就不需要进行逆变换，可以进一步加速训练和推理。

It is worth mentioning that in our current implementation of the FFT algorithm, input images which are not a power of 2 must be padded to the next highest power. For example, using input images of size 34 × 34 will be suboptimal in terms of speed since they must be padded to be 64 × 64. This limitation is not intrinsic to the FFT and we intend to extend our implementation to accept other sizes in the future. On the other hand, the fact that our method's speed is invariant to kernel size enables us to use larger kernels at different layers of the network. In future work we intend to thoroughly explore the effect of input image and kernel sizes on performance.

值得提到的是，在我们目前的FFT算法的实现中，输入图像的大小需要补零到2的整数次方大小。比如，使用输入大小34x34，性能就不会最佳，需要补零到64x64。这个限制并不是FFT内在的，我们计划将我们的实现拓展到支持其他大小。另一方面，我们的方法的速度对核大小是不变的，使得在网络的不同层，可以使用更大的核。在未来，我们计划彻底研究输入图像和核的大小在性能上的影响。