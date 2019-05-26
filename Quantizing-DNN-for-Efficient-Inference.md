# Quantizing deep convolutional networks for efficient inference: A whitepaper

Raghuraman Krishnamoorthi Google Inc.

## Abstract 摘要

We present an overview of techniques for quantizing convolutional neural networks for inference with integer weights and activations. 我们给出量化CNN，使用整数权重和激活进行推理的技术概览：

1. Per-channel quantization of weights and per-layer quantization of activations to 8-bits of precision post-training produces classification accuracies within 2% of floating point networks for a wide variety of CNN architectures (section 3.1). 训练过后，权重逐通道量化、激活逐层量化到8-bits精度，得到的分类准确率与浮点网络精度相差不超过2%，这对于很大范围的CNN架构都是这样（3.1节）。
2. Model sizes can be reduced by a factor of 4 by quantizing weights to 8-bits, even when 8-bit arithmetic is not supported. This can be achieved with simple, post training quantization of weights (section 3.1). 权重量化到8-bits，模型大小会降低4倍，即使在8-bit运算不支持的情况下。这可以通道很简单的、训练后权重量化得到（3.1节）。
3. We benchmark latencies of quantized networks on CPUs and DSPs and observe a speedup of 2x-3x for quantized implementations compared to floating point on CPUs. Speedups of up to 10x are observed on specialized processors with fixed point SIMD capabilities, like the Qualcomm QDSPs with HVX (section 6). 我们对量化网络在CPU和DSP上进行延迟基准测试，观察到量化实现比浮点实现在CPU上有2x到3x的加速。在专门的有定点SIMD功能的处理器上，有高达10x的加速，如Qualcomm QDSPs with HVX（第6节）。
4. Quantization-aware training can provide further improvements, reducing the gap to floating point to 1% at 8-bit precision. Quantization-aware training also allows for reducing the precision of weights to four bits with accuracy losses ranging from 2% to 10%, with higher accuracy drop for smaller networks (section 3.2). 针对量化的训练可以得到进一步的改进，8-bit精度下与浮点的差距降低到1%。针对量化的训练情况下，权重精度降低到4-bits，准确率损失在2%到10%之间，网络越小，准确率降低越高（3.2节）。
5. We introduce tools in TensorFlow and TensorFlowLite for quantizing convolutional networks (Section 3). 我们提出TensorFlow和TensorFlowLite中的量化卷积网络的工具（第3节）。
6. We review best practices for quantization-aware training to obtain high accuracy with quantized weights and activations (section 4). 我们回顾了针对量化的训练的最好工作，在量化的权重和激活下得到更高的准确率（第3节）。
7. We recommend that per-channel quantization of weights and per-layer quantization of activations be the preferred quantization scheme for hardware acceleration and kernel optimization. We also propose that future processors and hardware accelerators for optimized inference support precisions of 4, 8 and 16 bits (section 7). 我们推荐权重逐通道量化，激活逐层量化，作为硬件加速和核心优化的推荐方案。我们还提出将来的处理器和硬件加速器为优化推理支持4,8和16 bits（第7节）。

## 1 Introduction 介绍

Deep networks are increasingly used for applications at the edge. Devices at the edge typically have lower compute capabilities and are constrained in memory and power consumption. It is also necessary to reduce the amount of communication to the cloud for transferring models to the device to save on power and reduce network connectivity requirements. Therefore, there is a pressing need for techniques to optimize models for reduced model size, faster inference and lower power consumption.

深度网络越来越多的用于端侧的应用。端侧的设备通常有更低的计算能力，受限的内存和能耗。也有必要降低与云端的通信量，因为设备的通信模块需要节省电量，并降低网络连接需求。因此，有很大的必要对模型进行优化，以减小模型规模，更快的推理和降低能量消耗。

There is extensive research on this topic with several approaches being considered: One approach is to build efficient models from the ground up [1],[2] and [3]. Another technique is to reduce the model size by applying quantization, pruning and compression techniques [4], [5] and [6]. Faster inference has been achieved by having efficient kernels for computation in reduced precision like GEMMLOWP [7], Intel MKL-DNN [8] , ARM CMSIS [9], Qualcomm SNPE [10], Nvidia TensorRT [11] and custom hardware for fast inference [12], [13] and [14].

这个课题有很多研究，主要包括以下几种方法：一种是从头开始构建高效的模型[1,2,3]；另一种是通过量化、剪枝和压缩技术[4,5,6]来降低模型大小。更快的推理可以通过使用降低精度的高效核心进行计算，如GEMMLOWP[7], Intel MKL-DNN[8], ARM CMSIS[9], Qualcomm SNPE[10], NVidia TensorRT[11]和为快速推理的定制硬件[12,13,14]。

One of the simpler ways to reduce complexity of any model is to reduce the precision requirements for the weights and activations. This approach has many advantages: 降低任意模型的复杂度的一种较为简单的方法是，降低权重和激活的精度需要。这种方法有很多优点：

- It is broadly applicable across a range of models and use cases. One does not need to develop a new model architecture for improved speed. In many cases, one can start with an existing floating point model and quickly quantize it to obtain a fixed point quantized model with almost no accuracy loss, without needing to re-train the model. Multiple hardware platforms and libraries support fast inference with quantized weights and activations, so there is no need to wait for new hardware development.

- 适用于很多模型和使用情况。为改进速度，不需要开发新模型。在很多情况下，可以从已有的浮点模型开始，很快的进行量化，得到一个定点的量化模型，几乎没有准确率损失，不需要重新训练模型。多个硬件平台和库都支持使用量化权重和激活进行快速推理，所以不需要等待新的硬件开发。

- Smaller Model footprint: With 8-bit quantization, one can reduce the model size a factor of 4, with negligible accuracy loss. This can be done without needing any data as only the weights are quantized. This also leads to faster download times for model updates.

- 更小的模型占用空间：在8-bit量化的情况下，可以降低模型大小4倍，准确率损失可以忽略。不需要任何数据就可以这样做，因为只有权重被量化。这也可以为模型更新带来更少的下载时间。

- Less working memory and cache for activations: Intermediate computations are typically stored in cache for reuse by later layers of a deep network and reducing the precision at which this data is stored leads to less working memory needed. Having lower precision weights and activations allows for better cache reuse.

- 对于激活来说，更少的工作内存和缓存：中间计算结果一般存储在缓存中以便后面的层复用，降低存储数据的精度也就降低了所需的内存消耗。更低的权重和激活精度可以更好的进行缓存复用。

- Faster computation: Most processors allow for faster processing of 8-bit data.

- 更快的计算：多数处理器可以更快的计算8-bit数据。

- Lower Power: Moving 8-bit data is 4 times more efficient than moving 32-bit floating point data. In many deep architectures, memory access can dominate power consumption [2]. Therefore reduction in amount of data movement can have a significant impact on the power consumption.

- 更低的功耗：移动8-bit数据的效率是移动32-bit的浮点数据效率的4倍。在很多深度架构中，内存访问是主要的能耗部分[2]。所以更少的数据移动对能耗有显著的影响。

All the factors above translate into faster inference, with a typical speedup of 2-3x due to the reduced precision for both memory accesses and computations. Further improvements in speed and power consumption are possible with processors and hardware accelerators optimized for low precision vector arithmetic. 上面所有因素都会带来更快的推理，降低精度带来的内存访问效率提高和计算量降低，一般会带来2-3x的加速效果。处理器和硬件加速器对低精度向量运算进行优化的话，速度和能耗可以得到进一步的改进。

## 2 Quantizer Design 量化器设计

In this section, we review different design choices for uniform quantization. 本节中，我们回顾均匀量化的不同设计选择。

### 2.1 Uniform Affine Quantizer 均匀仿射量化器

Consider a floating point variable with range ($x_{min}, x_{max}$) that needs to be quantized to the range (0, $N_{levels}$ − 1) where $N_{levels}$ = 256 for 8-bits of precision. We derive two parameters: Scale (∆) and Zero-point(z) which map the floating point values to integers (See [15]). The scale specifies the step size of the quantizer and floating point zero maps to zero-point [4]. Zero-point is an integer, ensuring that zero is quantized with no error. This is important to ensure that common operations like zero padding do not cause quantization error.

考虑一个范围在($x_{min}, x_{max}$)的浮点变量，需要量化到范围(0, $N_{levels}$ − 1)，对于8-bit精度来说，$N_{levels}$ = 256。我们推导两个参数：尺度∆和零点z，将浮点数值映射到整数值（见[15]）。尺度指定了量化器的步长大小，浮点零映射到零点[4]。零点是一个整数，确保零量化的时候无误差。这很重要，要确保常见的操作，如补零不会导致量化误差。

For one sided distributions, therefore, the range ($x_{min}, x_{max}$) is relaxed to include zero. For example, a floating point variable with the range (2.1,3.5) will be relaxed to the range (0,3.5) and then quantized. Note that this can cause a loss of precision in the case of extreme one-sided distributions.

对于单边分布，范围($x_{min}, x_{max}$)只需要把零纳入进来。例如，范围在(2.1,3.5)的浮点变量，只需要把范围扩大到(0,3.5)，然后进行量化。注意在极端的单边分布情况下，这会带来精度损失。

Once the scale and zero-point are defined, quantization proceeds as follows: 一旦确定了尺度和零点，量化可以按照如下进行：

$$x_{int} = round(\frac{x}{∆}) + z$$(1)
$$x_Q = clamp(0, N_{levels} − 1, x_{int})$$(2)

where 其中

$$clamp(a,b,x) = \begin {cases} a, \quad x≤a \\ x, \quad a≤x≤b \\ b, \quad x≥b \end {cases}$$

The de-quantization operation is: 去量化运算为：

$$x_{float} = (x_Q − z)∆$$(3)

While the uniform affine quantizer allows for storing weights and activations at 8-bits of precision, there is an additional cost due to the zero-point. Consider a 2D convolution between a weight and an activation: 均匀仿射量化器可以将权重和激活以8-bit精度保存，但由零点的存在，有额外的代价。考虑权重和激活的2D卷积：

$$y(k,l,n) = ∆_w ∆_x conv(w_Q (k,l,m;n) − z_w , x_Q (k,l,m) − z_x)$$(4)
$$y(k,l,n) = conv(w_Q (k,l,m;n), x_Q (k,l,m)) − z_w \sum_{k=0}^{K-1} \sum_{l=0}^{K-1} \sum_{m=0}^{N-1} x_Q (k,l,m) − z_x \sum_{k=0}^{K-1} \sum_{l=0}^{K-1} \sum_{m=0}^{N-1} w_Q (k,l,m;n) + z_x z_w$$(5)

A naive implementation of convolution, by performing the addition of zero-point prior to the convolution, leads to a 2x to 4x reduction in the throughput due to wider (16/32-bit) operands. One can do better by using the equation above and noting that the last term is a constant and each of the other terms requires N multiplies, which is 3x more operations than the 8-bit dot product. This can be further improved by noting that the weights are constant at inference and by noting that the sum over activations is identical for all convolutional kernels of the same size. However, this requires optimizing convolution kernels. For an indepth discussion, please see [16].

卷积的一种朴素实现方法，在卷积之前加上零点，可以带来计算吞吐量2x到4x的降低，因为使用的是更宽(16/32-bit)算子。使用上面的公式还可以得到更好的结果，注意最后一项是一个常数，其他每一项都需要N个乘子，这比8-bit点乘多了3x的运算量。还要注意到，在推理时，权重是常数，所有激活的和对大小相同的卷积核是一样的。但是，这需要对卷积核进行优化。深度讨论请参考[16]。

### 2.2 Uniform symmetric quantizer 均匀对称量化器

A simplified version of the affine quantizer is the symmetric quantizer, which restricts zero-point to 0. With the symmetric quantizer, the conversion operations simplify to: 仿射量化器的一个简化版本是对称量化器，将零点限制在0上。在对称量化器上，转换运算简化为：

$$x_{int} = round(\frac{x}{∆})$$(7)
$$x_Q = clamp(−N_{levels}/2, N_{levels}/2 − 1, x_{int}) \quad if \: signed$$(8)
$$x_Q = clamp(0, N_{levels} − 1, x_{int}) \quad if \: unsigned$$(9)

For faster SIMD implementation, we further restrict the ranges of the weights. In this case, the clamping is modified to: 为更快用SIMD实现，我们进一步限制权重的范围。在这种情况下，区间限制函数修改为：

$$x_Q = clamp(−N_{levels}/2 − 1, N_{levels}/2 − 1, x_{int}) \quad if \: signed$$(10)
$$x_Q = clamp(0, N_{levels} − 2, x_{int}) \quad if \: unsigned$$(11)

Please see [4], Appendix B for more details. 详见[4]和附录B。

The de-quantization operation is: 去量化操作为： $x_{out} = x_Q ∆$

### 2.3 Stochastic quantizer 随机量化器

Stochastic quantization models the quantizer as an additive noise, followed by rounding. The stochastic quantizer is given by:  随机量化将量化器建模为加性噪声，和四舍五入。随机量化器为：

$$x_{int} = round(\frac {x+ϵ}{∆}) + z, ϵ ∼ U nif (-1/2, 1/2)$$
$$x_Q = clamp(0, N_{levels} − 1, x_{int})$$

The de-quantization operation is given by equation 3. Note that in expectation, the stochastic quantizer reduces to a pass-through of the floating point weights, with saturation for values outside the range. Therefore, this function is well behaved for purposes of calculating gradients. We do not consider stochastic quantization for inference as most inference hardware does not support it.

去量化运算由式3给出。注意，在期望上，随机量化器就是浮点权重的直通，但在范围之外有饱和。所以，这个函数对于计算梯度来说，效果很好。对于推理来说，我们不考虑随机量化，因为多数推理硬件都不支持。

### 2.4 Modeling simulated quantization in the backward pass 反向过程中模拟量化的建模

For Quantization-aware training, we model the effect of quantization using simulated quantization operations, which consist of a quantizer followed by a de-quantizer, i.e, 对于针对量化的训练，我们使用模拟量化运算对量化效果进行建模，包括一个量化器，随后跟着一个去量化器，即

$$x_{out} = SimQuant(x)$$(12)
$$= ∆ clamp(0, N_{levels} − 1, round(\frac{x}{∆}) − z)$$(13)

Since the derivative of a simulated uniform quantizer function is zero almost everywhere, approximations are required to model a quantizer in the backward pass. An approximation that has worked well in practice (see [5]) is to model the quantizer as specified in equation 14 for purposes of defining its derivative (See figure 1).

由于模拟均匀量化器函数的导数几乎到处都是0，所以需要在反向过程中对量化器进行建模的时候进行近似。实践中有不错的近似（见[5]），即对式14中的量化器进行建模，目的是定义其导数（见图1）。

$$x_{out} = clamp(x_{min}, x_{max}, x)$$(14)

Figure 1: Simulated Quantizer (top), showing the quantization of output values. Approximation for purposes of derivative calculation (bottom).

The backward pass is modeled as a ”straight through estimator” (see [5]). Specifically, 其反向过程建模为直通估计器（见[5]）。特别的，

$$δ_{out} = δ_{in} I_{x∈S} S : x : x min ≤ x ≤ x max$$(15)

where $δ_{in} = \frac {∂L}{∂w_{out}}$ is the backpropagation error of the loss with respect to the simulated quantizer output. 其中$δ_{in}$是损失对模拟量化器输出的反向传播误差。

### 2.5 Determining Quantizer parameters 确定量化器参数

The quantizer parameters can be determined using several criteria. For example, TensorRT [11] minimizes the KL divergence between the original and quantized distributions to determine the step size. In this work, we adopt simpler methods. For weights, we use the actual minimum and maximum values to determine the quantizer parameters. For activations, we use the moving average of the minimum and maximum values across batches to determine the quantizer parameters. For post training quantization approaches, one can improve the accuracy of quantized models by careful selection of quantizer parameters.

量化器的参数可以通过几个原则来确定。比如，TensorRT[11]最小化原始分布和量化分布之间的KL散度，来确定步长。在本文中，我们采用更简单的方法。对于权重，我们使用实际的最小和最大值来确定量化器参数。对于激活，我们采用多批次之间最小值和最大值的滑动平均来确定量化器参数。对于训练后的量化方法，可以通过仔细选择量化参数，来提高量化模型的准确率。

### 2.6 Granularity of quantization 量化的粒度

We can specify a single quantizer (defined by the scale and zero-point) for an entire tensor, referred to as per-layer quantization. Improved accuracy can be obtained by adapting the quantizer parameters to each kernel within the tensor [17]. For example, the weight tensor is 4 dimensional and is a collection of 3 dimensional convolutional kernels, each responsible for producing one output feature map. per-channel quantization has a different scale and offset for each convolutional kernel. We do not consider per-channel quantization for activations as this would complicate the inner product computations at the core of conv and matmul operations. Both per-layer and per-channel quantization allow for efficient dot product and convolution implementation as the quantizer parameters are fixed per kernel in both cases.

我们可以为一整个张量指定一个量化器（由尺度和零点定义），这称为逐层量化。可以通过为张量中每个核心修正量化参数[17]，来得到改进的准确率。比如，权重张量是四维的，是三维卷积核的集合，每个核负责产生一个输出特征图。逐通道量化对每个卷积核有着不同的尺度和偏移。对激活，我们不考虑逐通道的量化，因为这会使得内积计算变的复杂，这是卷积和矩阵相乘运算的核心。逐层和逐通道的量化可以得到高效的点积和卷积实现，因为量化器参数在两种情况下对每个核都是固定的。

## 3 Quantized Inference: Performance and Accuracy 量化推理：性能和准确率

Quantizing a model can provide multiple benefits as discussed in section 1. We discuss multiple approaches for model quantization and show the performance impact for each of these approaches.

如第1节所讨论，模型量化有几个好处。我们讨论讨论模型量化的几种方法，给出每种方法的性能影响。

### 3.1 Post Training Quantization 训练后量化

In many cases, it is desirable to reduce the model size by compressing weights and/or quantize both weights and activations for faster inference, without requiring to re-train the model. Post Training quantization techniques are simpler to use and allow for quantization with limited data. In this section, we study different quantization schemes for weight only quantization and for quantization of both weights and activations. We show that per-channel quantization with asymmetric ranges produces accuracies close to floating point across a wide range of networks.

在很多情况下，通过压缩权重和/或对权重和激活进行量化来压缩模型大小，以得到更快的推理，是非常理想的，不需要重新训练模型。训练后量化的技术使用起来更简单，在有限的数据下就可以进行量化。本节中，我们研究不同量化方案，用在只对权重进行量化，和对权重和激活都量化。我们证明，非对称范围的逐通道量化，可以得到与浮点接近的准确率，在很多网络中都可以得到这个结果。

#### 3.1.1 Weight only quantization 只对权重量化

A simple approach is to only reduce the precision of the weights of the network to 8-bits from float. Since only the weights are quantized, this can be done without requiring any validation data (See figure 2). A simple command line tool can convert the weights from float to 8-bit precision. This setup is useful if one only wants to reduce the model size for transmission and storage and does not mind the cost of performing inference in floating point.

一个很简单的方法是只降低网络中权重的精度，从浮点降到8-bits。由于只对权重量化，所以不需要任何验证数据（图2）。一个简单的命令行工具就可以将权重从浮点转化为8-bit精度。如果降低模型大小是为了传输和存储，而不关心用浮点精度进行推理，那么这种设置是有用的。

#### 3.1.2 Quantizing weights and activations 量化权重和激活

One can quantize a floating point model to 8-bit precision by calculating the quantizer parameters for all the quantities to be quantized. Since activations need to be quantized, one needs calibration data and needs to calculate the dynamic ranges of activations. (See figure 2) Typically, about 100 mini-batches are sufficient for the estimates of the ranges of the activation to converge.

将浮点模型量化到8-bit精度，可以计入所有需要量化的数值，然后计算量化参数。由于激活也需要量化，所以需要校准数据，计算激活的动态范围（见图2）。一般来说，大约100 mini-batches的数据就足可以估计激活的范围。

Figure 2: Overview of schemes for model quantization: One can quantize weights post training (left) or quantize weights and activations post training (middle). It is also possible to perform quantization aware training for improved accuracy. 模型量化的方案：训练后只量化权重（左），训练后量化权重和激活（中），针对量化进行训练（右）。

#### 3.1.3 Experiments 试验

For evaluating the tradeoffs with different quantization schemes, we study the following popular networks and evaluate the top-1 classification accuracy. Table 1 shows the wide variation in model size and accuracy across these networks. We note that Mobilenet-v1 [2] and Mobilenet-v2[1] architectures use separable depthwise and pointwise convolutions with Mobilenet-v2 also using skip connections. Inception-v3 [18] and NasNet [19] use network in network building blocks with NasNet determining the architecture via reinforcement learning techniques. Resnets [20] pioneered the idea of skip connections and consist of multiple blocks each making residual corrections to the main path with no transformations. Resnet-v2 [21] is an enhancement to the resnet architecture using pre-activation layers for improved accuracy. Note that all results are obtained using simulated quantization of weights and activations.

为评估不同量化方案的利弊，我们研究如下流行的网络，对其top-1分类准确率进行评估。表1给出了这些模型在规模上和准确率上的不同。我们注意到，Mobilenet-v1 [2] 和 Mobilenet-v2[1]架构使用了separable depthwise和pointwise卷积，MobileNet-v2还使用了跳跃连接。Inception-v3[18]和NasNet[19]使用network-in-network模块，NasNet通过强化学习技术来确定网络架构。ResNets[20]是跳跃连接的提出者，包括了多个模块，每个都与主路径有跳跃连接，没有变换。ResNet-v2[21]是ResNet架构的增强，使用了预激活层，以改进准确率。注意所有结果都使用了权重和激活的模拟量化技术得到的。

Table 1: Deep Convolutional networks: Model size and accuracy

Network | Parameters | top-1 Acc.
--- | --- | ---
Mobilenet-V1-0.25-128 | 0.47M | 0.415
Mobilenet-V2-1-224 | 3.54M | 0.719
Mobilenet-V1-1-224 | 4.25M | 0.709
Nasnet-Mobile | 5.3M | 0.74
Mobilenet-V2-1.4-224 | 6.06M | 0.749
Inception-V3 | 23.9M | 0.78
Resnet-v1-50 | 25.6M | 0.752
Resnet-v2-50 | 25.6M | 0.756
Resnet-v1-152 | 60.4M | 0.768
Resnet-v2-152 | 60.4M | 0.778

**Weight only quantization**: We first quantize only the weights post training and leave the activations un-quantized. From figure 2, we note that per-channel quantization is required to ensure that the accuracy drop due to quantization is small, with asymmetric, per-channel quantization providing the best accuracy.

**只量化权重**：我们首先进行训练后只量化权重，激活不进行量化。从图2中，我们注意到，需要逐通道量化来保证量化导致的准确率下降很小，非对称、逐层的量化可以得到最好的准确率。

Table 2: Weight only quantization: per-channel quantization provides good accuracy, with asymmetric quantization providing close to floating point accuracy.

Network | Asymmetric, per-layer | Symmetric, per-channel | Asymmetric, per-channel | Floating Point
--- | --- | --- | --- | ---
Mobilenetv1-1-224 | 0.001 | 0.591 | 0.704 | 0.709
Mobilenetv2-1-224 | 0.001 | 0.698 | 0.698 | 0.719
NasnetMobile-|-0.722 | 0.721 | 0.74 | 0.74
Mobilenetv2-1.4-224 | 0.004 | 0.74 | 0.74 | 0.749
Inceptionv3-|-0.78 | 0.78 | 0.78 | 0.78
Resnet-v1-50 | 0.75 | 0.751 | 0.752 | 0.752
Resnet-v2-50 | 0.75 | 0.75 | 0.75 | 0.756
Resnet-v1-152 | 0.766 | 0.763 | 0.762 | 0.768
Resnet-v2-152 | 0.761 | 0.76 | 0.77 | 0.778

**Weight and Activation Quantization**: Next, we quantize weights and activations to 8-bits, with per-layer quantization for activations. For weights we consider both symmetric and asymmetric quantizers at granularities of both a layer and a channel. We first show results for Mobilenetv1 networks and then tabulate results across a broader range of networks.

**量化权重和激活**：下一步，我们将权重和激活量化到8-bit，对激活采用逐层量化。对权重，我们考虑对称和非对称量化器，粒度包括层和通道。我们首先给出MobileNet-v1的结果，然后用表格给出更多网络的结果。

We also compare the post training quantization accuracies of popular convolutional networks: Inception-V3, Mobilenet-V2, Resnet-v1-50, Resnet-v1-152, Resnet-v2-50, Resnet-v2-152 and Nasnet-mobile on ImageNet in figure 4.

我们还比较了流行卷积网络训练后量化的准确率：Inception-V3, Mobilenet-V2, Resnet-v1-50, Resnet-v1-152, Resnet-v2-50, Resnet-v2-152 and Nasnet-mobile，如图4所示，在ImageNet上的准确率。

Figure 3: Comparison of post training weight and activation quantization schemes:Mobilenet-v1

Figure 4: Comparison of post training quantization schemes

Table 3: Post training quantization of weights and activations: per-channel quantization of weights and per-layer quantization of activations works well for all the networks considered, with asymmetric quantization providing slightly better accuracies.

Network | Asymmetric, per-layer | Symmetric, per-channel | Asymmetric, per-channel | Activation Only | Floating Point
--- | --- | --- | --- | --- | ---
Mobilenetv1-1-224 | 0.001 | 0.591 | 0.703 | 0.708 | 0.709
Mobilenetv2-1-224 | 0.001 | 0.698 | 0.697 | 0.7 | 0.719
NasnetMobile | 0.722 | 0.721 | 0.74 | 0.74 | 0.74
Mobilenetv2-1.4-224 | 0.004 | 0.74 | 0.74 | 0.742 | 0.749
Inceptionv3 | 0.78 | 0.78 | 0.78 | 0.78 | 0.78
Resnet-v1-50 | 0.75 | 0.751 | 0.751 | 0.751 | 0.752
Resnet-v2-50 | 0.75 | 0.75 | 0.75 | 0.75 | 0.756
Resnet-v1-152 | 0.766 | 0.762 | 0.767 | 0.761 | 0.768
Resnet-v2-152 | 0.761 | 0.76 | 0.76 | 0.76 | 0.778

We make the following observations: 我们做出如下观察：

1. Per-channel quantization can provide good accuracy and can be a good baseline for post training quantization of weights and activations, with asymmetric quantization providing close to floating point accuracy for all networks. 逐通道量化可以给出很好的准确率，可以作为权重和激活训练后量化的很好基准，非对称量化在所有网络上都可以得到更接近浮点准确率的结果。

2. Activations can be quantized to 8-bits with almost no loss in accuracy. The dynamic ranges of the activations are low due to a combination of: 激活量化为8-bits几乎不损失准确率。激活的动态范围很窄，因为如下两个因素：

(a) Batch normalization with no scaling: Used in Inception-V3, which ensures that the activations of all feature maps have zero mean and unit variance. 没有尺度的BN层：在Inception-v3中使用，这确保了所有特征图的激活都是0均值单位方差的。

(b) ReLU6: Used in Mobilenet-V1, which restricts the activations to be in a fixed range (0,6) for all feature maps, thereby removing large dynamic range variations. ReLU6使用在MobileNet-v1中，这使得所有特征图的激活的范围限制在(0,6)，所以不会有大的动态范围变化。

3. Networks with more parameters like Resnets and Inception-v3 are more robust to quantization compared to Mobilenets which have fewer parameters. 网络参数越多，如ResNets和Inception-v3，比参数少的网络对量化更稳健，如MobileNets。

4. There is a large drop when weights are quantized at the granularity of a layer, particularly for Mobilenet architectures. 如果权重以层为单位进行量化，准确率会大为下降，尤其是对于MobileNet框架。

5. Almost all the accuracy loss due to quantization is due to weight quantization. 几乎所有的准确率损失都是由权重量化造成的。

Weight quantization at the granularity of a layer causes large accuracy drops primarily due to batch normalization, which causes extreme variation in dynamic range across convolution kernels in a single layer. Appendix A has more details on the impact of batch normalization. Per-channel quantization side-steps this problem by quantizing at the granularity of a kernel, which makes the accuracy of per-channel quantization independent of the batch-norm scaling. However, the activations are still quantized with per-layer symmetric quantization.

以层为单位权重量化，会导致准确率大幅下降，这是因为BN的原因，这会导致同一层内不同卷积核的动态范围变化很极端。附录A详述了BN的影响。逐通道量化规避了这一问题，因为是在核这一粒度上进行的量化，所以逐层量化与BN的尺度是没有关系的。但是，激活仍然是按照逐层对称量化的。

Note that other approaches like weight regularization can also improve the accuracy of quantization post training, please see [22]. 其他方法，如权重正则化，也可以改进训练后量化的准确度，请看[22]。

### 3.2 Quantization Aware Training 针对量化的训练

Quantization aware training models quantization during training and can provide higher accuracies than post quantization training schemes. In this section, we describe how quantization is modeled during training and describe how this can be easily done using automatic quantization tools in TensorFlow. We also evaluate the accuracies obtained for different quantization schemes with quantization aware training and show that even per-layer quantization schemes show high accuracies post training at 8-bits of precision. We also show that at 4 bit precision, quantization aware training provides significant improvements over post training quantization schemes.

针对量化的训练在训练的过程中对量化进行建模，可以得到比训练后量化方案更高的准确率。在本节中，我们叙述训练过程中量化是怎样建模的，怎样使用TensorFlow中的自动量化工具轻松的建模。我们还评估了不同量化方案在针对量化的训练下得到的准确率，表明即使是逐层量化的方案，在训练后的8-bits量化下也可以给出很高的准确率。我们还在4-bit精度下进行了试验，针对量化的训练比训练后量化方案有明显的改进。

We model the effect of quantization using simulated quantization operations on both weights and activations. For the backward pass, we use the straight through estimator (see section 2.4) to model quantization. Note that we use simulated quantized weights and activations for both forward and backward pass calculations. However, we maintain weights in floating point and update them with the gradient updates. This ensures that minor gradient updates gradually update the weights instead of underflowing. The updated weights are quantized and used for subsequent forward and backward pass computation. For SGD, the updates are given by:

我们使用权重和激活的模拟量化运算来对量化的效果进行建模。对反向过程，我们使用直通估计器（见2.4节）来对量化建模。注意我们在前向和反向的计算过程中都使用模拟的量化权重和激活。但是，我们维持权重在浮点格式，使用梯度来进行更新。这确保了很小的梯度更新逐渐的更新权重，而不是下溢出。更新的权重被量化，并用于后续的前向和反向过程计算。对于SGD，更新由下式给出：

$$w_{float} = w_{float} - η \frac{∂L}{∂w_{out}} I_{w_{out}∈(w_{min}, w_{max})}$$(16)
$$w_{out} = SimQuant(w_{float})$$(17)

Quantization aware training is achieved by automatically inserting simulated quantization operations in the graph at both training and inference times using the quantization library at [23] for Tensorflow [24]. We follow the approach outlined in [4] closely, with additional enhancements on handling batch normalization and in modeling quantization in the backward pass. A simple one-line change to the training or evaluation code automatically inserts simulated quantization operations into the training or eval graph.

针对量化的训练是通过在训练和推理时，在图中自动插入模拟量化运算得到的，使用的是[23]中为TensorFlow[24]开发的库。我们使用最近[4]中列出的方法，还有处理BN和在反向过程中对量化建模的附加强化。训练或评估的代码中，只需要简单的修改一行，就可以在训练或评估图中增加模拟量化运算。

For training, the code snippet is: 对于训练的代码片段是
```
# Build forward pass of model.
...
loss = tf.losses.get_total_loss()

# Call the training rewrite which rewrites the graph in-place
# with FakeQuantization nodes and folds batchnorm for training.
# One can either fine tune an existing floating point model
# or train from scratch. quant_delay controls the onset
# of quantized training.

tf.contrib.quantize.create_training_graph(quant_delay=2000000)

# Call backward pass optimizer as usual.

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer.minimize(loss)
```

For evaluation, the code snippet is given below: 对于推理，代码片段是：
```
# Build eval model
...
logits, end_points = network_model(inputs,...)
# Call the eval rewrite which rewrites the graph in-place
# with FakeQuantization nodes and fold batchnorm for eval.

tf.contrib.quantize.create_eval_graph()
```

The high level conversion process is shown in figure 2. 高层的转换过程如图2所示。

The steps involved in training a quantized model are: 训练一个量化模型的步骤包括：

1. (Recommended): Fine tune from a floating point saved model: Start with a floating point pre-trained model or alternately train from scratch. 推荐：从一个浮点型的保存模型进行精调：从浮点的预训练模型开始，或从头开始训练。

2. Modify Estimator to add quantization operations: Add fake quantization operations to the model using the quantization rewriter at tf.contrib.quantize. 修改Estimator以增加量化运算：使用tf.contrib.quantize中的量化重写器，给模型增加假量化运算。

3. Train model: At the end of this process, we have a savedmodel with quantization information (scale, zero-point) for all the quantities of interest. (weights and activations). 训练模型：在这个过程的最后，我们有了一个保存好的模型savedmodel，包含所有感兴趣的量的量化信息（尺度、零点）。

4. Convert model: The savedmodel with range information is transformed into a flatbuffer file using the tensorflow converter (TOCO) at: tf.contrib.lite.toco_convert. This step creates a flatbuffer file that converts the weights into integers and also contains information for quantized arithmetic with activations. 转化模型：包含范围信息的savedmodel使用在tf.contrib.lite.toco_convert中tensorflow converter(TOCO)转换成一个flatbuffer文件。这一步骤生成一个flatbuffer文件，将权重转化成整数，也包含激活的量化计算信息。

5. Execute model: The converted model with integer weights can now be executed using the TFLite interpreter which can optionally execute the model in custom accelerators using the NN-API. One can also run the model on the CPU. 执行模型：转换的模型是整数权重的，现在可以使用TFLite解释器来执行，使用NN-API也可以在定制的加速器中运行模型。

A simple example showing the graph transformation for a convolutional layer is shown in figure 5. 卷积层的图变换的一个简单例子如图5所示。

Figure 5: Convolutional layer: Before and After Graph Transformation

#### 3.2.1 Operation Transformations for Quantization 量化的运算变换

It is important to ensure that all quantization related artifacts are faithfully modeled at training time. This can make trivial operations like addition, figure 6 and concatenation , figure 7 non-trivial due to the need to rescale the fixed point values so that addition/concatenation can occur correctly.

确保量化相关的所有artifacts在训练时都忠实的被建模了，这很重要。这会使得不重要的运算变得重要起来，如加法（图6）和拼接（图7），这是定点值的重新变换尺度的原因，只有这样加法和拼接这样的运算可以正确的执行。

In addition, it is important to ensure that fusion of operations at inference time is modeled correctly during training. For example, consider an add followed by a ReLU operation. In this case, one can fuse the addition and the ReLU operation at inference time in most platforms. To match this, fake quantization operations should not be placed between the addition and the ReLU operations.

另外，要确保推理时算子的融合在训练时得到正确的建模，这也非常重要。比如，考虑加法与ReLU的前后组合。在这种情况下，在多数平台上都可以在推理时将加法和ReLU算子进行融合。为与之相匹配，在加法和ReLU运算之间，不应当放入假量化运算。

Figure 6: Fixed point transformation of element-wise add

Figure 7: Fixed point transformation of concat

#### 3.2.2 Batch Normalization 批归一化

In this section, we describe several strategies for quantizing batch normalization layers. In section 4 and show that batch normalization with correction and freezing provides the best accuracy. 本节中，我们叙述几种量化BN层的策略。在第4部分中我们证明，带有矫正和冻结的BN可以得到最好的准确率。

Batch normalization [25], is a popular technique that normalizes the activation statistics at the output of every layer, reducing dependencies across layers while significantly improving model accuracy. BN[25]是一种流行的技术，在每一层的输出将激活的统计进行归一化，降低了层与层之间的依赖关系，显著改进了模型准确率。

Batch normalization is defined by the following equations: BN由下式定义

$$x_{bn} = γ(\frac{x − μ_B}{σ_B}) + β$$(18)

for training and 这是训练过程的

$$x_{bn} = γ(\frac{x − μ}{σ}) + β$$(19)

for inference. 这是推理过程的。

Where $μ_B$ and $σ_B$ are the batch mean and standard deviations. μ and σ are the long term mean and standard deviations and are computed as moving averages of batch statistic during training. 其中$μ_B$和$σ_B$是批平均和标准差。μ和σ是长期均值和标准差，由训练过程中的批次统计滑动平均计算得到。

For inference, we fold the batch normalization into the weights as defined by equations 20 and 21. Therefore, at inference there is no explicit batch normalization. The weights and biases are modified to account for batch normalization instead: 对于推理，我们将批归一化与权重归属到一起，由式20和21定义。所以，在推理时，没有显式的批归一化。权重和偏置是修改过的，用于BN。

$$W_{inf} = \frac{γW}{σ}$$(20)

$$Bias_{inf} = β − \frac{γμ}{σ}$$(21)

For quantized inference, we first consider folding batch norms and training them as shown in figure 8. We note that batch normalization uses batch statistics during training, but uses long term statistics during inference. Since the batch statistics vary every batch, this introduces undesired jitter in the quantized weights and degrades the accuracy of quantized models. (green curve in 14 and 15) A simple solution would be to switch to using long term moving averages during training, however, this eliminates batch normalization (i.e the mean and variance used do not correspond to the batch statistics) and causes instability in training. The graph rewriter implements a solution that eliminates the mismatch between training and inference with batch normalization (see figure 9):

对于量化的推理，我们首先考虑将BN与训练放到一起，如图8所示。BN在训练时使用的是批次统计信息，但在推理时使用的是长期统计数据。由于批次统计信息不同的批次是不一样的，这会在量化的权重中引入不想要的抖动，会降低量化模型的准确率（图14和15中的绿线）。简单的解决方案是，在训练时转而使用长期滑动平均，但是，这就不是批归一化了（即均值和方差不是批次统计数据），会在训练中导致不稳定。图重写器实现了一种解决方案，消除了使用BN在训练和推理时的不匹配（见图9）：

Figure 8: Baseline approach for folding batch norms for quantized inference

Figure 9: Folding Batch normalization layers for improved performance

1. We always scale the weights with a correction factor to the long term statistics prior to quantization. This ensures that there is no jitter in the quantized weights due to batch to batch variation. 我们永远用相关系数来来将权重的尺度在量化前改变为长期统计数据。这确保了在量化的权重中没有批次变化导致的抖动。

$$c = \frac{σ_B}{σ}$$(22)
$$w_{corrected} = c ×\frac{γW}{σ_B}$$(23)

2. During the initial phase of training, we undo the scaling of the weights so that outputs are identical to regular batch normalization. We also modify the bias terms correspondingly. 在训练的初始阶段，我们取消对权重的尺度变化，所以输出与常规BN完全一样。我们也相应的修改了偏置项。

$$y = conv(Q(w_{corrected}), x)$$(24)
$$y_{corrected} = y/c$$(25)
$$bias = β − γμ_B /σ_B$$(26)
$$bias_{corrected} = 0$$(27)

3. After sufficient training, switch from using batch statistics to long term moving averages for batch normalization, using the optional parameter freeze bn delay in create_experimental_training_graph() (about 300000 steps in figure 15 and 200000 in figure 14). Note that the long term averages are frozen to avoid instability in training. This corresponds to the normalization parameters used at inference and provides stable performance. 在足够的训练后，从使用批统计数据转换到使用长期滑动平均，进行批次归一化，使用可选的参数在create_experimental_training_graph()中冻结bn延迟（在图15中，大约是300000步，在图14中，是200000步）。注意，长期平均是冻结的，以避免训练中的不稳定性。这对应着在推理时的归一化参数，可以给出稳定的性能。

$$y = conv(Q(w_{corrected}), x)$$(28)
$$y_{corrected} = y$$(29)
$$bias = β − γμ_B /σ_B$$(30)
$$bias_{correction} = γ(μ_B /σ_B − μ/σ)$$(31)

#### 3.2.3 Experiments 试验

Quantization aware training closes the gap to floating point accuracy, even for perlayer quantization of weights. We repeat the same experiments for quantized weights and activations with training, starting from a floating point check-point and with batch normalization freezing and obtain the results shown in figures 11 and 10 and Table 4. 针对量化的训练缩小了与浮点准确率的差距，即使对于逐层的权重量化也是。我们重复量化权重、激活训练相同的试验，从浮点检查点、有BN冻结开始，得到的结果如图11、图10和表4所示。

All the experiments have the following settings: 所有的试验设置都如下所示：

- Fine tune from a floating point checkpoint, we used the models in [26]. 从浮点检查点开始精调，我们使用[26]中的模型。
- Use Stochastic Gradient Descent for fine tuning, with a step size of 1e-5. 使用SGD进行精调，步长为1e-5。

We note that: 我们注意到

1. Training closes the gap between symmetric and asymmetric quantization. 训练缩小了对称和非对称量化之间的差距。
2. Training allows for simpler quantization schemes to provide close to floating point accuracy. Even per-layer quantization shows close to floating point accuracy (see column 4 in Table 4) 训练后，更简单的量化方案也可以得到接近浮点的准确率。即使逐层量化，也可以得到接近浮点准确率的效果（见表4中的列4）。

Figure 10: Comparison of quantization-aware training schemes:Mobilenet-v1

Figure 11: Comparison of quantization-aware training schemes (different models)

Table 4: Quantization aware training provides the best accuracy and allows for simpler quantization schemes 表4：针对量化的训练可以得到最好的准确率，而且可以使用更简单的量化方案

Asym: asymmetric, p-l: per-layer, PTQ: post-training quantization, QAT: quantization-aware training, FP: floating point

Network | Asym,p-l(PTQ) | Sym,p-c(PTQ) | Asym,p-l(QAT) | Sym,p-c(QAT) | FP
--- | --- | --- | --- | --- | ---
Mobilenet-v1-1-224 | 0.001 | 0.591 | 0.70 | 0.707 | 0.709
Mobilenet-v2-1-224 | 0.001 | 0.698 | 0.709 | 0.711 | 0.719
Nasnet-Mobile | 0.722 | 0.721 | 0.73 | 0.73 | 0.74
Mobilenet-v2-1.4-224 | 0.004 | 0.74 | 0.735 | 0.745 | 0.749
Inception-v3 | 0.78 | 0.78 | 0.78 | 0.78 | 0.78
Resnet-v1-50 | 0.75 | 0.751 | 0.75 | 0.75 | 0.752
Resnet-v2-50 | 0.75 | 0.75 | 0.75 | 0.75 | 0.756
Resnet-v1-152 | 0.766 | 0.762 | 0.765 | 0.762 | 0.768
Resnet-v2-152 | 0.761 | 0.76 | 0.76 | 0.76 | 0.778

#### 3.2.4 Lower Precision Networks 低精度网络

We note that at 8-bits of precision, post training quantization schemes provide close to floating point accuracy. In order to better understand the benefits of quantization aware training, we perform experiments to assess performance at 4 bit quantization for weights and activations. 我们注意到，在8-bit精度上，训练后量化方案可以得到接近浮点的准确率。为更好的理解针对量化的训练，我们进行试验评估权重和激活的4-bit量化性能。

We perform the following experiments: 我们进行以下试验：

- Experiment 1: Per-channel quantization is significantly better than per-layer quantization at 4 bits. We show that per-channel quantization provides big gains over per-layer quantization for all networks. At 8-bits, the gains were not significant as there were sufficient levels to represent the weights with high fidelity. At four bits, the benefits of per-channel quantization are apparent, even for post training quantization (columns 2 and 3 of Table 5).

- 试验1：在4-bit量化的情况下，逐通道量化明显比逐层量化要好。我们展示了，对于所有网络，逐通道量化都比逐层量化的结果要好的多。在8-bit下，好的程度不是很明显，因为有足够多的等级来表示权重，保真度很高。在4-bit的情况下，逐通道量化的好处就很明显了，即使是训练后量化也是这样（表5的第2,3列）。

- Experiment 2: Fine tuning can provide substantial accuracy improvements at lower bitwidths. It is interesting to see that for most networks, one can obtain accuracies within 5% of 8-bit quantization with fine tuning 4 bit weights (column 4 of Table 5). The improvements due to fine tuning are also more apparent at 4 bits. Note that activations are quantized to 8-bits in these experiments.

- 试验2：精调可以在更低精度上得到很多准确率改进。对于多数网络，精调4-bit权重得到的准确率，与8-bit量化的不超过5%（表5，第4列）。由于精调得到的改进在4-bit下也更明显。注意，在这些试验中，激活是量化到8-bit的。

- Experiment 3: Lower precision activations: We investigate the accuracies obtained with 4-bit activations for all layers with and without fine tuning. Note that activations are quantized on a per-layer basis. The weights are quantized at 8-bits of precision with per-channel granularity. We note that fine tuning improves accuracy in this case also. The losses due to activation quantization are more severe than that of weight quantization (see Table 6). Note that the quantization granularity is different for activations and weights, so this is not a fair comparison of the impact of quantization. We hypothesize that quantizing activations introduces random errors as the activation patterns vary from image to image, while weight quantization is deterministic. This allows for the network to learn weight values to better compensate for the deterministic distortion introduced by weight quantization.

- 试验3：更低精度的激活：我们研究了所有层都是4-bit激活，在有精调和没有精调下，得到的准确率。注意，激活的量化是逐层的。权重是8-bit精度逐层量化的。我们注意到，精调在这些情况下也改进了准确率。激活量化导致的损失，比权重量化导致的损失要严重（见表6）。注意，激活和权重的量化粒度是不一样的，所以量化影响的比较是不公平的。我们假设，量化激活带来随机误差，因为激活的模式每幅图像都不一样，而权重量化则更具有确定性。这使得网络可以学习权重值，以更好的补偿权重量化带来的确定性的误差。

Table 5: 4-bit Weight Quantization: per-channel quantization outperforms per-layer quantization, with fine tuning providing big improvements. 4-bit权重量化：逐通道量化的准确率超过逐层量化，精调则会得到很大的改进。

Table 6: 4 bit Activation Quantization with and without fine tuning. Note that weights are quantized with symmetric per-channel quantization at 8-bits. We also show results for 4-bit per-channel quantization of weights with 8-bit activations to compare with 8-bit weights and 4-bit activations. 4-bit激活量化，有精调和没有精调的结果。注意，权重量化是8-bit对称逐通道量化。我们还给出权重4-bit逐通道量化，和8-bit激活量化的结果，与权重8-bit、激活4-bit量化的对比。

## 4 Training best practices

We experiment with several configurations for training quantized models: Our first experiment compares stochastic quantization with deterministic quantization. Subsequently, we study if training a quantized model from scratch provides higher accuracies than fine tuning from a floating point model. We also evaluate different methods for quantizing batch normalization layers and show that batch normalization with corrections provides the best accuracy. We also compare schemes that average weights during training with no averaging.

我们用几种训练量化模型的配置进行试验：我们的第一个试验比较的是随机量化和确定性量化；然后，我们研究一下，从头训练一个量化模型，是否会比精调浮点模型带来更高的准确率。我们还评估不同的量化BN层方法，证明了带修正的BN层给出最好的准确率。我们还比较了在训练过程中平均权重和不平均的方案。

1. **Stochastic Quantization does not improve accuracy**: Stochastic quantization determines floating point weights that provide robust performance under stochastic quantization, which causes the quantized weights to vary from mini-batch to mini-batch. At inference, quantization is deterministic, causing a mismatch with training. We observe that due to this mis-match, stochastic quantization underperforms determinstic quantization (figure 12), which can be compensated better during training. **随机量化不会改进准确率**：随机量化确定了浮点权重在随机量化的时候可以给出稳健的性能，使得量化的权重每个mini-batch都不同。在推理时，量化是确定性的，导致与训练不匹配。我们观察到，由于不匹配，随机量化没有确定性量化的效果好（见图12）。

Figure 12: Comparison of stochastic quantization vs deterministic quantization during training

2. **Quantizing a model from a floating point checkpoint provides better accuracy**: The question arises as to whether it is better to train a quantized model from scratch or from a floating point model. In agreement with other work [27], we notice better accuracy when we fine tune a floating point model as shown in figure 13. This is consistent with the general observation that it is better to train a model with more degrees of freedom and then use that as a teacher to produce a smaller model ([28]). **从浮点检查点中量化模型可以得到更高的准确率**：从头训练一个量化模型更好，还是从浮点模型量化更好。我们与其他工作[27]得到了一样的结论，从浮点模型精调可以得到更好的准确率，如图13所示。通常的结论是，以更大自由度来进行模型训练会更好，然后将其用作老师来生成更小的模型[28]，我们的结论与此一致。

Figure 13: Fine tuning a floating point checkpoint provides better fixed point accuracy

3. **Matching Batch normalization with inference reduces jitter and improves accuracy**. We show results for two networks. In the first experiment (see figure 14), we compare training with naive batch norm folding, batch renormalization and batch normalization with correction and freezing for Mobilenet-v1-1-224. We note stable eval accuracy and higher accuracy with our proposed approach. In the second experiment, we compare naive batch norm folding and batch normalization with correction and freezing for Mobilenet-v2-1-224. We note that corrections stabilize training and freezing batch norms provides additional accuracy gain, seen after step 400000 in figure 15. **将BN与推理匹配，会降低抖动，改进准确率**。我们给出两个网络的结果。第一个试验（见图14），我们在MobileNet-v1-1-224上进行比较，简单的BN折叠，批次重归一化，有修正的BN，和冻结方案。我们注意到，我们提出的方法得到稳定、更高的评估准确率。在第二个试验中，我们比较Mobilenet-v2-1-224在上述方案下的结果，我们注意到，修正使训练更稳定，冻结方案在400k步后，可以得到额外的准确率提高，如图15。

Figure 14: Mobilenet-v1-1-224: Comparison of Batch normalization quantization schemes: Batch normalization without corrections (green) shows a lot of jitter due to the changing scaling of weights from batch to batch. Batch renormalization (red) improves the jitter, but does not eliminate it. Quantizing the weights using moving average statistics reduces jitter, but does not eliminate it (orange). Freezing the moving mean and variance updates after step 200000 allows for quantized weights to adapt to the batch norm induced scaling and provides the best accuracy with minimal jitter (blue curve).

Figure 15: Mobilenet-v2-1-224: Impact of batch normalization corrections and freezing on accuracy. Quantizing without corrections shows high jitter (green curve). Correction with freezing show good accuracy (blue and red curves). The jitter in the eval accuracy drops significantly after moving averages are frozen (400000 steps). Note under performance of EMA weights (red curve) after sufficient training.

4. **Use Exponential moving averaging for quantization with caution**. Moving averages of weights [29] are commonly used in floating point training to provide improved accuracy [30]. Since we use quantized weights and activations during the back-propagation, the floating point weights converge to the quantization decision boundaries. Even minor variations in the floating point weights, between the instantaneous and moving averages can cause the quantized weights to be significantly different, hurting performance, see drop in accuracy for the EMA curve in figure 15. **使用指数滑动平均进行量化要小心**。权重滑动平均[29]在浮点训练中经常使用，可以得到更好的准确率[30]。由于我们在反向传播中使用量化权重和激活，浮点权重收敛到量化决策边界。即使是浮点权重很小的改变，可以导致量化权重的显著差异，使性能受到损失，可以在图15中看到EMA曲线的准确率下降。

## 5 Model Architecture Recommendations 模型架构推荐

In this section, we explore choices of activation functions and tradeoffs between precision and width of a network. 本节中，我们研究激活函数的选择，和网络精度与宽度的折中。

- **Do not constrain activation ranges**: One can get slightly better accuracy by replacing ReLU6 non-linearity with a ReLU and let the training determine the activation ranges (see figure 16). **不要限制激活的范围**：将ReLU6替换为ReLU，让训练来确定激活范围，可以得到略微更好的结果（见图16）。

Figure 16: Accuracy improvement of training with ReLU over ReLU6 for floating point and quantized mobilenet-v1 networks.

- **Explore tradeoff of width vs quantization**: An over-parameterized model is more amenable to quantization. Even for leaner architectures like mobilenet, one can tradeoff the depth multiplier with the precisions of the weights. We compare the accuracies obtained with 4 bit per-channel quantization of weights with 8-bit quantization across different depth multipliers in figure 17. Note that this comparison allows us to evaluate a depth vs quantization tradeoff (see [31]). It is interesting to see that one can obtain a further 25% reduction in the model size for almost the same accuracy by moving to 4 bit precision for the weights. **研究宽度和量化的折中关系**：参数越多的模型对量化越不敏感。即使是较小的架构，如MobileNet，在depth multiplier和权重精度之间也有折中关系。我们比较了4-bit逐通道权重量化，和8-bit量化，在不同depth multiplier时的结果，如图17所示。注意这种比较，使我们评估深度与量化的折中关系（见[31]）。通过将权重量化为4-bit，可以将模型进一步缩小25%，而且准确率几乎一样。

Figure 17: Width vs Precision tradeoff, illustrated for Mobilenet-v1 0.25 128, per-channel quantization of weights.

## 6 Run-time measurements 运行时测量结果

We measure the run-times (Table 7) on a single large core of the Google Pixel 2 device for both floating point and quantized models. We also measure run-times using the Android NN-API on Qualcomm’s DSPs. We see a speedup of 2x to 3x for quantized inference compared to float, with almost 10x speedup with Qualcomm DSPs.

我们在表7中，比较了在Google Pixel 2设备的大核上，浮点和量化模型的运行时间。我们还使用Android NN-API在高通的DSP上计算了运行时间。量化模型比浮点模型推理速度要快2x-3x，在高通DSP上快了几乎10x。

Table 7: Inference time measurements on Pixel2 phone in milliseconds on a single large core.

## 7 Neural network accelerator recommendations 神经网络加速器推荐

In order to fully realize the gains of quantized networks, we recommend that neural network accelerators consider the following enhancements: 为完全实现量化网络的好处，我们推荐神经网络加速器考虑下面的增强：

1. **Aggressive operator fusion**: Performing as many operations as possible in a single pass can lower the cost of memory accesses and provide significant improvements in run-time and power consumption. **更多的算子融合**：在一次操作中执行尽可能多的运算，可以降低内存访问的代价，在运行时间和能耗上得到显著的改进。

2. **Compressed memory access**: One can optimize memory bandwidth by supporting on the fly de-compression of weights (and activations). A simple way to do that is to support lower precision storage of weights and possibly activations. **压缩的内存访问**：通过支持权重（和激活）解压，可以优化内存带宽。支持更低精度存储权重（和可能的激活），是一种简单的方法。

3. **Lower precision arithmetic**: One can obtain further acceleration by supporting a range of precisions for arithmetic. Our recommendation is to support 4,8 and 16-bit weights and activations. While 4 and 8-bit precisions are sufficient for classification, higher precision support is likely needed for regression applications, like super-resolution and HDR image processing. **更低精度的代数运算**：通多支持多精度的代数运算，可以得到进一步的加速。我们推荐是要支持4，8和16-bit权重和激活。4,8-bit对于分类是足够的，回归类的应用可能需要更高精度的支持，如超分辨率和HDR图像处理。

4. **Per-layer selection of bitwidths**: We expect that many layers of a network can be processed at lower precision. Having this flexibility can further reduce model size and processing time. **逐层选择bit-width**：我们认为网络的很多层可以用更低的精度进行处理。有这种灵活性的话，可以进一步降低模型大小和处理时间。

5. **Per-channel quantization**: Support for per-channel quantization of weights is critical to allow for: **逐通道量化**：支持权重的逐通道量化，对于下面两点非常关键：

(a) Easier deployment of models in hardware, requiring no hardware specific fine tuning. 更容易在硬件上部署，不需要针对硬件的精调。

(b) Lower precision computation. 更低精度的计算。

## 8 Conclusions and further work 结论和未来的工作

Based on our experiments, we make the following conclusions: 基于我们的试验，我们得出如下结论：

**Quantizing models** 量化模型

1. Use symmetric-per-channel quantization of weights with post training quantization as a starting point. Optionally fine tune if there is an accuracy drop. 使用对称逐通道的权重量化和训练后量化作为开始。如果准确率有下降，那么可以选择性的进行精调。

2. Quantization aware training can narrow the gap to floating point accuracy and in our experiments, reduce the gap to within 5% of 8-bit quantized weights, even when all layers are quantized to 4 bits of precision. 在我们的试验中，针对训练的量化可以缩小与浮点准确率的差距；即使所有层都量化到4-bit精度，也可以达到与8-bit量化的权重的精度5%以内的差距。

**Performance** 性能

1. Quantized inference at 8-bits can provide 2x-3x speed-up on a CPU and close to 10x speedup compared to floating point inference on specialized processors optimized for low precision wide vector arithmetic, like the Qualcomm DSP with HVX. 8-bit的量化推理，与浮点推理相比，在CPU上可以达到加速2x到3x，在专用处理器上可以达到10x的加速，如专门为低精度宽向量代数运算优化的处理器，Qualcomm DSP with HVX。

2. One can obtain a model size reduction of 4x with no accuracy loss with uniform quantization. Higher compression can be obtained with non-uniform quantization techniques like K-means ([6]). 采用均匀量化，可以将模型大小降低4倍，准确率不变。更高的压缩率可以通过非均匀量化技术得到，如k-均值[6]。

**Training Techniques** 训练技术

1. Quantization aware training can substantially improve the accuracy of models by modeling quantized weights and activations during the training process. 针对训练的量化，通过在训练过程中模拟权重和激活的量化，可以极大的改进模型准确率。

2. It is critical to match quantized inference with the forward pass of training. 量化推理与训练的前向过程要匹配，这很关键。

3. Special handling of batch normalization is required to obtain improved accuracy with quantized models. BN层需要特别处理，以改进量化模型的准确率。

4. Stochastic quantization during training underperforms deterministic quantization. 训练过程中的随机量化，没有确定性的量化好。

5. Exponential Moving Averages of weights may under-perform instantaneous estimates during quantization aware training and must be used with caution. 指数滑动平均的权重在针对量化的训练中可能没有即时估计效果好，使用起来必须要小心。

**Model architectures for quantization** 量化的模型架构

1. There is a clear tradeoff between model size and compressibility. Larger models are more tolerant of quantization error. 模型大小和可压缩性有明显的折中关系。更大的模型对量化误差容忍性更好。

2. Within a single architecture, one can tradeoff feature-maps and quantization, with more feature maps allowing for lower bitwidth kernels. 在单个架构中，可以用特征图和量化，换取更多低精度核心的特征图。

3. One can obtain improved accuracy by not constraining the ranges of the activations during training and then quantizing them, instead of restricting the range to a fixed value. In our experiments, it was better to use a ReLU than a ReLU6 for the activations. 训练对激活的范围不进行限制，然后进行量化，可以提高准确率。在我们的试验中，使用ReLU比使用ReLU6作为激活效果更好。

Going forward, we plan to enhance our automated quantization tool to enable better quantization of networks by investigating the following areas: 更进一步，我们可以改进自动量化工具，通过研究下面的情况，来改进网络量化情况：

1. Regularization techniques to better control the dynamic ranges of weights and activations can provide further improvements. 研究正则化技术，更好的控制权重和激活的动态范围，可以得到进一步的改进。

2. Distilled training to further improve the accuracy of quantized models [32]. 研究蒸馏训练，以进一步改进量化模型的准确率[32]。

3. Per-layer quantization of weights and activations to provide further compression and performance gains on hardware accelerators. Reinforcement learning has been applied successfully towards this problem in [33]. 研究权重和激活的逐层量化，以在硬件加速器上得到进一步压缩和性能提升。强化学习已经成功的用于解决这个问题[33]。

## A Impact of Batch Normalization on Quantization BN对量化的影响

To understand the impact of batch normalization on the dynamic range of the folded weights (W), we consider the following metrics: 为理解BN对折叠权重动态范围的影响，我们考虑下面的度量标准：

1. SQNR: We calculate the Signal to quantization noise ratio defined as: 我们计算信号对量化噪声的比如下：

$$SQNR = 10log_{10} \frac{\sum W^2}{\sum (W − SimQuant(W))^2}$$

for different quantization schemes. We note that per-channel quantization provides significant improvement in SQNR over per-layer quantization, even if only symmetric quantization is used in the per-channel case. 对不同的量化方案。我们注意到，逐通道量化，比逐层量化的SQNR要好很多，即使是在逐通道的情况下使用对称量化。

2. Weight Power Distribution: We also plot the distribution of the sample weights, normalized by the average power, i.e we plot 我们还绘出了权重样本用平均能量归一化的分布图，即

$$histogram \frac{W^2}{E(W^2)}$$

for the weights before and after folding. We note that after folding, there are much larger outliers which severely degrade performance. 绘制的包括折叠前与折叠后的权重图。注意在折叠后，离群值多了很多，这会严重的降低性能。