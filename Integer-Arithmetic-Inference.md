# Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference

Benoit Jacob et al. Google Inc.

## Abstract

The rising popularity of intelligent mobile devices and the daunting computational cost of deep learning-based models call for efficient and accurate on-device inference schemes. We propose a quantization scheme that allows inference to be carried out using integer-only arithmetic, which can be implemented more efficiently than floating point inference on commonly available integer-only hardware. We also co-design a training procedure to preserve end-to-end model accuracy post quantization. As a result, the proposed quantization scheme improves the tradeoff between accuracy and on-device latency. The improvements are significant even on MobileNets, a model family known for run-time efficiency, and are demonstrated in ImageNet classification and COCO detection on popular CPUs.

智能移动设备越来越受欢迎，而基于深度学习的模型其计算量令人生畏，这就需要在设备上高效而准确的推理方案。我们提出了一种量化方案，使得推理可以只使用整数运算，在常见的整数型硬件上，这比浮点型的推理要高效的多。我们还配套设计了一种训练方法，量化后可以保持端到端的模型准确率。所以，我们提出的量化方案改进了准确率和设备上延迟的折中性能。甚至在MobileNet上这种改进都是非常明显的，我们在流行的CPUs上，在ImageNet分类和COCO目标检测上展示了这种改进效果。

## 1. Introduction

Current state-of-the-art Convolutional Neural Networks (CNNs) are not well suited for use on mobile devices. Since the advent of AlexNet [20], modern CNNs have primarily been appraised according to classification / detection accuracy. Thus network architectures have evolved without regard to model complexity and computational efficiency. On the other hand, successful deployment of CNNs on mobile platforms such as smartphones, AR/VR devices (HoloLens, Daydream), and drones require small model sizes to accommodate limited on-device memory, and low latency to maintain user engagement. This has led to a burgeoning field of research that focuses on reducing the model size and inference time of CNNs with minimal accuracy losses.

目前最好的CNNs在移动设备上是不太适用的。自动AlexNet的出现，现代CNN主要的贡献是提高了分类、检测准确率。所以网络架构的演变，并没有考虑到模型的复杂性和计算的效率。另一方面，将CNNs在移动平台上进行成功的部署，如智能手机、AR/VR设备(HoloLens, Daydream)，和无人机设备上，这需要小型模型，以适应设备上有限的内存，和为保持用户参与的低延迟。所以在最小的准确率损失下，降低模型大小和推理时间，这方面的研究越来越多。

Approaches in this field roughly fall into two categories. The first category, exemplified by MobileNet [10], SqueezeNet [16], ShuffleNet [32], and DenseNet [11], designs novel network architectures that exploit computation / memory efficient operations. The second category quantizes the weights and / or activations of a CNN from 32 bit floating point into lower bit-depth representations. This methodology, embraced by approaches such as Ternary weight networks (TWN [22]), Binary Neural Networks (BNN [14]), XNOR-net [27], and more [8, 21, 26, 33, 34, 35], is the focus of our investigation. Despite their abundance, current quantization approaches are lacking in two respects when it comes to trading off latency with accuracy.

这个领域的方法大致有两个类别。第一类，如MobileNet，SqueezeNet，ShuffleNet和DenseNet，主要是设计新型网络架构，以研究计算高效、内存利用高效的运算。第二类是对将CNN的权重和/或激活值进行量化，从32 bit的浮点数量化为低bit的表示。这种方法有很多工作支持，如Ternary weight networks (TWN [22]), Binary Neural Networks (BNN [14]), XNOR-net [27]和[8, 21, 26, 33, 34, 35]，是我们的研究重点。虽然研究众多，但目前的量化方法，在延迟与准确率的折中上，有两点缺陷。

First, prior approaches have not been evaluated on a reasonable baseline architecture. The most common baseline architectures, AlexNet [20], VGG [28] and GoogleNet [29], are all over-parameterized by design in order to extract marginal accuracy improvements. Therefore, it is easy to obtain sizable compression of these architectures, reducing quantization experiments on these architectures to proof-of-concepts at best. Instead, a more meaningful challenge would be to quantize model architectures that are already efficient at trading off latency with accuracy, e.g. MobileNets.

首先，之前的方法没有在合理的基准架构上进行评估。最常见的基准架构，AlexNet，VGG和GoogLeNet，在设计上都是参数过多的，以得到准确率的提升。因此，很容易对这些架构得到很好的压缩，在这些架构上对量化进行的试验，就仅仅停留在概念验证的阶段。相反，更有意义的挑战是，在一些已经取得了很好的延迟-准确率折中的模型上，如MobileNets，进行模型量化。

Second, many quantization approaches do not deliver verifiable efficiency improvements on real hardware. Approaches that quantize only the weights ([2, 4, 8, 33]) are primarily concerned with on-device storage and less with computational efficiency. Notable exceptions are binary, ternary and bit-shift networks [14, 22, 27]. These latter approaches employ weights that are either 0 or powers of 2, which allow multiplication to be implemented by bit shifts. However, while bit-shifts can be efficient in custom hardware, they provide little benefit on existing hardware with multiply-add instructions that, when properly used (i.e. pipelined), are not more expensive than additions alone. Moreover, multiplications are only expensive if the operands are wide, and the need to avoid multiplications diminishes with bit depth once both weights and activations are quantized. Notably, these approaches rarely provide on-device measurements to verify the promised timing improvements. More runtime-friendly approaches quantize both the weights and the activations into 1 bit representations [14, 27, 34]. With these approaches, both multiplications and additions can be implemented by efficient bit-shift and bit-count operations, which are showcased in custom GPU kernels (BNN [14]). However, 1 bit quantization often leads to substantial performance degradation, and may be overly stringent on model representation.

第二，很多量化方法没有给出在真实硬件上可验证的效率提升。只对权重进行量化的方法[2,4,8,33]考虑的基本只是设备上的存储量，而计算效率考虑的很少。值得提出来的例外是binary, ternary和bit-shift网络[14,22,27]。这些后面的方法使用的权重要么是0，要么是2的幂，这样乘法可以通过bit移位实现。但是，比特移位在特定的硬件上是很高效的，在现有的硬件上，都是乘法-加法指令，则没有多少好处，在这些设备上乘法如果合理的设计的话，比加法的耗费不会多很多。而且，乘法只在操作数很宽的时候，才会耗费很多，在权重和激活都进行量化的时候，需要避免乘法随着bit深度而消失。这些方法几乎都没有在设备上的实际测量试验，以验证陈述的耗时改进。更多运行时间友好的方法，对权重和激活都量化成了1 bit表示。在这些方法下，乘法和加法都可以使用高效的比特移位和bit-count运算实现，这在定制的GPU内核上进行了展现。但是，1 bit量化通常使性能下降很多，在模型表示上过于弱了一些。

In this paper we address the above issues by improving the latency-vs-accuracy tradeoffs of MobileNets on common mobile hardware. Our specific contributions are: 本文中，我们通过在常见的移动硬件上改进MobileNet的延迟-准确率折中，来解决上述问题。具体的贡献如下：

- We provide a quantization scheme (section 2.1) that quantizes both weights and activations as 8-bit integers, and just a few parameters (bias vectors) as 32-bit integers. 我们提出了一种量化方案（2.1节），将权重和激活都量化为8-bit整数，只有少数参数（偏置向量）量化为32-bit整数。

- We provide a quantized inference framework that is efficiently implementable on integer-arithmetic-only hardware such as the Qualcomm Hexagon (sections 2.2, 2.3), and we describe an efficient, accurate implementation on ARM NEON (Appendix B). 我们提出了一种量化的推理框架，可以只用整数代数运算的硬件上得到有效的实现，如高通Hexagon（2.2节，2.3节），我们还给出了在ARM NEON上的高效准确实现（附录B）。

- We provide a quantized training framework (section 3) co-designed with our quantized inference to minimize the loss of accuracy from quantization on real models. 我们提出了一个量化训练框架（第3节），和我们的量化推理共同设计出来，在对真实模型的量化中，可以使准确率损失达到最小。

- We apply our frameworks to efficient classification and detection systems based on MobileNets and provide benchmark results on popular ARM CPUs (section 4) that show significant improvements in the latency-vs-accuracy tradeoffs for state-of-the-art MobileNet architectures, demonstrated in ImageNet classification [3], COCO object detection [23], and other tasks. 我们将我们的框架用于基于MobileNet的高效分类和检测系统，在流行的ARM CPU上给出基准测试结果（第4节），对于目前最好的MobileNet架构，得到了明显的准确率-延迟折中改进，在ImageNet分类、COCO目标检测和其他任务中得到了证明。

Our work draws inspiration from [7], which leverages low-precision fixed-point arithmetic to accelerate the training speed of CNNs, and from [31], which uses 8-bit fixed-point arithmetic to speed up inference on x86 CPUs. Our quantization scheme focuses instead on improving the inference speed vs accuracy tradeoff on mobile CPUs.

我们的工作是从[7]中得到的灵感，利用低精度定点运算来加速CNN的训练，以及从[31]中得到启发，使用了8-bit定点运算来在X86 CPU上进行推理加速。我们的量化方案关注在移动CPUs上改进推理的准确率-速度折中。

## 2. Quantized Inference

### 2.1. Quantization scheme

In this section, we describe our general quantization scheme, that is, the correspondence between the bit-representation of values (denoted q below, for “quantized value”) and their interpretation as mathematical real numbers (denoted r below, for “real value”). Our quantization scheme is implemented using integer-only arithmetic during inference and floating-point arithmetic during training, with both implementations maintaining a high degree of correspondence with each other. We achieve this by first providing a mathematically rigorous definition of our quantization scheme, and separately adopting this scheme for both integer-arithmetic inference and floating-point training.

本节中，我们描述我们的一般量化方案，即数学上的实数值（表示为r，即实数值），和量化的bit表示（表示为q，即量化值）的对应关系。我们的量化方案是在推理阶段只用整数的代数运算实现的，在训练时则用浮点代数运算，两个实现保持互相之间的高度对应性。我们首先给出量化方案严格的数学定义，然后对整数运算的推理和浮点的训练分别采用这个方案。

The quantization scheme described here is the one adopted in TensorFlow Lite [5] and we will refer to specific parts of its code to illustrate aspects discussed below. 这里描述的量化方案是TensorFlow Lite采用的，我们会参考其部分代码来描述。

We had earlier described this quantization scheme in the documentation of gemmlowp [18]. That page may still be useful as an alternate treatment of some of the topics developed in this section, and for its self-contained example code. 我们更早的时候描述了这个方案，在gemmlowp的文档中。那个页面仍然是有用的，作为本节中一些话题的替代，并且还有自我包含的代码。

A basic requirement of our quantization scheme is that it permits efficient implementation of all arithmetic using only integer arithmetic operations on the quantized values (we eschew implementations requiring lookup tables because these tend to perform poorly compared to pure arithmetic on SIMD hardware). This is equivalent to requiring that the quantization scheme be an affine mapping of integers q to real numbers r, i.e. of the form

我们的量化方案的一个基本需求是，只在量化值上使用整数代数运算，来高效的实现所有代数运算（我们避开了需要查询表的运算，因为与SIMD硬件上纯代数运算相比，性能会很差）。这等价于，要求量化方案是整数值q到实数值r的仿射变换，即如下的形式

$$r = S(q-Z)$$(1)

for some constants S and Z. Equation (1) is our quantization scheme and the constants S and Z are our quantization parameters. Our quantization scheme uses a single set of quantization parameters for all values within each activations array and within each weights array; separate arrays use separate quantization parameters.

S和Z是特定常量。式(1)就是我们的量化方案，常数S和Z是我们的量化参数。我们的量化方案，对每个激活阵列和每个权重阵列中的所有值，使用一套量化参数；不同的阵列使用不同的量化参数。

For 8-bit quantization, q is quantized as an 8-bit integer (for B-bit quantization, q is quantized as an B-bit integer). Some arrays, typically bias vectors, are quantized as 32-bit integers, see section 2.4.

对于8-bit量化，q是量化成8-bit整数的（对于B-bit量化，q量化为B-bit整数）。一些阵列，如典型的偏置向量，量化为32-bit整数，见2.4节。

The constant S (for “scale”) is an arbitrary positive real number. It is typically represented in software as a floating-point quantity, like the real values r. Section 2.2 describes methods for avoiding the representation of such floating-point quantities in the inference workload.

常数S（对应尺度scale）是任意正实数值。在软件中一般表示为浮点量，如实数值r。2.2节描述的方法是在推理过程中避免使用这样的浮点量的表示。

The constant Z (for “zero-point”) is of the same type as quantized values q, and is in fact the quantized value q corresponding to the real value 0. This allows us to automatically meet the requirement that the real value r = 0 be exactly representable by a quantized value. The motivation for this requirement is that efficient implementation of neural network operators often requires zero-padding of arrays around boundaries.

常数Z（对应零点）是量化值q同样类型的数，实际上对应实数值0的量化值。这使实数值r=0自动表示为一个量化值，这个要求可以自动得到满足。这个要求的动机是，神经网络算子的高效实现，阵列的边缘需要补零。

Our discussion so far is summarized in the following quantized buffer data structure, with one instance of such a buffer existing for each activations array and weights array in a neural network. We use C++ syntax because it allows the unambiguous conveyance of types.

迄今为止，我们的讨论总结为，下列量化的buffer数据结构中，对于神经网络中的每个激活阵列和权重阵列，都会有一个对应的buffer实例。我们使用C++的语法，因为可以无疑义的表达类型。

```
template<typename QType>  // e.g. QType=uint8
struct QuantizedBuffer {
vector<QType> q;          // the quantized values
float S;                  // the scale
QType Z;                  // the zero-point
};
```
The actual data structures in the TensorFlow Lite [5] Converter are QuantizationParams and Array in this header file. As we discuss in the next subsection, this data structure, which still contains a floating-point quantity, does not appear in the actual quantized on-device inference code.

### 2.2. Integer-arithmetic-only matrix multiplication

We now turn to the question of how to perform inference using only integer arithmetic, i.e. how to use Equation (1) to translate real-numbers computation into quantized-values computation, and how the latter can be designed to involve only integer arithmetic even though the scale values S are not integers.

我们现在转到下面的问题，怎样只使用整数代数运算进行推理，即怎样使用式(1)将实数值的计算翻译到量化值的计算，怎样设计后者，才能只使用整数代数运算，即使尺度值S不是整数。

Consider the multiplication of two square N × N matrices of real numbers, r1 and r2, with their product represented by r3 = r1 r2. We denote the entries of each of these matrices rα (α = 1, 2 or 3) as r_α^{(i,j)} for 1 <= i, j <= N, and the quantization parameters with which they are quantized as (S α, Z α). We denote the quantized entries by q^α_{(i,j)}. Equation (1) then becomes:

考虑两个N×N矩阵的实数值的乘积，r1和r2，其乘积表示为r3 = r1 r2。我们将每个矩阵rα (α = 1, 2 or 3)的索引表示为r_α^{(i,j)}，1 <= i, j <= N，量化参数表示为(S α, Z α)。量化的值表示为q^α_{(i,j)}，那么式(1)就变成了：

$$r_α^{(i,j)} = S_α (q_α^{(i,j)} − Z_α)$$(2)

From the definition of matrix multiplication, we have 从矩阵乘积的定义中，我们有

$$S_3 (q_3^{(i,k)} − Z_3) = \sum^N_{j=1} S_1 (q_1^{(i,j)} − Z_1) S_2(q_2^{(j,k)} − Z_2)$$(3)

which can be rewritten as 可以重写为

$$q_3^{(i,k)} = Z_3 + M \sum^N_{j=1} (q_1^{(i,j)} − Z_1) (q_2^{(j,k)} − Z_2)$$(4)

where the multiplier M is defined as 其中乘子M定义为

$$M := \frac {S_1 S_2}{S_3}$$(5)

In Equation (4), the only non-integer is the multiplier M. As a constant depending only on the quantization scales S1, S2, S3, it can be computed offline. We empirically find it to always be in the interval (0, 1), and can therefore express it in the normalized form

在式(4)中，唯一的非整数是乘子M。作为一个常数，只依赖于量化尺度S1, S2, S3，可以离线计算得到。我们通过经验发现，M一直都是在(0,1)的范围内的，因为可以以下面的归一化形式表达

$$M = 2^{−n} M_0$$(6)

where M0 is in the interval [0.5, 1) and n is a non-negative integer. The normalized multiplier M0 now lends itself well to being expressed as a fixed-point multiplier (e.g. int16 or int32 depending on hardware capability). For example, if int32 is used, the integer representing M0 is the int32 value nearest to 2^31 M0. Since M0 > 0.5, this value is always at least 2^30 and will therefore always have at least 30 bits of relative accuracy. Multiplication by M0 can thus be implemented as a fixed-point multiplication. Meanwhile, multiplication by 2^−n can be implemented with an efficient bit-shift, albeit one that needs to have correct round-to-nearest behavior, an issue that we return to in Appendix B.

其中M0的取值范围为[0.5,1)，n是非负整数。归一化的乘子M0现在非常适合于表达为一个定点的乘子（如，int16或int32类型，视硬件能力而定）。比如，如果使用int32类型，表示M0的整数是最接近于2^31 M0的int32值。因为M0 > 0.5，这个值永远大于2^30，因此至少有30bit的相对精度。乘以M0因此可以实现为定点乘法。同时，乘以2^-n可以实现为高效的bit移位，尽管如此，需要有正确的四舍五入到最近值的行为，这个问题我们在附录B中讨论。

The computation discussed in this section is implemented in TensorFlow Lite [5] reference code for a fully-connected layer. 这里讨论的计算在Tensorflow Lite中实现的参考代码为一个全连接层。

### 2.3. Efficient handling of zero-points

In order to efficiently implement the evaluation of Equation (4) without having to perform 2N^3 subtractions and without having to expand the operands of the multiplication into 16-bit integers, we first notice that by distributing the multiplication in Equation (4), we can rewrite it as

为高效的实现式(4)的评估，不进行2N^3的减法，不将乘法的算数扩展成16-bit整数，我们首先注意到，将式(4)中的乘法展开，式子可以重写为

$$q_3^{(i,k)} = Z_3 + M (N Z_1 Z_2 - Z_1 a_2^{(k)} - Z_2 ā_1^{(i)} + \sum_{j=1}^N q_1^{(i,j)} q_2^{(j,k)}$$(7)

where 其中

$$a_2^{(k)} := \sum_{j=1}^N q_2^{(j,k)}, ā_1^{(i)} := \sum_{(j=1)}^N q_1^{(i,j)}$$(8)

Each $a_2^{(k)}$ or $ā_1^{(i)}$ takes only N additions to compute, so they collectively take only 2N^2 additions. The rest of the cost of the evaluation of (7) is almost entirely concentrated in the core integer matrix multiplication accumulation

每个$a_2^{(k)}$或$ā_1^{(i)}$需要计算N个加法，所有它们总计需要2N^2个加法。式(7)的计算剩下几乎都集中在核心的整数矩阵乘法累加上

$$\sum_{j=1}^N q_1^{(i,j)} q_2^{(j,k)}$$(9)

which takes 2N^3 arithmetic operations; indeed, everything else involved in (7) is O(N^2) with a small constant in the O. Thus, the expansion into the form (7) and the factored-out computation of $a_2^{(k)}$ and $ā_1^{(i)}$ enable low-overhead handling of arbitrary zero-points for anything but the smallest values of N, reducing the problem to the same core integer matrix multiplication accumulation (9) as we would have to compute in any other zero-points-free quantization scheme.

这需要2N^3次代数运算；确实，式(7)中其他的所有项的计算复杂度是O(N^2)。所以，扩展成式(7)的形式和分解出的$a_2^{(k)}$、$ā_1^{(i)}$，使得零点的处理计算量很小，将问题转化为相同的核心整数矩阵乘法累积运算(9)，这个形式和其他没有零点的量化方案一样。

### 2.4. Implementation of a typical fused layer

We continue the discussion of section 2.3, but now explicitly define the data types of all quantities involved, and modify the quantized matrix multiplication (7) to merge the bias-addition and activation function evaluation directly into it. This fusing of whole layers into a single operation is not only an optimization. As we must reproduce in inference code the same arithmetic that is used in training, the granularity of fused operators in inference code (taking an 8-bit quantized input and producing an 8-bit quantized output) must match the placement of “fake quantization” operators in the training graph (section 3).

我们继续2.3节的讨论，但现在显式的定义所有涉及到的量的数据类型，并修改量化的矩阵乘法(7)，以将与偏置的相加和激活函数直接合并到一起。整个层融合到一个单独的运算，这不止是一个优化。由于我们在推理代码中要复现训练中使用的相同的代数运算，因此推理代码中融合算子的粒度（以8-bit量化数据为输入，生成8-bit量化输出）需要和训练图中的替换的“虚假量化”算子相匹配（第3节）。

For our implementation on ARM and x86 CPU architectures, we use the gemmlowp library [18], whose GemmWithOutputPipeline entry point provides supports the fused operations that we now describe(The discussion in this section is implemented in TensorFlow Lite [5] for e.g. a Convolutional operator (reference code is self-contained, optimized code calls into gemmlowp [18])).

对于在ARM和x86 CPU架构上的实现，我们使用gemmlowp库，其GemmWithOutputPipeline入口点提供了融合运算的支持，我们现在进行描述（本节中的讨论，在Tensorflow Lite中有实现，如一个卷积算子）。

We take the q1 matrix to be the weights, and the q2 matrix to be the activations. Both the weights and activations are of type uint8 (we could have equivalently chosen int8, with suitably modified zero-points). Accumulating products of uint8 values requires a 32-bit accumulator, and we choose a signed type for the accumulator for a reason that will soon become clear. The sum in (9) is thus of the form:

我们令q1矩阵为权重，q2矩阵为激活。权重和激活的类型都是uint8（也可以等价的选择int8，只需要合理的修改零点）。uint8值的乘积累加需要一个32-bit的累加器，我们为累加器选择一个有符号的类型，原因马上就会明白。(9)中的求和因此是下面的形式：

$$int32 += uint8 * uint8.$$(10)

In order to have the quantized bias-addition be the addition of an int32 bias into this int32 accumulator, the bias-vector is quantized such that: it uses int32 as its quantized data type; it uses 0 as its quantization zero-point Z bias ; and its quantization scale S bias is the same as that of the accumulators, which is the product of the scales of the weights and of the input activations. In the notation of section 2.3,

为使与量化偏置的相加是int32的偏置加上int32的累加器，偏置向量的量化要满足如下条件：使用int32作为其量化数据类型；使用0作为其量化的零点Z偏置；其量化尺度S偏置是与累加器一样的，也就是权重和输入激活的尺度的乘积。在2.3节的符号表示下，

$$S_{bias} = S_1 S_2 , Z_{bias} = 0$$(11)

Although the bias-vectors are quantized as 32-bit values, they account for only a tiny fraction of the parameters in a neural network. Furthermore, the use of higher precision for bias vectors meets a real need: as each bias-vector entry is added to many output activations, any quantization error in the bias-vector tends to act as an overall bias (i.e. an error term with nonzero mean), which must be avoided in order to preserve good end-to-end neural network accuracy.

虽然偏置向量量化为32-bit值，但它们数量桑只占神经网络参数中的很小一部分。而且，偏置使用更高的精度复合真实的需求：因为每个偏置向量都与很多个输出激活相加，偏置向量中的任何量化误差都是整体的偏差（即，非零均值的误差项），这种情况必须避免，以保持好的端到端的神经网络准确率。

With the final value of the int32 accumulator, there remain three things left to do: scale down to the final scale used by the 8-bit output activations, cast down to uint8 and apply the activation function to yield the final 8-bit output activation.

有了累加器的int32的最终值，还有三件事要做：将最终值缩小到最终的尺度，即输出激活的8-bit值，转换到uint8类型，应用激活函数，得到最终的8-bit输出激活。

The down-scaling corresponds to multiplication by the multiplier M in equation (7). As explained in section 2.2, it is implemented as a fixed-point multiplication by a normalized multiplier M0 and a rounding bit-shift. Afterwards, we perform a saturating cast to uint8, saturating to the range [0, 255].

缩小对应着式(7)中的与乘子M相乘。如2.2节所解释，其实现是与归一化乘子M0的定点乘法，和bit移位运算，我们进行到uint8的饱和转换，饱和范围为[0,255]。

We focus on activation functions that are mere clamps, e.g. ReLU, ReLU6. Mathematical functions are discussed in appendix A.1 and we do not currently fuse them into such layers. Thus, the only thing that our fused activation functions need to do is to further clamp the uint8 value to some sub-interval of [0, 255] before storing the final uint8 output activation. In practice, the quantized training process (section 3) tends to learn to make use of the whole output uint8 [0, 255] interval so that the activation function no longer does anything, its effect being subsumed in the clamping to [0, 255] implied in the saturating cast to uint8.

我们关注的激活函数仅仅为clamp函数，如ReLU，ReLU6。数学函数如附录A.1讨论，我们现在不将其融合到这样的层中。因此，融合的激活函数需要做的唯一的事，就是在存储最终的uint8输出激活值之前，进一步将uint8值clamp到[0,255]的某子集。实践中，量化的训练过程（第3节）会倾向于学习使用uint8的整个输出区间[0,255]，这样激活函数就不会做任何事，求和结果clamp到[0,255]的效果，暗示着到uint8的饱和转换。

## 3. Training with simulated quantization

A common approach to training quantized networks is to train in floating point and then quantize the resulting weights (sometimes with additional post-quantization training for fine-tuning). We found that this approach works sufficiently well for large models with considerable representational capacity, but leads to significant accuracy drops for small models. Common failure modes for simple post-training quantization include: 1) large differences (more than 100×) in ranges of weights for different output channels (section 2 mandates that all channels of the same layer be quantized to the same resolution, which causes weights in channels with smaller ranges to have much higher relative error) and 2) outlier weight values that make all remaining weights less precise after quantization.

训练量化网络的常用方法是，在浮点时训练，然后将得到的权重量化（有时使用额外的量化后训练进行精调）。我们发现，这种方法对于表示能力很强的大型模型效果很好，但对于小型模型来说会导致明显的准确率下降。简单的训练后量化的常见失败模式包括：1)不同的输出通道的权重范围差异太大（超过100x）（第2节中同一层的所有通道强制量化为相同的分辨率，这会导致，通道中的权重如果范围较小，量化后的相对误差会较大），2)量化后，权重的离群值会使得所有其余的权重值精确度降低。

We propose an approach that simulates quantization effects in the forward pass of training. Backpropagation still happens as usual, and all weights and biases are stored in floating point so that they can be easily nudged by small amounts. The forward propagation pass however simulates quantized inference as it will happen in the inference engine, by implementing in floating-point arithmetic the rounding behavior of the quantization scheme that we introduced in section 2:

我们提出一种方法，在训练的前向过程中模拟量化效果。反向传播依然照旧，所有的权重和偏置都以浮点数据类型存储。但是，前向传播过程模拟量化的推理过程，就像推理引擎中发生的一样，实现的是浮点代数运算下量化方案的近似行为，我们在第2节中提出的：

- Weights are quantized before they are convolved with the input. If batch normalization (see [17]) is used for the layer, the batch normalization parameters are “folded into” the weights before quantization, see section 3.2. 权重在与输入卷积前进行量化。如果这一层使用了BN，那么BN的参数在量化前与权重折叠到一起，见3.2节。

- Activations are quantized at points where they would be during inference, e.g. after the activation function is applied to a convolutional or fully connected layer’s output, or after a bypass connection adds or concatenates the outputs of several layers together such as in ResNets. 激活就像在推理时一样被量化，如在激活函数应用于卷积或全连接层的输出后，或在旁路相加连接，或在几层的输出拼接在一起之后，如ResNet中。

For each layer, quantization is parameterized by the number of quantization levels and clamping range, and is performed by applying point-wise the quantization function q defined as follows:

对于每一层，量化的参数有量化级的数量和clamping范围，将下面的逐点量化函数q应用于量化：

$$clamp(r;a,b) := min(max(x,a),b); s(a,b,n) := \frac {b-a} {n-1}; q(r;a,b,n) := ⌊ \frac {clamp(r;a,b)-a} {s(a,b,n)} ⌉ s(a,b,n) + a$$(12)

where r is a real-valued number to be quantized, [a; b] is the quantization range, n is the number of quantization levels, and ⌊·⌉ denotes rounding to the nearest integer. n is fixed for all layers in our experiments, e.g. n = 2^8 = 256 for 8 bit quantization.

其中r是要量化的实值数，[a; b]是量化范围，n是量化级的数量，⌊·⌉表示四舍五入，n在我们的试验中对所有层为固定的数，如对于8-bit量化，n = 2^8 = 256。

### 3.1. Learning quantization ranges

Quantization ranges are treated differently for weight quantization vs. activation quantization: 量化范围对于权重量化和激活量化的处理是不一样的：

- For weights, the basic idea is simply to set a := min w, b := max w. We apply a minor tweak to this so that the weights, once quantized as int8 values, only range in [−127, 127] and never take the value −128, as this enables a substantial optimization opportunity (for more details, see Appendix B). 对于权重，基本的思想是，设a := min w, b := max w。我们进行一点小小的调整，使权重量化为int8值时，范围为[−127, 127]，取值不会达到-128，因为这会有很大的优化机会（详见附录B）。

- For activations, ranges depend on the inputs to the network. To estimate the ranges, we collect [a; b] ranges seen on activations during training and then aggregate them via exponential moving averages (EMA) with the smoothing parameter being close to 1 so that observed ranges are smoothed across thousands of training steps. Given significant delay in the EMA updating activation ranges when the ranges shift rapidly, we found it useful to completely disable activation quantization at the start of training (say, for 50 thousand to 2 million steps). This allows the network to enter a more stable state where activation quantization ranges do not exclude a significant fraction of values. 对于激活值，范围依赖于网络的输入。为估计范围，我们收集在训练时的激活值范围[a; b]，然后用平滑参数接近于1的指数滑动平均(EMA)对其集聚，这样观察到的范围在成千上万次训练步中得到了平滑。当范围快速变化时，EMA更新的激活范围有明显的延迟，我们发现在训练开始时（如从50K到2M步以内），完全不对激活进行量化反而很有用。这使得网络进入一种更稳定的状态，这样激活量化的范围不会超出部分数值太多。

In both cases, the boundaries [a; b] are nudged so that value 0.0 is exactly representable as an integer z(a, b, n) after quantization. As a result, the learned quantization parameters map to the scale S and zero-point Z in equation 1:

在两种情况下，范围[a; b]都进行略微调整，使数值0.0在量化后可以精确的表示为一个整数z(a, b, n)。结果是，学习到的量化参数对应着式(1)中的尺度参数S和零点Z：

$$S = s(a, b, n), Z = z(a, b, n)$$(13)

Below we depict simulated quantization assuming that the computations of a neural network are captured as a TensorFlow graph [1]. A typical workflow is described in Algorithm 1. Optimization of the inference graph by fusing and removing operations is outside the scope of this paper. Source code for graph modifications (inserting fake quantization operations, creating and optimizing the inference graph) and a low bit inference engine has been open-sourced with TensorFlow contributions in [19].

下面我们描述一下模拟量化，假设神经网络的计算就像一个Tensorflow图一样进行。一个典型的工作流如算法1表示。通过融合和移除运算来优化推理图，这是本文之外的工作。修改图的源码（插入虚假量化运算，创建和优化推理图）和低bit推理引擎，已经由Tensorflow开源[19]。

Algorithm 1 Quantized graph training and inference

- Create a training graph of the floating-point model.
- Insert fake quantization TensorFlow operations in locations where tensors will be downcasted to fewer bits during inference according to equation 12.
- Train in simulated quantized mode until convergence.
- Create and optimize the inference graph for running in a low bit inference engine.
- Run inference using the quantized inference graph.

Figure 1.1a and b illustrate TensorFlow graphs before and after quantization for a simple convolutional layer. Illustrations of the more complex convolution with a bypass connection in figure C.3 can be found in figure C.4.

图1.1a和1.1b描述了一个简单的卷积层量化前后的Tensorflow图。带有旁路连接的更复杂卷积的描述可以见图C.3和图C.4。

Note that the biases are not quantized because they are represented as 32-bit integers in the inference process, with a much higher range and precision compared to the 8 bit weights and activations. Furthermore, quantization parameters used for biases are inferred from the quantization parameters of the weights and activations. See section 2.4.

注意偏置没有被量化，因为它们在推理过程中被表示为32-bit整数，与8-bit权重和激活相比，范围和精度都要宽的多。而且，用于偏置的量化参数是从权重和激活的量化参数中推断得到的，见2.4节。

Typical TensorFlow code illustrating use of [19] follows:

```
from tf.contrib.quantize import quantize_graph as qg

g = tf.Graph()
with g.as_default():
    output = ...
    total_loss = ...
    optimizer = ...
    train_tensor = ...
if is_training:
    quantized_graph = qg.create_training_graph(g)
else:
    quantized_graph = qg.create_eval_graph(g)
# Train or evaluate quantized_graph.
```

### 3.2. Batch normalization folding

For models that use batch normalization (see [17]), there is additional complexity: the training graph contains batch normalization as a separate block of operations, whereas the inference graph has batch normalization parameters “folded” into the convolutional or fully connected layer’s weights and biases, for efficiency. To accurately simulate quantization effects, we need to simulate this folding, and quantize weights after they have been scaled by the batch normalization parameters. We do so with the following:

对于使用BN的模型，有一个额外的复杂性：训练图包含的BN是一个额外的运算模块，但推理图中的BN层与卷积层或全连接层的权重和偏置折叠到了一起，为了提高效率。为精确的模拟量化效果，我们需要模拟这种折叠效果，在权重与BN的参数折叠之后再进行量化。如下所示：

$$w_{fold} := \frac {γw}{\sqrt {EMA(σ_B^2)+ε}}$$(14)

Here γ is the batch normalization’s scale parameter, $EMA(σ_B^2)$ is the moving average estimate of the variance of convolution results across the batch, and ε is just a small constant for numerical stability.

这里γ是BN的尺度参数，$EMA(σ_B^2)$是整个批次的卷积结果方差的滑动平均估计，ε是一个小常数，增加数值稳定性。

After folding, the batch-normalized convolutional layer reduces to the simple convolutional layer depicted in figure 1.1a with the folded weights w fold and the corresponding folded biases. Therefore the same recipe in figure 1.1b applies. See the appendix for the training graph (figure C.5) for a batch-normalized convolutional layer, the corresponding inference graph (figure C.6), the training graph after batch-norm folding (figure C.7) and the training graph after both folding and quantization (figure C.8).

在折叠后，BN卷积层蜕化为简单的卷积层，如图1.1a所示，w为折叠的权重，还有对应的折叠偏置。因此，图1.1b就可以得到应用。见附录的BN卷积层的训练图（图C.5）对应的推理图（图C.6），BN折叠后的训练图（图C.7），和折叠和量化后的训练图（图C.8）。

## 4. Experiments

We conducted two set of experiments, one showcasing the effectiveness of quantized training (Section. 4.1), and the other illustrating the improved latency-vs-accuracy tradeoff of quantized models on common hardware (Section. 4.2). The most performance-critical part of the inference workload on the neural networks being benchmarked is matrix multiplication (GEMM). The 8-bit and 32-bit floating-point GEMM inference code uses the gemmlowp library [18] for 8-bit quantized inference, and the Eigen library [6] for 32-bit floating-point inference.

我们进行两类试验，一类展示量化训练的有效性（4.1节），另一类展示量化模型在常见硬件上的延迟-准确率折中效果的改进（4.2节）。神经网络推理工作中性能最关键的部分是矩阵乘积(GEMM)。8-bit和32-bit浮点GEMM推理，在8-bit量化推理时使用gemmlowp库，在32-bit浮点推理时使用Eigen库。

### 4.1. Quantized training of Large Networks

We apply quantized training to ResNets [9] and InceptionV3 [30] on the ImageNet dataset. These popular networks are too computationally intensive to be deployed on mobile devices, but are included for comparison purposes. Training protocols are discussed in Appendix D.1 and D.2.

对ResNet和InceptionV3在ImageNet数据集上的训练，我们使用量化训练。这些流行的网络计算量非常大，不能在移动设备上部署，只是为了进行比较。训练方案见附录D.1和D.2。

#### 4.1.1 ResNets

We compare floating-point vs integer-quantized ResNets for various depths in table 4.1. Accuracies of integer-only quantized networks are within 2% of their floating-point counterparts. 我们在表4.1中比较了各种深度下的浮点ResNet和量化的整形ResNet。整形量化网络的准确率比浮点的准确率，降低幅度在2%以内。

Table 4.1: ResNet on ImageNet: Floating-point vs quantized network accuracy for various network depths.

ResNet depth | 50 | 100 | 150
--- | --- | --- | ---
Floating-point accuracy | 76.4% | 78.0% | 78.8%
Integer-quantized accuracy | 74.9% | 76.6% | 76.7%

We also list ResNet50 accuracies under different quantization schemes in table 4.2. As expected, integer-only quantization outperforms FGQ [26], which uses 2 bits for weight quantization. INQ [33] (5-bit weight floating-point activation) achieves a similar accuracy as ours, but we provide additional run-time improvements (see section 4.2).

我们还将不同量化方案下的ResNet50准确率列在表4.2中。和预期一样，整数量化的性能超过了FGQ[26]，使用了2-bit进行权重量化。INQ[33]（5-bit权重浮点激活）取得了去我们类似的准确率，但我们有额外的运行时间改进（见4.2节）。

Table 4.2: ResNet on ImageNet: Accuracy under various quantization schemes, including binary weight networks (BWN [21, 15]), ternary weight networks (TWN [21, 22]), incremental network quantization (INQ [33]) and fine-grained quantization (FGQ [26])

Scheme | BWN | TWN | INQ | FGQ | Ours
--- | --- | --- | --- | --- | ---
Weight bits | 1 | 2 | 5 | 2 | 8
Activation bits | float32 | float32 | float32 | 8 | 8
Accuracy | 68.7% | 72.5% | 74.8% | 70.8% | 74.9%

#### 4.1.2 Inception v3 on ImageNet

We compare the Inception v3 model quantized into 8 and 7 bits, respectively. 7-bit quantization is obtained by setting the number of quantization levels in equation 12 to n = 2^7. We additionally probe the sensitivity of activation quantization by comparing networks with two activation nonlinearities, ReLU6 and ReLU. The training protocol is in Appendix D.2.

我们比较了InceptionV3模型量化为8-bit和7-bit的效果。7-bit量化，即式(12)中的量化级设为n = 2^7。我们另外探索了激活量化的敏感度，比较了两种激活非线性，ReLU6和ReLU。训练方案如附录D.2所示。

Table 4.3 shows that 7-bit quantized training produces model accuracies close to that of 8-bit quantized training, and quantized models with ReLU6 have less accuracy degradation. The latter can be explained by noticing that ReLU6 introduces the interval [0, 6] as a natural range for activations, while ReLU allows activations to take values from a possibly larger interval, with different ranges in different channels. Values in a fixed range are easier to quantize with high precision.

表4.3表明，7-bit量化训练得到的模型准确率，与8-bit量化训练接近，采用ReLU6的量化模型准确率下降更少一些。后者这个现象，可以这样解释，ReLU6带来激活范围的自然限制到[0, 6]，而ReLU使激活的取值范围更大一些，不同的通道中范围不一样。固定范围的数值更好量化，精度也更高。

Table 4.3: Inception v3 on ImageNet: Accuracy and recall 5 comparison of floating point and quantized models.

Act | type | Acc mean | Acc stddev | recall mean | recall stddev
--- | --- | --- | --- | --- | ---
ReLU6 | floats | 78.4% | 0.1% | 94.1% | 0.1%
ReLU6 | 8 bits | 75.4% | 0.1% | 92.5% | 0.1%
ReLU6 | 7 bits | 75.0% | 0.3% | 92.4% | 0.2%
ReLU | floats | 78.3% | 0.1% | 94.2% | 0.1%
ReLU | 8 bits | 74.2% | 0.2% | 92.2% | 0.1%
ReLU | 7 bits | 73.7% | 0.3% | 92.0% | 0.1%

### 4.2. Quantization of MobileNets

MobileNets are a family of architectures that achieve a state-of-the-art tradeoff between on-device latency and ImageNet classification accuracy. In this section we demonstrate how integer-only quantization can further improve the tradeoff on common hardware.

MobileNets是在设备上的延迟和ImageNet分类准确率取得了目前最好的折中的一族模型架构。本节中，我们证明了，整数量化可以进一步在常见设备上改进折中。

#### 4.2.1 ImageNet

We benchmarked the MobileNet architecture with varying depth-multipliers (DM) and resolutions on ImageNet on three types of Qualcomm cores, which represent three different micro-architectures: 1) Snapdragon 835 LITTLE core, (figure. 1.1c), a power-efficient processor found in Google Pixel 2; 2) Snapdragon 835 big core (figure. 4.1), a high-performance core employed by Google Pixel 2; and 3) Snapdragon 821 big core (figure. 4.2), a high-performance core used in Google Pixel 1.

我们在ImageNet上，测试了不同深度乘子(DM)和不同分辨率输入的MobileNet架构，使用三种Qualcomm核，代表了不同的微架构：1)Snapdragon 835小核（图1.1c），Google Pixel 2中的高效处理器；2)Snapdragon 835大核（图4.1），Google Pixel 2中的高性能核心；3)Snapdragon 821大核（图4.2），Google Pixel 1中使用的高性能核。

Integer-only quantized MobileNets achieve higher accuracies than floating-point MobileNets given the same runtime budget. The accuracy gap is quite substantial (∼ 10%) for Snapdragon 835 LITTLE cores at the 33ms latency needed for real-time (30 fps) operation. While most of the quantization literature focuses on minimizing accuracy loss for a given architecture, we advocate for a more comprehensive latency-vs-accuracy tradeoff as a better measure. Note that this tradeoff depends critically on the relative speed of floating-point vs integer-only arithmetic in hardware. Floating-point computation is better optimized in the Snapdragon 821, for example, resulting in a less noticeable reduction in latency for quantized models.

在相同的运行时间预算下，整数量化的MobileNets比浮点型的MobileNets准确率更高。在Snapdragon 835上小核上，在33ms的延迟下，达到实时效果的运算(30 fps)下，准确率的差距还是非常大的（～10%）。大多数量化的文献关注的是，在给定的架构下，如何最小化准确度的损失，我们看中的则是，更综合的延迟-准确率折中，作为一个更好的度量。注意，这种折中严重依赖硬件上的浮点和整形运算的速度比。比如，浮点计算在Snapdragon 821上优化更好一些，那么量化模型的延迟降低的就很少。

Figure 4.1: ImageNet classifier on Qualcomm Snapdragon 835 big cores: Latency-vs-accuracy tradeoff of floating-point and integer-only MobileNets.

Figure 4.2: ImageNet classifier on Qualcomm Snapdragon 821: Latency-vs-accuracy tradeoff of floating-point and integer-only MobileNets.

#### 4.2.2 COCO

We evaluated quantization in the context of mobile real time object detection, comparing the performance of quantized 8-bit and float models of MobileNet SSD [10, 25] on the COCO dataset [24]. We replaced all the regular convolutions in the SSD prediction layers with separable convolutions (depthwise followed by 1 × 1 projection). This modification is consistent with the overall design of MobileNets and makes them more computationally efficient. We utilized the Open Source TensorFlow Object Detection API [12] to train and evaluate our models. The training protocol is described in Appendix D.3. We also delayed quantization for 500 thousand steps (see section 3.1), finding that it significantly decreases the time to convergence.

我们在移动设备上的实时目标检测的环境下评估量化的效果，在COCO数据集上比较8-bit量化和浮点MobileNet-SSD的性能。我们将SSD预测层中所有的常规卷积替换为可分离卷积（分层卷积与1×1投影的结合）。这个改动与MobileNet的总体设计一致，使其计算更加高效。我们使用开源的Tensorflow Object Detection API来训练和评估我们的模型。训练方案如附录D.3所述。我们还进行了500K步的延迟量化（见3.1节），发现这会显著降低收敛的时间。

Table 4.4 shows the latency-vs-accuracy tradeoff between floating-point and integer-quantized models. Latency was measured on a single thread using Snapdragon 835 cores (big and LITTLE). Quantized training and inference results in up to a 50% reduction in running time, with a minimal loss in accuracy (−1.8% relative).

表4.4给出了在浮点和整形模型的延迟-准确率折中。延迟的测量是在Snapdragon 835核(big and LITTLE)上单线程环境下的。量化训练和推理降低了50%的运行时间，准确率下降非常小（相对降低了1.8%）。

Table 4.4: Object detection speed and accuracy on COCO dataset of floating point and integer-only quantized models. Latency (ms) is measured on Qualcomm Snapdragon 835 big and LITTLE cores.

DM | Type | mAP | LITTLE (ms) | big (ms)
--- | --- | --- | --- | ---
100% | floats | 22.1 | 778 | 370
100% | 8-bits | 21.7 | 687 | 272
50% | floats | 16.7 | 270 | 121
50% | 8-bits | 16.6 | 146 | 61

#### 4.2.3 Face detection

To better examine quantized MobileNet SSD on a smaller scale, we benchmarked face detection on the face attribute classification dataset (a Flickr-based dataset used in [10]). We contacted the authors of [10] to evaluate our quantized MobileNets on detection and face attributes following the same protocols (detailed in Appendix D.4).

为在更小的尺度上更好的检验量化MobileNet-SSD模型，我们在一个人脸属性分类数据集（[10]中使用的基于Flickr的数据集）上测试人脸识别的性能。我们联系了[10]的作者，在人脸检测和人脸属性上评估了量化的MobileNets，使用了相同的评估方案（详见附录D.4）。

As indicated by tables 4.5 and 4.6, quantization provides close to a 2× latency reduction with a Qualcomm Snapdragon 835 big or LITTLE core at the cost of a ∼ 2% drop in the average precision. Notably, quantization allows the 25% face detector to run in real-time (1K/28 ≈ 36 fps) on a single big core, whereas the floating-point model remains slower than real-time (1K/44 ≈ 23 fps).

如表4.5和4.6所示，在Snapdragon 835 big or LITTLE核上，量化降低了接近2x的延迟，代价是平均精度降低了约~2%。值得注意的是，量化使DM 25%的人脸检测器在单个大核上可以以实时方式运行(1K/28 ≈ 36 fps)，而浮点模型则比实时更慢一些（1K/44 ≈ 23 fps）。

We additionally examine the effect of multi-threading on the latency of quantized models. Table 4.6 shows a 1.5 to 2.2× speedup when using 4 cores. The speedup ratios are comparable between the two cores, and are higher for larger models where the overhead of multi-threading occupies a smaller fraction of the total computation.

我们还检验了多线程对量化模型的延迟效果。表4.6表明，在使用4核的时候，会有1.5到2.2x的加速效果。加速率在两类核之间是类似的，对于大型模型加速率更高，多线程的消耗在整体运算中占的比例更小。

Table 4.5: Face detection accuracy of floating point and integer-only quantized models. The reported precision / recall is averaged over different precision / recall values where an IOU of x between the groundtruth and predicted windows is considered a correct detection, for x in {0.5, 0.55, . . . , 0.95}.

DM | type | Precision | Recall
--- | --- | --- | ---
100% | floats | 68% | 76%
100% | 8-bits | 66% | 75%
50% | floats | 65% | 70%
50% | 8-bits | 62% | 70%
25% | floats | 56% | 64%
25% | 8-bits | 54% | 63%

Table 4.6: Face detection: latency of floating point and quantized models on Qualcomm Snapdragon 835 cores.

DM | type | LITTLE 1 | 2 | 4 | big 1 | 2 | 4
--- | --- | --- | --- | --- | --- | --- | ---
100% | floats | 711 | - | - | 337 | - | -
100% | 8-bits | 372 | 238 | 167 | 154 | 100 | 69
50% | floats | 233 | - | - | 106 | - | -
50% | 8-bits | 134 | 96 | 74 | 56 | 40 | 30
25% | floats | 100 | - | - | 44 | - | -
25% | 8-bits | 67 | 52 | 43 | 28 | 22 | 18

#### 4.2.4 Face attributes

Figure 4.3 shows the latency-vs-accuracy tradeoff of face attribute classification on the Qualcomm Snapdragon 821. Since quantized training results in little accuracy degradation, we see an improved tradeoff even though the Qualcomm Snapdragon 821 is highly optimized for floating point arithmetic (see Figure 4.2 for comparison).

图4.3展示了人脸属性分类在Qualcomm Snapdragon 821上的延迟-准确率折中。由于量化训练的结果准确率降低很小，我们可以看到，即使Snapdragon 821是为浮点运算高度量化的，这个折中仍然得到了很大改进。

Figure 4.3: Face attribute classifier on Qualcomm Snapdragon 821: Latency-vs-accuracy tradeoff of floating-point and integer-only MobileNets.

**Ablation study**. To understand performance sensitivity to the quantization scheme, we further evaluate quantized training with varying weight and activation quantization bit depths. The degradation in average precision for binary attributes and age precision relative to the floating-point baseline are shown in Tables 4.7 and 4.8, respectively. The tables suggest that 1) weights are more sensitive to reduced quantization bit depth than activations, 2) 8 and 7-bit quantized models perform similarly to floating point models, and 3) when the total bit-depths are equal, it is better to keep weight and activation bit depths the same.

**分离试验研究**。为理解性能对量化方案的敏感度，我们进一步使用不同的bit深度对权重和激活进行量化，评估量化训练的性能。相对于浮点基准，量化方案的二值属性和年龄精度的平均精度的降低，分别如表4.7和4.8所示。这些表格说明，1)权重对于降低量化bit深度更敏感，2)8-bit和7-bit量化模型与浮点模型表现类似，3)当总bit深度相等时，最好保持权重的bit深度和激活的bit深度相同。

Table 4.7: Face attributes: relative average category precision of integer-quantized MobileNets (varying weight and activation bit depths) compared with floating point.

Table 4.8: Face attributes: Age precision at difference of 5 years for quantized model (varying weight and activation bit depths) compared with floating point.

## 5. Discussion

We propose a quantization scheme that relies only on integer arithmetic to approximate the floating-point computations in a neural network. Training that simulates the effect of quantization helps to restore model accuracy to near-identical levels as the original. In addition to the 4× reduction of model size, inference efficiency is improved via ARM NEON-based implementations. The improvement advances the state-of-the-art tradeoff between latency on common ARM CPUs and the accuracy of popular computer vision models. The synergy between our quantization scheme and efficient architecture design suggests that integer-arithmetic-only inference could be a key enabler that propels visual recognition technologies into the realtime and low-end phone market.

我们提出了一个量化方案，只用整数代数运算来近似神经网络中的浮点运算。训练时模拟量化的效果，可以帮助模型准确率恢复到与浮点级类似。除了模型大小降低了4x，推理效率也可以通过基于ARM NEON的实现得到改进。这些改进得到了目前最好的折中效果，即在常见的ARM CPUs上的延迟，和流行的计算机视觉模型的准确率之间的折中。量化方案和高效架构的协同，说明整数代数的推理效果非常好，使视觉识别技术在低端手机市场也可以达到实时的效果。

## A. Appendix: Layer-specific details

### A.1. Mathematical functions

Math functions such as hyperbolic tangent, the logistic function, and softmax often appear in neural networks. No lookup tables are needed since these functions are implemented in pure fixed-point arithmetic similarly to how they would be implemented in floating-point arithmetic.

数学函数，如双曲余弦，logistic函数，和softmax，在神经网络中很常见。其实现不需要查询表，因为这些函数都是用纯定点代数实现的，与其用浮点代数的实现是类似的。

### A.2. Addition

Some neural networks use a plain Addition layer type, that simply adds two activation arrays together. Such Addition layers are more expensive in quantized inference compared to floating-point because rescaling is needed: one input needs to be rescaled onto the other’s scale using a fixed-point multiplication by the multiplier M = S1/S2 similar to what we have seen earlier (end of section 2.2), before the actual addition can be performed as a simple integer addition; finally, the result must be rescaled again to fit the output array’s scale.

一些神经网络会使用最简单的加法层，只是简单的将两个激活阵列相加。与浮点相加相比，这种相加层在量化推理中计算量却很大，因为需要重新确定数值幅度：一个输入的幅度需要调整到另一个的幅度上，使用定点乘法，与乘子M=S1/S2相乘（见2.2节末），然后再用简单的整数相加得到实际相加的效果；最后，结果的幅度还要重新调整到输出阵列的幅度上。

### A.3. Concatenation

Fully general support for concatenation layers poses the same rescaling problem as Addition layers. Because such rescaling of uint8 values would be a lossy operation, and as it seems that concatenation ought to be a lossless operation, we prefer to handle this problem differently: instead of implementing lossy rescaling, we introduce a requirement that all the input activations and the output activations in a Concatenation layer have the same quantization parameters. This removes the need for rescaling and concatenations are thus lossless and free of any arithmetic.

拼接层也有同样的问题。因为这种uint8值的幅度修改会是一种有损运算，但拼接运算看起来应当是无损的，那么我们就用一种不同的方式来处理这个问题：我们不进行有损的幅度改变，而是提出，拼接层中所有的输入激活和输出激活必须有相同的量化参数。这就不需要重新改变幅度了，拼接也就成了无损的，不需要进行任何代数运算。

## B. Appendix: ARM NEON details

## C. Appendix: Graph diagrams

## D. Experimental protocols

### D.1. ResNet protocol

### D.2. Inception protocol

### D.3. COCO detection protocol

### D.4. Face detection and face attribute classification protocol