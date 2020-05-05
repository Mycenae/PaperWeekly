# Machine Learning Systems are Stuck in a Rut

Paul Barham, Michael Isard Google Brain

## 0. Abstract

In this paper we argue that systems for numerical computing are stuck in a local basin of performance and programmability. Systems researchers are doing an excellent job improving the performance of 5-year-old benchmarks, but gradually making it harder to explore innovative machine learning research ideas.

本文我们认为，数值计算系统陷入了性能和可编程性的局部盆地。系统研究者在改进5年期基准测试上做的工作非常好，但在探索新的机器学习研究思想时逐渐越来越困难。

We explain how the evolution of hardware accelerators favors compiler back ends that hyper-optimize large monolithic kernels, show how this reliance on high-performance but inflexible kernels reinforces the dominant style of programming model, and argue these programming abstractions lack expressiveness, maintainability, and modularity; all of which hinders research progress.

我们解释了，硬件加速器的演化怎样更倾向于对大型单一核进行超优化的后台编译器，展示了这种对高性能但不灵活的核的依赖性，强化了编程模型的主导性样式，并说明这些编程抽象缺乏表达性，可维护性和模块性；所有这些都阻碍了研究的进展。

We conclude by noting promising directions in the field, and advocate steps to advance progress towards high-performance general purpose numerical computing systems on modern accelerators.

我们对这个领域中有希望的研究方向进行总结，并支持在现代加速器上向高性能通用目标的数值计算系统进行推进的步骤。

## 1 Compiling for modern accelerators 编译现代加速器

We became interested in this paper’s subject when trying to improve an implementation of Capsule networks [1] to scale up to larger datasets. Capsule networks are an exciting machine learning research idea where scalar-valued “neurons” are replaced by small matrices, allowing them to capture more complex relationships. Capsules may or may not be the “next big thing” in machine learning, but they serve as a representative example of a disruptive ML research idea.

当我们试图改进Capsule网络的实现，对其放大以适应更大的数据集，这是我们就对本文的主题产生了兴趣。Capsule网络是一个很激动人心的机器学习研究思想，其中标量值的神经元替换为了小型矩阵，是其可以捕获更复杂的关系。Capsules可能是机器学习中的下一个大事，也可能不是，但它们是引起混乱的ML研究思想的一个代表性例子。

Although our convolutional Capsule model requires around 4 times fewer floating point operations (FLOPS) with 16 times fewer training parameters than the convolutional neural network (CNN) we were comparing it to, implementations in both TensorFlow[2] and PyTorch[3] were much slower and ran out of memory with much smaller models. We wanted to understand why.

虽然我们的卷积Capsule模型比CNN比较起来，需要的FLOPs少了4倍，训练参数少了16倍，但其用TensorFlow和PyTorch的实现却慢的多，很小的模型就会用掉很多的内存。我们想理解，这是为什么。

### 1.1 New ideas often require new primitives

We won’t discuss the full details of Capsule networks in this paper, but for our purposes it is sufficient to consider a simplified form of the inner loop, which is similar to the computation in a traditional CNN layer but operating on 4×4 matrices rather than scalars.

我们在本文中不讨论Capsule网络的细节，但为了达到目标，只考虑内部循环中的一个简化形式就足够了，这与传统CNN层中的计算类似，但是在4×4的矩阵上运算的，而不是标量。

A basic building block of current machine learning frameworks is the strided 2D convolution. Most frameworks provide a primitive operation that accepts N input images of size H×W, where each pixel has a “depth” of Ci channels. Informally, for a “kernel size” K=3 and “stride” S=2, conv2d computes a weighted sum of overlapping 3×3 patches of pixels centered at every other (x,y) coordinate, to produce N smaller images with pixel depth Co (Figure 1). Mathematically, this can be expressed as follows:

目前机器学习框架的一个基本模块是带步长的2D卷积。多数框架给出的原语操作，接受N个H×W大小的图像作为输入，其中每个像素的深度有Ci个通道。非正式的来说，对于一个核心大小K=3，有步长S=2，conv2d计算的是，在每个(x,y)坐标点上，将3×3大小的图像块与坐标点中心重合，然后计算一个加权求和，生成N个更小的图像，像素深度为Co，见图1。数学上，这可以表示为下式：

$$∀n,x,y,c_o: O_{x,y}^{n,c_o} = \sum_{k_x} \sum_{k_y} \sum_{c_i} I_{sx+k_x,sy+k_y}^{n,c_i} · K_{k_x,k_y}^{c_i,c_o}$$(1)

Figure 1. Conv2D operation with 3×3 kernel, stride=2

where · denotes scalar multiplication, and O, I, and K are all 4-dimensional arrays of scalars. The resulting code is little more than 7 nested loops around a multiply-accumulate operation, but array layout, vectorization, parallelization and caching are extremely important for performance [5].

其中·表示标量相乘，O, I和K都是4维标量阵列。得到的代码是7层嵌套循环，里面是一个乘法累加的运算，但矩阵的排布，向量化，并行化和缓存对于性能来说非常重要。

The analogous computation for convolutional Capsules sums weighted “pose” matrices in 3×3 convolution patches to form “votes”: 卷积Capusule的类似计算，是在3×3卷积块中将加权的pose矩阵进行求和，形成投票：

$$∀n,x,y,c_o: V_{x,y}^{n,c_o} = \sum_{k_x} \sum_{k_y} \sum_{c_i} P_{sx+k_x,sy+k_y}^{n,c_i} · W_{k_x,k_y}^{c_i,c_o}$$(2)

where · now denotes matrix multiplication and V, P, and W are 4-dimensional arrays of 4×4 matrices, or equivalently, 6-dimensional arrays of scalars.

现在·表示矩阵乘法，V, P和W是4×4矩阵的4维阵列，或等价的说，是6维标量阵列。

The following sections explain why ML frameworks make it hard to run the Capsule computation efficiently. 下面的小结解释了为什么ML框架很难高效的运行Capsule计算。

## 2 Compiling kernels is hard

Convolutional Capsule primitives can be implemented reasonably efficiently on CPU (see Table 1) but problems arise on accelerators (e.g., GPU and TPU). Performance on accelerators matters because almost all current machine learning research, and most training of production models, uses them. The marginal cost to perform a particular ML training or large-scale inference workload in a given time is much lower using accelerators than CPUs.

卷积Capsule原语在CPU上实现还是比较高效的（见表1），但在加速器上（如GPU和TPU）问题就出现了。在加速器上的性能很重要，因为几乎所有目前的机器学习研究，以及大多数生产模型的训练都在使用。在加速器上进行特定ML训练，或大规模推理任务时间，比在CPU上少很多。

Table 1. Convolutional Capsules Microbenchmark

Compiler | Device | Compilation | Execution
--- | --- | --- | ---
gcc | x86 (1 cores) | 500ms | 64.3ms
gcc -fopenmp | x86 (6 cores) | 500ms | 11.7ms
PlaidML | GTX1080 | 560ms | 604ms
Tensor Comp. | GTX1080 | 3.2s | 225ms
Tensor Comp. | GTX1080 | 64s | 18.3ms
Tensor Comp. | GTX1080 | 1002s | 1.8ms
CUDA | GTX1080 | 48h | 1.9ms

Accelerators have been very successful for machine learning workloads because the computationally expensive part of training tasks is written as dense linear algebra over multi-dimensional arrays. Dense linear algebra is regular compared to workloads that CPUs are designed for, and comparatively easy to parallelize. Consequently people have built increasingly complex accelerators designed for regular parallel computation. Example accelerator features include “warps”, blocks, and grids of threads, very wide vector arithmetic units (ALUs), and systolic array multipliers (MXUs). As we explain next, it is hard to get good accelerator performance even on these regular computations. While frequently occurring computations receive attention and are well optimized, the performance of non-standard computations like convolutional Capsules suffers.

加速器对于机器学习来说非常成功，因为训练任务中计算量很大的部分，都是在多维阵列上的密集线性代数运算。与CPU设计承担的任务相比，密集线性代数是非常规则的，相对较容易并行化。结果是，人们构建了越来越复杂的加速器，其设计目的都是规则的并行计算。加速器特征的例子包括warps，blocks和线程网格，非常宽的向量代数单元(ALUs)，和收缩的阵列乘法器(MXUs)。就像我们接下来要解释的，在这些常规计算上，很难得到很好的加速器性能。经常发生的计算会得到关注，可以进行很好的优化，而非标准计算的性能，如卷积Capsules则会非常差。

### 2.1 Compiling for accelerators

A major reason that it’s hard to get good performance from regular computations is that the compiler has to consider the memory system of an accelerator as well as the ALUs. In an attempt to prevent data bottlenecks, accelerators’ parallel capabilities have become tightly coupled with the memory system. For example [6]: peak ALU performance on GPUs requires “coalesced loads” where all 32 threads of a warp simultaneously access different values in the same cache line; implementations must be tailored to the sizes and strides implied by the organization of memory banks; and efficient programs must make use of all values loaded in a single memory access which may have large granularity.

从常规计算中很难得到很好的性能，一个主要原因是，编译器要考虑加速器的存储系统，还要考虑ALUs。为防止数据瓶颈，加速器的并行能力，与存储系统是紧密关联的。比如[6]：在GPUs上的峰值ALU性能需要合并的负载，其中一个warp的32个线程要在同样的缓存线中访问不同的值；实现要定制大小和步长，复合内存banks的组织中内含的值；高效的程序需要使用所有装载到单块内存访问的值，这可能有很大的粒度性。

In general accelerator code must perform explicit scheduling through the memory hierarchy rather than relying on transparent multi-level caches. Often memory access granularities require threads to cooperatively load each others’ values and then exchange them; so that code also contains complex instruction scheduling across loop iterations. While matching memory accesses to the parallel ALUs results in good hardware utilization, any mismatch can lead to orders of magnitude of performance slowdown [6]. Avoiding this slowdown requires tuning kernel parameters for e.g., padding, strides, and dimension layout, for each generation of each accelerator.

总体上，加速器代码需要通过内存等级体系进行显式的安排，而不能依靠透明的多层次缓存。通常内存访问粒度，需要线程来配合的装载每个线程的值，然后进行交换；这样代码也会在循环迭代中包含复杂的指令安排。内存访问与并行ALUs的匹配，会得到很好的硬件利用率，而任何不匹配都会导致性能几何级数的下降。避免这种下降，需要对每一代每个加速器调整核的参数，如补零，步长，和维度排布。

For “stencil computations” like convolution in which input values are reused by overlapping computation windows, scheduling loads and stores to optimize memory bandwidth is very challenging and has given rise to sophisticated tools such as Halide [7]. The data-reuse pattern in a convolutional capsule has several additional dimensions of complexity.

对于模板计算，如卷积，其中输入值在重叠的计算窗口中重复使用，安排从内存中装载和存储到内存中的工作，以优化内存带宽，这是非常有挑战性的工作，有一些很复杂的工具来进行，如Halide。卷积Capsule中数据重用的模式，则有几个额外维度的复杂度。

### 2.2 The monolithic kernel approach 单一大核的方法

Because of the difficulty of tuning parameters analytically, and the combinatorial number of choices, high-performance back ends for accelerators expend a lot of development effort on a small set of computational “kernels” (generally, isolated loop nests), such as 2D convolution and batch matrix multiplication, that dominate performance profiles of benchmarks. For each of these kernels the back end maintainers spend hours or days searching for the best algorithm and parameter settings for a small representative set of operand shapes, and then use heuristics or auto-tuning to select one of these pre-tuned implementations at runtime.

因为解析的调节参数的困难性，以及选择数量组合起来非常庞大，加速器高性能的后端，在很少几个计算核上（一般来说，是孤立的循环嵌套）耗费了大量开发努力，如2D卷积和batch矩阵乘积，这两者占据了基准测试中性能图表的主要部分。对每个这样的核，后端维护者耗费大量时间，搜索最好的算法和参数设置，得到少数几个操作数形状的代表性集合，然后使用启发式的或自动调节，在运行时来选择这些预先调节好的实现中的一个。

### 2.3 Compiling custom kernels 编译定制核

It is surprisingly common for machine learning papers to propose new primitives that cannot be computed efficiently with existing kernels. Compilers like Tensor Comprehensions (TC) [8] and PlaidML [9] have been developed to allow end-users to write such custom kernels, and both provide DSLs with concise syntax that resembles the math, e.g., compare the TC implementation of a capsule primitive in Figure 2 with Equation 2.

机器学习文章中会提出新的原语，不能用现有的核进行高效的计算，这非常常见。像Tensor Comprehensions(TC)和PlaidML这样的编译器，开发出来，允许终端用户自己编写这样的定制核，为DSL提供简洁的像数学一样的语法，如，比较图2中的capsule原语的TC实现与式2比较。

Figure 2. Tensor Comprehensions Capsules Code

Despite the obvious appeal, the current state-of-the-art is that these tools are only really suitable for compiling small code fragments: compilation times are often long and the resulting code quality frequently does not come close to peak performance.

虽然有明显的吸引力，但目前最好的状况是，这些工具只适用于编译小型代码片段：编译时间通常太长，得到的代码质量难以接近峰值性能。

Figure 3a illustrates the difficulty that current compiler frameworks have generating GPU code for conventional 2D convolution whose performance competes with the carefully tuned library implementation in cuDNN [10]. In each case we wrote the conv2d implementation (Eqn. 1) using the lowest-level primitives available in the framework. TC uses a genetic search algorithm over optimization parameters and comes within a factor of 8 of cuDNN’s performance, but only after an hour of search. Final performance is also very dependent on the memory layout of the input operands (NCHW vs NHWC). TVM [11] has a similar autotuned convolution template that does not match cuDNN performance after 30 minutes of search (Fig. 3b). PlaidML [9] gets the best performance, just under 4× slower than cuDNN, with rapid compilation time, but uses heuristics that, as we show next, are brittle when the computation is more complex than simple convolution. TVM also has a hand-scheduled conv2d kernel for these operand shapes, but it is almost 19× slower than cuDNN.

图3a描述了这种困难，即目前的编译器框架对传统的2D卷积生成的GPU代码，其性能与cuDNN中仔细调节过的库实现接近。在每个例子中，我们所写的conv2d实现（式1），使用的都是框架中可用的最底层的原语。TC对优化参数使用遗传搜索算法，得到了cuDNN性能的1/8，但只用一小时的搜索。最终的性能，也与输入操作数的内存排布非常有关(NCHW vs NHWC)。TVM有一个类似的自动调节的卷积模板，在30分钟的搜索后，仍然未达到cuDNN的性能（图3b）。PlaidML得到了最佳的性能，比cuDNN慢了4倍，编译速度很快，但使用的是启发式的方法，就像我们下面展示的，当计算比简单的卷积更复杂时，是很脆弱的。TVM对这些操作数的形状，也有一个手动安排的conv2d核，但比cuDNN慢了接近19倍。

Returning to our motivating Capsules example, we next tried implementing custom kernels for the core Capsules primitives (Eqn. 2). As a baseline, compiling the obvious C++ loop nests around a 4×4 matmul function with gcc produces good quality vectorized code that runs in around 60ms on a single x86 core and 11.7ms when parallelized across 6 cores with OpenMP. A hand-written CUDA implementation runs in 1.9ms but took over two days to manually tune.

回到我们的Capsule例子，我们下一步尝试对核心的Capsule原语（式2），实现定制的核。作为基准，用gcc在4×4的矩阵乘法函数上编译C++循环嵌套，会得到质量很高的向量化的代码，在单x86核上的运行时间大约是60ms，当用OpenMP在6核上并行，会得到11.7ms的运行时间。手写的CUDA实现运行时间为1.9ms，但需要超过两天的时间来手动调整。

Though PlaidML compiles as fast at gcc, the resulting kernel executes much slower. Tensor Comprehensions takes nearly 3 minutes to find a kernel that outperforms the CPU, but eventually discovers a schedule that runs in 1.8ms (see Table 1 and Fig. 3c).

虽然PlaidML的编译速度与gcc一样快，得到的核的运行速度却慢很多。Tensor Comprehensions需要大约3分钟的时间来找到一个核，性能超过CPU，但最终发现的计划运行时间在1.8ms（见表1和图3c）。

Our interpretation of these results is that current frameworks excel at workloads where it makes sense to manually tune the small set of computations used by a particular model or family of models. Unfortunately, frameworks become poorly suited to research, because there is a performance cliff when experimenting with computations that haven’t previously been identified as important. While a few hours of search may be acceptable before production deployment, it is unrealistic to expect researchers to put up with such compilation times (recall this is just one kernel in what may be a large overall computation); and even if optimized kernels were routinely cached locally, it would be a major barrier to disseminating research if anyone who downloaded a model’s source code had to spend hours or days compiling it for their hardware before being able to experiment with it.

我们对这些结果的解释是，目前的框架比较擅长的工作是，手动调节好的小型计算集合，这些计算是用于特定模型或特定模型族的。不幸的是，框架对研究工作适应的不好，因为当对之前没有定义为重要的计算进行试验时，性能就会有断崖式的下降。几个小时的搜索可能在生产部署之前是可以接受的，但要研究者接受这样的编译时间，就不太现实了（回忆一下，这只是一个核心，只是一个大型计算的一部分）；甚至即使是优化的核心是例行在本地部署，要传播研究工作的话，下载了模型源码的人需要耗费几个小时甚至几天来编译，才能用来进行试验，这无疑是一个主要的阻碍。

### 2.4 ML framework APIs are inflexible

As we will discuss in more detail in later sections, it is not straightforward to use custom computations in ML frameworks like TensorFlow and PyTorch. As a consequence, the easiest and best performing way to implement convolutional Capsules in both TensorFlow and Pytorch is to target high-level operations that are already supported by those frameworks. The best implementation we have found is the same for both frameworks:

我们在后面的小节中会更详细的讨论，在ML框架中，如TensorFLow或PyTorch，使用定制的计算，并不是简单的事情。结果是，在TensorFlow和PyTorch中实现卷积Capusule，最简单性能最好的方法是，面向这些框架已经支持的高层运算。我们发现的最佳实现，对于两个框架都是一样的：

- Materialize all of the 3x3 image patches (of 4x4 matrices), which with stride 2 almost doubles the input size. 将所有3x3的图像块实现（成4x4的矩阵），在步长为2时这几乎使输入大小加倍。
- Shuffle this large tensor to rearrange dimension order to suit the matrix multiplication operator. 将大型张量进行shuffle，以重新排布维度顺序，适应矩阵相乘的算子。
- Perform a large yet inefficient batch matrix multiplication (of many tall/skinny matrices). 进行一个大型但效率不高的batch矩阵相乘（很多很高很瘦的矩阵）。
- Shuffle the data layout back. 将数据排布shuffle回去。
- Sum the resulting tensor over 3 (non-adjacent) dimensions. 对得到的张量在三个不相邻的维度上进行求和。

To make things worse, between each layer of a Capsules model, votes are “routed” using the Expectation-Maximization (EM) algorithm [12], which repeatedly computes weighted means and variances of the votes using reductions over additional dimensions that could theoretically be fused into the above code (i.e., simply change a ∀ to a Σ in Eqn. 2).

使得事情更糟的是，在一个Capsule模型的每一层之间，投票是使用EM算法进行的，不断重复的计算投票的加权平均值和方差，使用额外维度的reduction，理论上是可以融入上述代码的（在式2中将∀变为Σ）。

Unfortunately, neither framework is able to fuse the final reduction into the batch matrix multiplication (a pre-optimized kernel) which greatly increases the required memory bandwidth as well as intermediate storage requirements . To compute two relatively small quantities, the APIs force us to copy, rearrange and materialize to memory two orders of magnitude more data than strictly necessary.

不幸的是，没有哪个框架可以将最终的reduction与批矩阵相乘（一个预优化的核）融合起来，这会极大的增加需要的内存带宽，以及中间存储需求。为计算两个相对小的量，APIs迫使我们拷贝、重新安排并实现到内存中，大小比严格需要的多两个数量级。

It is issues like the above that have prevented us from finding any high performance implementation of convolutional Capsules to date. 就是上述的这些问题，阻碍了我们找到卷积Capsule的高效实现。

## 3 Compiling programs is harder

In the preceding section we discussed the difficulty of performance-tuning a non-standard kernel. In practice programs must evaluate large graphs of kernels, and strategies for evaluating these graphs introduce more opportunities for optimization. In this section we discuss some of those strategies and point out ways in which the use of inflexible monolithic kernel implementations can constrain their optimizations.

在前一节中，我们讨论了对一个非标准的核进行性能调节的难点。在实践中，程序要计算大型图的核，计算这些图的策略，给优化带来了更多的机会。本节中，我们讨论这些策略中的一部分，指出使用不灵活的单个大核的实现的方法，会对其优化进行约束。

### 3.1 Layout

Even for ML models as simple as ResNet [13], a sequential chain of 2D convolutions that has been extensively benchmarked, operands have different shapes so different convolution invocations may have different optimal parameters. If the layout differs between the producer and consumer operations of an intermediate value then the value must be expensively “transposed” (converted to a different layout). The use of pre-optimized kernels makes good layout assignment harder by constraining every operator to use one of a small number of layouts that have been chosen to be optimal in isolation. If kernel layouts were not constrained in this way a layout assignment algorithm might choose to use a “compromise” layout that is suboptimal for any given operator, but preferable to transposing intermediate values. In practice there are so few choices of layout available that frameworks like XLA [14] and TVM [11] do not attempt a global layout assignment, and instead choose fixed layouts for expensive operators like convolution, then propagate those layouts locally through the operator graph inserting transposes where necessary.

即使是对于像ResNet这样简单的ML模型，2D卷积的顺序链已经被广泛的测试过，操作数有着不同的形状，所以不同的卷积可能有不同的最优参数。如果一个中间值在生产者和消费者运算的排布不一样，那么这个值必须要很费劲的转置（转换到一个不同的排布上去）。预优化的核的使用，使好的排布的指定更难，因为这约束了每个算子使用很少几种排布中的一个，这些排布已经证明是最优的。如果核的排布没有以这种方式约束，那么排布指定算法可能会选择使用一个折中的排布，这对任何给定的算子都是次优的，但对中间值的转置却比较好。在实践中，排布选择是非常少的，像XLA和TVM这样的框架都不试图进行全局排布指定，反而对于很昂贵的算子，如卷积，采用固定的排布，然后将这些排布局部的传播到算子图中，在需要的地方加入转置。

### 3.2 Numerical precision

ML computations are unusual compared to many computing workloads in that they typically involve approximate floating- or fixed-point numbers. A target metric such as test set prediction accuracy may be attainable in more or less time, using more or less energy, by varying the choice of precision and/or non-uniform quantization used to represent intermediate values [15]. Precision can also affect the computation/communication bottleneck. When kernel implementations are fixed in library code it is not practical to include versions for all combinations of precision of inputs and outputs, reducing the opportunity to experiment with and optimize for different choices of quantized and low-precision types.

ML计算与很多计算任务相比，因为一般是近似浮点或定点数的运算，所以是不寻常的。目标度量，如测试集预测准确率，时间或多或少，使用的能力多一点少一点，都可以得到，只要将精度的选择和/或非一致性量化的选择进行改变。精度也会影响计算/通信的瓶颈。当核的实现在库代码中就已经固定时，要包含所有版本的输入输出精度组合，是不现实的，降低了要进行试验的机会，降低了优化不同量化的低精度类型选项的机会。

### 3.3 Interdependent global optimizations 相互依赖的全局优化

We also note some additional whole-program optimizations which would be particularly useful for machine learning computations. The following optimizations are not made more difficult by the use of monolithic kernels, but are perhaps neglected because of the peephole optimization mindset adopted by current frameworks.

我们还要说明一些额外的整个程序的优化，对于机器学习计算来说尤其有用。下列优化，在使用大型单一核时，不会更困难，但由于目前框架采用的优化集很有限，所以忽略了这些。

**Common-subexpression elimination (CSE)**: CSE is surprisingly important for ML frameworks because of the computational structure introduced by backpropagation. Many backward pass operations use “activations” computed in the forward pass, and in the case of deep or recurrent neural networks the consumer of the activation is often far from the producer. Standard CSE favors materializing the activations to memory, but the limited amount of accelerator memory and its slowness relative to ALUs means that it may frequently be preferable to recompute activations instead. Thus CSE introduces another combinatorial search problem for frameworks: choosing which values should be materialized. Heuristics have been proposed to choose between materialization and recomputation [14, 16], but we are not aware of a system that tries to automatically make globally optimal choices about which values to materialize.

**通用子表达式的消除(CSE)**：CSE对ML框架非常重要，因为反向传播带来的计算结构。很多反向过程的运算使用了正向过程计算的激活值，在深度神经网络或循环神经网络的情形中，激活值的消费者，通常与生产者距离很远。标准CSE喜欢将这些激活值存储到内存中，但加速器内存是有限的，而且相对于ALUs是很慢的，意味着可能重新计算激活值更可选一些。因此CSE提出了框架的另一种组合式的搜索问题：选择那些值需要被materialized。已经提出了一些启发式的规则，选择是进行materialization还是重新计算，但我们还不知道有哪个系统，可以自动的得到全局最优的选项，得到哪些值需要进行materialize。

**Distributed execution**: Because of their data-parallel structure, machine learning computations can often usefully be distributed over multiple accelerators. In practice, compilers and machine learning frameworks typically expose distributed parallelism using mechanisms that are disjoint from those used within a device, for example offering only collective reduction and manual point-to-point messaging between devices. Despite initial research in this area [17], to our knowledge no framework tries to jointly optimize the choice of which fragments of a computation should run on which device with the choice of how to structure the subcomputations within a device.

**分布式执行**：因为数据并行的结构，在多个加速器上分布式进行机器学习计算，通常会很有用。实践中，编译器和机器学习框架将分布式并行暴露，使用的机制与在一个设备中使用的是无关的，比如只给出集体缩减和手动点对点的设备之间的信息。尽管在这个领域有初始性的研究，据我们所知，没有哪个框架对下列问题进行联合优化，即一个计算的哪些片段应当在哪些设备上运行，还有在一个设备中怎样对子运算进行分解。

### 3.4 Manual vs automatic search strategies

As explained in the preceding section, it is already hard to compile a single kernel in isolation, on a single device, with layout fixed in advance and relatively few choices for materialization. Optimizing an entire computation means also picking layouts and materialization points, and potentially distribution strategies, making the search space much larger.

在前一小节中已经解释了，在单个设备上，排布先确定好，要进行materialization的选择也相对很少，要独立编译单独一个核就已经很困难了，对整个计算进行优化，意味着选择排布和要materialization的点，还有潜在的分布式策略，这使得搜索空间会极大。

Machine learning training algorithms are particularly dependent on automatic optimization strategies because the majority of the computation is typically performed in gradient operators that are synthesized from the “forward pass” implementation that appears in the source code. Since the code to compute the gradient is not written by the programmer, there is limited opportunity for programmers to guide its optimization: for example, there are no identifiers in scope for intermediate values computed as part of the gradient, to which programmers could manually assign layouts. Even in the absence of auto-generated gradients it is hard to write modular code with the manual optimization annotations used by, e.g., Halide [7] and TVM [11].

机器学习训练算法，尤其依赖于自动优化策略，因为大部分运算一般都是在梯度算子上的，这是从前向过程实现中合成得到的。这些计算梯度的代码不是由程序员写的，因此程序员要引导这个优化过程的机会非常有限：比如，对于计算得到的梯度的一部分的中间值，是没有标识符的，所以程序员无法手动指定排布。即使不是自动生成的梯度，要写模块化的代码进行手动优化标注，且被Halide和TVM使用，这也是很困难的。

Recent research shows growing interest in automatic whole-program optimization techniques [18–20], but approaches are preliminary and typically focus on optimizing only one aspect of a program at a time. There is no doubt that multi-dimensional whole program optimization is a hard task, but we can perhaps take some hope from the recent success of hybrid search/learning approaches such as AlphaGo [21] that show promise in finding good solutions within huge combinatorial search spaces. It seems likely that it will be necessary to architect machine learning frameworks with automatic optimizers in mind before it will be possible to make the best use of whole-program optimization.

最近的研究表明，自动的全程序优化技术兴趣越来越高，但方法都是很初步的，一般聚焦在一次只优化一个程序的一方面。毫无疑问，多维全程序优化是一个很难的工作，但我们可能从最近的混合搜索/学习方法中得到一些希望，如AlphaGo，在巨型组合搜索空间中找到了好的解决方案。似乎在进行全程序优化之前，有必要构建带有自动优化功能的机器学习框架。

## 4 Evolving programming models

Thus far we have concentrated on code generation for accelerators, without much attention to programming models. We first observe that numerical computation benefits from features that are not present in traditional programming languages. Automatic differentiation is one such feature, and numerical programs are also unusual in that they are naturally written using functions that are polymorphic over the rank (number of dimensions) of their arguments. Consider again the standard convolution expression (Eqn. 1). The computations for each element of the batch (n) and output channel (Co) dimensions are independent, and a natural way to express convolution would be in terms of a subcomputation

迄今为止，我们聚焦在加速器的代码生成上，并没有多关注编程模型。我们首先观察到，数值计算受益的特征，在传统编程语言中并不存在。自动微分是一个这样的特征，数值程序也不同寻常，它们是很自然的使用函数写成的，这些函数对其参数的阶数（维度数量）是polymorphic的。考虑标准卷积表达式（式1）。对batch(n)和输出通道(Co)维的每个元素的计算是独立的，表达卷积的一个自然方式是用子计算的方式

$$∀x,y: O_{x,y} = \sum_{k_x} \sum_{k_y} \sum_{c_i} I_{sx+k_x,sy+k_y}^{c_i} · K_{k_x,k_y}^{c_i}$$(3)

written in terms of 3-dimensional inputs I and K. A language could then automatically “lift” the function across batch and output channels if it were applied to inputs with more dimensions. Numerical languages at least as far back as APL [22] have included lifting for rank polymorphism, but there are plenty of open research questions on how to integrate such polymorphism with modern modular types and languages.

这是以三维输入I和K写成的。如果要应用到更多维度的输入上，一种语言就可以自动的从batch和输出通道中将函数提出来。数值语言，像早期的APL，已经包含了对秩多态的lifting，但在怎样将这种多态整合到现代的模块化的类型和语言中，还有很多开放的研究问题

Recall that back ends are structured around calls to large monolithic kernels. In this section we argue that this back-end design approach is slowing progress in the maintainability, debuggability, and expressiveness of programming models. Worse, the resulting brake on innovation in languages is in itself reducing the incentive for back-end developers to improve on the current situation.

回忆一下，后端是按照大型单一核的调用组成结构的。本节中，我们认为，这种后端的设计，在程序模型的可维护性，可调试性和表达能力上，减缓了整个过程。更差的是，在语言的创新上得到的刹车，正在减少后端开发者改进目前的情况的刺激措施。

### 4.1 Opaque operators hurt extensibility

One consequence of monolithic back-end kernels is that front ends choose the kernel or “operator” as a point of abstraction. In popular frameworks like TensorFlow [23] and PyTorch [3], user programs are written in Python and call into operators that are written in terms of back end-specific languages and libraries such as C++, MKL [24], CUDA [25], and cuDNN [10], or sometimes lower-level but portable domain-specific languages such as Tile [9] and Tensor Comprehensions [8]. When existing operators are not sufficient for a task, the user must descend into a lower-level language to write a new operator, and typically also manually write its gradient, a process which can be difficult and error-prone.

单个大型后端核的一个后果是，前端选择核或“算子”作为一种抽象。在流行的框架中，如TensorFlow和PyTorch，用户的程序是用Python写的，调用算子是用后端特定的语言和库写的，如C++，MKL，CUDA和cuDNN，或者有时候是低层但可移植的领域特定的语言，如Tile，和Tensor Comprehensions。当现有的算子不足以完成一个任务时，用户必须下沉到一个更底层的语言来写一个新的算子，而且一般还需要手写其梯度，这个过程可能很困难，并且容易出错。

There are frameworks, such as Julia [26], which nominally use the same language to represent both the graph of operators and their implementations, but back-end designs can diminish the effectiveness of such a front end. In Julia, while 2D convolution is provided as a native Julia library, there is an overloaded conv2d function for GPU inputs which calls NVidia’s cuDNN kernel. Bypassing this custom implementation in favor of the generic code essentially hits a “not implemented” case and falls back to a path that is many orders of magnitude slower.

有一些框架，如Julia，名义上使用相同的语言来表示算子图及其实现，但后端设计会削弱这样一种前端的有效性。在Julia中，2D卷积是作为一个本地Julia库的，但对于GPU输入的情况，有一个重载的conv2d函数，调用的是NVidia的cuDNN核。绕过这个定制的实现，选用通用代码，实质上是一种“未实现”的情况，回到了一条会慢很多个数量级的路径上。

### 4.2 Opaque operators hurt modularity

A more subtle problem with monolithic kernels is that frameworks “commit” to a particular interface for an operator. As we observed in the introduction to this section, the conv2d operator includes a batch dimension n as well as the expected height, width and channels of the input image. Historically, conv2d has been used in mini-batch stochastic gradient descent when training CNNs and the kernel parameters can potentially be reused for each batch element of the image. Supplying a batch of images rather than a single image began as a performance optimization that has become fixed in the API.

单一大核的一个更微妙的问题是，框架对一个算子只给出了一个特定的接口。就像我们在本节简介中观察到的，conv2d算子包含的，除了期望的输入图像的高度，宽度和通道数，还有batch这个维度。历史上，conv2d在mini-batch SGD中使用，进行CNNs的训练，核参数对图像的每个batch元素都可以重用。给出一批图像，而不是单个图像，开始作为一种性能优化，这种形式在API中已经固定下来。

Convolutions are in fact used in many other computations whose input may naturally have 3 or more than 4 dimensions, but to call conv2d the programmer must first transform the input into a “view” in which all the data-parallel dimensions are placed in the batch “slot”, and afterwards re-transform the output to restore the computation’s meaningful dimensions. These transforms can involve shuffling elements in memory, and have potentially large performance cost. Users thus add transforms in as few places as possible, which can make it very hard to keep track of the meaning of the dimensions of intermediate quantities.

卷积实际上是用在很多其他计算中的，其输入很自然的有3个或4个维度，但为了调用conv2d，程序员必须首先将输入转换成一种视图，其中所有的数据并行维度都放入batch slot中，然后将输出重新变换，恢复计算的有意义维度。这些变换涉及到在内存中元素的移动，会性能有很大的代价。用户因此在尽可能少的地方增加变换，这使得要追踪中间量的维度的意义，非常困难。

As we suggested earlier, an alternative at the language level would be to supply a lower-dimensional conv2d function and lift it to higher dimensions where necessary. We believe this would greatly improve readability and maintainability. Monolithic back-end kernels are an obstacle to such an approach, because the language would have to automatically discover patterns that were equivalent to the monolithic kernels under some dimension ordering, and then automatically reorder dimensions before calling the kernel. Any failure of this pattern-matching would cause a severe performance cliff. We are encouraged by recent attempts to support a restricted form of lifting [27] but we believe progress would be much more rapid if there were a compiler that could emit efficient code for general-purpose lifted functions.

如同我们前面提到的，语言层次的一种替换方法是，提供一种更低维度的conv2d函数，在需要的地方将其提升到更高的维度中。我们相信这会极大的改进可读性和可维护性。单一大型后端核是这种方法的一个障碍，因为这种语言需要自动发现等价于单一大型核在一些维度排序下的模式，然后在调用核之前自动重拍维度。这种模式匹配的任何失败，都会导致严重的性能断崖式下降。我们受到最近的支持受限形式的lifting的努力的鼓励，但我们相信，如果有一个编译器可以对通用目标的lifted函数释放高效的代码，这个过程会快的多。

### 4.3 Ordered dimensions considered harmful

People have frequently advocated for “named dimensions”, whereby the dimensions of an array are associated with a textual name along with, or instead of, their numeric position [28–31]. Named dimensions improve readability by making it easier to determine how dimensions in the code correspond to the semantic dimensions described in, e.g., a research paper. We believe their impact could be even greater in improving code modularity, as named dimensions would enable a language to move away from fixing an order on the dimensions of a given tensor, which in turn would make function lifting more convenient. (In APL, for example, where dimensions are strictly ordered, the rightmost dimensions are always lifted and an argument must be transposed to ensure the correct ordering before the function is called.)

人们经常支持“维度命名”，因此一个阵列的维度是与伴随的textual名字相关的，或其数值位置。命名的维度，通过更容易确定代码中的维度与语义维度如何对应，改进了可读性。我们相信，其影响在改进代码模块性上甚至更高，因为命名的维度可以使一种语言可以不用对给定的张量固定维度顺序，这又会使函数lifting更加方便。（比如，在APL中，其维度是有严格顺序的，最右边的维度永远是lifted，一个参数必须转置，来保证在函数调用前确保正确的顺序）

In order to efficiently implement programs with named or unordered dimensions, we believe it will be necessary to rethink the design of the back end, for example adopting an IR that operates over unordered sets of dimensions, followed by a device-specific lowering to a concrete (ordered and optionally padded/tiled) memory layout. We are pleased to note that several projects [11, 14, 32] have taken preliminary steps in this direction by decoupling to some extent the dimension order of the source code from that of the lowering, although they still require the front end to specify ordered dimensions for every array.

为用命名的维度或未命名的维度高效的实现程序，我们相信有必要重新思考后端的设计，比如采用一种IR，在未排序的维度集合上进行运算，然后再用一种设备相关的lowering得到一种具体的内存排布（排序的，有选择的补零/tiled）。我们很高兴指出，有几个项目已经在这个方向上开始了初始的步子，将源码中的维度顺序，与lowering中的，在一定程度上解除其耦合关系，虽然他们还需要前端来对每个阵列指定有顺序的维度。

## 5 A way forward

It is hard to experiment with front end features like named dimensions, because it is painful to match them to back ends that expect calls to monolithic kernels with fixed layout. On the other hand, there is little incentive to build high quality back ends that support other features, because all the front ends currently work in terms of monolithic operators. An end-to-end tool chain for machine learning requires solutions from many specialist disciplines. Despite impressive and sometimes heroic efforts on some of the sub-problems, we as a community should recognize that we aren’t doing a great job of tackling the end-to-end problem in an integrated way. There are many sub-problems that we could be working on in an independent but coordinated way. In this spirit we present a possible research agenda:

很难用前端的特征如命名的维度来进行试验，因为要将其与后端进行匹配，后端期待的调用的是大型单一核，排布固定，这会非常痛苦。另一方面，构建高质量、支持其他特征的后端的动机不足，因为所有前端目前都是以大型单一算子的形式进行工作的。一种机器学习端到端的工具链，需要很多专家级的原则的解决方案。尽管在一些子问题中有很令人印象深刻、有时候英雄式的努力，作为一个团体来说，我们应当承认，我们在解决端到端问题上并没有完成的很好。有很多子问题，我们可以以一种独立但协调的方式进行工作。根据这种思想，我们提出一种可能的研究日程：

- Language design, including automatic differentiation, using purely named dimensions and kernels expressed within the language syntax. 语言设计，包括自动微分，使用纯粹的命名维度和核，在语言语法中进行表达。
- A back end IR defining a graph of layout-agnostic general-purpose loop nests. 一种后端IR，定义了一个图，包含与排布无关的通用目标的循环嵌套。
- Transformation passes for the above IR that lower it to a concrete CSE strategy, with the layout of each materialized intermediate. 将上述IR进行变换，并传递lower到一个具体的CSE策略，并包含每个materialized中间表示的排布。
- Compilation passes that generate accelerator code given the lowered IR above, producing adequate code quickly, and close to peak performance after searching. 给定上述lowered IR，生成的加速器代码，进行编译，迅速生成足够的代码，在搜索后接近峰值性能。

We do not want to minimize the thought and engineering that has gone into current machine learning tool chains, and clearly they are valuable to many. Our main concern is that the inflexibility of languages and back ends is a real brake on innovative research, that risks slowing progress in this very active field. We urge our colleagues to bear this in mind when designing accelerators, tool chains, and especially benchmarks.

我们不希望最小化已经进入目前机器学习工具链的思想和工程，很明显对很多人都很宝贵。我们主要的考虑是，语言和后端的不灵活性，是研究创新的一个阻碍，在这个活跃的领域中，很可能减缓这个过程。我们敦促我们的同事，在设计加速器、工具链，尤其是基准测试时，心里要记得这件事。