# TVM: An Automated End-to-End Optimizing Compiler for Deep Learning

Tianqi Chen et al. University of Washington, AWS, Shanghai Jiao Tong University, UC Davis, Cornell

## Abstract 摘要

There is an increasing need to bring machine learning to a wide diversity of hardware devices. Current frameworks rely on vendor-specific operator libraries and optimize for a narrow range of server-class GPUs. Deploying workloads to new platforms – such as mobile phones, embedded devices, and accelerators (e.g., FPGAs, ASICs) – requires significant manual effort. We propose TVM, a compiler that exposes graph-level and operator-level optimizations to provide performance portability to deep learning workloads across diverse hardware back-ends. TVM solves optimization challenges specific to deep learning, such as high-level operator fusion, mapping to arbitrary hardware primitives, and memory latency hiding. It also automates optimization of low-level programs to hardware characteristics by employing a novel, learning-based cost modeling method for rapid exploration of code optimizations. Experimental results show that TVM delivers performance across hardware back-ends that are competitive with state-of-the-art, hand-tuned libraries for low-power CPU, mobile GPU, and server-class GPUs. We also demonstrate TVM’s ability to target new accelerator back-ends, such as the FPGA-based generic deep learning accelerator. The system is open sourced and in production use inside several major companies.

越来越需要将机器学习部署到各种各样的硬件设备中。目前的框架依赖与大公司相关的算子库，并为很少的服务器级的GPU进行优化。将工作部署到新的平台上，如移动手机，嵌入式设备，和加速器（如FPGA, ASIC）上，需要更多的人力劳动。我们提出的TVM是一个编译器，进行了图级和算子级的优化，使深度学习工作在不同的硬件后端上都可以进行移植。TVM专门解决深度学习方面的优化挑战，如高层算子融合，映射到任意的硬件上，隐藏内存延迟等。TVM还自动优化低层程序在硬件特征上的表现，采用了一个新的基于学习代价建模方法，以进行代码优化的快速探索。试验结果表明，TVM在不同硬件后端上得到类似的性能，与目前最好的、手动调节的库可以类比，包括低能耗CPU，移动GPU，服务器级GPU等。我们还展示了TVM在新的加速器后端上的能力，如基于FPGA的通用深度学习加速器。系统是开源的，在几个主要的公司中的生产环境已经使用。

## 1 Introduction 引言

Deep learning (DL) models can now recognize images, process natural language, and defeat humans in challenging strategy games. There is a growing demand to deploy smart applications to a wide spectrum of devices, ranging from cloud servers to self-driving cars and embedded devices. Mapping DL workloads to these devices is complicated by the diversity of hardware characteristics, including embedded CPUs, GPUs, FPGAs, and ASICs (e.g., the TPU [21]). These hardware targets diverge in terms of memory organization, compute functional units, etc., as shown in Figure 1.

深度学习模型模型现在可以识别图像，处理自然语言，在很有挑战的策略游戏中击败人类。越来越需要将智能应用部署到各种不同的设备上，从云端服务器到自动驾驶汽车和嵌入式设备。将深度学习工作映射到这些设备上是非常复杂的，因为硬件特性非常多，包括嵌入式CPU，GPU，FPGA和ASIC（如TPU[21]）。这些硬件的特点非常不同，如内存组织，计算性能单元等，如图1所示。

Figure 1: CPU, GPU and TPU-like accelerators require different on-chip memory architectures and compute primitives. This divergence must be addressed when generating optimized code.

图1：CPU，GPU和类TPU的加速器需要不同的片上内存架构和计算原语。在生成优化代码的时候必须处理这种多样性。

Current DL frameworks, such as TensorFlow, MXNet, Caffe, and PyTorch, rely on a computational graph intermediate representation to implement optimizations, e.g., auto differentiation and dynamic memory management [3, 4, 9]. Graph-level optimizations, however, are often too high-level to handle hardware back-end-specific operator-level transformations. Most of these frameworks focus on a narrow class of server-class GPU devices and delegate target-specific optimizations to highly engineered and vendor-specific operator libraries. These operator-level libraries require significant manual tuning and hence are too specialized and opaque to be easily ported across hardware devices. Providing support in various DL frameworks for diverse hardware back-ends presently requires significant engineering effort. Even for supported back-ends, frameworks must make the difficult choice between: (1) avoiding graph optimizations that yield new operators not in the predefined operator library, and (2) using unoptimized implementations of these new operators.

现在的深度学习框架，如TensorFlow, MXNet, Caffe和PyTorch，依赖的是一个计算图中间表示来实现优化，如自动微分和动态内存管理[3,4,9]。但图级的优化通常太高层，难以处理特定硬件后端的算子级的变换。多数这些框架聚焦在少数几种服务器级的GPU设备，将对特定目标的优化交给高度工程化的针对特定大厂的算子库。这些算子级的库需要很多手工调节工作，所以是非常专门化的，不能很容易的在不同硬件设备之间进行移植。在各种深度学习框架中为不同硬件后端提供支持现在需要非常多的工程努力。即使对于已支持的后端，框架也必须做出以下艰难的选择：(1)防止图优化产生新的算子不在预定义的算子库中，(2)使用这些新算子的未优化实现。

To enable both graph- and operator-level optimizations for diverse hardware back-ends, we take a fundamentally different, end-to-end approach. We built TVM, a compiler that takes a high-level specification of a deep learning program from existing frameworks and generates low-level optimized code for a diverse set of hardware back-ends. To be attractive to users, TVM needs to offer performance competitive with the multitude of manually optimized operator libraries across diverse hardware back-ends. This goal requires addressing the key challenges described below.

为对不同的硬件后端实现图级和算子级的优化，我们采用了一种根本上不同的、端到端的方法。我们构建了一个编译器TVM，其输入是从各种已有框架下的深度学习程序的高层规范，生成的对不同硬件后端的底层优化的代码。为吸引用户，TVM需要得到在不同硬件后端下手工优化的算子库类似的性能。为达到这个目标，需要解决下述的关键挑战。

**Leveraging Specific Hardware Features and Abstractions**. DL accelerators introduce optimized tensor compute primitives [1, 12, 21], while GPUs and CPUs continuously improve their processing elements. This poses a significant challenge in generating optimized code for a given operator description. The inputs to hardware instructions are multi-dimensional, with fixed or variable lengths; they dictate different data layouts; and they have special requirements for memory hierarchy. The system must effectively exploit these complex primitives to benefit from acceleration. Further, accelerator designs also commonly favor leaner control [21] and offload most scheduling complexity to the compiler stack. For specialized accelerators, the system now needs to generate code that explicitly controls pipeline dependencies to hide memory access latency – a job that hardware performs for CPUs and GPUs.

**利用特定的硬件特征和抽象**。深度学习加速器引入了优化的张量计算原语[1,12,21]，同时GPU和CPU持续改进其处理元素。这对生成给定算子描述的优化代码提出了不小的挑战。硬件指令的输入是多维的，长度是固定的或可变的；它们使用不同的数据分布方式；它们对内存层次有特别的需求。系统需要有效的利用这些复杂的原语，以从加速器中获益。而且，加速器设计也通常更倾向于较少的控制[21]，将大多数调度的复杂任务交给编译栈。对特定的加速器来说，系统现在需要生成显式的控制流水依赖，以隐藏内存访问延迟的代码，这对于CPU和GPU来说，是硬件执行的工作。

**Large Search Space for Optimization**. Another challenge is producing efficient code without manually tuning operators. The combinatorial choices of memory access, threading pattern, and novel hardware primitives creates a huge configuration space for generated code (e.g., loop tiles and ordering, caching, unrolling) that would incur a large search cost if we implement black box auto-tuning. One could adopt a predefined cost model to guide the search, but building an accurate cost model is difficult due to the increasing complexity of modern hardware. Furthermore, such an approach would require us to build separate cost models for each hardware type.

**大型搜索优化空间**。另一个挑战是不需要手工调节算子而生成高效的代码。内存访问、线程模式和新的硬件原语的组合形成了生成代码非常大的配置空间（如，loop tiles and ordering，缓存，unrolling），如果我们进行黑盒自动调节，会有很大的搜索代价。可以采用预定义的代价模型来对搜索进行导引，但构建一个准确的代价模型非常困难，因为现代硬件的复杂度不断增加。而且，这样一种方法需要我们对每种硬件类型都构建不同的代价模型。

TVM addresses these challenges with three key modules. (1) We introduce a tensor expression language to build operators and provide program transformation primitives that generate different versions of the program with various optimizations. This layer extends Halide [32]’s compute/schedule separation concept by also separating target hardware intrinsics from transformation primitives, which enables support for novel accelerators and their corresponding new intrinsics. Moreover, we introduce new transformation primitives to address GPU-related challenges and enable deployment to specialized accelerators. We can then apply different sequences of program transformations to form a rich space of valid programs for a given operator declaration. (2) We introduce an automated program optimization framework to find optimized tensor operators. The optimizer is guided by an ML-based cost model that adapts and improves as we collect more data from a hardware backend. (3) On top of the automatic code generator, we introduce a graph rewriter that takes full advantage of high- and operator-level optimizations.

TVM用三个关键模块解决这些挑战。(1)我们引入一种张量表达式语言来构建算子，提供程序变换原语，生成不同版本的各种优化程序。这一层拓展了Halide[32]的计算/调度分离概念，也分离了目标硬件特征与变换原语，所以可以支持新型加速器及其对应的新特性。而且，我们引入了新的变换原语，来解决GPU相关的挑战，并可以部署在特定的加速器的。然后我们就可以应用不同的程序变换序列，对给定的算子声明，形成多种可用程序。(2)我们引入了一种自动程序优化框架，找到优化的张量算子。优化器由基于机器学习的代价模型指引，随着我们从硬件后端收集了更多的数据，会逐渐适应并改进。(3)在自动代码生成器之上，我们引入了图重写器，可以充分利用高层和算子层的优化优势。

By combining these three modules, TVM can take model descriptions from existing deep learning frameworks, perform joint high- and low-level optimizations, and generate hardware-specific optimized code for backends, e.g., CPUs, GPUs, and FPGA-based specialized accelerators.

这三个模块结合到一起，TVM可以从现有深度学习框架中获得模型描述，进行高层和底层的联合优化，生成对特定硬件后端优化的代码，如CPU，GPU，和基于FPGA的特定加速器。

This paper makes the following contributions: 本文做出了以下贡献：

- We identify the major optimization challenges in providing performance portability to deep learning workloads across diverse hardware back-ends. 我们给出了在不同硬件后端上进行深度学习工作移植性的主要优化挑战。

- We introduce novel schedule primitives that take advantage of cross-thread memory reuse, novel hardware intrinsics, and latency hiding. 我们引入了新的调度原语，利用了跨线程内存重使用、新的硬件特征和延迟隐藏的优势。

- We propose and implement a machine learning based optimization system to automatically explore and search for optimized tensor operators. 我们提出并实现了一种基于机器学习的优化系统，可以自动探索并搜索优化的张量算子。

- We build an end-to-end compilation and optimization stack that allows the deployment of deep learning workloads specified in high-level frameworks (including TensorFlow, MXNet, PyTorch, Keras, CNTK) to diverse hardware back-ends (including CPUs, server GPUs, mobile GPUs, and FPGA-based accelerators). The open-sourced TVM is in production use inside several major companies. 我们构造了一个端到端的编译和优化栈，可以将特定高层框架(including TensorFlow, MXNet, PyTorch, Keras, CNTK)的深度学习任务，部署到各种硬件后端上（包括CPU，服务器GPU，移动端GPU和基于FPGA的加速器）。开源的TVM在几个主要公司里已经得到了生产环境的使用。

We evaluated TVM using real world workloads on a server-class GPU, an embedded GPU, an embedded CPU, and a custom generic FPGA-based accelerator. Experimental results show that TVM offers portable performance across back-ends and achieves speedups ranging from 1.2× to 3.8× over existing frameworks backed by hand-optimized libraries. 我们使用真实世界的任务，在服务器级GPU、嵌入式GPU、嵌入式CPU和定制的通用FPGA加速器上评估TVM。试验结果表明，TVM在不同后端上给出了可移植的性能，与现有的手工优化库的框架后端比，有1.2x到3.8x的加速效果。

## 2 Overview 概览

This section describes TVM by using an example to walk through its components. Figure 2 summarizes execution steps in TVM and their corresponding sections in the paper. The system first takes as input a model from an existing framework and transforms it into a computational graph representation. It then performs high-level dataflow rewriting to generate an optimized graph. The operator-level optimization module must generate efficient code for each fused operator in this graph. Operators are specified in a declarative tensor expression language; execution details are unspecified. TVM identifies a collection of possible code optimizations for a given hardware target’s operators. Possible optimizations form a large space, so we use an ML-based cost model to find optimized operators. Finally, the system packs the generated code into a deployable module.

本节通过一个例子，来描述TVM的各个部件。图2总结了TVM的执行步骤，以及对应本文中的相应章节。系统首先以现有框架中的模型作为输入，将其变换为一个计算图表示。然后进行高层数据流重写，以生成优化的图。算子级的优化模块必须对图中每个融合的算子生成高效的代码。算子用一种声明式的张量表达式语言进行指定；执行的细节则没有指定。TVM对给定的目标硬件算子，给出可能的代码优化集合。可能的优化形成了一个很大的空间，所以我们使用一种基于机器学习的代价模型来寻找优化的算子。最后，系统将生成的代码打包成可部署的模块。

Figure 2: System overview of TVM. The current stack supports descriptions from many deep learning frameworks and exchange formats, such as CoreML and ONNX, to target major CPU, GPU and specialized accelerators.

**End-User Example**. In a few lines of code, a user can take a model from existing deep learning frameworks and call the TVM API to get a deployable module: 终端用户例子，用几行代码，用户就可以从现有的深度学习框架中引入输入，调用TVM的API得到可部署的模型

```
import tvm as t
# Use keras framework as example, import model
graph, params = t.frontend.from_keras(keras_model)
target = t.target.cuda()
graph, lib, params = t.compiler.build(graph, target, params)
```

This compiled runtime module contains three components: the final optimized computational graph (graph), generated operators (lib), and module parameters (params). These components can then be used to deploy the model to the target back-end: 这个编译好的运行时模块包含三个部件：最后优化的计算图(graph)，生成的算子(lib)，和参数模块(params)。这些模块可以用于部署模型到目标后端：

```
import tvm.runtime as t
module = runtime.create(graph, lib, t.cuda(0))
module.set_input(**params)
module.run(data=data_array)
output = tvm.nd.empty(out_shape, ctx=t.cuda(0))
module.get_output(0, output)
```

TVM supports multiple deployment back-ends in languages such as C++, Java and Python. The rest of this paper describes TVM’s architecture and how a system programmer can extend it to support new back-ends. TVM支持多种部署后端，支持语言包括C++, JAVA和Python。本文剩下的部分描述了TVM的架构，以及一个系统程序员怎样将其拓展到新的后端支持。

## 3 Optimizing Computational Graphs 优化计算图

Computational graphs are a common way to represent programs in DL frameworks [3, 4, 7, 9]. Figure 3 shows an example computational graph representation of a two-layer convolutional neural network. The main difference between this high-level representation and a low-level compiler intermediate representation (IR), such as LLVM, is that the intermediate data items are large, multi-dimensional tensors. Computational graphs provide a global view of operators, but they avoid specifying how each operator must be implemented. Like LLVM IRs, a computational graph can be transformed into functionally equivalent graphs to apply optimizations. We also take advantage of shape specificity in common DL workloads to optimize for a fixed set of input shapes.

计算图是在深度学习框架中表示程序的常用方式[3,4,7,9]。图3给出了计算图表示两层卷积神经网络的例子。高层表示和底层编译器中间表示(IR, e.g., LLVM)的主要区别是，中间数据项是大型多维张量。计算图给出了算子的全局视野，但没有给出每个算子是怎样实现的。与LLVM IR类似，计算图可以转化为功能上等价的图，以进行优化。我们还利用了常用深度学习任务中的形状特定类型，以对固定集合的输入大小进行优化。

Figure 3: Example computational graph of a two-layer convolutional neural network. Each node in the graph represents an operation that consumes one or more tensors and produces one or more tensors. Tensor operations can be parameterized by attributes to configure their behavior (e.g., padding or strides).

图3.两层卷积神经网络的计算图例子。图中的每个节点代表了一个运算，以一个或多个张量为输入，生成一个或多个张量。张量运算可以用属性进行参数化，以配置其行为（如，补0或步长）。

TVM exploits a computational graph representation to apply high-level optimizations: a node represents an operation on tensors or program inputs, and edges represent data dependencies between operations. It implements many graph-level optimizations, including: operator fusion, which fuses multiple small operations together; constant-folding, which pre-computes graph parts that can be determined statically, saving execution costs; a static memory planning pass, which pre-allocates memory to hold each intermediate tensor; and data layout transformations, which transform internal data layouts into back-end-friendly forms. We now discuss operator fusion and the data layout transformation.

TVM利用计算图表示来进行高层优化：一个节点表示在张量或程序输入上的运算，边代表数据在运算之间的依赖关系。TVM实现了很多图级的优化，包括：算子融合，将多个小的运算融合到一起；常量折叠，即预先计算好图中可以静态确定的部分，节省运行代价；静态的内存规划pass，即预先分配好内存，以保留给每个中间张量；数据分布变换，将内部数据分布变换到后端友好的形式。我们现在讨论算子融合，和数据分布变换。

**Operator Fusion**. Operator fusion combines multiple operators into a single kernel without saving the intermediate results in memory. This optimization can greatly reduce execution time, particularly in GPUs and specialized accelerators. Specifically, we recognize four categories of graph operators: (1) injective (one-to-one map, e.g., add), (2) reduction (e.g., sum), (3) complex-out-fusable (can fuse element-wise map to output, e.g., conv2d), and (4) opaque (cannot be fused, e.g., sort). We provide generic rules to fuse these operators, as follows. Multiple injective operators can be fused into another injective operator. A reduction operator can be fused with input injective operators (e.g., fuse scale and sum). Operators such as conv2d are complex-out-fusable, and we can fuse element-wise operators to its output. We can apply these rules to transform the computational graph into a fused version. Figure 4 demonstrates the impact of this optimization on different workloads. We find that fused operators generate up to a 1.2× to 2× speedup by reducing memory accesses.

**算子融合**。算子融合将多个算子综合成单个核，中间结果不在内存中保留。这种优化可以极大的减少执行时间，尤其是在GPU和专有加速器中。特别的，我们识别4类图算子：(1)单射的（一对一映射，如加法）；(2)reduction（如，求和）；(3)complex-out-fusable（可以逐元素融合映射到输出，如conv2d）；(4)opaque（不能融合的，如排序）。我们给出融合这些算子的通用规则，如下。多个单射算子可以融合成另一个单射算子。一个reduction算子可以与输入单射算子融合到一起（如，将尺度和求和融合到一起）。像conv2d这样的算子是complex-out-fusable，我们可以将逐元素的算子与其输出融合。我们可以应用这些规则，将计算图变换为融合的版本。图4证明了这种优化在不同任务上的影响。我们发现，融合的算子通过减少内存访问，可以得到1.2×到2×的加速效果。

Figure 4: Performance comparison between fused and non-fused operations. TVM generates both operations. Tested on NVIDIA Titan X. 图4：融合和非融合算子的性能比较。TVM可以生成这两种算子。

**Data Layout Transformation**. There are multiple ways to store a given tensor in the computational graph. The most common data layout choices are column major and row major. In practice, we may prefer to use even more complicated data layouts. For instance, a DL accelerator might exploit 4 × 4 matrix operations, requiring data to be tiled into 4 × 4 chunks to optimize for access locality.

**数据分布变换**。在计算图中存储给定的张量有多种方法。最常见的数据分布选择是列为主和行为主。实践中，我们可能会使用更复杂的数据分布。例如，一个深度学习加速器可能会进行4×4的矩阵运算，需要数据以4×4的块堆叠起来，以优化访问局部性。

Data layout optimization converts a computational graph into one that can use better internal data layouts for execution on the target hardware. It starts by specifying the preferred data layout for each operator given the constraints dictated by memory hierarchies. We then perform the proper layout transformation between a producer and a consumer if their preferred data layouts do not match.

数据分布优化将计算图转化为另一种形式，使用更好的内部数据分布，可以更好的在目标硬件上执行。其开始于指定每个算子在给定内存层级决定的限制下，偏好的数据分布。我们然后在producer和consumer之间进行合理的分布变换，如果它们偏好的数据分布不匹配的话。

While high-level graph optimizations can greatly improve the efficiency of DL workloads, they are only as effective as what the operator library provides. Currently, the few DL frameworks that support operator fusion require the operator library to provide an implementation of the fused patterns. With more network operators introduced on a regular basis, the number of possible fused kernels can grow dramatically. This approach is no longer sustainable when targeting an increasing number of hardware back-ends since the required number of fused pattern implementations grows combinatorially with the number of data layouts, data types, and accelerator intrinsics that must be supported. It is not feasible to handcraft operator kernels for the various operations desired by a program and for each back-end. To this end, we next propose a code generation approach that can generate various possible implementations for a given model’s operators.

高层图优化可以极大的改进深度学习任务的效率，但它们还是只与算子库提供的效率一样。目前，仅有几个深度学习框架支持算子融合，而且需要算子库提供融合算子的实现。随着更多的网络算子以常规方式提出，可能的融合核心会急剧增长。当目标硬件后端数量增加时，这种方法是不可持续的，因为需要的融合算子数量实现，还要考虑要支持的数据分布形式、数据类型和加速器特性，会以指数级增长。对一个程序需要的各种算子，为各种硬件后端手工设计算子核心，这是不可行的。为此，下面我们提出一种代码生成方法，可以对给定模型的算子生成各种可能的实现。

## 4 Generating Tensor Operations 生成张量算子

TVM produces efficient code for each operator by generating many valid implementations on each hardware back-end and choosing an optimized implementation. This process builds on Halide’s idea of decoupling descriptions from computation rules (or schedule optimizations) [32] and extends it to support new optimizations (nested parallelism, tensorization, and latency hiding) and a wide array of hardware back-ends. We now high-light TVM-specific features.

TVM会对每个算子生成高效的代码，方法是在每个硬件后端上生成很多可用的实现，然后选择一个优化的实现。这个过程是构建在Halide的思想之上的，即描述与计算规则（或调度优化）分离[32]，并将其进行了拓展，以支持新的优化（嵌套并行，张量化，和延迟隐藏），和支持更多的硬件后端。现在我们强调叙述一下TVM特有的特征。

### 4.1 Tensor Expression and Schedule Space 张量表达式和调度空间

We introduce a tensor expression language to support automatic code generation. Unlike high-level computation graph representations, where the implementation of tensor operations is opaque, each operation is described in an index formula expression language. The following code shows an example tensor expression to compute transposed matrix multiplication:

我们引入一种张量表达式语言，以支持自动代码生成。高层计算图表示中的张量算子实现是不明确的，与之不同的是，张量表达式中每个算子是以一种索引公式表达式语言所描述的。下面的代码就是张量表达式的例子，计算的是转置的矩阵乘法：

```
m, n, h = t.var('m'), t.var('n'), t.var('h')
A = t.placeholder((m, h), name='A')
B = t.placeholder((n, h), name='B')
k = t.reduce_axis((0, h), name='k')
C = t.compute((m, n), lambda y, x: t.sum(A[k, y] * B[k, x], axis=k))
```

Each compute operation specifies both the shape of the output tensor and an expression describing how to compute each element of it. Our tensor expression language supports common arithmetic and math operations and covers common DL operator patterns. The language does not specify the loop structure and many other execution details, and it provides flexibility for adding hardware-aware optimizations for various backends. Adopting the decoupled compute/schedule principle from Halide [32], we use a schedule to denote a specific mapping from a tensor expression to low-level code. Many possible schedules can perform this function.

每个计算运算指定了输出张量的形状，和一个表达式，描述的是怎样计算其中的每个元素。我们的张量表达式语言支持通常的代数和数学运算，也支持常见的深度学习算子模式。这个语言并没有指定循环结构和其他很多执行的细节，为对各种后端增加感知硬件的优化提供了灵活性。采用Halide[32]的分离的计算/调度原则，我们使用一个schedule来表示从张量表达式到底层代码的特定映射。很多可能的schedule可以执行这个函数。

We build a schedule by incrementally applying basic transformations (schedule primitives) that preserve the program’s logical equivalence. Figure 5 shows an example of scheduling matrix multiplication on a specialized accelerator. Internally, TVM uses a data structure to keep track of the loop structure and other information as we apply schedule transformations. This information can then help generate low-level code for a given final schedule.

我们创建一个schedule的方法是，递增的应用保持程序逻辑等价性的基本变换（shcedule原语）。图5是一个在特殊加速器上调度矩阵相乘的例子。在内部，TVM在我们应用调度变换时，使用一个数据结构来追踪循环结构和其他信息。这些信息可以为给定的最终调度帮助生成底层代码。

Figure 5: Example schedule transformations that optimize a matrix multiplication on a specialized accelerator.

Our tensor expression takes cues from Halide [32], Darkroom [17], and TACO [23]. Its primary enhancements include support for the new schedule optimizations discussed below. To achieve high performance on many back-ends, we must support enough schedule primitives to cover a diverse set of optimizations on different hardware back-ends. Figure 6 summarizes the operation code generation process and schedule primitives that TVM supports. We reuse helpful primitives and the low-level loop program AST from Halide, and we introduce new primitives to optimize GPU and accelerator performance. The new primitives are necessary to achieve optimal GPU performance and essential for accelerators. CPU, GPU, TPU-like accelerators are three important types of hardware for deep learning. This section describes new optimization primitives for CPUs, GPUs and TPU-like accelerators, while section 5 explains how to automatically derive efficient schedules.

我们的张量表达式从Halide[32], Darkroom[17]和TACO[23]中汲取了思想。其基本改进包括支持下面所述的新调度优化。为在很多后端上得到高性能，我们必须支持足够多的调度原语，以覆盖不同硬件后端的多种优化。图6总结了算子代码生成过程和TVM支持的调度原语。我们从Halide中复用有帮助的原语和底层循环程序AST，并引入了新的原语以优化GPU和加速器性能。新的原语是取得最佳的GPU性能所必须的，对加速器也是非常重要的。CPU，GPU和类TPU的加速器是三种重要的深度学习硬件类型。本节描述了CPU，GPU和类TPU加速器上的新优化原语，第5节解释了怎样自动导出高效的调度。

Figure 6: TVM schedule lowering and code generation process. The table lists existing Halide and novel TVM scheduling primitives being used to optimize schedules for CPUs, GPUs and accelerator back-ends. Tensorization is essential for accelerators, but it can also be used for CPUs and GPUs. Special memory-scope enables memory reuse in GPUs and explicit management of on-chip memory in accelerators. Latency hiding is specific to TPU-like accelerators.

### 4.2 Nested Parallelism with Cooperation 有合作的嵌套并行

Parallelism is key to improving the efficiency of compute-intensive kernels in DL workloads. Modern GPUs offer massive parallelism, requiring us to bake parallel patterns into schedule transformations. Most existing solutions adopt a model called nested parallelism, a form of fork–join. This model requires a parallel schedule primitive to parallelize a data parallel task; each task can be further recursively subdivided into subtasks to exploit the target architecture’s multi-level thread hierarchy (e.g., thread groups in GPU). We call this model shared-nothing nested parallelism because one working thread cannot look at the data of its sibling within the same parallel computation stage.

并行化是改进深度学习工作中计算密集核心的效率的关键。现代GPU提供了巨量的并行化，需要我们将并行模式转换称调度变换。多数现有的解决方案采用的方法是嵌套并行化，这是一种fork-join的形式。这个模型需要一个并行调度原语，使数据并行任务并行化；每个任务能够进一步递归分解成子任务，以利用目标架构的多层次线程层级关系（如，GPU中的线程组）。我们称这个模型为“未共享嵌套并行化”，因为一个工作的线程与其同一并行计算阶段的兄弟线程并不公用任何数据。

An alternative to the shared-nothing approach is to fetch data cooperatively. Specifically, groups of threads can cooperatively fetch the data they all need and place it into a shared memory space. (Halide recently added shared memory support but without general memory scope for accelerators.) This optimization can take advantage of the GPU memory hierarchy and enable data reuse across threads through shared memory regions. TVM supports this well-known GPU optimization using a schedule primitive to achieve optimal performance. The following GPU code example optimizes matrix multiplication.

未共享方法的一个替代是合作取数据。特别的，线程组合作取来都需要的数据，将其放入共享的内存空间中。（Halide最近增加了共享内存的支持，但没有用于加速器的通用内存范围。）这种优化可以利用GPU内存的层次结构，使跨线程间共享内存区域的数据复用成为可能。TVM支持这种GPU优化，使用了一种调度原语来得到最佳性能。下面的GPU代码例子优化了矩阵相乘。

```
for thread_group (by, bx) in cross(64, 64):
    for thread_item (ty, tx) in cross(2, 2):
        local CL[8][8] = 0
        shared AS[2][8], BS[2][8]
        for k in range(1024):
            for i in range(4):
                AS[ty][i*4+tx] = A[k][by*64+ty*8+i*4+tx]
            for each i in 0..4:
                BS[ty][i*4+tx] = B[k][bx*64+ty*8+i*4+tx]
            memory_barrier_among_threads ()
            for yi in range(8):
                for xi in range(8):
                    CL[yi][xi] += AS[yi] * BS[xi]
            for yi in range(8):
                for xi in range(8):
                    C[yo*8+yi][xo*8+xi] = CL[yi][xi]
```

Figure 7 demonstrates the impact of this optimization. We introduce the concept of memory scopes to the schedule space so that a compute stage (AS and BS in the code) can be marked as shared. Without explicit memory scopes, automatic scope inference will mark compute stages as thread-local. The shared task must compute the dependencies of all working threads in the group. Additionally, memory synchronization barriers must be properly inserted to guarantee that shared loaded data is visible to consumers. Finally, in addition to being useful to GPUs, memory scopes let us tag special memory buffers and create special lowering rules when targeting specialized DL accelerators.

图7给出了这种优化的影响。我们为调度空间提出了内存范围的概念，这样一个计算阶段（代码中的AS和BS）可以标记为共享。没有显式的内存范围的话，自动范围推理会将这个计算阶段标记为thread-local。共享的任务必须计算所有分组中工作线程的依赖关系。另外，必须加入合理的内存同步障碍，以保证共享的载入数据对consumers可见。最后，除了对GPU有用，内存范围使我们可以标记特殊的内存缓存，在面向特殊的深度学习加速器时，生成特殊的lowering rules（代码移植到底层设备时的规则）。

Figure 7: Performance comparison between TVM with and without cooperative shared memory fetching on matrix multiplication workloads. time: cuBLAS<TVM<TVM w/o coop.

### Tensorization 张量化

DL workloads have high arithmetic intensity, which can typically be decomposed into tensor operators like matrix-matrix multiplication or 1D convolution. These natural decompositions have led to the recent trend of adding tensor compute primitives [1, 12, 21]. These new primitives create both opportunities and challenges for schedule-based compilation; while using them can improve performance, the compilation framework must seamlessly integrate them. We dub this tensorization: it is analogous to vectorization for SIMD architectures but has significant differences. Instruction inputs are multi-dimensional, with fixed or variable lengths, and each has different data layouts. More importantly, we cannot support a fixed set of primitives since new accelerators are emerging with their own variations of tensor instructions. We therefore need an extensible solution.

深度学习任务的算术强度非常高，一般可以分解为张量运算，如矩阵矩阵相乘或1D卷积。这些自然的分解，带来了最近增加张量计算原语[1,12,21]的趋势。这些新的原语对于基于调度的编译形成了机会和挑战；使用它们，可以改进性能，编译框架必须无缝的整合它们。我们称之为张量化：类比于SIMD架构的向量化，但有明显的区别。指令输入是多维的，有固定的或者变化的长度，每个都有不同的数据分布。更重要的是，我们不能支持固定原语集合了，因为新的加速器不断出现，其张量指令各有变化。所以我们需要一个可扩展的解决方案。

We make tensorization extensible by separating the target hardware intrinsic from the schedule with a mechanism for tensor-intrinsic declaration. We use the same tensor expression language to declare both the behavior of each new hardware intrinsic and the lowering rule associated with it. The following code shows how to declare an 8 × 8 tensor hardware intrinsic.

我们将目标硬件的特性与调度分开，使用的是一种张量式的声明机制，这样使张量化可扩展。我们使用同样的张量表达式语言，来声明每种新硬件特性的行为，和与之相关的lowering rule。下面的代码是怎样声明一个8×8的张量硬件本征。

```
w, x = t.placeholder((8, 8)), t.placeholder((8, 8))
k = t.reduce_axis((0, 8))
y = t.compute((8, 8), lambda i, j: t.sum(w[i, k] * x[j, k], axis=k))

def gemm_intrin_lower(inputs, outputs):
    ww_ptr = inputs[0].access_ptr(“r")
    xx_ptr = inputs[1].access_ptr("r")
    zz_ptr = outputs[0].access_ptr("w")
    compute = t.hardware_intrin("gemm8x8", ww_ptr, xx_ptr, zz_ptr)
    reset = t.hardware_intrin("fill_zero", zz_ptr)
    update = t.hardware_intrin("fuse_gemm8x8_add", ww_ptr, xx_ptr, zz_ptr)
    return compute, reset, update

gemm8x8 = t.decl_tensor_intrin(y.op, gemm_intrin_lower)
```

Additionally, we introduce a tensorize schedule primitive to replace a unit of computation with the corresponding intrinsics. The compiler matches the computation pattern with a hardware declaration and lowers it to the corresponding hardware intrinsic.

另外，我们提出了一种张量化的调度原语，以替代一个带有对应本征的计算单元。编译器将计算模式与硬件声明相匹配，并lower到对应的硬件本征上。

Tensorization decouples the schedule from specific hardware primitives, making it easy to extend TVM to support new hardware architectures. The generated code of tensorized schedules aligns with practices in high-performance computing: break complex operations into a sequence of micro-kernel calls. We can also use the tensorize primitive to take advantage of handcrafted micro-kernels, which can be beneficial in some platforms. For example, we implement ultra low precision operators for mobile CPUs that operate on data types that are one- or two-bits wide by leveraging a bit-serial matrix vector multiplication micro-kernel. This micro-kernel accumulates results into progressively larger data types to minimize the memory footprint. Presenting the micro-kernel as a tensor intrinsic to TVM yields up to a 1.5× speedup over the non-tensorized version.

张量化将调度与特定硬件原语分开，使TVM很容易拓展支持新的硬件架构。生成的张量化调度代码与高性能计算上的实践相一致：将复杂的算子分解成微核心调用序列。我们也可以使用这些张量原语，来利用手工设计的微核心的优势，在一些平台上这是有好处的。比如，我们在移动CPU上实现极低精度算子，在单比特和双比特宽的数据类型上进行计算，利用比特序列的矩阵向量乘积微核心。这种微核心将结果累积成逐渐变大的数据类型，以最小化内存容量。提出微核心作为TVM的张量本征，比非张量化版本有1.5×的加速。

# 4.4 Explicit Memory Latency Hiding 显式的内存延迟隐藏

Latency hiding refers to the process of overlapping memory operations with computation to maximize utilization of memory and compute resources. It requires different strategies depending on the target hardware back-end. On CPUs, memory latency hiding is achieved implicitly with simultaneous multithreading [14] or hardware prefetching [10, 20]. GPUs rely on rapid context switching of many warps of threads [44]. In contrast, specialized DL accelerators such as the TPU [21] usually favor leaner control with a decoupled access-execute (DAE) architecture [35] and offload the problem of fine-grained synchronization to software.

延迟隐藏指的是将内存操作与计算叠加到一起，以最大化利用内存和计算资源。依赖不同的目标硬件后端上，需要不同的策略。在CPU上，内存延迟隐藏是用同时多线程[14]或硬件预取[10,20]技术隐式实现的。GPU则依靠的是迅速切换很多warps线程的上下文[44]。比较起来，专用的深度学习加速器，如TPU[21]通常倾向于较少的控制，采用的是分离的访问-执行(decoupled access-excute, DAE)架构[35]，将细粒度的同步问题交给软件实现。

Figure 9 shows a DAE hardware pipeline that reduces runtime latency. Compared to a monolithic hardware design, the pipeline can hide most memory access overheads and almost fully utilize compute resources. To achieve higher utilization, the instruction stream must be augmented with fine-grained synchronization operations. Without them, dependencies cannot be enforced, leading to erroneous execution. Consequently, DAE hardware pipelines require fine-grained dependence enqueuing/dequeuing operations between the pipeline stages to guarantee correct execution, as shown in Figure 9’s instruction stream.

图9所示的是一个DAE硬件流程，可以降低运行时的延迟。比单片式硬件设计相比，这个流程可以隐藏多数内存访问的开销，几乎可以全部利用计算资源。为获得更高的利用率，指令流必须用细粒度的同步操作扩充起来。没有这些，就不能加入依赖关系，带来错误的执行结果。结果是，DAE硬件流程需要流程阶段间细粒度的依赖关系enqueuing/dequeuing操作，以保证正确的执行，如图9的指令流所示。

Figure 9: Decoupled Access-Execute in hardware hides most memory access latency by allowing memory and computation to overlap. Execution correctness is enforced by low-level synchronization in the form of dependence token enqueueing/dequeuing actions, which the compiler stack must insert in the instruction stream.

Programming DAE accelerators that require explicit low-level synchronization is difficult. To reduce the programming burden, we introduce a virtual threading scheduling primitive that lets programmers specify a high-level data parallel program as they would a hardware back-end with support for multithreading. TVM then automatically lowers the program to a single instruction stream with low-level explicit synchronization, as shown in Figure 8. The algorithm starts with a high-level multi-threaded program schedule and then inserts the necessary low-level synchronization operations to guarantee correct execution within each thread. Next, it interleaves operations of all virtual threads into a single instruction stream. Finally, the hardware recovers the available pipeline parallelism dictated by the low-level synchronizations in the instruction stream.

对需要显式的底层同步的DAE加速器进行编程是很困难的。为降低编程负担，我们提出了一种虚拟线程调度原语，使程序员可以指定一种高层的数据并行程序，可以使用支持多线程的硬件后端。TVM然后会自动将程序lower到单指令流，有底层的显式同步，如图8所示。算法开始于高层多线程程序调度，然后加入必须的底层同步操作，以保证每个线程内的正确执行。下一步，将各种虚拟线程的操作交叠，称为一个单指令流。最后，硬件恢复可用的流程并行化，这由指令流中的底层同步决定。

Figure 8: TVM virtual thread lowering transforms a virtual thread-parallel program to a single instruction stream; the stream contains explicit low-level synchronizations that the hardware can interpret to recover the pipeline parallelism required to hide memory access latency.

**Hardware Evaluation of Latency Hiding**. We now demonstrate the effectiveness of latency hiding on a custom FPGA-based accelerator design, which we describe in depth in subsection 6.4. We ran each layer of ResNet on the accelerator and used TVM to generate two schedules: one with latency hiding, and one without. The schedule with latency hiding parallelized the program with virtuals threads to expose pipeline parallelism and therefore hide memory access latency. Results are shown in Figure 10 as a roofline diagram [47]; roofline performance diagrams provide insight into how well a given system uses computation and memory resources for different benchmarks. Overall, latency hiding improved performance on all ResNet layers. Peak compute utilization increased from 70% with no latency hiding to 88% with latency hiding.

**延迟隐藏的硬件评估**。我们现在在一个定制的基于FPGA的加速器设计上证明延迟隐藏技术的有效性，细节在6.4小节中详述。我们在加速上运行ResNet的每一层，使用TVM来生成两种调度方案：一个有延迟隐藏，一个没有。有延迟隐藏的调度方案用虚拟线程使程序并行化，因此隐藏了内存访问的延迟。结果如图10所示，是一个roofline的形状[47]；这可以说明一个系统在不同基准测试中对计算资源和内存资源的使用情况。总体上，延迟隐藏在所有ResNet层中都改进了性能。顶峰计算利用率从没有延迟隐藏的70%提高到了有延迟隐藏的88%.

Figure 10: Roofline [47] of an FPGA-based DL accelerator running ResNet inference. With latency hiding enabled by TVM, performance of the benchmarks is brought closer to the roofline, demonstrating higher compute and memory bandwidth efficiency.

## 5 Automating Optimization 自动优化

Given the rich set of schedule primitives, our remaining problem is to find optimal operator implementations for each layer of a DL model. Here, TVM creates a specialized operator for the specific input shape and layout associated with each layer. Such specialization offers significant performance benefits (in contrast to handcrafted code that would target a smaller diversity of shapes and layouts), but it also raises automation challenges. The system needs to choose the schedule optimizations – such as modifying the loop order or optimizing for the memory hierarchy – as well as schedule-specific parameters, such as the tiling size and the loop unrolling factor. Such combinatorial choices create a large search space of operator implementations for each hardware back-end. To address this challenge, we built an automated schedule optimizer with two main components: a schedule explorer that proposes promising new configurations, and a machine learning cost model that predicts the performance of a given configuration. This section describes these components and TVM’s automated optimization flow (Figure 11).

有了调度原语的丰富集合，我们剩下的问题就是为一个深度学习模型的每一层寻找最优的算子实现。这里，TVM对特定的输入形状和分布创建了一种专门的算子。这种专门化带来了显著的性能提升（与手工设计的代码比较，这种会面向较少的形状和分布变化），但这也带来了自动化的挑战。这个系统需要选择调度优化方案，如修正循环顺序，或为内存层次结构优化，以及选择与调度相关的参数，比如堆叠大小，和循环unrolling因子。这种选择组合会产生很大的算子实现搜索空间，对于每种硬件后端都是。为解决这种挑战，我们构造了一种自动调度优化器，包括两个部件：一个调度探索器，负责提出有希望的新配置，一个机器学习代价模型，负责预测给定配置的性能。本节叙述这些部件，和TVM的自动优化流（图11）。

Figure 11: Overview of automated optimization framework. A schedule explorer examines the schedule space using an ML-based cost model and chooses experiments to run on a distributed device cluster via RPC. To improve its predictive power, the ML model is updated periodically using collected data recorded in a database.

图11.自动优化框架概览。调度探索器使用基于机器学习的代价模型来检查调度空间，选择试验在分布式设备集群上通过RPC运行。为改进预测能力，机器学习模型使用收集并保存在数据库中的数据进行周期性的更新。

### 5.1 Schedule Space Specification 调度空间规范

We built a schedule template specification API to let a developer declare knobs in the schedule space. The template specification allows incorporation of a developer’s domain-specific knowledge, as necessary, when specifying possible schedules. We also created a generic master template for each hardware back-end that automatically extracts possible knobs based on the computation description expressed using the tensor expression language. At a high level, we would like to consider as many configurations as possible and let the optimizer manage the selection burden. Consequently, the optimizer must search over billions of possible configurations for the real world DL workloads used in our experiments.

我们构建一个调度模板规范API，使开发者在一个调度空间中声明节点。模板规范允许加入开发者的领域相关的知识，需要的时候可以指定可能的模块。我们还为每种硬件后端创建了通用大师模板，可以基于用张量描述语言表达的计算描述自动提取可能的节点。在高层次上，我们可能会考虑尽可能多的配置，使优化器来管理选择的任务。结果是，优化器必须为我们试验中真实世界的深度学习工作搜索上十亿种可能的配置。

### 5.2 ML-Based Cost Model 基于机器学习的代价模型

One way to find the best schedule from a large configuration space is through blackbox optimization, i.e., auto-tuning. This method is used to tune high performance computing libraries [15, 46]. However, auto-tuning requires many experiments to identify a good configuration.

从大型配置空间中寻找最佳调度的一种方法是通过黑盒优化，即自动调优。这种方法是用于调节高性能计算库的[15,46]。但是，自动调优需要很多试验来辨别一个好的配置。

An alternate approach is to build a predefined cost model to guide the search for a particular hardware back-end instead of running all possibilities and measuring their performance. Ideally, a perfect cost model considers all factors affecting performance: memory access patterns, data reuse, pipeline dependencies, and threading patterns, among others. This approach, unfortunately, is burdensome due to the increasing complexity of modern hardware. Furthermore, every new hardware target requires a new (predefined) cost model.

另一种方法是构建一种预定义的代价模型，来在特定的硬件后端上指引搜索，而不是运行所有可能性，并度量其性能。理想情况下，一个完美的代价模型会考虑所有影响性能的因素：内存访问模式，数据复用，流水线依赖关系，线程模式，以及其他。很不幸，这种方法由于现代硬件越来越复杂，负担非常大。而且，每一种新的硬件目标需要一种新的预定义模型。

We instead take a statistical approach to solve the cost modeling problem. In this approach, a schedule explorer proposes configurations that may improve an operator’s performance. For each schedule configuration, we use an ML model that takes the lowered loop program as input and predicts its running time on a given hardware back-end. The model, trained using runtime measurement data collected during exploration, does not require the user to input detailed hardware information. We update the model periodically as we explore more configurations during optimization, which improves accuracy for other related workloads, as well. In this way, the quality of the ML model improves with more experimental trials. Table 1 summarizes the key differences between automation methods. ML-based cost models strike a balance between auto-tuning and predefined cost modeling and can benefit from the historical performance data of related workloads.

所以我们采用了一种统计方法来解决代价模型的问题。在这种方法中，调度探索器提出可能改进一个算子性能的配置。对每个调度配置，我们使用一个机器学习模型，以lowered循环程序为输入，预测其在给定硬件后端上的运行时间。这个模型是用探索过程中收集的运行时间度量数据进行训练的，不需要用户输入硬件信息的细节。我们在优化时探索更多配置的时，周期性的更新模型这会改进其他相关的工作的准确率。通过这种方法，机器学习模型的质量随着更多的试验尝试得到改进。表1总结了自动化方法之间的关键差别。基于机器学习的代价模型在自动调优和预定义的代价模型中取得了平衡，可以从相关工作的历史表现数据中受益。

Table 1: Comparison of automation methods. Model bias refers to inaccuracy due to modeling.

Method Category | Data Cost | Model Bias | Need Hardware Info | Learn from History
--- | --- | --- | --- | ---
Blackbox auto-tuning | high | none | no | no
Predefined cost model | none | high | yes | no
ML based cost model | low | low | no | yes

**Machine Learning Model Design Choices**. We must consider two key factors when choosing which ML model the schedule explorer will use: quality and speed. The schedule explorer queries the cost model frequently, which incurs overheads due to model prediction time and model refitting time. To be useful, these overheads must be smaller than the time it takes to measure performance on real hardware, which can be on the order of seconds depending on the specific workload/hardware target. This speed requirement differentiates our problem from traditional hyperparameter tuning problems, where the cost of performing measurements is very high relative to model overheads, and more expensive models can be used. In addition to the choice of model, we need to choose an objective function to train the model, such as the error in a configuration’s predicted running time. However, since the explorer selects the top candidates based only on the relative order of the prediction (A runs faster than B), we need not predict the absolute execution times directly. Instead, we use a rank objective to predict the relative order of runtime costs.

**机器学习模型设计选择**。我们在选择调度探索器会使用哪个深度学习模型时，必须考虑两个关键因素：质量和速度。调度探索器频繁的查询代价模型，由于模型预测时间和模型重新适配的时间，会带来耗费。这些耗费必须比在实际硬件上测量性能的时间要少，随着特定的目标工作/硬件不同，可能是几秒级的时间，这样才可以有用。这个速度将我们的问题与传统的超参数调优问题区分开来，超参数调优的性能测量代价相对于模型耗时非常高，所以我们可以使用更复杂的模型。除了模型的选择，我们还需要选择目标函数来训练模型，比如配置预测的运行时间的误差。但是，由于探索器选择只基于预测的相对顺序来选择最高候选（A比B运行速度要快），我们不需要直接预测绝对执行时间。相反，我们使用一个级别目标函数来预测运行代价的相对顺序。

We implement several types of models in our ML optimizer. We employ a gradient tree boosting model (based on XGBoost [8]), which makes predictions based on features extracted from the loop program; these features include the memory access count and reuse ratio of each memory buffer at each loop level, as well as a one-hot encoding of loop annotations such as “vectorize”, “unroll”, and “parallel.” We also evaluate a neural network model that uses TreeRNN [38] to summarize the loop program’s AST without feature engineering. Figure 13 summarizes the workflow of the cost models. We found that tree boosting and TreeRNN have similar predictive quality. However, the former performs prediction twice as fast and costs much less time to train. As a result, we chose gradient tree boosting as the default cost model in our experiments. Nevertheless, we believe that both approaches are valuable and expect more future research on this problem.

我们在我们的机器学习优化器中实现几种类型的模型。我们采用一个gradient tree boosting模型（基于XGBoost[8]），基于从循环程序中提取的特征进行预测；这些特征包括每个循环级别的内存访问计数和每块内存缓存的复用率，以及循环标注的独热码（如vectorize, unroll和parallel）。我们还评估了一个神经网络模型，使用的是TreeRNN[38]，以不使用特征工程来总结循环程序AST。图13总结了代价模型的工作流。我们发现tree boosting和TreeRNN有类似的预测质量。但是，前者预测的速度快2倍，而且训练时间很少。结果是，我们选择gradient tree boosting作为我们试验中的默认代价模型。尽管如此，我们相信两种方法都非常有价值，希望在这个问题上有更多的研究。

Figure 13: Example workflow of ML cost models. XGBoost predicts costs based on loop program features. TreeRNN directly summarizes the AST.

On average, the tree boosting model does prediction in 0.67 ms, thousands of times faster than running a real measurement. Figure 12 compares an ML-based optimizer to blackbox auto-tuning methods; the former finds better configurations much faster than the latter.

平均来说，tree boosting模型预测时间为0.67ms，这比进行一次真实的测量快数千倍。图12比较了基于机器学习的优化器和黑盒自动调优方法；前者比后者可以用快得多的速度找到更好的配置。

Figure 12: Comparison of different automation methods for a conv2d operator in ResNet-18 on TITAN X. The ML-based model starts with no training data and uses the collected data to improve itself. The Y-axis is the speedup relative to cuDNN. We observe a similar trend for other workloads.

### 5.3 Schedule Exploration 调度探索

Once we choose a cost model, we can use it to select promising configurations on which to iteratively run real measurements. In each iteration, the explorer uses the ML model’s predictions to select a batch of candidates on which to run the measurements. The collected data is then used as training data to update the model. If no initial training data exists, the explorer picks random candidates to measure.

一旦我们选择一个代价模型，我们就可以将其用于选择有希望的配置，在实际测量中迭代运行。在每次迭代中，探索器使用机器学习模型的预测来选择一批候选，在这些候选上运行测量。收集的数据然后用作训练数据来更新模型。如果没有初始训练数据，探索器选择随机的候选来测量。

The simplest exploration algorithm enumerates and runs every configuration through the cost model, selecting the top-k predicted performers. However, this strategy becomes intractable with large search spaces. Instead, we run a parallel simulated annealing algorithm [22]. The explorer starts with random configurations, and, at each step, randomly walks to a nearby configuration. This transition is successful if cost decreases as predicted by the cost model. It is likely to fail (reject) if the target configuration has a higher cost. The random walk tends to converge on configurations that have lower costs as predicted by the cost model. Exploration states persist across cost model updates; we continue from the last configuration after these updates.

最简单的探索算法是枚举，并通过代价模型运行每个配置，选择最高的k个预测的执行者。但是，这种策略很难应对大型搜索空间。相反，我们运行一个并行的模拟退火算法[22]。探索器开始于随机配置，在每一步中，随机行走到一个附近的配置。如果代价模型预测的代价降低，这种迁移就是成功的。如果目标配置代价更高，就很可能失败（拒绝）。随机行走在代价模型预测的更低代价的配置上更可能会收敛。探索状态在代价模型更新的时候会持续下去；在这些更新之后，我们从上一个配置继续。

###　5.4　Distributed Device Pool and RPC　分布式设备和RPC

A distributed device pool scales up the running of on-hardware trials and enables fine-grained resource sharing among multiple optimization jobs. TVM implements a customized, RPC-based distributed device pool that enables clients to run programs on a specific type of device. We can use this interface to compile a program on the host compiler, request a remote device, run the function remotely, and access results in the same script on the host. TVM’s RPC supports dynamic upload and runs cross-compiled modules and functions that use its runtime convention. As a result, the same infrastructure can perform a single workload optimization and end-to-end graph inference. Our approach automates the compile, run, and profile steps across multiple devices. This infrastructure is especially critical for embedded devices, which traditionally require tedious manual effort for cross-compilation, code deployment, and measurement.

分布式设备池可以扩展硬件上尝试的运行，多个优化任务之间的细粒度资源分享成为可能。TVM实现了一个定制的基于RPC的分布式设备池，使clients可以在指定设备类型上运行程序。我们可以使用这些接口在host编译器上编译一个程序，然后请求一个远程设备，以远程方式运行这个函数，并在host上相同的脚本中访问结果。TVM的RPC支持动态上传并运行交叉编译的模块和函数，使用其运行时惯例。结果是，相同的基础设施可以进行单工作优化和端到端的图推理。我们的方法将编译、运行和优化步骤在多个设备上都自动化了。这个基础设施对于嵌入式设备来说非常关键，传统上需要非常繁杂的手工工作来进行交叉编译、代码部署和测量。

## 6 Evaluation 评估

TVM’s core is implemented in C++ (∼50k LoC). We provide language bindings to Python and Java. Earlier sections of this paper evaluated the impact of several individual optimizations and components of TVM, namely, operator fusion in Figure 4, latency hiding in Figure 10, and the ML-based cost model in Figure 12. We now focus on an end-to-end evaluation that aims to answer the following questions:

TVM的核心是由C++实现的(~50k LoC, lines of code)。我们对Python和Java提供了语言绑定。本文的前面小节评估了几种单个优化的影响，和TVM的部件，即图4中的算子融合，图10中的延迟隐藏，和图12中的基于机器学习的代价模型。我们现在聚焦在端到端的评估上，其目标是回答以下几个问题：

- Can TVM optimize DL workloads over multiple platforms? TVM可以在多个平台上优化深度学习工作吗？
- How does TVM compare to existing DL frameworks (which rely on heavily optimized libraries) on each back-end? TVM与其他已有的深度学习框架（非常依赖于优化的库）在每个后端上相比如何？
- Can TVM support new, emerging DL workloads (e.g., depthwise convolution, low precision operations)? TVM可以支持新的正出现的深度学习工作（如depthwise convolution, 低精度操作）吗？
- Can TVM support and optimize for new specialized accelerators? TVM可以支持并优化新的专门的加速器吗？

To answer these questions, we evaluated TVM on four types of platforms: (1) a server-class GPU, (2) an embedded GPU, (3) an embedded CPU, and (4) a DL accelerator implemented on a low-power FPGA SoC. The benchmarks are based on real world DL inference workloads, including ResNet [16], MobileNet [19], the LSTM Language Model [48], the Deep Q Network (DQN) [28] and Deep Convolutional Generative Adversarial Networks (DCGAN) [31]. We compare our approach to existing DL frameworks, including MxNet [9] and TensorFlow [2], that rely on highly engineered, vendor-specific libraries. TVM performs end-to-end automatic optimization and code generation without the need for an external operator library.

为回答这些问题，我们在四种平台上评估TVM：(1)服务器级的GPU，(2)嵌入式GPU，(3)嵌入式CPU，(4)在一个低功耗FPGA SoC上实现的深度学习加速器。基准测试是基于真实世界深度学习推理工作的，包括ResNet[16], MobileNet[19], LSTM语言模型[48]，Deep Q Network(DQN)[28]和深度卷积生成式对抗网络(DCGAN)[11]。我们将我们的方法与现有的深度学习框架进行比较，包括MXNet[9]和TensorFlow[2]，它们依赖高度工程化的，与特定供应公司相关的库。TVM进行端到端的自动优化和代码生成，不需要外部算子库。

### 6.1 Server-Class GPU Evaluation 服务器级的GPU评估

We first compared the end-to-end performance of deep neural networks TVM, MXNet (v1.1), Tensorflow (v1.7), and Tensorflow XLA on an Nvidia Titan X. MXNet and Tensorflow both use cuDNN v7 for convolution operators; they implement their own versions of depthwise convolution since it is relatively new and not yet supported by the latest libraries. They also use cuBLAS v8 for matrix multiplications. On the other hand, Tensorflow XLA uses JIT compilation.

我们首先比较深度神经网络在TVM，MXNet(v1.1)，TensorFlow(v1.7)和TensorFlow XLA在NVidia TitanX上端到端的性能。MXNet和TensorFlow都使用cuDNN v7进行卷积操作；它们都有depthwise convolution的自己实现的版本，因为这是相对较新的算子，并没有最新的库支持。它们还使用cuBLAS v8进行矩阵乘积。另一方面，TensorFlow XLA使用JIT编译。

Figure 14 shows that TVM outperforms the baselines, with speedups ranging from 1.6× to 3.8× due to both joint graph optimization and the automatic optimizer, which generates high-performance fused operators. DQN’s 3.8 x speedup results from its use of unconventional operators (4×4 conv2d, strides=2) that are not well optimized by cuDNN; the ResNet workloads are more conventional. TVM automatically finds optimized operators in both cases.

从图14中可以看出，TVM超过了这些基准线，加速范围从1.6×到3.8×，这是联合图优化、自动优化器的结果，生成了高性能的融合算子。DQN的3.8倍加速结果，是使用非传统的算子(4×4 conv2d, stride=2)的结果，这在cuDNN上的优化不是很好；ResNet的工作更传统一些。TVM在两种情况下都可以自动找到优化的算子。

Figure 14: GPU end-to-end evaluation for TVM, MXNet, Tensorflow, and Tensorflow XLA. Tested on the NVIDIA Titan X.

To evaluate the effectiveness of operator level optimization, we also perform a breakdown comparison for each tensor operator in ResNet and MobileNet, shown in Figure 15. We include TensorComprehension (TC, commit: ef644ba) [42], a recently introduced auto-tuning framework, as an additional baseline. TC results include the best kernels it found in 10 generations × 100 population × 2 random seeds for each operator (i.e., 2000 trials per operator). 2D convolution, one of the most important DL operators, is heavily optimized by cuDNN. However, TVM can still generate better GPU kernels for most layers. Depthwise convolution is a newly introduced operator with a simpler structure [19]. In this case, both TVM and TC can find fast kernels compared to MXNet’s handcrafted kernels. TVM’s improvements are mainly due to its exploration of a large schedule space and an effective ML-based search algorithm.

为评估算子级优化的有效性，我们对ResNet和MobileNet中的每个张量算子进行了分解对比，如图15所示。我们将TensorComprehension(TC, commit:ef644ba)[42]作为额外的基准，这是一个最近提出的自动调优框架。TC的结果包括其在10 generations × 100 population × 2 random seeds中为每个算子发现的最好的核心（即，每个算子2000次尝试）。2D卷积是一个最重要的深度卷积算子，cnDNN进行了重度优化。但是，TVM仍然可以对多数层生成更好的GPU核。Depthwise卷积是新近提出的算子，结构更简单[19]。在这种情况下，TVM和TC与MXNet的手工设计核心比，可以找到更快的核心。TVM的实现主要是由于其探索了一个大型的调度空间，和一个有效的基于机器学习的搜索算法。

Figure 15: Relative speedup of all conv2d operators in ResNet-18 and all depthwise conv2d operators in MobileNet. Tested on a TITAN X. See Table 2 for operator configurations. We also include a weight pre-transformed Winograd [25] for 3x3 conv2d (TVM PT).

Table 2: Configurations of all conv2d operators in ResNet-18 and all depthwise conv2d operators in MobileNet used in the single kernel experiments. H/W denotes height and width, IC input channels, OC output channels, K kernel size, and S stride size. All ops use “SAME” padding. All depthwise conv2d operations have channel multipliers of 1.

### 6.2 Embedded CPU Evaluation 嵌入式CPU评估

We evaluated the performance of TVM on an ARM Cortex A53 (Quad Core 1.2GHz). We used Tensorflow Lite (TFLite, commit: 7558b085) as our baseline system. Figure 17 compares TVM operators to hand-optimized ones for ResNet and MobileNet. We observe that TVM generates operators that outperform the hand-optimized TFLite versions for both neural network workloads. This result also demonstrates TVM’s ability to quickly optimize emerging tensor operators, such as depthwise convolution operators. Finally, Figure 16 shows an end-to-end comparison of three workloads, where TVM outperforms the TFLite baseline.

我们在ARM Cortex A53 (Quad Core 1.2GHz)上评估TVM的性能。我们使用TensorFlow Lite(TFLite, commit: 7558b085)作为基准系统。图17比较了TVM算子与手工优化的ResNet和MobileNet算子。我们观察到，TVM生成的算子超过了手工优化的TFLite版的算子，在两个神经网络工作中都是。这个结果也说明，TVM可以快速优化新出现的张量算子，如depthwise convolution算子。最后，图16给出了三项工作的端到端比较，其中TVM超过了TFLite基准。

Figure 16: ARM A53 end-to-end evaluation of TVM and TFLite.

Figure 17: Relative speedup of all conv2d operators in ResNet-18 and all depthwise conv2d operators in mobilenet. Tested on ARM A53. See Table 2 for the configurations of these operators.

**Ultra Low-Precision Operators**. We demonstrate TVM’s ability to support ultra low-precision inference [13, 33] by generating highly optimized operators for fixed-point data types of less than 8-bits. Low-precision networks replace expensive multiplication with vectorized bit-serial multiplication that is composed of bitwise and popcount reductions [39]. Achieving efficient low-precision inference requires packing quantized data types into wider standard data types, such as int8 or int32. Our system generates code that outperforms hand-optimized libraries from Caffe2 (commit: 39e07f7) [39]. We implemented an ARM-specific tensorization intrinsic that leverages ARM instructions to build an efficient, low-precision matrix-vector microkernel.We then used TVM’s automated optimizer to explore the scheduling space.

**超低精度算子**。我们对少于8-bits的定点数据类型生成高度优化的算子，证明了TVM可以支持超低精度的推理[13,33]。低精度网络将耗时的乘积运算替换为向量化的bit-serial乘积，这种运算由bitwise and popcount reduction组成[39]。进行有效的低精度推理，需要将量化的数据类型打包成更宽的标准数据类型，如int8或int32。我们的系统生成的代码，其性能超过了caffe2(commit:39e07f7) [39]手工优化的库。我们实现了一个ARM专用的张量化本征，利用的ARM的指令来构建一个高效的低精度矩阵向量微核心。然后我们使用TVM的自动化优化器来探索调度空间。

Figure 18 compares TVM to the Caffe2 ultra low-precision library on ResNet for 2-bit activations, 1-bit weights inference. Since the baseline is single threaded, we also compare it to a single-threaded TVM version. Single-threaded TVM outperforms the baseline, particularly for C5, C8, and C11 layers; these are convolution layers of kernel size 1×1 and stride of 2 for which the ultra low-precision baseline library is not optimized. Furthermore, we take advantage of additional TVM capabilities to produce a parallel library implementation that shows improvement over the baseline. In addition to the 2-bit+1-bit configuration, TVM can generate and optimize for other precision configurations that are unsupported by the baseline library, offering improved flexibility.

图18比较了TVM和Caffe2的超低精度库，使用ResNet 2-bit激活，1-bit权重推理。由于基准是单线程的，我们也与单线程版的TVM相比。单线程的TVM超过了基准，尤其是C5, C8和C11层；这些卷积层的核心大小为1×1，步长2，其超低精度基准库没有优化。而且，我们利用的TVM的其他能力来产生了一个并行库实现，比基准有所改进。除了2-bit+1-bit的配置，TVM可以生成和优化其他精度的配置，基准库则不支持这些，这就体现了改进的灵活性。

Figure 18: Relative speedup of single- and multi-threaded low-precision conv2d operators in ResNet. Baseline was a single-threaded, hand-optimized implementation from Caffe2 (commit: 39e07f7). C5, C3 are 1x1 convolutions that have less compute intensity, resulting in less speedup by multi-threading.

### 6.3 Embedded GPU Evaluation 移动GPU的评估

For our mobile GPU experiments, we ran our end-to-end pipeline on a Firefly-RK3399 board equipped with an ARM Mali-T860MP4 GPU. The baseline was a vendor-provided library, the ARM Compute Library (v18.03). As shown in Figure 19, we outperformed the baseline on three available models for both float16 and float32 (DCGAN and LSTM are not yet supported by the base-line). The speedup ranged from 1.2× to 1.6×.

对于移动GPU的试验，我们在Firefly-RK3399板子上运行端到端的流程，板子上有一个ARM Mali-T860MP4 GPU。基准是一个大公司提供的库，ARM Compute Library(v18.03)。如图19所示，我们在三个可用的模型上超过了基准，在两个精度float16和float32上都是。加速效果从1.2×到1.6×。

Figure 19: End-to-end experiment results on Mali-T860MP4. Two data types, float32 and float16, were evaluated.

### 6.4 FPGA Accelerator Evaluation FPGA加速器评估

**Vanilla Deep Learning Accelerator**. We now relate how TVM tackled accelerator-specific code generation on a generic inference accelerator design we prototyped on an FPGA. We used in this evaluation the Vanilla Deep Learning Accelerator (VDLA) – which distills characteristics from previous accelerator proposals [12, 21, 27] into a minimalist hardware architecture – to demonstrate TVM’s ability to generate highly efficient schedules that can target specialized accelerators. Figure 20 shows the high-level hardware organization of the VDLA architecture. VDLA is programmed as a tensor processor to efficiently execute operations with high compute intensity (e.g, matrix multiplication, high dimensional convolution). It can perform load/store operations to bring blocked 3-dimensional tensors from DRAM into a contiguous region of SRAM. It also provides specialized on-chip memories for network parameters, layer inputs (narrow data type), and layer outputs (wide data type). Finally, VDLA provides explicit synchronization control over successive loads, computes, and stores to maximize the overlap between memory and compute operations.

**传统深度学习加速器**。我们现在处理TVM怎样在一个通用推理加速器上进行加速器相关的代码生成。我们在这个评估中，使用传统深度学习加速器(VDLA)来证明TVM生成高效率调度的能力，目标就是专用的加速器。图20所示的是VDLA架构的高层硬件组织。VDLA是一个张量处理器，可以高效的执行高计算强度的运算（如矩阵乘积，高维卷积）。它可以执行载入/存储操作，将成块的三维张量从DRAM中转移到SRAM的连续区域中。它也可以为网络参数、层的输入（窄数据类型）和层的输出（宽数据类型）提供专用的片上存储。最后，VDLA为连续的载入、计算和存储提供显式的同步控制，以最大化重叠内存操作和计算操作。

Figure 20: VDLA Hardware design overview.

**Methodology**. We implemented the VDLA design on a low-power PYNQ board that incorporates an ARM Cortex A9 dual core CPU clocked at 667MHz and an Artix-7 based FPGA fabric. On these modest FPGA resources, we implemented a 16 × 16 matrix-vector unit clocked at 200MHz that performs products of 8-bit values and accumulates them into a 32-bit register every cycle. The theoretical peak throughput of this VDLA design is about 102.4GOPS/s. We allocated 32kB of resources for activation storage, 32kB for parameter storage, 32kB for microcode buffers, and 128kB for the register file. These on-chip buffers are by no means large enough to provide sufficient on-chip storage for a single layer of ResNet and therefore enable a case study on effective memory reuse and latency hiding.

We built a driver library for VDLA with a C runtime API that constructs instructions and pushes them to the target accelerator for execution. Our code generation algorithm then translates the accelerator program to a series of calls into the runtime API. Adding the specialized accelerator back-end took ∼2k LoC in Python.

**方法**。我们在一个低功耗PYNQ板子上实现VDLA设计，包含了一个ARM Cortex A9双核CPU，主频667MHz，和一个基于Artix-7的FPGA fabric。在这些中等的FPGA资源上，我们实现了一个16×16矩阵-向量单元，时钟为200MHz，进行8-bit的乘积，每个循环中将其累积到一个32-bit寄存器中。这种VDLA设计的理论峰值吞吐量为102.4 GOPS/s。我们为激活存储分配了32kB的资源，参数存储分配了32kB，为代码缓存32kB，寄存器文件128kB。这些片上缓存肯定不足以为ResNet的单个层提供足够片上存储的，因此可以在此研究内存复用和延迟隐藏的有效性。

We built a driver library for VDLA with a C runtime API that constructs instructions and pushes them to the target accelerator for execution. Our code generation algorithm then translates the accelerator program to a series of calls into the runtime API. Adding the specialized accelerator back-end took ∼2k LoC in Python.

我们为VDLA开发了一个驱动库，和一个C运行时API，构建指令并将其压入到目标加速器上，以供执行。我们的代码生成算法将加速器程序翻译成运行时API的一系列调用。增加专用加速器的后端，用Python有～2k LoC代码量。

**End-to-End ResNet Evaluation**. We used TVM to generate ResNet inference kernels on the PYNQ platform and offloaded as many layers as possible to VDLA. We also used it to generate both schedules for the CPU only and CPU+FPGA implementation. Due to its shallow convolution depth, the first ResNet convolution layer could not be efficiently offloaded on the FPGA and was instead computed on the CPU. All other convolution layers in ResNet, however, were amenable to efficient offloading. Operations like residual layers and activations were also performed on the CPU since VDLA does not support these operations.

**端到端的ResNet评估**。我们使用TVM来在PYNQ平台生成ResNet推理核心，将尽可能多的层的任务都offload给VDLA。我们还用其生成只为CPU实现的调度和为CPU+FPGA实现的调度。由于其卷积深度较浅，第一个ResNet卷积层不能被有效的offload到FPGA上，而是用CPU计算的。ResNet中所有其他卷积层是可以有效的offload的。残差层和激活这样的运算也是用CPU运算的，因为VDLA不支持这些运算。

Figure 21 breaks down ResNet inference time into CPU-only execution and CPU+FPGA execution. Most computation was spent on the convolution layers that could be offloaded to VDLA. For those convolution layers, the achieved speedup was 40×. Unfortunately, due to Amdahl’s law, the overall performance of the FPGA accelerated system was bottlenecked by the sections of the workload that had to be executed on the CPU. We envision that extending the VDLA design to support these other operators will help reduce cost even further. This FPGA-based experiment showcases TVM’s ability to adapt to new architectures and the hardware intrinsics they expose.

图21将ResNet推理的时间分解为只用CPU运算的和CPU+FPGA运算的。多数计算都集中在卷积层中，这可以offload给VDLA。对于这些卷积层，得到的加速效果可以达到40×。不幸的是，由于Amdahl定律，FPGA加速系统的总体性能有一个瓶颈，即必须在CPU上执行的工作部分。我们可以预见，拓展VDLA设计，以支持其他算子，将会进一步降低损耗。这种基于FPGA的试验展现了TVM适应新框架和新的硬件本征的能力。

Figure 21: We offloaded convolutions in the ResNet workload to an FPGA-based accelerator. The grayed-out bars correspond to layers that could not be accelerated by the FPGA and therefore had to run on the CPU. The FPGA provided a 40x acceleration on offloaded convolution layers over the Cortex A9.

## 7 Related Work 相关工作

Deep learning frameworks [3, 4, 7, 9] provide convenient interfaces for users to express DL workloads and deploy them easily on different hardware back-ends. While existing frameworks currently depend on vendor-specific tensor operator libraries to execute their workloads, they can leverage TVM’s stack to generate optimized code for a larger number of hardware devices.

深度学习框架[3,4,7,9]提供了方便的接口供用户进行深度学习任务，可以很容易的在不同的硬件后端部署。现有的框架目前依赖特定公司提供的张量算子库来执行其工作，他们还可以利用TVM栈来为更多硬件设备生成优化的代码。

High-level computation graph DSLs are a typical way to represent and perform high-level optimizations. Tensorflow’s XLA [3] and the recently introduced DLVM [45] fall into this category. The representations of computation graphs in these works are similar, and a high-level computation graph DSL is also used in this paper. While graph-level representations are a good fit for high-level optimizations, they are too high level to optimize tensor operators under a diverse set of hardware back-ends. Prior work relies on specific lowering rules to directly generate low-level LLVM or resorts to vendor-crafted libraries. These approaches require significant engineering effort for each hardware back-end and operator-variant combination.

高层计算图DSL是一种表示和进行高层优化的典型方法。TensorFlow的XLA[3]和最近提出的DLVM[45]都是这个类别的。这些工作中计算图的表示是类似的，本文中也使用了高层计算图DSL。图级的表示可以很好的适配高层优化，但是这太高层了，无法在很多硬件后端上优化张量算子。之前的工作依靠特定的lowering规则来直接生成底层LLVM或求助于工厂定制的库。这些方法在每种硬件后端和算子变化组合上都需要大量的工程努力。

Halide [32] introduced the idea of separating computing and scheduling. We adopt Halide’s insights and reuse its existing useful scheduling primitives in our compiler. Our tensor operator scheduling is also related to other work on DSL for GPUs [18, 24, 36, 37] and polyhedral-based loop transformation [6,43]. TACO [23] introduces a generic way to generate sparse tensor operators on CPU. Weld [30] is a DSL for data processing tasks. We specifically focus on solving the new scheduling challenges of DL workloads for GPUs and specialized accelerators. Our new primitives can potentially be adopted by the optimization pipelines in these works.

Halide[32]提出了将计算和调度分离的概念。我们采用了Halide的思想，在我们的编译器中复用其现有的调度原语。我们的张量算子调度与GPU上的DSL[18,24,36,37]和polyhedral-based loop transformation[6,43]也有关系。TACO[23]提出了一种在CPU上生成稀疏张量算子的通用方法。Weld[30]是数据处理任务的DSL。我们特别聚焦于解决新的深度学习任务的调度挑战，在GPU和专用加速器上。我们的新原语可能会被这些工作中的优化流程所采用。

High-performance libraries such as ATLAS [46] and FFTW [15] use auto-tuning to get the best performance. Tensor comprehension [42] applied black-box auto-tuning together with polyhedral optimizations to optimize CUDA kernels. OpenTuner [5] and existing hyper parameter-tuning algorithms [26] apply domain-agnostic search. A predefined cost model is used to automatically schedule image processing pipelines in Halide [29]. TVM’s ML model uses effective domain-aware cost modeling that considers program structure. The based distributed schedule optimizer scales to a larger search space and can find state-of-the-art kernels on a large range of supported back-ends. More importantly, we provide an end-to-end stack that can take descriptions directly from DL frameworks and jointly optimize together with the graph-level stack.

像ATLAS[46]和FFTW[15]这样的高性能库使用自动调优来得到最佳性能。Tensor Comprehension[42]使用黑盒自动调优和polyhedral优化来优化CUDA核。OpenTuner[5]和现有的超参数调优算法[26]使用领域相关的搜索。Halide[29]使用一种预定义的代价模型来自动调度图像处理流程。TVM的机器学习模型使用有效的领域相关的代价模型，考虑了程序结构。这种调度优化器可以搜索更大的空间，在更大范围支持的后端上找到最好的核心。更重要的是，我们提供了一种端到端的栈，可以直接从深度学习框架中获取描述，对图级的栈进行联合优化。

Despite the emerging popularity of accelerators for deep learning [11, 21], it remains unclear how a compilation stack can be built to effectively target these devices. The VDLA design used in our evaluation provides a generic way to summarize the properties of TPU-like accelerators and enables a concrete case study on how to compile code for accelerators. Our approach could potentially benefit existing systems that compile deep learning to FPGA [34,40], as well. This paper provides a generic solution to effectively target accelerators via tensorization and compiler-driven latency hiding.

尽管新出现了很多深度学习加速器[11,21]，但仍然不明确的是，怎样构建一个编译栈，可以高效的在这些设备上运行。我们评估中使用的VDLA设计提供了一种通用途径来总结类TPU加速器的性质，使怎样进行加速器的代码编译的实例研究成为可能。我们的方法可能会使现有的系统受益，编译深度学习工作到FPGA上[34,40]。本文提出了一种通用解决方案，可以有效的通过张量化和编译器驱动的延迟隐藏来使用加速器。

## 8 Conclusion 结论

We proposed an end-to-end compilation stack to solve fundamental optimization challenges for deep learning across a diverse set of hardware back-ends. Our system includes automated end-to-end optimization, which is historically a labor-intensive and highly specialized task. We hope this work will encourage additional studies of end-to-end compilation approaches and open new opportunities for DL system software-hardware co-design techniques.

我们提出一种端到端的编译栈，为深度学习在广泛的硬件后端上解决基本优化挑战。我们的系统包括了自动的端到端优化，历史上这是一个需要大量工作和非常转业的工作。我们希望本文会鼓励更多端到端编译方法的研究，开启深度学习系统软件-硬件协同设计技术的新机会。