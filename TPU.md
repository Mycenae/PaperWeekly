# In-Datacenter Performance Analysis of a Tensor Processing Unit

Norman P. Joupp et. al. Google

## 0. Abstract

Many architects believe that major improvements in cost-energy-performance must now come from domain-specific hardware. This paper evaluates a custom ASIC—called a ​Tensor Processing Unit (TPU)​ — deployed in datacenters since 2015 that accelerates the inference phase of neural networks (NN). The heart of the TPU is a 65,536 8-bit MAC matrix multiply unit that offers a peak throughput of 92 TeraOps/second (TOPS) and a large (28 MiB) software-managed on-chip memory. The TPU's deterministic execution model is a better match to the 99th-percentile response-time requirement of our NN applications than are the time-varying optimizations of CPUs and GPUs (caches, out-of-order execution, multithreading, multiprocessing, prefetching, …) that help average throughput more than guaranteed latency. The lack of such features helps explain why, despite having myriad MACs and a big memory, the TPU is relatively small and low power. We compare the TPU to a server-class Intel Haswell CPU and an Nvidia K80 GPU, which are contemporaries deployed in the same datacenters. Our workload, written in the high-level TensorFlow framework, uses production NN applications (MLPs, CNNs, and LSTMs) that represent 95% of our datacenters' NN inference demand. Despite low utilization for some applications, the TPU is on average about 15X - 30X faster than its contemporary GPU or CPU, with TOPS/Watt about 30X - 80X higher. Moreover, using the GPU’s GDDR5 memory in the TPU would triple achieved TOPS and raise TOPS/Watt to nearly 70X the GPU and 200X the CPU.

很多架构师相信，能耗比的主要改进现在必须来自于领域专用的硬件。本文评估了一种定制的ASIC，称为TPU，自从2015年就部署到了数据中心中，对神经网络的推理阶段进行了加速。TPU的核心，是65536个8位MAC矩阵乘法单元，峰值吞吐率可以达到92 TOPS，软件可以管理的片上内存很大，可以达到28 MiB。TPU确定性的执行模型，非常适合神经网络应用的99th百分位的响应时间要求，在CPUs和GPUs中是时间变化的优化（缓存，乱序执行，多线程，多进程，预取等）则不太适应，会将吞吐率平均到许诺的延迟之下。TPU缺少这些特征，也解释了其尽管有无数的MACs和很大的内存，但是相对较小，能耗也较低。我们将TPU与服务器级的Intel Haswell CPU和NVIDIA K80 GPU进行比较，它们都在相同的数据中心中进行了部署。我们的工作量是用高层的TensorFlow框架写的，使用的是生产中的神经网络应用(MLPs, CNNs, LSTMs)，代表了我们数据中心中NN推理需求的95%。尽管对一些应用利用率很低，TPU平均比对应的GPU或GPU快了15X-30X，TOPS/Watt高了30X-80X。而且，在TPU中使用GPU的GDDR5会使TOPS增加3倍，TOPS/Watt比GPU要快70X，比CPU快200X。

**Index terms** – DNN, MLP, CNN, RNN, LSTM, neural network, domain-specific architecture, accelerator

## 1. Introduction to Neural Networks

The synergy between the large data sets in the cloud and the numerous computers that power it has enabled a renaissance in machine learning. In particular, deep neural networks (DNNs) have led to breakthroughs such as reducing word error rates in speech recognition by 30% over traditional approaches, which was the biggest gain in 20 years [17]; cutting the error rate in an image recognition competition since 2011 from 26% to 3.5% [30, 56, 22]; and beating a human champion at Go [53].

云上的大型数据集，与很多计算机的协同作用，使得机器学习得到复兴。特别是，DNNs已经带来了一个突破，比如在语音识别中错误率比传统方法降低了30%，这是20年来的最大提升；将图像识别比赛中的错误率从2011年的26%降低到3.5%；在围棋比赛中击败了一个人类冠军。

Neural networks (NN) target brain-like functionality and are based on a simple artificial neuron: a nonlinear function (such as max(0, value)) of a weighted sum of the inputs. These pseudo neurons are collected into layers, with the outputs of one layer becoming the inputs of the next in the sequence. The “deep” part of DNN comes from going beyond a few layers, as the large data sets in the cloud allowed more accurate models to be built by using extra and larger layers to capture higher levels of patterns or concepts, and GPUs provided enough computing to develop them.

NN的目标是与大脑类似的功能，基于简单的人工神经元：输入的加权和的非线性函数，比如max(0, value)。这些伪神经元形成层，一层的输出是下一层的输入。DNN的深度部分是说，其层数更多，因为云中的大型数据集使得可以使用额外更大的层，来捕获更高层的模式或概念，构建更精确的模型，而GPUs则给出了足够的计算能力来开发这些模型。

The two phases of NN are called training (or learning) and inference (or prediction), and they refer to development versus production. The developer chooses the number of layers and the type of NN, and training determines the weights. Virtually all training today is in floating point, which is one reason GPUs have been so popular. A step called quantization transforms floating-point numbers into narrow integers—often just 8 bits—which are usually good enough for inference. Eight-bit integer multiplies can be 6X less energy and 6X less area than IEEE 754 16-bit floating-point multiplies, and the advantage for integer addition is 13X in energy and 38X in area [15].

NN的两个阶段称为训练和推理，或学习和预测，这是指开发和生产。开发者选择NN的层数和类型，训练则确定了权重。今天所有的训练都是以浮点数的形式进行的，这也是GPUs这么流行的原因之一。量化的步骤，将浮点数转换成很窄的整数，通常只有8 bits，这对于推理来说已经足够好了。8-bits整数乘法比16-bit的浮点数乘法，能耗低了6X，面积小了6X，整数相加则比浮点数相加能耗低了13X，面积小了38X。

Three kinds of NNs are popular today: 今天有三种NNs非常流行：

1. Multi-Layer Perceptrons (MLP): Each new layer is a set of nonlinear functions of a weighted sum of all outputs (fully connected) from the prior one. 多层感知机：每个新层是前一层的所有输出的加权和的非线性函数。

2. Convolutional Neural Networks (CNN): Each layer is a set of nonlinear functions of weighted sums at different coordinates of spatially nearby subsets of outputs from the prior layer, which allows the weights to be reused. 卷积神经网络：每个层是前一层的输出的空域附近的集合的加权和的非线性函数，这使得权重可以重用。

3. Recurrent Neural Networks (RNN): Each subsequent layer is a collection of nonlinear functions of weighted sums of outputs and the previous state. The most popular RNN is Long Short-Term Memory (LSTM). The art of the LSTM is in deciding what to forget and what to pass on as state to the next layer. The weights are reused across time steps. 循环神经网络：每个后续层是输出和之前状态的加权和的非线性函数。最流行的RNN是LSTM。LSTM的特点是，确定到下一层时，哪些要忘记，哪些要留下。权重在不同时间步骤中是重用的。

Table 1 shows two examples of each of the three types of NNs—which represent 95% of NN inference workload in our datacenters—that we use as benchmarks. Typically written in TensorFlow [1], they are surprisingly short: just 100 to 1500 lines of code. Our benchmarks are small pieces of larger applications that run on the host server, which can be thousands to millions of lines of C++ code. The applications are typically user facing, which leads to rigid response-time limits.

表1给出了三种NNs的例子，这代表了我们数据中心中95%的NN推理任务，我们将之用作基准测试。一般用TensorFlow写的话会很短，只需要100到1500行代码。我们的基准测试是跑在服务器上的更大型应用的片段，这些大型应用可能是数千到上百万行C++代码。这些应用通常是面向用户的，这要求的响应时间非常刚性。

Each model needs between 5M and 100M weights (9th column of Table 1), which can take a lot of time and energy to access. To amortize the access costs, the same weights are reused across a batch of independent examples during inference or training, which improves performance.

每个模型需要5M到100M权重，访问起来需要大量的时间和能量。为摊销访问代价，在训练或推理时，一个批次中独立的样本要复用同样的权重，这改进了性能。

This paper describes and measures the Tensor Processing Unit (TPU) and compares its performance and power for inference to its contemporary CPUs and GPUs. Here is a preview of the highlights:

本文描述并测量了TPU，并将其在推理时的性能与CPU和GPU进行了比较。这里一些亮点预览：

• Inference apps usually emphasize response-time over throughput since they are often user facing. 推理的应用强调响应时间，而不太强调吞吐量，因为通常是面向用户的。

• As a result of latency limits, the K80 GPU is just a little faster for inference than the Haswell CPU, despite it having much higher peak performance and memory bandwidth. 延迟限制的结果是，K80 GPU的推理时间只比Haswell CPU快一点点，尽管其有相当大的峰值性能和内存带宽。

• While most architects are accelerating CNNs, they are just 5% of our datacenter workload. 多数架构师都在加速CNNs，但这只占我们数据中心工作量的5%。

• The TPU is about 15X – 30X faster at inference than the K80 GPU and the Haswell CPU. 在推理时，TPU比K80 GPU和Haswell CPU快了15X到30X。

• Four of the six NN apps are memory bound; if the TPU were revised to have the same memory as the K80 GPU, it would be about 30X – 50X faster than the GPU and CPU. 6个NN应用中的4个是内存bound；如果TPU经过修改，与K80 GPU有相同的内存，将会比GPU和CPU快30X到50X。

• Despite having a much smaller and lower power chip, the TPU has 25 times as many MACs and 3.5 times as much on-chip memory as the K80 GPU. 与K80 GPU相比，尽管TPU小了很多，能耗也低了很多，但MACs多了25倍，片上内存多了3.5倍。

• The performance/Watt of the TPU is 30X – 80X that of its contemporary CPUs and GPUs; a revised TPU with K80 memory would be 70X – 200X better. TPU的性能/Watt比是CPUs和GPUs的30X到80X；内存修改后，将会是80X到200X。

## 2. TPU Origin, Architecture, and Implementation

Starting as early as 2006, we discussed deploying GPUs, FPGAs, or custom ASICs in our datacenters. We concluded that the few applications that could run on special hardware could be done virtually for free using the excess capacity of our large datacenters, and it’s hard to improve on free. That changed in 2013 when a projection showed people searching by voice for three minutes a day using speech recognition DNNs would double our datacenters’ computation demands, which would be very expensive using conventional CPUs. Thus, we started a high-priority project to produce a custom ASIC quickly for inference (and bought off-the-shelf GPUs for training). The goal was to improve cost-performance by 10X over GPUs. Given this mandate, in just 15 months the TPU was designed, verified [55], built, and deployed in datacenters. (Space limits the amount and the level of detail on the TPU in this paper; see [46], [47], [48], [49], [57], and [60] for more.)

最早从2006年开始，我们讨论在数据中心中部署GPUs, FPGAs, 或定制ASICs。我们得出结论，仅有的几个需要在特殊硬件上运行的应用，使用我们的大型数据中心的大量计算能力，几乎可以自由运行，但是要自由的改进，就很难了。在2013年，一个预测表明，人们在一天中只使用三分钟的语音搜索，其语音识别的DNNs会使我们数据中心的计算需求翻倍，使用传统的CPUs，这会非常昂贵，这就改变之前的结论。因此，我们开始了一个高优先级项目，快速生产一种定制ASIC进行推理（购买拆箱即用的GPUs进行训练）。目标是比GPUs的性价比提高10X。给了这个命令后，15个月就将TPU设计了出来，进行了验证，生产了出来并在数据中心中进行了部署。

Rather than be tightly integrated with a CPU, to reduce the chances of delaying deployment, the TPU was designed to be a coprocessor on the PCIe I/O bus, allowing it to plug into existing servers just as a GPU does. Moreover, to simplify hardware design and debugging, the host server sends TPU instructions for it to execute rather than the TPU fetching them itself. Hence, the TPU is closer in spirit to an FPU (floating-point unit) coprocessor than it is to a GPU.

为了不延误部署，TPU没有集成到CPU中，而是设计成了在PCIe IO总线上的一个协处理器，使其可以出入到现有的服务器中，就像一个GPU一样。而且，为了简化硬件设计和调试，主服务器发送TPU指令来给其执行，而不是TPU自己去取指令。因此，TPU更接近于一个FPU协处理器，而不是一个GPU。

The goal was to run whole inference models in the TPU to reduce interactions with the host CPU and to be flexible enough to match the NN needs of 2015 and beyond, instead of just what was required for 2013 NNs. Figure 1 shows the block diagram of the TPU.

目标是在TPU中运行整个推理模型，减少与主CPU的交互，而且要足够灵活，匹配2015年及以后的NN的需求，而不是只满足2013的NNs的需求。图1展示了TPU的模块图。

The TPU instructions are sent from the host over the PCIe Gen3 x16 bus into an instruction buffer. The internal blocks are typically connected together by 256-byte-wide paths. Starting in the upper-right corner, the Matrix Multiply Unit is the heart of the TPU. It contains 256x256 MACs that can perform 8-bit multiply-and-adds on signed or unsigned integers. The 16-bit products are collected in the 4 MiB of 32-bit Accumulators below the matrix unit. The 4 MiB holds 4096, 256-element, 32-bit accumulators. The matrix unit produces one 256-element partial sum per clock cycle. We picked 4096 by first noting that the operations per byte needed to reach peak performance (roofline knee in Section 4) is ~1350, so we rounded that up to 2048 and then duplicated it so that the compiler could use double buffering while running at peak performance.

TPU指令从主服务器上经过PCIe Gen3 x16总线发送到指令缓存中的。内部模块是由256 byte宽的通道连接在一起的。从右上角开始，矩阵相乘单元是TPU的核心，包含了256x256个MACs，可以对有符号或无符号的整数进行8-bit乘法和加法。16-bit的乘积是在矩阵单元下面的4 Mib的32-bit累加器上进行的。这4 MiB是4096个256元素的32-bit累加器。矩阵单元在每个时钟周期中产生一个256元素的部分和。我们选择4096是因为，每个byte需要达到峰值性能的运算时大约1350，所以四舍五入到2048，然后翻倍，这样编译器可以使用双倍buffering，同时运行在峰值性能。

When using a mix of 8-bit weights and 16-bit activations (or vice versa), the Matrix Unit computes at half-speed, and it computes at a quarter-speed when both are 16 bits. It reads and writes 256 values per clock cycle and can perform either a matrix multiply or a convolution. The matrix unit holds one 64 KiB tile of weights plus one for double buffering (to hide the 256 cycles it takes to shift a tile in). This unit is designed for dense matrices. Sparse architectural support was omitted for time-to-deployment reasons. The weights for the matrix unit are staged through an on-chip Weight FIFO that reads from an off-chip 8 GiB DRAM called Weight Memory (for inference, weights are read-only; 8 GiB supports many simultaneously active models). The weight FIFO is four tiles deep. The intermediate results are held in the 24 MiB on-chip Unified Buffer, which can serve as inputs to the Matrix Unit. A programmable DMA controller transfers data to or from CPU Host memory and the Unified Buffer.

当混合使用8 bit权重和16 bit激活值时，矩阵单元以半速进行计算，当两者都是16 bits时，以四分之一的速度计算。每个时钟周期可以读取并写入256个值，可以进行一次矩阵乘法或一次卷积。矩阵单元可以保存两个64 KiB权重，其中一个用于缓存（以隐藏将一个tile挪进来需要的256个周期）。这个单元是为密集矩阵设计的。我们忽略了稀疏架构的支持，因为到部署所给的时间很有限。矩阵单元所用的权重用一个片上权重FIFO来暂存，从一个片外的8 GiB内存中读取数据，称为权重内存（对于推理，权重是只读的，8 GiB支持很多同时的活跃模型）。权重的FIFO有4个tiles深。中间结果保存在24 MiB的片上统一Buffer中，作为矩阵单元的输入。可编程的DMA控制器将数据在CPU主内存和统一Buffer之间进行传输。

Figure 2 shows the floor plan of the TPU die. The 24 MiB Unified Buffer is almost a third of the die and the Matrix Multiply Unit is a quarter, so the datapath is nearly two-thirds of the die. The 24 MiB size was picked in part to match the pitch of the Matrix Unit on the die and, given the short development schedule, in part to simplify the compiler (see Section 7). Control is just 2%. Figure 3 shows the TPU on its printed circuit card, which inserts into existing servers like an SATA disk.

图2展示了TPU芯片上的平面布置图。24 MiB的统一Buffer几乎占了芯片的1/3，矩阵乘法单元大约是1/4，所以datapath是芯片的几乎2/3。选择24 MiB，部分是为了匹配矩阵单元在芯片上的pitch，因为开发计划很紧张，部分是为了简化编译器。控制部分只占了2%。图3展示了TPU与电路板，可以像一个SATA硬盘一样插入现有的服务器中。

As instructions are sent over the relatively slow PCIe bus, TPU instructions follow the CISC tradition, including a repeat field. The average clock cycles per instruction (CPI) of these CISC instructions is typically 10 to 20. It has about a dozen instructions overall, but these five are the key ones:

因为指令是通过相对低速的PCIe总线发送的，TPU指令遵循CISC传统，包含了一个重复域。这些CISC指令的平均每指令时钟周期一般是10-20。总计大约有12个指令，但下面5个是最关键的：

1. Read_Host_Memory reads data from the CPU host memory into the Unified Buffer (UB). 读取主内存指令，从CPU主内存中将数据读到统一Buffer中。
2. Read_Weights reads weights from Weight Memory into the Weight FIFO as input to the Matrix Unit. 读取权重指令，将权重从权重内存读取到权重FIFO中，作为矩阵单元的输入。
3. MatrixMultiply/Convolve causes the Matrix Unit to perform a matrix multiply or a convolution from the Unified Buffer into the Accumulators. A matrix operation takes a variable-sized Bx256 input, multiplies it by a 256x256 constant weight input, and produces a Bx256 output, taking B pipelined cycles to complete. 矩阵乘法/卷积指令，让矩阵单元执行一次矩阵乘法或一次卷积，输入为统一Buffer，输出到累加器中。一个矩阵运算的输入为可变大小的Bx256，乘以一个常量的256x256权重输入，生成一个Bx256的输出，耗时B个流水线周期来完成。
4. Activate performs the nonlinear function of the artificial neuron, with options for ReLU, Sigmoid, and so on. Its inputs are the Accumulators, and its output is the Unified Buffer. It can also perform the pooling operations needed for convolutions using the dedicated hardware on the die, as it is connected to nonlinear function logic. 激活指令，进行人工神经元的非线性函数运算，可选的有ReLU，Sigmoid，等等。其输入是累加器，输出是统一Buffer。这个指令也可以进行卷积所需的池化运算，因为这是与非线性函数逻辑相关的。
5. Write_Host_Memory writes data from the Unified Buffer into the CPU host memory. 写主内存指令，将数据从统一Buffer写到CPU主内存中。

The other instructions are alternate host memory read/write, set configuration, two versions of synchronization, interrupt host, debug-tag, nop, and halt. The CISC MatrixMultiply instruction is 12 bytes, of which 3 are Unified Buffer address; 2 are accumulator address; 4 are length (2 dimensions for convolutions); and the rest are opcode and flags.

其他指令是候补的主内存读写，设置配置，两个版本的同步，中断宿主，debug-tag，nop和停止。CISC MatrixMultiply指令是12 bytes，其中3个是统一Buffer的地址；2个是累加器的地址；4个是长度（卷积的2个维度）；剩余的是操作代码和flags。

The philosophy of the TPU microarchitecture is to keep the matrix unit busy. It uses a 4-stage pipeline for these CISC instructions, where each instruction executes in a separate stage. The plan was to hide the execution of the other instructions by overlapping their execution with the MatrixMultiply instruction. Toward that end, the Read_Weights instruction follows the decoupled-access/execute philosophy [54], in that it can complete after sending its address but before the weights are fetched from Weight Memory. The matrix unit will stall if the input activation or weight data is not ready.

TPU微架构的哲学是保持矩阵单元忙碌。它对这些CISC指令使用了4阶段流水线，其中每个指令在一个单独的阶段执行。计划是将其他指令的执行隐藏起来，使其执行与MatrixMultiply指令重叠起来。为此，Read_Weights指令按照解耦的访问/执行的哲学进行，在其发出地址后，再权重从权重内存中获取得到之前就可以结束。如果输入激活或权重数据没有准备好，矩阵单元就停止。

We don’t have clean pipeline overlap diagrams, because our CISC instructions can occupy a station for 1000s of clock cycles, unlike the traditional RISC pipeline with 1 per stage. Situations occur when the activations for one network layer must complete before the matrix multiplications of the next layer can begin; we see a RAW pipeline stall where the matrix unit waits for explicit synchronization before safely reading from the Unified Buffer.

我们没有流水线重叠的图，因为我们的CISC指令可以占据1000s的时钟周期，与传统的RISC流水线不同，每个阶段只有一个。当一个网络层的激活值必须计算结束，才能开始下一层的矩阵相乘计算时，就会出现一些情况；在矩阵单元等待显式的同步信号，以安全的从统一Buffer中读取时，我们就会看到一个RAW流水线停止的情况。

As reading a large SRAM uses much more power than arithmetic, the matrix unit uses systolic execution to save energy by reducing reads and writes of the Unified Buffer [31, 44, 40]. It relies on data from different directions arriving at cells in an array at regular intervals where they are combined. Figure 4 shows that data flows in from the left, and the weights are loaded from the top. A given 256-element multiply-accumulate operation moves through the matrix as a diagonal wavefront. The weights are preloaded, and take effect with the advancing wave alongside the first data of a new block. Control and data are pipelined to give the illusion that the 256 inputs are read at once, and that they instantly update one location of each of 256 accumulators. From a correctness perspective, software is unaware of the systolic nature of the matrix unit, but for performance, it does worry about the latency of the unit.

由于读取大型SRAM比代数运算使用的能量更多，矩阵单元使用收缩性的执行来节省能量，减少对统一Buffer的读取和写入。它依靠不同方向的数据以规则的间隔到达阵列中的cells，在这里将其结合到一起。图4展示了数据从左边流入，权重从上面载入。一个给定的256个元素的相乘累加运算，在矩阵中的穿越，像一个对角的波前。权重是预先载入的，与推进的波一起，与一个新模块的第一波数据进行运算。控制和数据是流水线化的，会得到幻觉，256个输入是一次性读取的，而且它们立刻更新了256个累加器的每个的一个位置。从正确性的角度，软件是意识不到矩阵单元的收缩性本质的，但对于性能来说，不用担心单元的延迟。

The TPU software stack had to be compatible with those developed for CPUs and GPUs so that applications could be ported quickly to the TPU. The portion of the application run on the TPU is typically written in TensorFlow and is compiled into an API that can run on GPUs or TPUs [33]. Like GPUs, the TPU stack is split into a User Space Driver and a Kernel Driver. The Kernel Driver is lightweight and handles only memory management and interrupts. It is designed for long-term stability.

TPU软件栈必须与给CPUs和GPUs开发的要兼容，这样应用可以快速迁移到TPU。在TPU上运行的应用部分，一般是由TensorFlow写的，编译成一个API，可以在GPUs或TPUs上运行。与GPUs相似，TPU栈分割成一个用户空间的驱动和一个核心驱动。核心驱动是轻量的，只处理内存管理和中断。这是为了长期稳定性设计的。

The User Space driver changes frequently. It sets up and controls TPU execution, reformats data into TPU order, translates API calls into TPU instructions, and turns them into an application binary. The User Space driver compiles a model the first time it is evaluated, caching the program image and writing the weight image into the TPU’s weight memory; the second and following evaluations run at full speed. The TPU runs most models completely from inputs to outputs, maximizing the ratio of TPU compute time to I/O time. Computation is often done one layer at a time, with overlapped execution allowing the matrix multiply unit to hide most non-critical-path operations.

用户空间驱动频繁的变化。其设置和控制TPU的执行，将数据重新格式化，成TPU的顺序，将API调用翻译成TPU的指令，将其转变成一个应用的二进制。用户空间驱动在一个模型第一次计算时对其进行编译，将程序镜像进行缓存，将权重镜像写入TPU的权重内存；第二次和后续的计算，则全速进行。TPU运行多数模型都是从输入到输出，将TPU的计算时间与I/O时间的比最大化。计算通常是每层一次，重叠的执行使矩阵相乘单元将多数非关键路径运算隐藏起来。

## 3. CPU, GPU, and TPU Platforms

The six production applications in Table 1 are our workload for this paper. As mentioned above, these six are representative of 95% of TPU use in our datacenters. Ironically, deploying and measuring popular small DNNs like AlexNet or VGG is difficult on production machines. However, one of our CNNs derives from Inception V2, which is widely used.

表1中的6个生产应用是本文的工作负荷。如同上述，这6个是我们数据中心中95%的TPU使用代表。讽刺的是，在生产机器上部署并测量流行的小型DNNs是很困难的，如AlexNet或VGG。但是，从Inception V2推导出的一个CNN也是广泛使用的。

The benchmark platforms are server-class computers that were available in 2015 when the TPUs were deployed. This restriction meant that they must include at least SECDED protection of internal SRAM as well as external DRAM memory like the TPU, which excludes some choices such as the Nvidia Maxwell GPU. For our company to purchase and deploy them, they also had to be sensibly configured machines, and not awkward artifacts assembled solely to win benchmarks.

基准测试平台是2015年可用的服务器级的计算机，TPUs在此时在这些服务器上部署。这种限制意味着，必须包括内部SRAM和外部DRAM的至少SECDED保护，比如TPU，这就排除了一些选项，比如NVIDIA Maxwell GPU。因为我们公司要购买并部署，它们必须要是可有意义的配置的机器，不是只为了赢得基准测试的奇怪的人造物。

Table 2 lists our choices. The traditional CPU server is represented by an 18-core, dual-socket Haswell processor from Intel. This platform is also the host server for GPUs or TPUs. Haswell was fabbed in an Intel 22nm process. Both the CPU and GPU are very large dies: about 600 mm2!

表2列出了我们的选项。传统的CPU服务器是18核双插座的Haswell处理器。这个平台也是GPUs或TPUs的宿主服务器。Haswell是用Intel 22nm工艺制造的。CPU和GPU的芯片都很大，大约600mm2。

The 2.3 GHz CPU clock rate doesn’t include Turbo mode because it seldom occurs in our datacenters for NN apps. Haswell has different clock rates depending on whether programs use AVX instructions, which our NN apps often use. The higher clock rate of Turbo mode (for programs that avoid AVX) occurs when they don’t use all their cores. Thus, another reason Turbo mode is rare in our datacenters is that our apps typically do use all the cores, plus they can run other datacenter jobs to fill any idle cores.

2.3 GHz CPU时钟速率并没有包括Turbo模式，因为在我们的数据中心中对于NN应用很少发生。Haswell有不同的时钟速率，这依赖于程序是否使用AVX指令，我们的NN应用经常使用。Turbo模式的更高时钟速率发生时，不能使用所有核心。因此，在我们的数据中心中Turbo模式很稀少的另一个原因是，我们的应用一般使用所有的核心，它们可以在任何其他空闲的核心上运行其他数据中心任务。

The GPU accelerator is the Nvidia K80. Each K80 card contains two dies and offers SECDED on internal memory and DRAM. Nvidia states that the “K80 Accelerator dramatically lowers datacenter cost by delivering application performance with fewer, more powerful servers” [38]. NN researchers frequently used K80s in 2015, and they were chosen for new cloud-based GPU offerings as recently as September 2016 [7]. Up to eight K80 dies can be installed in four cards on this server, which is the configuration we benchmark.

GPU加速器是NVIDIA K80。每个K80卡包含两个芯片，在内部存储和DRAM上都有SECDED。NVIDIA表明，K80加速器极大的降低了数据中心代价，用更少更强力的服务器给出了应用性能。NN研究者在2015年频繁的使用K80，最近在2016年9月被选做新的基于云的GPU服务。在这个服务器上，可以安装四张卡，8个K80芯片，这是我们进行基准测试的配置。

As the number of dies per server varies between 2 to 8, we usually show results normalized per die (Figures 5–8, Figures 10–11, and Tables 3, 5, and 7), but we occasionally show whole systems (Figure 9). We hope this distinction is clear.

每个服务器的芯片数量从2到8个，我们通常给出每个芯片的结果，但也会给出整个系统的性能。希望这个区别是明显的。

## 4. Performance: Rooflines, Response-Time, and Throughput

To illustrate the performance of the six apps on the three processors, we adapt the Roofline Performance model from high-performance computing (HPC) [58]. This simple visual model is not perfect, yet it offers insights into the causes of performance bottlenecks. The assumption behind the model is that applications don’t fit in on-chip caches, so they are either computation-limited or memory bandwidth-limited. For HPC, the Y-axis is performance in floating-point operations per second, thus the peak computation rate forms the “flat” part of the roofline. The X-axis is operational intensity, measured as floating-point operations per DRAM byte accessed. Memory bandwidth is bytes per second, which turns into the “slanted” part of the roofline since (FLOPS/sec)/ (FLOPS/Byte) = Bytes/sec. Without sufficient operational intensity, a program is memory bandwidth-bound and lives under the slanted part of the roofline.

为描述6个应用在3个处理器上的性能，我们调整了高性能计算HPC中的Roofline性能模型。这种简单的视觉模型并不是完美的，但是给性能瓶颈的原因提供了很多洞见。模型后的假设是，应用不适应片上的缓存，所以它们要么受到了计算能力的限制，或者受到了内存带宽的限制。对高性能计算，Y轴是性能，即每秒的浮点数计算次数，因此峰值计算速率形成了roofline平坦的部分。X轴是运算强度，每个访问的DRAM byte中的浮点运算数量。内部带宽是每秒的bytes数量，会转换成roofline中倾斜的部分，因为(FLOPS/sec)/ (FLOPS/Byte) = Bytes/sec。没有足够的运算强度，一个程序就是受到内存带宽限制的，处于roofline的倾斜部分。

The gap between the actual operations per second of an application and the ceiling directly above it shows the potential benefit of further performance tuning while leaving operational intensity untouched; of course, optimizations that increase operational intensity (such as cache blocking) may yield even greater benefit.

一个应用实际的每秒运算量，和其上的ceiling之间的差距，表明了在运算强度不变的情况下，进一步进行性能调节可能得到的可能好处；当然，增加运算强度的优化（比如cache blocking）会带来更大的好处。

To use the Roofline model for the TPU, when NN applications are quantized, we first replace floating-point operations with integer operations. As weights do not normally fit in on-chip memory for NN applications, the second change is to redefine operational intensity to be integer operations per byte of weights read (see the tenth column of Table 1).

为对TPU使用Roofline模型，当NN应用量化过后，我们首先将浮点运算替换为整数运算。因为在NN应用中权重一般与片上内存并不适配，第二个变化是将运算强度重新定义为，每读取一个byte的整数运算量（见表1的第10列）。

Figure 5 shows the Roofline model for a single TPU die on log-log scales. The TPU has a long “slanted” part of its roofline, where operational intensity means that performance is limited by memory bandwidth rather than by peak compute. Five of the six applications are happily bumping their heads against the ceiling: the MLPs and LSTMs are memory bound, and CNNs are computation bound. CNN1, despite a high operational intensity, is running at only 14.1 TOPS while CNN0 runs at 86 TOPS.

图5展示了log-log尺度下，单个TPU芯片的Roofline模型。TPU的roofline有很长的斜坡部分，其中运算强度意思是，性能受到内存带宽局限，而不是受到峰值计算能力局限。6个应用中的5个，即那些MLPs和LSTMs是受到内存限制的，CNNs是受到计算能力限制的。CNN1尽管其运算强度很高，其消耗的计算量仅仅是14.1 TOPS，而CNN0则是86 TOPS。

Table 3 explains what happened with CNN1, based on the performance counters that give us partial visibility into TPU operation. The TPU spends less than half of its cycles performing matrix operations for CNN1 (column 7, row 1). On each of those active cycles, only about half of the 65,536 MACs hold useful weights because some layers in CNN1 have shallow feature depths. About 35% of cycles are spent waiting for weights to load from memory into the matrix unit, which occurs during the 4 fully connected layers that run at an operational intensity of just 32 (see the last fallacy in Section 8). This leaves roughly 19% of cycles not explained by the matrix-related counters. Because of overlapped execution on the TPU, we do not have exact accounting for those cycles, but we can see that 23% of cycles have stalls for RAW dependences in the pipeline, and 1% are spent stalled for input over the PCIe bus.

表3解释了CNN1发生了什么，性能计数器使我们对TPU运算是部分可见的。TPU少于一半的周期在进行CNN1的矩阵运算。在这些活跃周期中的每个，这65536个MACs只有一半有有用的权重，因为CNN1中的一些层的特征深度很浅。大约35%的周期用在等待权重从内存装载到矩阵单元中，这在4个全连接层中会发生，其中的运算强度只有32（见第8部分）。这只剩下19%的周期，与矩阵相关的计数器未进行解释。因为TPU中的执行使重叠的，我们没有对这些周期进行严格的技术，但我们可以看到，23%的周期有因为流水线中的RAW依赖关系停止，1%因为PCIe总线输入而停止过。

Figures 6 and 7 show rooflines for a single Haswell die and for a single K80 die. The six NN applications are generally further below their ceilings than was the TPU in Figure 5. Response time is the reason. Many of these NN applications are parts of end-user-facing services. Researchers have demonstrated that small increases in response time cause customers to use a service less [51]. Hence, while training may not have hard response time deadlines, inference usually does. That is, inference prefers latency over throughput.

图6和图7展示了单个Haswell芯片和单个K80芯片的rooflines。与图5中的TPU情况比较，6个NN应用一般来说都在ceiling之下。响应时间是主要原因。很多这些NN应用都是面向最终用户的服务的一部分。研究者证明了，响应时间的很小的增加，都会使用户更少的使用这个服务。因此，训练不一定有很硬的响应时间要求，推理通常有。即，推理对延迟要求更高，对吞吐量要求没那么高。

For example, the 99th-percentile response time limit for MLP0 was 7 ms, which was required by the application developer. (The inferences per second and 7 ms latency include the server host time as well as the accelerator time.) Table 4 shows that Haswell and the K80 run at just 42% and 37%, respectively, of the highest throughput achievable for MLP0 if the response time limit was relaxed. These bounds affect the TPU as well, but at 80% it is operating much closer to its highest MLP0 throughput. As compared to CPUs and GPUs, the single-threaded TPU has none of the sophisticated microarchitectural features that consume transistors and energy to improve the average case but not the 99th-percentile case: no caches, branch prediction, out-of-order execution, multiprocessing, speculative prefetching, address coalescing, multithreading, context switching, and so forth. Minimalism is a virtue of domain-specific processors.

比如，MLP0的响应时间限制的99th百分位是7ms，这是应用开发者要求的。（每秒的推理数和7ms延迟包括服务器宿主时间以及加速器时间）表4展示了Haswell和K80分别运行在可取得的最高吞吐量的42%和37%，最高吞吐量是在不计算响应时间的限制的情况下得到的。这些限制也影响TPU，但是运行在80%的水平上的，非常接近于其最高的MLP0吞吐量。与CPUs和GPUs比较，单线程的TPU没有复杂的微架构特征，来消耗晶体管和能量来改进平均情况：没有缓存，没有分支预测，没有乱序执行，没有多进程，没有推测预取，没有地址合并，没有多线程，没有上下文切换，等等。极简主义是领域专用处理器的特点。

Table 3 shows TPU performance, but it doesn’t account for host server time, which can be divided into running the host share of the application and talking to the TPU. Table 5 lists the second part, but the first part is hard. Queueing theory shows that long input queues raise throughput—by ensuring that the computer is never idle—but stretch response time. Thus, most applications keep their input queues empty. Alas, we can’t measure when the TPU is idle since it is waiting for the CPU to do its portion of the application or because the CPU is also idle due to an empty input queue.

表3展示了TPU的性能，但并没有计入宿主服务器时间，可以分为，运行应用在服务器的部分，和与TPU的通信。表5列出了第二部分，但第一部分是很难的。队列理论表明，很长的输入队列会提升吞吐量，这可以确保计算机永远不会空闲，但会加长响应时间。因此，多数应用会将其输入队列空置。我们无法测量TPU何时是闲置的，因为在等待CPU运行应用的部分，或因为CPU也是空闲的，因为输入队列是空的。

Table 6 gives the bottom line of relative inference performance per die including the host server overhead for the two accelerators versus the CPU. The next-to-last column shows the geometric mean of the relative performance for the six NN applications, which suggests the K80 die is 1.1X the speed of a Haswell die, that the TPU die is 14.5 times as fast, and thus the TPU die is 13.2 times as fast as the GPU die. Figure 8 shows their relative speeds visually.

表6给出了两个加速器相对于CPU，每个芯片的相对推理性能，包括了宿主服务器的开销。倒数第二列展示了6个NN应用的相对性能的几何均值，说明K80芯片是Haswell芯片速度的1.1倍，TPU芯片的速度是14.5倍，因此TPU芯片的速度是GPU芯片的13.2倍。图8给出了其相对速度。

Recall that architects use the geometric mean when they don’t know the actual mix of programs that will be run [23]. For this study, however, we do know the mix (Table 1). The weighted mean in the last column of Table 6 using the actual mix increases the GPU to 1.9X and the TPU to 29.2X, so the TPU die is now 15.3 times as fast as the GPU die.

回忆一下，架构师在不知道要运行的程序的实际混合是什么样子的，就使用几何平均。但对这个研究，我们确实知道混合的具体情况（表1）。表6中最后一列的加权均值，使用的是实际的混合，使GPU的效率到了1.9X，TPU到了29.2X，所以TPU芯片是GPU芯片速度的15.3倍。

## 5. Cost-Performance, TCO, and Performance/Watt

When buying computers by the thousands, cost-performance trumps performance. The best cost metric in a datacenter is total cost of ownership (TCO). The actual price we pay for thousands of chips depends on negotiations between the companies involved. For business reasons, we can’t publish such price information or data that might let them be deduced. However, power is correlated with TCO, and we can publish Watts per server, so we use performance/Watt as our proxy for performance/TCO in this paper. In this section, we compare whole servers rather than single dies, which Table 2 lists in the “Benchmarked Server” columns.

当购买数千台计算机时，性价比超过了性能。数据中心中最好的价格度量，是总体拥有成本(TCO)。我们为数千个芯片所付出的实际价格，依赖于涉及到的公司之间的谈判。由于商业原因，我们不能发表这些价格信息或数据。但是，能量是与TCO相关的，我们可以发表每个服务器的Watt数，所以我们使用性能每Watt作为性能每TCO的代理。本节中，我们比较整个服务器，而不是单个芯片，在表2中基准测试服务器列列出的。

Figure 9 shows the geometric and weighted mean performance/Watt for the K80 GPU and TPU relative to the Haswell CPU. We present two different calculations of performance/Watt. The first (“total”) includes the power consumed by the host CPU server when calculating performance/Watt for the GPU and TPU. The second (“incremental”) subtracts the host CPU server power from the GPU and TPU beforehand.

图9展示了K80 GPU和TPU相对于Haswell CPU的每Watt性能的几何和加权均值。我们给出每Watt性能的两种不同的计算。第一种（总计）包括宿主CPU服务器消耗的能量，第二种（增量）只计算了GPU和TPU的能耗。

For total-performance/Watt, the K80 server is 1.2X – 2.1X Haswell. For incremental-performance/Watt, when Haswell server power is omitted, the K80 server is 1.7X – 2.9X. The TPU server has 17X – 34X better total-performance/Watt than Haswell, which makes the TPU server 14X – 16X the performance/Watt of the K80 server. The relative incremental-performance/Watt— which was our company’s justification for a custom ASIC—is 41X – 83X for the TPU, which lifts the TPU to 25X – 29X the performance/Watt of the GPU.

对于总计的每Watt性能，K80服务器是Haswell的1.2X到2.1X。对于增量的每Watt性能，当Haswell服务器能量被忽略时，K80服务器是1.7X到2.9X。总计每Watt性能，TPU服务器是Haswell的17X-34X，也就是K80服务器的14X-16X。相对的增量每Watt性能，TPU相对于CPU是41X-83X，相对于GPU是25X-29X。

## 6. Energy Proportionality

Thermal Design Power (TDP) affects the cost of provisioning power, as you must supply sufficient power and cooling when hardware is at full power. However, the cost of electricity is based upon the average consumed as the workload varies during the day. [6] found that servers are 100% busy less than 10% of the time and advocated energy proportionality: servers should consume power proportional to the amount of work performed.

TDP影响供应能量的代价，因为硬件在全速运行时，必须供给足够的能量和冷却。但是，电的代价是基于平均消耗，因为一天中的工作负载在变化。[6]发现，服务器在不到10%的时间中是满负荷的，主张能量相称性：服务器消耗的能量应当与进行的工作成比例。

The estimate of power consumed in the prior section is based upon the fraction of the TDP that has been seen in our datacenters. We measured performance and power of servers including CPUs, TPUs, and GPUs as the offered workload utilization varies from 0% to 100%, collected in buckets of 10% delta of workload [32]. Figure 10 shows server power divided by the number of dies per server for the three chips by varying CNN0’s workload.

前一节估计的能量消耗，是基于我们数据中心中看到的TDP。我们测量的服务器性能和能量，包括CPUs，TPUs和GPUs，因为工作负载的利用率的变化从0%到100%。图10展示了每个服务器上的服务器能耗除以芯片数量，对CNN0的工作负载进行变化。

We see that the TPU has the lowest power —40W per die— but it has poor energy proportionality: at 10% load, the TPU uses 88% of the power it uses at 100%. (The short design schedule prevented inclusion of many energy-saving features.) Not surprisingly, Haswell is the best at energy proportionality of the group: it uses 56% of the power at 10% load as it does at 100%. The K80 is closer to the CPU than the TPU, using 66% of the full load power at 10% workload. LSTM1, which is not computation bound, performs similarly: at 10% load the CPU uses 47% of full power, the GPU uses 78%, and the TPU uses 94%.

我们看到，TPU的能耗最低，每个芯片40W，但能耗相称性很差：在10%的工作量时，使用88%的能量。（设计周期短，所以没有包括很多能耗节省的特征）并不令人惊讶的是，Haswell的能耗相称性最好：在工作负载10%时，使用的能量是56%。K80比TPU更接近CPU，在工作负载10%时，使用66%的能量。LSTM1，并不受到计算能力限制，性能类似：在10%的工作负载时，CPU使用47%的能量，GPU使用78%，TPU使用94%。

What happens to the server power usage when running CNN0 if it becomes a host to accelerators? When the GPU and TPU are at 100% load, the CPU server uses 52% of full power for the GPU and 69% for the TPU. (The CPU does more work for the TPU because it is running so much faster than the GPU.) Consequently, the Haswell server plus four TPUs use <20% additional power but run CNN0 80 times faster than the Haswell server alone (4 TPUs vs. 2 CPUs).

当服务器是加速器的宿主时，如果运行CNN0，那么服务器的能耗会怎样呢？当GPU和TPU满负载运行时，CPU服务器对GPU使用52%的能量，对TPU使用69%。（CPU为TPU做了更多工作，因为运行速度比GPU快太多。）结果是，Haswell服务器和4个TPUs使用的能量比Haswell服务器多了不到20%，但运行CNN0的速度快了80倍。

## 7. Evaluation of Alternative TPU Designs

Like an FPU, the TPU coprocessor is relatively easy to evaluate, so we created a performance model for our six applications. Table 7 shows the differences between the model results and the hardware performance counters, which average below 10%. We then modeled performance as we varied the memory bandwidth, the clock rate and number of accumulators, and the matrix multiply unit size.

与FPU类似，TPU协处理器评估起来相对容易，所以我们对6个NN应用创建了一个性能模型。表7给出了模型结果和硬件性能计数器之间的差异，平均小于10%。我们然后变化内存带宽，时钟速率和加速器数量，和矩阵乘法单元数量，对性能进行了建模。

Figure 11 shows the mean performance sensitivity of TPU die as we scale these parameters over the range for 0.25x to 4x. It plots weighted means, but the geometric means look similar. In addition to evaluating the impact of only raising clock rates (clock in Figure 11), we also plot a design (clock+) where the clock rate is increased and the number of accumulators is correspondingly scaled so the compiler can keep more memory references in flight. Likewise, we plot matrix unit expansion if we increase the number of accumulators with the square of the rise in one dimension (matrix+), since the number of multipliers in the matrix grows in both dimensions, as well as just increasing the matrix unit alone (matrix).

图11展示了TPU的平均性能敏感度，参数在0.25x到4x的范围内变动。所画出的是加权平均，但是几何平均看起来也很类似。除了评估只增加时钟速率的影响，我们还画出了一个设计(clock+)，其中时钟速率是增加的，而且加速器的数量也相应的增加了，这样编译器可以保持更多的内存in flight。类似的，如果我们增加了加速器的数量，我们还会画出了矩阵单元扩张，因为乘法器的数量是按照平方的速度增加的。

First, increasing memory bandwidth (memory) has the biggest impact: performance improves 3X on average when memory increases 4X. Second, clock rate has little benefit on average with or without more accumulators. The reason is the MLPs and LSTMs are memory bound but only the CNNs are compute bound. While hard to see in Figure 11, since it shows only the weighted mean of all six DNNs, increasing the clock rate by 4X has almost no impact on MLPs and LSTMs but improves performance of CNNs by about 2X. Third, the average performance in Figure 11 slightly degrades when the matrix unit expands from 256x256 to 512x512 for all apps, whether or not they get more accumulators. The issue is analogous to internal fragmentation of large pages, only worse since it’s in two dimensions. Consider the 600x600 matrix used in LSTM1. With a 256x256 matrix unit, it takes 9 steps to tile 600x600, for a total of 18 us of time. The larger 512x512 unit requires only four steps, but each step takes four times longer, for 32 us of time. Our CISC instructions are long, so decode is insignificant and does not hide the overhead of loading from the DRAM.

首先，增加内存带宽的影响最大：当内存增加4X时，性能增加了3X。第二，时钟速度几乎没有影响。原因是，MLPs和LSTMs是受到内存约束的，只有CNNs是受到计算能力限制的。在图11中很难看到，因为只展示了所有6个DNNs的加权平均，时钟速度增加4X，对MLPs和LSTMs几乎没有影响，但CNNs的性能增加了2X。第三，当矩阵单元从256x256增加到512x512时，对所有应用的平均性能反而略有下降，不论其是否有了更多的加速器使用。这个问题与大型pages的内部碎片化类似，而且还会更差一些，因为这是二维的。考虑LSTM1中使用的600x600矩阵。用256x256的矩阵单元，tile 600x600需要9步，总计18us。更大的512x512单元只需要4步，但是每步需要4倍的时间，总计32us。我们的CISC指令太长了，所以解码不重要，不会将从DRAM中加载的开销隐藏掉。

Table 8 shows the utilization of the 24 MiB Unified Buffer, which was initially sized to allow MLPs to run at batch sizes up to 2048. We recently improved the storage allocator for the Unified Buffer, which reduces the memory needed for the largest of the six applications to 14 MiB. For the first 18 months of deployment, the TPU used its full capacity while the new allocator was being developed. Now the extra capacity adds margin for adopting bigger models.

表8展示了24 MiB的统一Buffer的利用率，初始确定这个大小，是让MLPs可以在batch size在2048的大小时，也可以使用。我们最近改进了统一Buffer的存储分配器，将6个应用中最大的需要的内存降低到了14 MiB。对于开始18个月的部署，TPU完全利用了其能力，而新的分配器开发出来了。现在额外的能力可以使得采用更大的模型成为可能。

We next used the performance model to evaluate a hypothetical TPU die (TPU’) that could be designed in the same process technology if we had more than 15 months. More aggressive logic synthesis and block design might have increased the clock rate by 50%. Designing an interface circuit for GDDR5 memory, as in the K80, would improve Weight Memory bandwidth by more than a factor of five, shifting its roofline ridge point from 1350 to 250. As Figure 11 shows, increasing clock rate to 1050 MHz but not helping memory makes little change. If we left the clock at 700 MHz but used GDDR5 for Weight Memory, the geometric mean increase jumps to 2.6 and the weighted mean to 3.9. Doing both raises the geometric mean (2.9) but not the weighted mean, so TPU’ just has faster memory.

我们下一步用性能模型来评估一个假设的TPU芯片(TPU')，用相同的工艺进行设计，但是假设我们有超过15个月的时间来设计。更激进的逻辑综合和模块设计可能会使时钟速率增加50%。为GDDR5内存设计一个接口电路，就像K80中的一样，会改进权重内存的带宽，使其提升5倍，将roofline的ridge点从1350变到250。如图11所示，将时钟速率提升到1050MHz，但并不提升内存，这不会有帮助。如果我们让时钟为700MHz，但使用GDDR5作为权重内存，几何平均会提升到2.6，加权均值会到3.9。两者都做，会将几何平均提升到2.9，但加权平均不会有变化，所以TPU'只是有更快的内存。

Figure 11 does not include host server time. We used Table 5 to calculate time for the host server interaction overhead for the TPU. Adding that same extra time drops TPU’ means from 2.6 to 1.9 and 3.9 to 3.2. This change is both optimistic, since it doesn’t include CPU time to run its share of the app, and pessimistic, as we likely would aggressively tune the host code given a 3X faster TPU’.

图11并没有包含宿主服务器时间。我们使用表5来计算宿主服务器与TPU的互动消耗。加上了这些额外的时间，使得TPU'的均值从2.6下降到1.9，从3.9下降到3.2。这个变化是乐观的，因为并没有包括CPU运行应用的部分的时间，同时也是悲观的，因为给定一个3X快的TPU'，我们很可能会激进的调整宿主的代码。

Replacing just the DDR3 Weight Memory with the K80- equivalent GDDR5 memory requires doubling the number of memory channels to four. This improvement would expand die size by about 10%. However, higher memory bandwidth reduces pressure on the Unified Buffer, so reducing the Unified Buffer to 14 MiB could gain back 10% in area. GDDR5 would also increase the TPU system power budget from 861 Watts to about 900 Watts, as there are 4 TPUs per server.

将DDR3的权重内存替换成K80等价的GDDR5内存，需要将内存通道数量翻倍到4。这种改进会将芯片大小增加10%。但是，更高的内存带宽降低了统一Buffer的压力，所以降低统一Buffer的大小到14MiB，会得到10%的面积。GDDR5也会提升TPU的系统能量预算，从861 Watts到大约900 Watts，因为每个服务器上有4个TPUs。

Figure 9 above shows the relative total-performance /Watt/die of TPU’ leaps to 31X – 86X over Haswell and 25X – 41X over the K80. The incremental metric soars to 69X–196X over Haswell and 42X – 68X over the K80.

图9展示了TPU'对Haswell的相对总计性能/Watt/die为31X到86X，相对于K80为25X-41X。对Haswell的增量度量会提升到69X-196X，对K80的会提升到42X-68X。

## 8. Discussion

This section follows the fallacy and pitfall with rebuttal style of [23].

● Fallacy: NN inference applications in datacenters value throughput as much as response time. 对于数据中心中的神经网络推理应用，吞吐量和响应时间一样重要。

We were surprised that our developers had strong response-time demands, as some suggested in 2014 that batch sizes would be large enough for the TPU to reach peak performance or that latency requirements would not be as tight. One driving application was off-line image processing, and the intuition was that if interactive services also wanted TPUs, most of them would just accumulate larger batches. Even the developers of one application in 2014 that cared about response time (LSTM1) said the limit was 10 ms in 2014, but shrank it to 7 ms when they actually ported it to the TPU. The unexpected desire for TPUs by many such services combined with the impact on and preference for low response time changed the equation, with application writers often opting for reduced latency over waiting for bigger batches to accumulate. Fortunately, the TPU has a simple and repeatable execution model to help meet the response-time targets of interactive services and such high peak throughput that even small batch sizes result in higher performance than contemporary CPUs and GPUs.

我们很惊讶的发现，开发者对于响应时间的需求很强，因为一些人在2014年建议，对于TPU来说batch sizes会足够大，可以达到峰值性能，或者说延迟的需求不会那么紧要。一个驱动型的应用是离线的图像处理，直觉是，如果互动型的服务也需要TPUs，多数也会累积成更大的batches。2014年，一个对响应时间很在意的应用(LSTM1)的开发者说，在2014年的极限是10ms，但是当迁移到TPU时，缩减到了7ms。很多这种服务队TPUs的意外的期待，与对低响应时间的倾向，改变了情况，应用开发者经常需要降低延迟，而不是等待累积更大的batches。幸运的是，TPU的执行模型简单又可重复，可以达到互动模型的响应时间目标，也可以达到这样高的峰值吞吐量，即使很小的batch sizes也会比同时的CPUs和GPUs得到更好的性能。

● Fallacy: The K80 GPU architecture is a good match to NN inference. K80的GPU架构可以很好的匹配NN的推理。

GPUs have traditionally been seen as high-throughput architectures that rely on high-bandwidth DRAM and thousands of threads to achieve their goals. This perspective helps explain why the K80 is only a little faster at inference than Haswell and much slower than the TPU. Successors to the K80 will surely include optimizations to improve peak inference performance, but given their throughput-oriented architectural approach, it may be more challenging for GPUs to meet the strict latency limits. And as Section 7 shows, there is plenty of headroom to improve the TPU, so it’s not an easy target.

GPUs传统上被认为是高吞吐量的架构，依赖于高带宽的DRAM和几千个线程来达到其目标。这个角度解释了，为什么K80在推理上只比Haswell快了一点点，而比TPU慢了很多。K80的后继者，当然会进行优化以改进峰值推理性能，但由于其倾向于吞吐量的架构方法，要让GPUs满足其严格的延迟限制，就非常有挑战性。第7部分显示，TPU可以改进的空间还有很大，所以这个目标并不容易。

● Pitfall: Architects neglected important NN tasks. 架构师忽略了重要的NN任务。

We are pleased by the attention that the architecture community is paying to NN: 15% of the papers at ISCA 2016 were on hardware accelerators for NN [2, 11 , 13, 21, 29, 34, 35, 45, 52]! Alas, all nine papers looked at CNNs, and only two mentioned other NNs. CNNs are more complex than MLPs and are prominent in NN competitions [50], which might explain their allure, but they are only about 5% of our datacenter NN workload. While CNNs may be common in edge devices, the volume of convolutional models hasn’t yet caught up with MLPs and LSTMs in the datacenter. We hope that architects try to accelerate MLPs and LSTMs with at least as much gusto.

架构团体现在非常注意NN，我们很高兴看到这一点：ISCA 2016中15%的文章是NN的硬件加速器！所有的9篇文章都关注CNNs，只有2个提到了其他类型的NN。CNNs比MLPs更加复杂，在NN的竞赛中非常引人注目，这也解释了其吸引力，但是它们只占了我们数据中心中NN工作量的5%。CNNs在边缘设备中会更加常见，卷积模型在数据中心中的体量，还没有达到MLPs和LSTMs。我们希望架构师用相同的注意力来加速MLPs和LSTMs。

● Pitfall: For NN hardware, Inferences Per Second (IPS) is an inaccurate summary performance metric. 对于NN硬件，每秒推理数并不是一个精确的总结性的性能度量。

Our results show that IPS is a poor overall performance summary for NN hardware, as it’s simply the inverse of the complexity of the typical inference in the application (e.g., the number, size, and type of NN layers). For example, the TPU runs the 4-layer MLP1 at 360,000 IPS but the 89-layer CNN1 at only 4,700 IPS, so TPU IPS vary by 75X! Thus, using IPS as the single-speed summary is even more misleading for NN accelerators than MIPS or FLOPS are for regular processors [23], so IPS should be even more disparaged. To compare NN machines better, we need a benchmark suite written at a high-level to port it to the wide variety of NN architectures. Fathom is a promising new attempt at such a benchmark suite [3].

我们的结果表明，IPS对于NN硬件来说并不是一个很好的总体性能总结，因为很容易改变应用中典型的推理复杂度（如，数量，大小，和NN层的类型）。比如，TPU运行4层的MLP1的速度为360000 IPS，但对于89层的CNN1则只有4700 IPS，所以TPU IPS变化了75X。因此，使用IPS作为NN加速器的单速度总结，比使用MIPS或FLOPS作为常规处理器的指标，会更加误导人，所以不应当使用IPS。为更好的比较NN硬件，我们需要一个用更高层次写的基准测试包，以移植到更多的NN架构中。Fathom是一个很有希望的新尝试[3]。

● Fallacy: The K80 GPU results would be much better if Boost mode were enabled. 如果使用了Boost模式，K80 GPU的结果会好很多。

We didn’t use K80 Boost mode but measured its impact on LSTM1. Boost mode increased the clock rate by a factor of up to 1.6—from 560 to 875 MHz—which increased performance by 1.4X, but it also raised power by 1.3X. The net gain in performance/Watt is 1.1X, and thus Boost mode would have a minor impact on LSTM1.

我们没有使用K80的Boost模式，但用LSTM1测量了其影响。Boost模式提高了时钟速度，最多提高了1.6倍，从560提升到了875MHz，性能提升了1.4X，但能耗也提升了1.3X。每Watt性能净增值为1.1X，因此Boost模式对LSTM1的影响不会很大。

● Fallacy: CPU and GPU results would be similar to the TPU if we used them more efficiently or compared to newer versions. 如果我们更高效的使用或用新版的进行比较，CPU和GPU的结果会与TPU类似。

We originally had 8-bit results for just one DNN on the CPU, due to the significant work to use AVX2 integer support efficiently. The benefit was ~3.5X. It was less confusing (and less space) to present all CPU results in floating point, rather than having one exception, with its own roofline. If all DNNs had similar speedup, performance/Watt ratio would drop from 41X – 83X to 12X – 24X. The new 16-nm, 1.5GHz, 250W P40 GPU can perform 47 Tera 8-bit ops/sec, but wasn’t available in early 2015, so isn’t contemporary with our three platforms. We also can’t know the fraction of P40 peak delivered within our rigid time bounds. (It also doesn’t offer SECDED on internal memory, so we can’t deploy it in our datacenters.) If we compared newer chips, Section 7 shows that we could triple performance of the 28-nm, 0.7GHz, 40W TPU just by using the K80’s GDDR5 memory (at a cost of an additional 10W).

我们开始在CPU上只对一个DNN有8-bit的结果，因为高效的使用AVX2整数支持，效果真的很好。增效是~3.5X。将所有CPU结果用浮点数的形式给出，而不是有一个例外，有其自己的roofline，这样会更好一些。如果所有的DNNs有类似的加速，每Watt性能比会从41X-83X下降到12X-24X。新的16-nm, 1.5GHz, 250W P40 GPU可以达到47TOPS的8-bit性能，但在2015年早期并不可用，所以与我们的三个平台并不是同时的。我们也不能知道，P40在我们严格的时间限制内能达到的峰值。（其在内部存储中也没有实现SECDED，所以我们不能在我们的数据中心中部署。）如果我们比较更新的芯片，第7部分表明，我们在28-nm, 0.7GHz, 40W TPU上，只需要使用K80的GDDR5内存，就可以将性能提升到3倍（功耗会额外提升10W）。

● Pitfall: Performance counters added as an afterthought for NN hardware. 性能计数器是NN硬件的事后想法。

The TPU has 106 performance counters, and we would like even more (see Table 3). The raison d'etre for NN accelerators is performance, and it is way too early in their evolution to have good intuition about what is going on.

TPU有106个性能计数器，我们甚至想要更多（见表3）。NN加速器存在的理由是性能，在其演化中，要靠直觉知道在发生什么，现在还太早。

● Fallacy: After two years of software tuning, the only path left to increase TPU performance is hardware upgrades. 在2年的软件调节后，TPU性能的提升的唯一路径是硬件升级。

The performance of CNN1 on the TPU could improve if developers and compiler writers did more work to match CNN1 to the TPU hardware. For example, developers could reorganize the applications to aggregate multiple short batches out of the convolution layers into a single, deeper batch (from 32 to 128) for the four fully connected layers. Such a single layer would improve utilization of the matrix unit (see Table 3). As CNN1 currently runs more than 70 times faster on the TPU than the CPU, the CNN1 developers are already very happy, so it’s not clear whether or when such optimizations would be performed.

如果开发者和编译器开发者做更多的工作，以使CNN1与TPU硬件更加匹配，那么CNN1在TPU上的性能可以进一步改进。比如，开发者可以重新组织应用，将卷积层的短batches累积成单个更深的batch（从32到128），用于4个全连接层。这样的单个层会改进矩阵单元的利用率（见表3）。由于CNN1目前在TPU上的运行，比在CPU上快了70倍，CNN1开发者已经很开心了，所以这种优化什么时候会进行，是否会进行，我们都不清楚。

● Pitfall: Being ignorant of architecture history when designing a domain-specific architecture. 当设计一个领域专用架构时，不需要知道架构设计的历史。

Ideas that didn’t fly for general-purpose computing may be ideal for domain-specific architectures. For the TPU, three important architectural features date back to the early 1980s: systolic arrays [31], decoupled-access/execute [54], and CISC instructions [41]. The first reduced the area and power of the large matrix multiply unit, the second fetches weights concurrently during operation of the matrix multiply unit, and the third better utilizes the limited bandwidth of the PCIe bus for delivering instructions. History-aware architects could have a competitive edge.

不用于通用目标计算的思想，可能对于领域专用的架构是理想的。对于TPU，三个重要的架构特征，可以追溯到1980s早期：收缩性的阵列，访问/执行解耦，和CISC指令。第一个降低了大型矩阵乘法单元的面积和能量，第二个在矩阵乘法单元运算时，同时进行取权重，第三个更好的利用了PCIe总线的有限带宽，以传送指令。意识到历史的架构师，会更加有优势。

## 9. Related Work

Two survey articles document that custom NN ASICs go back at least 25 years [25, 4]. For example, CNAPS chips contained a 64 SIMD array of 16-bit by 8-bit multipliers, and several CNAPS chips could be connected together with a sequencer [19]. The Synapse-1 system was based upon a custom systolic multiply-accumulate chip called the MA-16, which performed sixteen 16-bit multiplies at a time [44]. The system concatenated several MA-16 chips together and had custom hardware to do activation functions.

两篇综述文章记录了，定制的NN ASICs至少可以回到25年前。比如，CNAPS芯片包含了一个64 SIMD阵列的16-bit乘以8-bit的乘法器，几个CNAPS芯片可以用一个sequencer连接到一起。Synapse-1系统是基于定制的收缩性乘法累加芯片的，称为MA-16，可以一次进行16个16-bit的乘法。系统拼接了几个MA-16的芯片，有定制的硬件来进行激活函数。

Twenty-five SPERT-II workstations, accelerated by the T0 custom ASIC, were deployed starting in 1995 to do both NN training and inference for speech recognition [5]. The 40-Mhz T0 added vector instructions to the MIPS instruction set architecture. The eight-lane vector unit could produce up to sixteen 32-bit arithmetic results per clock cycle based on 8-bit and 16-bit inputs, making it 25 times faster at inference and 20 times faster at training than a SPARC-20 workstation. They found that 16 bits were insufficient for training, so they used two 16-bit words instead, which doubled training time. To overcome that drawback, they introduced “bunches” (batches) of 32 to 1000 data sets to reduce time spent updating weights, which made it faster than training with one word but no batches.

25个SPERT-II工作站，由T0定制ASIC加速，自从1995年开始部署，进行NN训练和推理，用于语音识别。40MHz的T0给MIPS指令集架构增加了向量指令。8车道的向量单元会在每个时钟周期内，基于8-bit和16-bit输入，产生最多16个32-bit代数结果，使其比SPARC-20工作站在推理上快了25倍，训练上快了20倍。他们发现，16-bit用于训练是不够的，所以他们使用2个16-bit word，这使训练时间加倍。为克服这个缺陷，他们引入了32到1000个数据集的bunches，以降低在更新权重上花费的时间，这比用1个word但是没有batches的训练更快。

The more recent DianNao family of NN architectures minimizes memory accesses both on the chip and to external DRAM by having efficient architectural support for the memory access patterns that appear in NN applications [28, 11]. All use 16-bit integer operations and all designs synthesized down to layout, but no chips were fabricated. The original DianNao uses an array of 64 16-bit integer multiply-accumulate units with 44 KB of on-chip memory and is estimated to be 3 mm2 (65 nm), to run at 1 GHz, and to consume 0.5W. Most of this energy went to DRAM accesses for weights, so one successor DaDianNao (“big computer”) includes eDRAM to keep 36 MiB of weights on chip. The goal was to have enough memory in a multichip system to avoid external DRAM accesses. The follow-on PuDianNao (“general computer”) is aimed more at support vector machines. Another offshoot is ShiDianNao (“vision computer”) aimed at CNNs, which avoids DRAM accesses by connecting the accelerator directly to the sensor.

最近的DianNao族NN架构，最小化了片上和外部DRAM的内存访问，对于NN应用中出现的内存访问模式有高效的架构支持。它们都使用了16-bit整数运算，所有的设计都综合到layout，但并没有制造芯片。原始的DianNao使用了64个16-bit的整数乘法-累加单元，有44KB的片上内存，大约为3mm^2，运行在1GHz上，功耗0.5W。大部分能量用于权重的DRAM访问，所以一个后继者DaDianNao包含了36 MiB的片上权重。目标是在多芯片系统中有足够的内存，以避免外部DRAM的访问。后续的PuDianNao目标是更多的支持向量机器。另一个分支是ShiDianNao，目标是CNNs，通过将加速器直接连接到传感器，来避免DRAM访问。

The Convolution Engine is also focused on CNNs for image processing [43]. This design deploys 64 10-bit multiply-accumulator units and customizes a Tensilica processor estimated to run at 800 MHz in 45 nm. It is projected to be 8X to 15X more energy-area efficient than an SIMD processor, and within 2X to 3X of custom hardware designed just for a specific kernel.

卷积引擎也聚焦在用于图像处理的CNNs。这个设计部署了64个10-bit乘法累加器单元，定制了一个Tensilica处理器，估计运行在800MHz，45nm。比一个SIMD处理器在能量-面积上要高效8X-15X，对于只为一个特殊核心的定制硬件设计要高效2X-3X。

The Fathom benchmark paper seemingly reports results contradictory to ours, with the GPU running inference much faster than the CPU [3]. However, their CPU and GPU are not server-class, the CPU has only four cores, the applications do not use the CPU’s AVX instructions, and there is no response-time cutoff [8].

Fathom基准测试的文章给出的结果，似乎与我们的矛盾，GPU运行的推理比CPU要快很多。但是，其CPU和GPU并不是服务器级的，CPU只有4个核，应用没有使用CPU的AVX指令，也没有响应时间的cutoff。

Catapult [42] is the most widely deployed example of using reconfigurability to support DNNs, which many have proposed. They chose FPGAs over GPUs to reduce power as well as the risk that latency-sensitive applications wouldn’t map well to GPUs. FPGAs can also be re-purposed, such as for search, compression, and network interface cards [Put15]. The TPU project actually began with FPGAs, but we abandoned them when we saw that the FPGAs of that time were not competitive in performance compared to the GPUs of that time, and the TPU could be much lower power than GPUs while being as fast or faster, potentially making it much better than FPGAs and GPUs.

使用可重配置性来支持DNNs，Catapult是部署最广泛的例子，很多都推荐。他们选择FPGAs来降低能耗，以及对延迟敏感的应用不会很好的适应GPUs的风险。FPGAs也可以改变其目的，比如搜索，压缩，和网络接口卡。TPU项目实际上是从FPGAs开始的，但是我们看到，那时的FPGAs与那时的GPUs相比，在性能上并没有竞争力，我们就抛弃了FPGAs，而且TPU在比GPU同样快或更快的情况下，可以能耗更低，使其比FPGAs和GPUs要可能更好。

Although first published in 2014, Catapult is a TPU contemporary since it deployed 28-nm Stratix V FPGAs into datacenters concurrently with the TPU in 2015. Catapult has a 200 MHz clock, 3,926 18-bit MACs, 5 MiB of on-chip memory, 11 GB/s memory bandwidth, and uses 25 Watts. The TPU has a 700 MHz clock, 65,536 8-bit MACs, 28 MiB, 34 GB/s, and typically uses 40 Watts. A revised version of Catapult was deployed at larger scale in 2016 [9].

虽然是在2014年发表的，但Catapult是与TPU同时的，因为其部署了28-nm的Strtix V FPGAs到数据中心中，与2015年的TPU是同时的。Catapult的时钟为200MHz，有3926个18-bit的MACs，5MiB片上内存，11GB/s的内存带宽，功耗25Watts。TPU有700MHz的时钟，65536个8-bit的MACs，28MiB片上内存，34GB/s的带宽，功耗为40Watts。Catapult的一个修改版以更大的规模在2016年进行了部署。

Catapult V1 runs CNNs—using a systolic matrix multiplier—2.3X as fast as a 2.1 GHz, 16-core, dual-socket server [39]. Using the next generation of FPGAs (14-nm Arria 10) of Catapult V2, performance might go up to 7X, and perhaps even 17X with more careful floor-planning [40]. Although it’s apples versus oranges, a current TPU die runs its CNNs 40X to 70X versus a somewhat faster server (Tables 2 and 6). Perhaps the biggest difference is that to get the best performance the user must write long programs in the low-level hardware-design-language Verilog [36] versus writing short programs using the high-level TensorFlow framework. That is, reprogrammability comes from software for the TPU rather than from firmware for the FPGA.

Catapult V1使用收缩性矩阵乘法器，运行CNNs的速度与2.1 GHz, 16-core, dual-socket服务器一样快。使用下一代FPGAs (14-nm Arria 10)的Catapult V2，性能会提升到7X，如果进行更仔细的平面布置，可能会达到17X。当前的TPU芯片与一个更快的服务器相比，性能是其40X-70X。可能更大的差异是，要得到最好的性能，用户使用Catapult需要用底层的Verilog写很长的程序，而用TPU则只需要用高层的TensorFlow。因此，可重编程性对于TPU来说是来自软件，而对于FPGA来说是来自于固件。

## 10. Conclusion

Despite living on an I/O bus and having relatively low memory bandwidth that limits utilization of the TPU—four of the six NN applications are memory-bound—a small fraction of a big number can nonetheless be relatively large, as the Roofline performance model demonstrates. This result suggests a “Cornucopia Corollary” to Amdahl’s Law: low utilization of a huge, cheap resource can still deliver high, cost-effective performance.

尽管要依靠I/O总线，相对较低的内存带宽，限制了TPU的利用，6个NN应用中的4个都是受到内存限制的，大量应用中的一小部分会相对较大，Roofline性能模型这样显示。这个结果说明：大量便宜资源的较低利用率，仍然可以给出很高的性价比很好的性能。

The TPU leverages the order-of-magnitude reduction in energy and area of 8-bit integer systolic matrix multipliers over 32-bit floating-point datapaths of a K80 GPU to pack 25 times as many MACs (65,536 8-bit vs. 2,496 32-bit) and 3.5 times the on-chip memory (28 MiB vs. 8 MiB) while using less than half the power of the K80 in a relatively small die. This larger memory helps increase the operational intensity of applications to let them utilize the abundant MACs even more fully.

TPU与K80相比，8-bit整数收缩性矩阵乘法器的数量比32-bit浮点数datapaths要多25倍，片上内存是3.5X (28 MiB vs. 8 MiB)，能耗低了一半。更大的内存提升了应用的运算强度，使其更充分的利用富余的MACs。

We found that despite a recent emphasis on CNNs in the architecture community, they constitute only about 5% of the representative NN workload for our datacenters, which suggests more attention should be paid to MLPs and LSTMs. Repeating history, it’s similar to when many architects concentrated on floating-point performance when most mainstream workloads turned out to be dominated by integer operations.

我们发现，架构团体尽管强调CNNs，但它们只占我们数据中心典型NN任务的5%，这说明应该更加关注MLPs和LSTMs。历史上有重复，当大多数架构师都关注浮点数的性能时，其实主要的工作量在于整数运算中。

We observed that inferences per second (IPS) is more a function of the NN than of the underlying hardware, and so IPS is an even worse single performance metric for NN processors than MIPS and MFLOPS are for CPUs and GPUs.

我们观察到，IPS更多的是NN的功能，而不是底层硬件的功能，所以IPS作为NN处理器的性能指标，比MIPS和MFLOPS作为CPUs和GPUs的指标还要差。

We also learned that inference applications have serious response-time bounds because they are often part of user facing applications, thus NN architectures need to perform well when coping with 99th-percentile latency deadlines. While the K80 may excel at training, on average it is just a little faster than Haswell at inference for our workload, perhaps because of its emphasis on throughput rather than latency; that conflicts with the strict response-time deadlines of our inference applications.

我们还学习到，推理应用有很强的响应时间需求，因为通常都是面向用户的应用，因此NN架构在处理99th百分位延迟的deadlines时，需要处理的很好。K80在训练上性能很好，对我们的工作任务，在推理时只比Haswell性能好了一点点，可能是因为只强调了吞吐量，而不是延迟；这与推理应用对响应时间严格的要求相矛盾。

The TPU die leverages its advantage in MACs and on-chip memory to run short programs written using the domain-specific TensorFlow framework 15 times as fast as the K80 GPU die, resulting in a performance per Watt advantage of 29 times, which is correlated with performance per total cost of ownership. Compared to the Haswell CPU die, the corresponding ratios are 29 and 83. While future CPUs and GPUs will surely run inference faster, a redesigned TPU using circa 2015 GPU memory would go two to three times as fast and boost the performance/Watt advantage to nearly 70 over the K80 and 200 over Haswell.

TPU芯片利用其在MACs和片上内存的优势，运行用领域专用的TensorFlow框架写的短程序，速度是K80芯片的最快15倍，每Watt性能比为29倍。与Haswell CPU芯片相比，对应的比率为29倍和83倍。未来的CPUs和GPUs会更快的运行推理，但重新设计的TPU，使用circa 2015 GPU内存，会将速度提升快2-3倍，将每Watt性能的优势提升到接近对K80的70倍和对Haswell的200倍。

In summary, the TPU succeeded because of the large—but not too large—matrix multiply unit; the substantial software-controlled on-chip memory; the ability to run whole inference models to reduce dependence on its host CPU; a single-threaded, deterministic execution model that proved to be a good match to 99th-percentile response time limits; enough flexibility to match the NNs of 2017 as well as of 2013; the omission of general-purpose features that enabled a small and low power die despite the larger datapath and memory; the use of 8-bit integers by the quantized applications; and that applications were written using TensorFlow, which made it easy to port them to the TPU at high-performance rather than them having to be rewritten to run well on the very different TPU hardware.

总结起来，TPU成功是因为很大的矩阵乘法单元，但是不是太大；很大的软件控制的片上内存；可以运行整个推理模型的能力，以降低对宿主CPU的依赖性；单线程，确定性的执行模型，可以很好的匹配99th百分位的响应时间限制；足够的灵活性，可以匹配2017年的NNs，以及2013年的NNs；忽略了通用目标的特征，使得面积小低能耗的芯片与更大的datapath和内存成为可能；量化应用使用8-bit的整数运算；应用是用TensorFlow写的，使其很容易以高性能迁移到TPU中，而不需要重新写代码，以在非常不同的TPU硬件中运行。

Order-of-magnitude differences between products are rare in computer architecture, which may lead to the TPU becoming an archetype for domain-specific architectures. We expect that many will build successors that will raise the bar even higher.

产品之间几个量级之间的差距，在计算机架构中是少见的，这会导致TPU成为领域专用架构的典型。我们期望有更多的后继者。