# NAAS: Neural Accelerator Architecture Search

Yujun Lin, Mengtian Yang, Song Han @ MIT & SJTU

## 0. Abstract

Data-driven, automatic design space exploration of neural accelerator architecture is desirable for specialization and productivity. Previous frameworks focus on sizing the numerical architectural hyper-parameters while neglect searching the PE connectivities and compiler mappings. To tackle this challenge, we propose Neural Accelerator Architecture Search (NAAS) that holistically searches the neural network architecture, accelerator architecture and compiler mapping in one optimization loop. NAAS composes highly matched architectures together with efficient mapping. As a data-driven approach, NAAS rivals the human design Eyeriss by 4.4× EDP reduction with 2.7% accuracy improvement on ImageNet under the same computation resource, and offers 1.4× to 3.5× EDP reduction than only sizing the architectural hyper-parameters.

神经加速器架构的数据驱动的自动设计空间探索，对专用性和生产力来说非常理想。之前的框架聚焦在确定数值架构超参数的大小，而忽视搜索PE的连接性和编译器的映射。为处理这个挑战，我们提出了神经加速器架构搜索(NAAS)，在一个优化循环中全面搜索神经网络架构，加速器架构和编译器映射。NAAS由高度匹配的架构，和高效的映射组成。作为一个数据驱动的方法，NAAS超过了人类设计Eyeriss，EDP降低了4.4x，在相同的计算资源下，在ImageNet上的准确率提高了2.7%，与只确定架构超参数的方法比，EDP降低了1.4x到3.5x。

## 1. Introduction

Neural architecture and accelerator architecture co-design is important to enable specialization and acceleration. It covers three aspects: designing the neural network, designing the accelerator, and the compiler that maps the model on the accelerator. The design space of each dimension is listed in Table I, with more than 10^800 choices for a 50-layer neural network. Given the huge design space, data-driven approach is desirable, where new architecture design evolves as new designs and rewards are collected. Recent work on hardware-aware neural architecture search (NAS) and auto compiler optimization have successfully leverage machine learning algorithms to automatically explore the design space. However, these works only focuses on off-the-shelf hardware [1]–[6], and neglect the freedom in the hardware design space.

神经架构和加速器架构的联合设计非常重要，可以实现专用和加速。这覆盖了三个方面：设计神经网络，设计加速器，和编译器，将模型映射到加速器上。每个维度的设计空间列在表I中，对于一个50层的神经网络来讲，有超过10^800个选择。由于设计空间巨大，数据驱动的方法是很理想的，其中新的架构设计演化为新的设计，并收集其回报。最近在与硬件相关的神经架构搜索(NAS)和自动编译器优化上的工作，已经成功的利用了机器学习算法，来自动探索设计空间。但是，这些工作只关注了开箱即用的硬件[1-6]，忽略了硬件设计空间的自由度。

Table I: Nerual-Accelerator architechure search space

Aspect | List

--- | ---

Accelerator | Compute Array Size (#rows#columns); (Input/Weight/Output) Buffer Size; PE Inter-connection (Dataflow)

Compiler Mapping | Loop Order, Loop Tiling Sizes

Neural Network | #Layers, #Channels, Kernel Size, Block Structure, Input Size

The interactions between the neural architecture and the accelerator architecture is illustrated in Table II. The correlations are complicated and vary from hardware to hardware: for instance, tiled input channels should be multiples of #rows of compute array in NVDLA, while #rows is related to the kernel size in Eyeriss. It is important to consider all the correlations and make them fit. A tuple of perfectly matched neural architecture, accelerator architecture, and mapping strategy will improve the utilization of the compute array and on-chip memory, maximizing efficiency and performance.

神经架构和加速器架构的相互作用如表II所示。其关系很复杂，随着硬件的不同而不同：比如，在NVDLA中，tiled输入通道应道是计算阵列的#rows的倍数，而在Eyeriss中，#rows与核的大小有关。考虑所有的关系，使其匹配，这非常重要。完美匹配的神经架构，加速器架构和映射策略的组合，会改进计算阵列和片上存储的利用，最大化效率和性能。

Table II: The complicated correlation between nerual and accelerator design space. It differs from accelerator to accelerator: N is NVDLA and E is Eyeriss.

The potential of exploring both neural and accelerator architecture has been proven on FPGA platforms [7]–[10] where HLS is applied to generate FPGA accelerator. Earlier work on accelerator architecture search [11]–[13] only search the architectural sizing while neglecting the PE connectivity (e.g., array shape and parallel dimensions) and compiler mappings, which impact the hardware efficiency.

对神经架构和加速器架构进行共同探索的潜力，在FPGA平台上已经证明了[7-10]，其中使用了HLS来生成FPGA加速器。更早的在加速器架构搜索[11-13]上的工作，只搜索了架构的大小，而忽视了PE连接性（如，阵列的形状和并行的维度）和编译器的设置，这影响了硬件效率。

We push beyond searching only hardware hyper-parameters and propose the Neural Accelerator Architecture Search (NAAS), which fully exploits the hardware design space and compiler mapping strategies at the same time. Unlike prior work [11] which formulate the hardware parameter search as a pure sizing optimization, NAAS models the co-search as a two-level optimization problem, where each level is a combination of indexing, ordering and sizing optimization. To tackle such challenges, we propose an encoding method which is able to encode the non-numerical parameters such as loop order and parallel dimension chosen as numerical parameters for optimization. As shown in Figure 1, the outer loop of NAAS optimizes the accelerator architecture while the inner loop optimizes the compiler mapping strategies.

我们不仅搜索硬件的超参数，提出NAAS，同时完全利用了硬件设计空间和编译映射策略。之前的工作[11]将硬件参数搜索表述为纯大小优化，与之不同的是，NAAS将联合搜索建模为一个两级优化问题，其中每个层级都是索引，排序和大小优化的组合。为处理这样的挑战，我们提出一种编码方法，可以将非数值参数，比如循环顺序和并行维度，编码为数值参数进行优化。如图1所示，NAAS的外层循环优化加速器架构，而内部循环优化编译器映射策略。

Combining both spaces greatly enlarges the optimization space: within the same #PEs and on-chip memory resources as EdgeTPU there are at least 10^11 hardware candidates and 10^17 mapping candidates for each layer, which composes 10^(11+50·17) = 10861 possible combinations in the joint search space for ResNet-50, while there are only 104 hardware candidates in NASAIC’s design space. To efficiently search over the large design space, NAAS leverages the biologically-inspired evolution-based algorithm rather than meta-controller-based algorithm to improve the sample efficiency. It keeps improving the quality of the candidate population by ruling out the inferior and generating from the fittest. Thanks to the low search cost, NAAS can be easily integrated with hardware-aware NAS algorithm by adding another optimization level (Figure 1), achieving the joint search.

将两个空间结合在一起，极大的增大了优化空间：在相同的#PEs和片上存储资源下，如EdgeTPU，对每一层都至少有10^11个硬件候选和10^17个映射候选，对ResNet-50来说，这在联合搜索空间中构成了10^(11+50·17) = 10861个可能的组合，而在NASAIC的设计空间中，只有104个硬件候选。为高效的搜索大型设计空间，NAAS利用了生物启发的基于演化的算法，而不是基于元控制器的算法，来改进样本效率。算法持续改进候选种群的质量，将较低质量的排除掉，从最优的中生成子代。NAAS的搜索代价很低，可以很容易的与硬件NAS算法集成到一起，加入另一个优化层次（图1），得到联合搜索。

Extensive experiments verify the effectiveness of our framework. Under the same #PE and on-chip memory constraints, the NAAS is able to deliver 2.6×, 4.4× speedup and 2.1×, 1.4× energy savings on average compared to Eyeriss [14], NVDLA [15] design respectively. Integrated with Once-For-All NAS algorithm [4], NAAS further improves the top-1 accuracy on ImageNet by 2.7% without hurting the hardware performance. Using the similar compute resources, NAAS achieves 3.0×, 1.9× EDP improvements compared to Neural-Hardware Architecture Search [12], and NASAIC [11] respectively.

大量试验验证了我们框架的有效性。在相同的#PE和片上存储约束下，NAAS与Eyeriss和NVDLA的设计相比，分别可以得到平均2.6x，4.4x的加速，和2.1x，1.4x的能量节省。与Once-For-All NAS算法相结合，NAAS进一步将ImageNet的top-1准确率提高了2.7%，而没有降低硬件性能。在类似的计算资源下，NAAS与Neural-Hardware Architecture Search [12], and NASAIC [11]相比，分别获得了3.0x，1.9x的EDP改进。

## 2. Nerual Accelerator Architechture Search

Figure 1 shows the optimization flow of Neural Accelerator Architecture Search (NAAS). NAAS explores the design space of accelerators, and compiler’s mappings simultaneously.

图1展示了NAAS的优化流程。NAAS同时探索了加速器的设计空间和编译器的映射。

### 2.1. Accelerator Architecture Search

**a) Design Space**: The accelerator design knobs can be categorized into two classes:

设计空间：加速器设计的点，可以分类下面两类：

1) Architectural Sizing: the number of processing elements (#PEs), private scratch pad size (L1 size), global buffer size (L2 size), and memory bandwidth.

架构大小的确定：PE的数量，私有便签的大小，全局缓冲区大小，和内存带宽。

2) Connectivity Parameters: the number of array dimensions (1D, 2D or 3D array), array size at each dimension, and the inter-PE connections.

连接性参数：阵列维度(1D, 2D或3D阵列)，每个维度的阵列大小，和PE间连接。

Most state-of-art searching frameworks only contains architectural sizing parameters in their design space. These sizing parameters are numerical and can be easily embedded into vectors during search. On the other hand, PE connectivity is difficult to encode as vectors since they are not numerical numbers. Moreover, changing the connectivity requires re-designing the compiler mapping strategies, which extremely increase the searching cost. In NAAS, besides the architectural sizing parameters which are common in other frameworks, we introduce the connectivity parameters into our search space, making it possible to search among 1D, 2D and 3D array as well, and thus our design space includes almost the entire accelerator design space for neural network accelerators.

多数目前最好的搜索框架在其设计空间中只包含架构大小参数。这些大小参数都是数值的，可以在搜索的时候很容易的嵌入到向量中。另一方面，PE连接性很难编码为向量，因为这并不是数值量。而且，改变连接性意味着重新设计编译器映射策略，这极大的增加了搜索代价。在NAAS中，除了在其他框架中常见的架构大小参数，我们还引入了连接性参数到搜索空间中，使其可以在1D，2D和3D阵列中搜索，因此我们的设计空间几乎包括神经网络加速器的整个加速器设计空间。

**b) Encoding**: We first model the PE connectivity as the choices of parallel dimensions. For example, parallelism in input channels (C) means a reduction connection of the partial sum register inside each PE. Parallelism in output channels means a broadcast to input feature register inside each PE. The most straight-forward method to encode the parallel dimension choice is to enumerate all possible parallelism situations and choose the index of the enumeration as the encoding value. However, since the increment or decrement of indexes does not convey any physical information, it is hard to be optimized.

编码：我们首先将PE连接性建模为并行维度的选择。比如，输入通道中的并行性C，意味着在每个PE中部分和寄存器的缩减连接。输出通道中的并行性，意味着在每个PE中广播到输入特征寄存器。编码并行维度选择的最直接的方法，是枚举所有的并行性情况，选择枚举的索引作为编码值。但是，由于索引的递增或递减不会传递任何物理信息，很难进行优化。

To solve this problem, we proposed the “importance-based” encoding method for choosing parallelism dimensions in the dataflow and convert the indexing optimization into the sizing optimization. For each dimension, our optimizer will generate an importance value. To get the corresponding parallel dimensions, we first collect all the importance value, then sort them in decreasing order, and select the first k dimensions as the parallel dimensions of a k-D compute array. As shown in the left of Figure 3, the generated candidate is a 2D array with size 16 × 16. To find the parallel dimension for this 2D array candidate, The importance values are first generated for 6 dimensions in the same way as other numerical parameters in the encoding vector. We then sort the value in decreasing order and determine the new order of the dimensions. Since the importance value of “C” and “K” are the largest two value, we finally select “C” and “K” as the parallel dimensions of this 2D array. The importance value of the dimension represents the priority of the parallelism: a larger value indicates a higher priority and a higher possibility to be paralleled in the computation loop nest, which contains higher relativity with accelerator design compared to indexes of enumerations.

为解决这个问题，我们提出基于重要性的编码方法，在数据流中选择并行性维度，将索引优化转化成大小优化问题。对每个维度，我们的优化器会生成一个重要性值。为得到相应的并行维度，我们首先收集所有的重要性值，以降序进行排列，选择前k个维度作为k-D计算阵列的并行维度。如图3左所示，生成的候选是2D阵列，大小16 × 16。为找到这个2D阵列候选的并行维度，首先对6维生成重要性值，与编码向量中的其他数值参数一样。我们然后将值按照降序进行排列，确定新的维度顺序。由于C和K的重要性值是最大的两个值，我们最后选择C和K作为这个2D阵列的并行维度。维度的重要性值表示并行性的优先级：更大的值表示更高的优先级，以及在计算循环嵌套中并行化更高的可能性，与枚举的索引相比，这对加速器设计的相关性要更高一些。

For other numerical parameters, we use the straight-forward encoding method. The whole hardware encoding vector is shown in Figure 2, which contains all of the necessary parameters to represent an accelerator design paradigm.

对其他的数值参数，我们使用直接编码的方法。整体硬件编码向量如图2所示，包含了所有必须的参数，来表示一个加速器设计范式。

**c) Evolution Search**: We leverage the evolution strategy [17] to find the best solution during the exploration. In order to take both latency and energy into consideration, we choose the widely used metric Energy-Delay Product (EDP) to evaluate a given accelerator configuration on a specific neural network workload. At each evolution iteration, we first sample a set of candidates according to a multivariate normal distribution in [0, 1]^|θ|. Each candidate is represented as a |θ|-dimension vector. These candidates are then projected to hardware encoding vectors and decoded into accelerator design. We rule out the invalid accelerator samples and keep sampling until the candidate set reaches a predefined size (population size) in our experiments. To evaluate the candidate performance, we need to perform mapping strategy search in Section II-B on each benchmark and adopt the best searched result as the EDP reward of this candidate. After evaluating all the candidates on the benchmarks, we update the sampling distribution based on the relative ordering of their EDP. Specifically, we select the top solutions as the “parents” of the next generation and use their center to generate the new mean of the sampling distribution. We update the covariance matrix of the distribution to increase the likelihood of generating samples near the parents [17]. We then repeat such sampling and updating process.

演化搜索：我们利用演化策略来在探索中找到最佳解。为将延迟和能量都进行考虑，我们选择广泛使用的度量EDP来对特定的神经网络workload评估给定的加速器配置。在每次演化迭代中，我们首先根据多变量正态分布[0, 1]^|θ|采样候选集。每个候选都表示为一个|θ|维向量。这些候选然后投影到硬件编码向量，解码到加速器设计中。我们剔除掉无效的加速器样本，继续采样，直到候选集达到我们试验中预定义的大小（种群大小）。为评估候选性能，我们需要在每个基准测试上进行2.2中的映射策略搜索，采用搜索到的最好结果，作为这个候选的EDP回报。在基准测试上评估了所有的候选后，我们基于这些EDP的相对顺序，更新采样分布。具体的，我们选择最好的解作为下一次生成的父辈，用其中心来生成新的采样分布的均值。我们更新分布的协方差矩阵，以增加生成接近父辈的样本的可能性。然后我们重复这样的采样过程和更新过程。

Figure 4 shows the statistics of energy-delay products of hardware candidates in the population. As the optimization continues, the EDP mean of NAAS candidates decreases while that of random search remains high, which indicates that NAAS gradually improves the range of hardware selections.

图4展示了种群中硬件候选的EDP的统计值。随着优化过程的进行，NAAS候选的EDP均值降低了，而随机搜索的值则保持很高，这说明NAAS逐渐的改进了硬件选择的范围。

### 2.2. Compiler Mapping Strategy Search

The performance and energy efficiency of deep learning accelerators also depend on how to map the neural network task on the accelerator. The search space of compiler mapping strategy is much larger than accelerator design, since different convolution layers may not share the same optimal mapping strategy. Hence we optimize the mapping for each layer independently using the similar evolution-based search algorithm to accelerator design search in Section II-A0c.

深度学习加速器的性能和功耗效率，还依赖于怎样将神经网络任务映射到加速器上。编译器映射策略的搜索空间比加速器设计要大的多，因为不同的卷积层的最优映射策略可能不一样。因此，我们对每一层进行独立的映射最优化，使用的是与加速器设计搜索类似的基于演化的搜索算法。

The compiler mapping strategy consists of two components: the execution order and the tiling size of each for-loop dimension. Similar to the accelerator design search, the order of for-loop dimensions is non-trivial. Rather than enumerating all of the possible execution orders and using indexes as encoding, we use the similar “importance-based” encoding methods in Section II-A0b. For each dimension of the array (corresponding to each level of for-loop nests), the mapping optimizer will assign each convolution dimension with an importance value. The dimension with the highest importance will become the outermost loop while the one with the lowest importance will be placed at the innermost in the loop nests. The right of Figure 3 gives an example. The optimizer firstly generates the importance values for 6 dimensions, then sort the value in decreasing order and determine the corresponding order of the dimensions. Since “C” and “R” dimension have largest value 5, they will become the outermost loops. “S” dimension has the smallest value 1, so it is the innermost dimension in the loops. This strategy is interpretable, since the importance value represents the data locality of the dimension: the dimension labeled as most important has the best data locality since it is the outermost loop, while the dimension labeled as least important has the poorest data locality therefore it is the innermost loop.

编译器映射策略由两个部分组成：执行顺序，和每个for循环维度的tiling大小。与加速器设计搜索类似，for循环维度的顺序也是有意义的。我们没有枚举所有可能的执行顺序，使用索引作为编码，而是使用了与上节类似的基于重要性的编码方法。对于阵列的每个维度（对应着for循环嵌套的每个层次），映射优化器会对每个卷积维度指定一个重要性值。最高重要性的维度会变成最外层的循环，而最低重要性的维度，会被放到循环嵌套中最内层。图3的右边给出了一个例子。优化器首先对6个维度生成重要性值，然后按照降序排列，确定维度的对应顺序。由于C和R维度的值最大，为5，它们会成为最外层的循环。S维度的值最小，为1，所以这是循环的最内层维度。这种策略是可解释的，因为重要性值表示维度的数据局部性：标记为最重要的维度，有最好的数据局部性，因为这是最外层的循环，而标记为最不重要的维度，其数据局部性最差，因此这是最内层的循环。

As for tiling sizes, since they are highly related to the network parameters, we use the scaling ratio rather than the absolute tiling value. Hence, the tiling sizes are still numerical parameters and able to adapt to different networks. The right part of figure 3 illustrates the composition of the mapping encoding vector. Note that for PE level we need to ensure that there is only one MAC in a PE, so we only search the loop order at PE level. For each array level, the encoding vector contains both the execution order and tiling size for each for-loop dimension.

至于tiling大小，因为这与网络参数高度相关，我们使用缩放比，而不使用绝对的tiling值。因此，tiling大小仍然是数值参数，可以适应不同的网络。图3的右边，给出了映射编码向量的组成。注意，对于PE的层次，我们需要确保，在一个PE中只有一个MAC，所以我们只在PE层次搜索循环顺序。对每个阵列层次，编码向量包含每个for循环维度的执行顺序和tiling大小。

### 2.3. Integrated with Neural Architecture Search

Thanks to the low search cost, we can integrate our framework with neural architecture search to achieve neural-accelerator-compiler co-design. Figure 1 illustrates integrating NAAS with NAS. The joint design space is huge, and in order to improve the search efficiency, we choose and adapt Once-For-All NAS algorithm for NAAS. First, NAAS generates a pool of accelerator candidates. For each accelerator candidate, we sample a network candidate from NAS framework which satisfies the pre-defined accuracy requirement. Since each subnet of Once-For-All network is well trained, the accuracy evaluation is fast. We then apply the compiler mapping strategy search for the network candidate on the corresponding accelerator candidate. NAS optimizer will update using the searched EDP as a reward. Until NAS optimizer reaches its iteration limitations, and feedback the EDP of the best network candidate to accelerator design optimizer. We repeated the process until the best-fitted design is found. In the end, we obtain a tuple of matched accelerator, neural network, and its mapping strategy with guaranteed accuracy and lowest EDP.

由于搜索成本很低，我们可以将我们的框架与神经架构搜索集成到一起，得到神经网络-加速器-编译器的联合设计。图1描述了将NAAS与NAS集成到一起。联合设计空间是巨大的，为改进搜索效率，我们对NAAS选择了Once-For-All算法并进行了修改。首先，NAAS生成了加速器候选池。对每个加速器候选，我们从NAS框架中得到一个网络候选样本，要满足预定义的准确率要求。由于Once-For-All网络的每个子网络都是训练好的，准确率评估是很快的。我们然后对网络候选，在对应的加速器候选上，应用编译器映射策略搜索。NAS优化器会使用搜索得到的EDP作为回报进行更新。直到NAS优化器达到其迭代限制，将最好的候选网络的EDP反馈给加速器设计优化器。我们重复这个过程，直到找到最适应的设计。最后，我们得到匹配的加速器，神经网络，和其映射策略的组合，有保证的准确率，和最低的EDP。

## 3. Evaluation

We evaluate Neural Accelerator Architecture Search’s performance improvement step by step: 1) the improvement from applying NAAS given the same hardware resource; 2) performance of NAAS which integrates the Once-For-All to achieve the neural-accelerator-compiler co-design.

我们逐步评估NAAS的性能改进：1)在给定相同硬件资源下，应用NAAS的改进；2)NAAS集成了Once-For-All后，达到神经网络-加速器-编译器的联合设计后的性能。

### 3.1. Evaluation Environment

**a) Design Space of NAAS**: We select four different resource constraints based on EdgeTPU, NVDLA [15], Eyeriss [14] and Shidiannao [18]. When comparing to each baseline architecture, NAAS is conducted within corresponding computation resource constraint including the maximum #PEs, the maximum total on-chip memory size, and the NoC bandwidth. NAAS searches #PEs at stride of 8, buffer sizes at stride of 16B, array sizes at stride of 2.

NAAS的设计空间：我们选择四种不同的资源限制，基于EdgeTPU，NVDLA，Eyeriss和Shidiannao。当与每种基准架构比较时，NAAS都在对应的计算资源约束下进行，包括最大#PE，最大总计片上内存，和NoC带宽。NAAS搜索#PEs时步长为8，搜索缓冲区大小时步长为16B，搜索阵列大小时步长为2。

**b) CNN Benchmarks**: We select 6 widely-used CNN models as our benchmarks. The benchmarks are divided into two sets: classic large-scale networks (VGG16, ResNet50, UNet) and light-weight efficient mobile networks (MobileNetV2, SqueezeNet, MNasNet). Five deployment scenarios are divided accordingly: we conduct NAAS for large models with more hardware resources (EdgeTPU, NVDLA with 1024 PEs), and for small models with limited hardware resources (ShiDianNao, Eyeriss, and NVDLA with 256 PEs).

CNN基准测试：我们选择6个广泛使用的CNN模型作为基准测试。基准测试分为两个集合：经典的大规模网络(VGG16, ResNet50, UNet)和轻量级高效移动网络(MobileNetV2, SqueezeNet, MNasNet)。相应的分成5种部署场景：我们对大模型用更多的硬件资源进行NAAS(EdgeTPU, 1024 PEs的NVDLA)，对小模型使用有限的硬件资源(ShiDianNao, Eyeriss, 256 PEs的NVDLA)。

**c) Design Space in Once-For-All NAS**: When integrating with Once-For-All NAS, the neural architecture space is modified from ResNet-50 design space following the open-sourced library [4]. There are 3 width multiplier choices (0.65, 0.8, 1.0) and 18 residual blocks at maximum, where each block consists of three convolutions and has 3 reduction ratios (0.2, 0.25, 0.35). Input image size ranges from 128 to 256 at stride of 16. In total there are 1013 possible neural architectures.

Once-For-All NAS的设计空间：当与Once-For-All NAS集成到一起时，神经架构空间按照开源库[4]从ResNet-50的设计空间进行修改。有3种宽度乘子选项(0.65, 0.8, 1.0)，和最多18个残差模块，每个模块有3个卷积，有3个缩减率(0.2, 0.25, 0.35)。输入图像大小从128到256，步长16。总计有1013种可能的神经架构。

### 3.2. Improvement from NAAS

Figure 5 shows the speedup and energy savings of NAAS using the same hardware resources compared to the baseline architectures. When running large-scale models, NAAS delivers 2.6×, 2.2× speedup and 1.1×, 1.1× energy savings on average compared to EdgeTPU and NVDLA-1024. Though NAAS tries to provide a balanced performance on all benchmarks by using geomean EDP as reward, VGG16 workload sees the highest gains from NAAS. When inferencing light-weight models, NAAS achieves 4.4×, 1.7×, 4.4× speedup and 2.1×, 1.4×, 4.9× energy savings on average compared to Eyeriss, NVDLA-256, and ShiDianNao. Similar to searching for large models, different models obtain different benefits from NAAS under different resource constraints.

图5给出了NAAS与基准架构相比，在使用相同的硬件资源下，加速和能耗的节省。在运行大模型时，NAAS与EdgeTPU和NVDLA-1024相比，平均给出了2.6x，2.2x的加速，和1.1x，1.1x的能耗节省。虽然NAAS试图在所有基准测试上给出均衡的性能，但是在VGG16上NAAS得到的收益最多。当推理轻量级模型时，NAAS与Eyeriss，NVDLA-256和ShiDianNao相比，有4.4x，1.7x，4.4x的加速，和2.1x，1.4x，4.9x的功耗节省。与搜索大模型类似，不同的模型在不同的资源限制下，从NAAS中受益不同。

Figure 7 demonstrates three examples of searched architectures. When given different computation resources, for different NN models, NAAS provides different solutions beyond numerical design parameter tuning. Different dataflow parallelisms determine the different PE micro-architecture and thus PE connectivies and even feature/weight/partial-sum buffer placement.

图7展示了搜索到的架构的三个例子。当给定不同的计算资源时，对不同的NN模型，NAAS给出不同的解，不止包括数值设计参数的调整。不同的数据流并行性决定了不同的PE微架构，和PE连接性，甚至是特征/权重/部分和缓冲区布置。

Figure 8 illustrates the benefits of searching connectivity parameters and mapping strategy compared to searching architectural sizing only. NAAS outperforms architectural sizing search by 3.52×, 1.42×EDP reduction on VGG and MobileNetV2 within EdgeTPU resources, as well as 2.61×, 1.62×improvement under NVDLA-1024 resources.

图8给出了搜索连接性参数和映射策略的收益，与只搜索架构大小相比的结果。NAAS与架构大小搜索相比，在EdgeTPU的资源下，在VGG和MobileNetV2中，EDP下降为其3.52x，1.42x，在NVDLA-1024的资源下，比值为2.61x和1.62x。

Figure 9 further shows the EDP reduction using different encoding methods for non-numerical parameters in hardware and mapping encoding vectors. Our proposed importance-based encoding method significantly improves the performance of optimization by reducing EDP from 1.4×to 7.4×.

图9进一步展示了，对硬件中的非数值参数和映射编码向量，使用不同的编码方法下的EDP下降。我们提出的基于重要性的编码方法显著改进了优化性能，EDP下降达到了1.4x到7.4x。

### 3.3. More Improvement from Integrating NAS

Different from accelerator architectures, neural architectures have much more knobs to tune (e.g., network depths, channel numbers, input image sizes), providing us with more room to optimize. To illustrate the benefit of NAAS with NAS, we evaluate on ResNet50 with hardware resources similar to Eyeriss. Figure 10 shows that NAAS (accelerator only) outperforms Neural-Hardware Architecture Search (NHAS) [12] (which only searches the neural architecture and the accelerator architectural sizing) by 3.01× EDP improvement. By integrating with neural architecture search, NAAS achieves 4.88× EDP improvement in total as well as 2.7% top-1 accuracy improvement on ImageNet dataset than Eyeriss running ResNet50.

与加速器架构不同，神经网络架构可以调节的点更多（如，网络深度，通道数量，输入图像大小），可以优化的空间更大。为展示NAAS与NAS集成的优势，我们在ResNet50和与Eyeriss类似的硬件资源下进行评估。图10展示了NAAS（只搜索架构）超过了NHAS[12]（只搜索神经网络架构和加速器架构的大小），EDP改进了3.01x。NAAS与NAS集成到一起，与Eyeriss运行ResNet50相比，得到了4.88x的EDP改进，而且ImageNet top-1准确率改进了2.7%。

**a) Comparison to NASAIC**: We also compare our NAAS with previous work NASAIC [11] in Table III. NASAIC adopts DLA [15], ShiDianNao [18] as source architectures for its heterogeneous accelerator, and only searches the allocation of #PEs and NoC bandwidth. In contrast, we explore more possible improvements by adapting both accelerator design searching and mapping strategy searching. Inferencing the same network searched by NASAIC, NAAS outperforms NASAIC by 1.88× EDP improvement (3.75× latency improvement with double energy cost) using the same design constraints.

与NASAIC的比较：我们还将NAAS与之前的工作NASAIC[11]进行比较，如表III所示。NASAIC对其异质加速器采用DLA[15]，ShiDianNao[18]作为资源架构，只搜索#PEs和NoC带宽的分配。比较之下，我们探索更多可能的改进，修改加速器设计搜索和映射策略搜索。NASAIC搜索得到的结果对相同的网络进行推理，NAAS在相同的设计约束下，EDP比NASAIC改进了1.88x（延迟改进了3.75x，功耗为2x）。

**b) Search Cost**: Table IV reports the search cost of our NAAS compared to NASAIC and NHAS when developing accelerator and network for N development scenarios, where “AWS Cost” is calculated based on the price of on-demand P3.16xlarge instances, and “CO2 Emission” is calculated based on Strubell et al. [19]. NASAIC relies on a meta-controller-based search algorithm which requires training every neural architecture candidates from scratch. NHAS [12] decouples training and searching in NAS but it also requires retraining the searched network for each deployment which costs 16N GPU days. The high sample efficiency of NAAS makes it possible to integrate the Once-For-All NAS in one search loop. As a conservative estimation, NAAS saves more than 120× search cost compared to NASAIC on ImageNet.

搜索代价：表IV给出了NAAS与NASAIC和NHAS相比，在N个开发场景下，开发加速器和网络的搜索代价，AWS价格的计算是基于按需的P3.16xlarge的情况，CO2 Emission的计算是基于Strubell等[19]。NASAIC是基于元控制器的搜索算法，需要从头训练每个神经网络架构候选。NHAS[12]在NAS中将训练和搜索分离，但还需要对搜索到的网络进行重新训练然后进行部署，需要16N GPU天。NAAS的样本效率很高，使其能够与Once-For-All NAS在一个搜索循环中集成。保守估计的话，NAAS与NASAIC相比，节省了超过120x的搜索代价。

## 4. Related Works

**a) Accelerator Design-Space Exploration**: Ealier work focuses on fine-grained hardware resource assignment for deployment on FPGAs [7], [10], [13], [20]–[22]. Several work focuses on co-designing neural architectures and ASIC accelerators. Yang et al. [11] (NASAIC) devise a controller that simultaneously predicts neural architectures as well as the selection policy of various IPs in a heterogeneous accelerator. Lin et al. [12] focuses on optimizing the micro architecture parameters such as the array size and buffer size for given accelerator design while searching the quantized neural architecture. Besides, some work focuses on optimizing compiler mapping strategy on fixed architectures. Chen et al. [23] (TVM) designed a searching framework for optimizing for-loop execution on CPU/GPU or other fixed platforms. Mu et al. [24] proposed a new searching algorithm for tuning mapping strategy on fixed GPU architecture. On the contrary, our work explores not only the sizing parameters but also the connectivity parameters and the compiler mapping strategy. We also explore the neural architecture space to further improve the performance. Plenty of work provides the modeling platform for design space exploration [25]–[28]. We choose MAESTRO [28] as the accelerator evaluation backend.

加速器设计空间探索：早期的工作聚焦在FPGAs上部署的细粒度硬件资源指定。有几个工作聚焦在联合设计神经网络架构和ASIC加速器。Yang等(NASAIC)[11] 设计出了一种控制器，在异质加速器中，同时预测神经架构，以及选择各种IPs的策略。Lin等[12]聚焦在对给定的加速器设计优化微架构参数，比如阵列大小，和缓冲区大小，搜索量化的神经网络架构。除此以外，一些工作聚焦在在固定的架构中优化编译器映射策略。Chen等(TVM)[23]设计了一种搜索框架，在CPU/GPU或其他固定的平台上优化for循环的执行。Mu等[24]提出了一个新的搜索算法，在固定的GPU架构上调节映射策略。我们的工作不止探索了大小的参数，还有连接性的参数，和编译器映射策略。我们还探索了神经架构空间，以进一步改进性能。很多工作为设计空间探索提供了建模的平台[25-28]。我们选择MAESTRO[28]作为加速器评估的后端。

**b) AutoML and Hardware-Aware NAS**: Researchers have looked to automate the neural network design using AutoML. Grid search [28], [29] and reinforcement learning with meta-controller [2], [11], [30] both suffer from prohibitive search cost. One-shot-network-based frameworks [1], [3], [31] achieved high performance at a relatively low search cost. These NAS algorithms require retraining the searched networks while Cai et al. [4] proposed Once-For-All network of which the subnets are well trained and can be directly extracted for deployment. Recent neural architecture search (NAS) frameworks started to incorporate the hardware into the search feedback loop [1]–[4], [8], [9], though they have not explored hardware accelerator optimization.

AutoML和硬件敏感的NAS：研究者使用AutoML来自动化神经网络的设计。网格搜索和带有元控制器的强化学习的搜索代价都非常高。基于one-shot-network的框架以相对较低的搜索代价得到了很高的性能。这些NAS算法需要重新训练搜索到的网络，而Cai等[4]提出了Once-For-All网络，其子网络是训练好的，可以直接用于部署。最近NAS框架开始已经开始将硬件纳入到搜索反馈循环里[1-4,8,9]，但是还没有探索硬件加速器优化。

## 5. Conclusion

We propose an evolution-based accelerator-compiler co-search framework, NAAS. It not only searches the architecture sizing parameters but also the PE connectivity and compiler mapping strategy. Integrated with the Once-for-All NAS algorithm, it explores the search spaces of neural architectures, accelerator architectures, and mapping strategies together while reducing the search cost by 120× compared with previous work. Extensive experiments verify the effectiveness of NAAS. Within the same computation resources as Eyeriss [14], NAAS provides 4.4× energy-delay-product reduction with 2.7% top-1 accuracy improvement on ImageNet dataset compared to directly running ResNet-50 on Eyeriss. Using the similar computation resources, NAAS integrated with NAS achieves 3.0×, 1.9× EDP improvements compared to NHAS [12], and NASAIC [11] respectively.

我们提出了一种基于演化的加速器编译器联合搜索框架，NAAS。不仅搜索架构大小参数，还包括PE连接性和编译器映射策略。与Once-For-All NAS算法集成后，探索神经网络架构，加速器架构和映射策略一起的搜搜空间，与之前的工作相比，搜索代价降低了120x。广泛的试验验证了NAAS的有效性。在与Eyeriss相同的计算资源下，与在Eyeriss上直接运行ResNet-50相比，NAAS的EDP降低了4.4x，ImageNet的top-1准确率改进了2.7%。使用类似的计算资源，与NHAS[12]，NASAIC[11]相比，NAAS与NAS集成，获得了3.0x，1.9x的EDP改进。