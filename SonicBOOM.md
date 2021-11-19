# SonicBOOM: The 3rd Generation Berkeley Out-of-Order Machine

Jerry Zhao, Ben Korpan, Abraham Gonzalez, Krste Asanovic @ Berkeley

# 0. Abstract

We present SonicBOOM, the third generation of the Berkeley Out-of-Order Machine (BOOM). SonicBOOM is an open-source RTL implementation of a RISC-V superscalar out-of-order core and is the fastest open-source core by IPC available at time of publication.

我们给出了SonicBOOM，BOOM的第三代版本。SonicBOOM是RISC-V超标量乱系核的一个开源RTL实现，是目前发表时以IPC计最快的开源核。

SonicBOOM provides a state-of-the-art platform for research in high-performance core design by providing substantial microarchitectural improvements over BOOM version 2. The most significant performance gains are enabled by optimizations to BOOM’s execution path and a redesign of the instruction fetch unit with a new hardware implementation of the state-of-the-art TAGE branch predictor algorithm. Additionally, SonicBOOM provides the first open implementation of a load-store unit that can provide multiple loads per cycle. With these optimizations and new features, SonicBOOM can achieve 2x higher IPC on SPEC CPU benchmarks compared to any prior open-source out-of-order core. Additionally, SonicBOOM achieves 6.2 CoreMark/MHz, making it the fastest currently available open-source core by IPC.

SonicBOOM为高性能核设计的研究，给出了一个目前最好的平台，相对于BOOMv2，给出了实质性的架构改进。最显著的性能提升，是通过对BOOM的执行路径的优化，和取指单元的重新设计得到的，包含了对目前最好的TAGE分支预测算法的新硬件实现。另外，SonicBOOM给出的LSU的实现，是目前第一个开放的可以在一个周期内进行多次load的实现。有了这些优化和新特征，SonicBOOM与之前任何开源乱序核相比，在SPEC CPU基准测试上，都会得到高2x的IPC。另外，SonicBOOM获得了6.2 CoreMark/MHz，使其成为以IPC来说目前最快的开源核。

**Keywords** superscalar, out-of-order, microarchitecture, branch-prediction, open-source, RISC-V

## 1. Introduction

As the need for general-purpose computing power increases, the deployment of high-performance superscalar, out-of-order cores has expanded beyond datacenters and workstations, into mobile and edge devices. In addition, the recent disclosure of microarchitectural timing attacks [12, 14] on speculating cores injects a new concern into the design space. Architects must now consider security, in addition to power, performance, and area, when evaluating new designs.

随着通用目标计算能力的需求增加，高性能超标量乱序核的部署，已经走出了数据中心和工作站，到了移动和边缘设备上。另外，最近暴露了对预测核的微架构时序攻击，使设计空间有了新的考虑。架构师在评估新设计的时候，除了能耗，性能，和面积，现在必须要考虑安全。

For those studying these problems, an open-source hardware implementation of a superscalar out-of-order core is an invaluable resource. Compared to open-source software models of high-performance cores, like gem5 [5], MARSSx86 [18], Sniper [9], or ZSim [22], an open-source hardware implementation provides numerous advantages. Unlike a software model, a hardware implementation can demonstrate precise microarchitectural behaviors execute real applications for trillions of cycles, and empirically provide power and area measurements. Furthermore, an open hardware implementation provides a baseline platform as a point of comparison for new microarchitectural optimizations.

对于研究这些问题的人，超标量乱序核的一种开源硬件实现是很宝贵的资源。与高性能核的开源软件模型相比，比如gem5，MARSSx86，Sniper，或ZSim，开源硬件实现会有很多优势。与软件模型不同的是，硬件实现可以展示执行数十亿周期的真实应用时，精确的微架构行为，从经验上给出能耗和面积度量。而且，开源硬件实现给出了一个基准平台，是新的架构优化的比较点。

While the growth in the number of open-source hardware development frameworks in recent years may seem to provide the solution to this problem [2, 4], most of these frameworks only provide support for simple in-order cores, like Rocket [3], Ariane [31], Black Parrot [21], or PicoRV32 [29]. Without a full-featured, high-performance implementation of a superscalar out-of-order core, these frameworks cannot generate modern mobile or server-class SoCs.

近些年，开源硬件开发框架逐渐增长，这似乎对这个问题提出了解决方案，但大多数这些框架都只对简单的顺序核给出支持，如Rocket，Ariane，Black Parrot或PicoRV32。这些框架没有包含超标量乱序核的完整特征的高性能实现，不能生成现代移动或服务器级的SoC。

Although there has also been an explosion in the number of open-source out-of-order cores in recent years, these fail to address the demand for an open high-performance implementation. Neither BOOMv2 [11], riscy-OOO [32], nor RSD [17] demonstrate substantial performance advantages over open-source in-order cores. The lack of features like 64-bit support, compressed-instruction support, accurate branch prediction, or superscalar memory accesses, in some of these designs further inhibits their usefulness to architects interested in studying application-class cores.

虽然最近几年开源乱序核的数量有了爆炸性增长，但这些都没有处理开源高性能实现的需求。BOOMv2，riscy-OOO和RSD与开源顺序核相比，都没有根本的性能优势。缺少64-bit的支持，压缩指令的支持，准确的分支预测，或超标量内存访问，这些设计对于对研究应用级的核的架构师来说，缺少有用性。

To address these concerns, we developed the third generation of the BOOM core, named SonicBOOM. Table 1 displays how SonicBOOM demonstrates improved performance and extended functionality compared to prior open-source out-of-order cores. Furthermore, the configuration of SonicBOOM depicted in Figure 1 is the fastest publicly available open-source core (measured by IPC at time of publication).

为处理这些考虑，我们开发了第三代BOOM核，名为SonicBOOM。表1展示了SonicBOOM的改进性能和功能，与之前的开源乱序核进行了比较。而且，图1中展示了SonicBOOM的配置，是目前最快的开源核（在发表时以IPC进行度量）。

## 2. BOOM History

We describe prior iterations of the BOOM core, and how the development history motivated a new version of BOOM. Figure 2 depicts the evolution of the BOOM microarchitecture towards SonicBOOM.

我们描述了BOOM核的迭代，一级开发历史怎么促进新版BOOM的诞生。图2展示了BOOM微架构演进到SonicBOOM的过程。

### 2.1 BOOM Version 1

BOOM version 1 (BOOMv1) was originally developed as an educational tool for UC Berkeley’s undergraduate and graduate computer architecture courses. BOOMv1 closely followed the design of the MIPS R10K [30], and featured a short, simple pipeline, with a unified register file and issue queue. BOOMv1 was written in the Chisel hardware description language, and heavily borrowed existing components of the Rocket in-order core in its design, including the frontend, execution units, MMU, L1 caches, and parts of the branch predictor. While BOOMv1 could achieve commercially-competitive performance in simulation, its design had unrealistically few pipeline stages, and was not physically realisable.

BOOMv1开始是为了UC Berkeley的本科生和研究生的计算机架构课程开发的，作为一个教学工具。BOOMv1基本是按照MIPS R10K的设计来的，流水线短，简单，统一的寄存器组和发射队列。BOOMv1是用Chisel写的，重度借鉴了Rocket顺序核设计的已有组成部分，包括前端，执行单元，MMU，L1缓存，和分支预测的一部分。BOOMv1在仿真时能得到与商用可竞争的性能，其设计流水线数量很少，不能在物理上实现。

### 2.2 BOOM Version 2

BOOM version 2 (BOOMv2) improved the design of BOOMv1 to be more suitable for fabrication and physical design flows. In order to be physically realizable, BOOMv2 added several pipeline stages to the frontend and execution paths, resolving critical paths in BOOMv1. In addition, the floating point register file and execution units were partitioned into an independent pipeline, and the issue queues were split into separate queues for integer, memory, and floating-point operations. BOOMv2 was fabricated within the BROOM test chip in TSMC 28nm [6].

BOOMv2改进了BOOMv1的改进版，更适合于制造和物理设计流程。为了在物理上能够实现，BOOMv2对前端和执行路径增加了几个流水线级，解决了BOOMv1中的关键路径。另外，浮点寄存器组和浮点执行单元分割成了独立的流水线，发射队列为整数操作，内存操作和浮点运算分成了独立的队列。BOOMv2以TSMC 28nm工艺在BROOM测试芯片中进行了制造。

### 2.3 BOOM Version 3

BOOM version 3 (SonicBOOM) builds upon the performance and physical design lessons from previous BOOM versions in order to support a broader set of software stacks and address the main performance bottlenecks of the core while maintaining physical realizability. We identify key performance bottlenecks within the instruction fetch unit, execution backend, and load/store unit. We also provide new implementations of many structures which, in BOOMv2, were borrowed from the Rocket in-order core. New implementations in SonicBOOM were designed from the ground-up for integration within a superscalar out-of-order core.

BOOMv3在之前BOOM版本的性能和物理设计教训之上构建，支持更广泛的软件栈，解决了核的主要性能瓶颈问题，同时保持了物理可实现性。我们在取指单元，执行后端和LSU中发现了关键的性能瓶颈。我们还给出了很多结构的新实现，在BOOMv2中这些结构都是从Rocket顺序核借鉴来的。SonicBOOM的新实现是从头开始设计的，集成到了超标量乱序核中。

## 3. Instruction Fetch

BOOMv2’s instruction fetch unit was limited by the lack of support for compressed 2-byte RISC-V (RVC) instructions, as well as a restrictively tight coupling between the fetch unit and the branch predictors. SonicBOOM addresses both of these issues, with a new frontend capable of decoding RVC instructions, and a new advanced pipelined TAGE-based branch predictor.

BOOMv2的取指单元缺少对2-byte RISC-V压缩指令的支持，而且取指单元和分支预测耦合过于紧密。SonicBOOM处理了这些问题，新的前端可以解码RVC指令，以及一个新的基于TAGE的高级流水线分支预测器。

### 3.1 Compressed Instructions

The C-extension to the RISC-V ISA [28] provides additional support for compressed 2-byte forms of common 4-byte instructions. Since this extension substantially reduces code-size, it has become the default RISC-V ISA subset for packaged Linux distributions like Fedora and Debian. These distributions additionally include thousands of pre-compiled software packages, providing a rich library of applications to run without requiring complex cross-compilation flows.

RISC-V ISA的C拓展对压缩的2-byte指令提供了额外支持，这是普通4-byte指令的压缩版。由于这个拓展极大的压缩了指令大小，这是紧凑Linux发行版的默认RISC-V ISA子集，如Fedora和Debian。这些发行版额外的包含了数千个预编译的软件包，有丰富的应用库可以运行，不需要复杂的交叉编译流。

To support the growing community-driven ecosystem of pre-packaged RISC-V software, SonicBOOM includes a new superscalar fetch unit which supports the C-extension. SonicBOOM’s new fetch unit decodes a possible 2-byte or 4-byte instruction for every 2-byte parcel. An additional pipeline stage after the frontend-decoders shifts the decoded instructions into a dense fetch-packet to pass into the SonicBOOM backend.

为支持逐渐增加的预打包RISC-V软件生态，SonicBOOM包含了新的超标量取指单元，支持C扩展。SonicBOOM的新取指单元对2-byte和4-byte指令每2-byte的包进行解码。前端解码器后面的额外流水线，将解码的指令转移到一个密集取指包，以传到SonicBOOM的后端。

### 3.2 Branch Prediction

Branch prediction is a critical component contributing to the performance of out-of-order cores. Hence, improving branch prediction accuracy in SonicBOOM was a first-order concern. The tight integration between the fetch unit, branch target buffer (BTB), and branch predictor within BOOMv2 restricted the addition of new features and optimizations within the fetch pipeline. Bug fixes and new features to the fetch unit frequently degraded branch predictor accuracy between BOOMv1 and BOOMv2.

分支预测是一个关键组成组分，对乱序核的性能贡献很大。因此，在SonicBOOM中改进分支预测准确率，是第一等考虑。在BOOMv2中，取指单元，BTB和分支预测器的耦合非常紧密，这限制了新特征和优化在取指流水中的加入。对取指单元的bug修复和新特征的加入，频繁的降低BOOMv1和BOOMv2的预测器的准确率。

SonicBOOM’s fetch pipeline was redesigned with a more general and flexible interface to a pipelined hierarchical branch predictor module. Compared to BOOMv2, SonicBOOM supports a single-cycle next-line predictor, and also provides substantial machinery for managing branch predictor state, local histories, and global histories.

SonicBOOM的取指流水进行了重新设计，接口更加通用灵活，对接流水线层次化分支预测器模块。与BOOMv2相比，SonicBOOM支持单周期下一行预测器，还有很多可以用于管理分支预测状态，局部历史和全局历史。

The branch predictor was re-written to integrate cleanly with the superscalar banked fetch unit in BOOM. In BOOMv2, the banked ICache was partnered to an unbanked branch predictor, resulting in frequent aliasing of branches between the even/odd ICache banks in the predictor memories. The final result was that the branch predictor capacity was effectively halved in BOOMv2, as the mis-configuration forced the predictor to learn two entries (one for each bank) for some branches. In SonicBOOM, the branch predictor is banked to match the ICache.

分支预测器进行了重写，以与BOOM中的超标量banked取指单元进行干净的整合。在BOOMv2中，banked ICache与unbanked分支预测器进行匹配，在预测器内存中的奇偶ICache banks之间，存在频繁的别名分支。最终的结果是，BOOMv2中的分支预测器的能力基本上只有一半，因为错误的配置迫使预测器对一些分支学习到两个entries（每个bank一个）。在SonicBOOM中，分支预测器也进行了bank以匹配ICache。

Additionally, SonicBOOM rectifies the minimum 2-cycle redirect penalty in BOOMv2 by adding a small micro BTB (uBTB). This uBTB (sometimes called "next-line-predictor", or "L0 BTB") redirects the PC in a single cycle from a small fully-associative buffer of branch targets, drastically improving fetch throughput on small loops.

另外，SonicBOOM修正了BOOMv2中的2-cycle重定向最小惩罚，加上了一个uBTB。这个uBTB（有时成为下一行预测器，或L0 BTB），在一个周期中将PC从一个小的全相联分支目标buffer中进行了重新定向，极大的改进了小型循环的取指吞吐量。

The most significant contribution to overall core performance is the inclusion of a high-performance TAGE [24] branch predictor, with a speculatively updated and repaired global-history vector driving predictions. Unlike BOOMv2, SonicBOOM carefully maintains a superscalar global history vector across speculative fetches and mis-speculation, enabling more accurate predictions. The SonicBOOM predictors were also redesigned to be superscalar, as we observed aliasing between branches in the same fetch-packet significantly degraded prediction accuracy for those branches in BOOMv2.

对核总体性能最显著的贡献，是加入了一个高性能TAGE分支预测器，这就有了一个推测更新并修复的全局历史向量来驱动预测。与BOOMv2不一样的是，SonicBOOM在推测取指和误推测之间，仔细的维护了一个超标量全局历史向量，预测可以更加准确。SonicBOOM预测器也重新设计成了超标量的，因为我们观察到，在同一个取指包中分支间的别名，会使BOOMv2中的这些分支中的预测准确率显著下降。

SonicBOOM additionally provides new repair mechanisms to restore predictor state after misspeculation. Specifically, the loop predictor and return-address-stack (RAS) are snapshotted and repaired on mispredict [26]. Compared with the original unrepaired RAS from BOOMv2, the SonicBOOM RAS achieves 10x fewer mis-predicts, with 98% prediction accuracy on ret instructions.

BOOMv2还额外提供了新的修复机制，在误预测之后，恢复预测器的状态。具体的，循环预测器和RAS在误预测时都进行了快照并进行修复。与BOOMv2中原始的未修复的RAS相比，SonicBOOM RAS的误预测少了10倍，对ret指令的预测准确率达到了98%。

SonicBOOM also changed the branch resolution mechanism. In BOOMv2, only a single branch per cycle could be resolved, as branch units must read the single-ported fetch-PC queue to determine a branch target. We observed that this limitation limited scalability towards wider frontends with high-throughput trace caches, as branch-dense code might include multiple branches in a fetch packet. In SonicBOOM, we add support for superscalar branch resolution, with multiple branch-resolution units. An additional pipeline stage is inserted after writeback to only read the fetch-PC queue to determine the target address for the oldest mispredicted branch in a vector of resolved branches. While this increases our branch-to-branch latency to 12 cycles, the additional scheduling flexibility provided by multiple branch units overall improved performance on relevant workloads, as the schedulers could more aggressively schedule branches, instead of waiting for a single branch unit to become available.

SonicBOOM还改变了分支解析的机制。在BOOMv2中，每个周期只能解析一个分支，因为分支单元必须读取单端口取指PC队列，来确定一个分支目标。我们观察到，这限制了向更宽的前端的可扩展性，有更高吞吐量的trace缓存，因为分支密集的代码会在一个取指包中包含多个分支。在SonicBOOM中，我们对超标量分支解析增加了支持，有多个分支解析单元。在写回后，还加入了一个额外的流水线阶段，只读取取指PC队列，来确定解析的分支的向量中最老的误预测的分支中的目标地址。这将分支到分支的延迟增加到了12个周期，但多分支单元给出的额外调度灵活性，对相关的workload改进了性能，因为调度器可以更激进的调度分支，而不是等待单个分支单元可用。

## 4. Execute

We provide two major new features in SonicBOOM’s execute pipeline. Support for the RoCC accelerator interface enables integration of custom accelerators into the BOOM pipeline. The short-forwards branch (SFB) optimization improves IPC by recoding difficult-to-predict branches into internal predicated microOps.

SonicBOOM的执行流水线中有两个新的主要特征。支持RoCC加速器接口，可以集成定制的加速器到BOOM流水线中。SFB优化改进了IPC，将难以预测的分支重新编码到内部断言uOp中。

### 4.1 RoCC Instructions

The Rocket Custom Coprocessor interface (RoCC) was originally designed as a tightly-integrated accelerator interface for the Rocket in-order core. When implemented, the Rocket in-order core will send the operands and opcodes of any custom accelerator instruction through the RoCC interface to the accelerator, which can write data directly into core registers. Additionally, the accelerator can also access ports to the L1, L2, outer memory, and MMU.

RoCC最开始设计是用作紧密集成的加速器接口，用在Rocket顺序核中。实现时，Rocket顺序核将任意的定制加速器的指令的opcodes和操作数通过RoCC接口送到加速器，这将数据直接写到核寄存器中。另外，加速器也可以访问端口到L1，L2，外部存储和MMU。

The design of the RoCC interface has proven to be very useful for accelerator research. A wide variety of tightly-integrated accelerators have been designed for this interface, accelerating a diverse set of tasks including machine learning [7], vector computation [13], garbage collection [15], page fault handling [20], memory copying [16], and cryptography [23].

RoCC接口的设计，在加速器的研究中非常有用。很多紧密集成的加速器设计用于这个接口，加速了很多任务，包括机器学习，向量计算，垃圾回收，页出错处理，内存拷贝，和密码学。

Nevertheless, some of the workloads accelerated by these accelerators have been limited by the performance of the host Rocket core, as the host core must be able to fetch and issue enough instructions to saturate the accelerator. We implemented support for the RoCC interface in SonicBOOM to provide a platform for accelerator research on top of high performance out-of-order cores. Unlike Rocket, SonicBOOM speculatively decodes RoCC instructions and holds their operands in a “RoCC queue” until they pass the architectural commit point, at which time they can be issued into the accelerator. As a result, SonicBOOM can more aggressively drive a custom RoCC accelerator, compared to the Rocket core.

尽管如此，这些加速器加速的一些workloads受到Rocket核的性能的限制，因为宿主核需要能够取到并发射足够的指令，来使加速器饱和。我们在SonicBOOM中实现了RoCC接口的支持，在高性能乱序核上，为加速器研究提供了一个平台。与Rocket不同，SonicBOOM推测解码RoCC指令，将其操作数放在RoCC队列中，直到它们通过了架构commit点，在这个时刻它们会被发射到加速器中。结果是，与Rocket核相比，SonicBOOM可以更激进的驱动一个定制RoCC加速器。

### 4.2 Short-forwards Branch Optimizations

A frequent code pattern is a data-dependent branch over a short basic block. The data-dependent branches in these sequences are often challenging to predict, and naive execution of these code sequences would result in frequent branch mispredictions and pipeline flushes.

一种频繁的代码模式是，在一小段代码块中有一种依赖于数据的分支。在这些序列中，依赖于数据的分支通常是难以预测的，这些代码序列的简单执行，会导致频繁的分支预测错误，和流水线冲刷。

While the base RISC-V ISA does not provide conditional-move or predicated instructions, which can effectively replace unpredictable short-forwards branches, we observe that we can dynamically optimize for these cases in the microarchitecture. Specifically, we introduce additional logic to detect short-forwards in fetched instructions, and decode them into internal “set-flag” and “conditional-execute” micro-ops. This is similar to the predication support in the IBM Power8 microarchitecture [25].

基础的RISC-V ISA并没有给出条件移动或条件执行指令，这都可以有效的替换无法预测的短的向前的分支，我们观察到，我们可以对这些情况，在微架构中进行动态的优化。具体的，我们引入了额外的逻辑，在取到的指令中检测短的向前跳转指令，并将其解码成内部的set-flag和conditional-execute微操作。这与IBM Power8微架构中的条件支持很类似。

The “set-flag” micro-op replaces the original “branch” micro-op, and instead writes the outcome of the branch to a renamed predicate register file. Renaming the predicate register file is necessary to support multiple in-flight short-forwards-branch sequences. The “conditional-execute” micro-ops read the predicate register file to determine whether to execute their original operation, or to perform a copy operation from the stale physical register to the destination physical register. In the example in Figure 3, the short basic block consisting of two mv instructions are internally recoded as "conditional-moves" within SonicBOOM, while the unpredictable bge branch can be recoded as a "set-flag" operation.

Set-flag微操作替换了原来的分支微操作，将分支的输出写到一个重命名的predicate寄存器组中。Predicate寄存器组的重命名是必须的，以支持多个在飞行中的短的向前跳转的分支序列。Conditional-execute微操作读取predicate寄存器组，来是否执行原始运算，或执行一个复制运算，从旧的物理寄存器到目的物理寄存器。在图3的例子中，短的基础块包含两个mv指令，在SonicBOOM内部被重新编码为conditional-moves，而不可预测的bge分支被重新编码为set-flag运算。

We observe that this optimization provides up to 1.7x IPC on some code sequences. As an example, SonicBOOM achieves 6.15 CoreMark/MHz with the SFB optimization enabled, compared to 4.9 CoreMark/MHz without.

我们观察到，这种优化对一些代码序列给出了最高1.7x的IPC。比如，SonicBOOM在开启了SFB优化后，得到了6.15 CoreMark/MHz，而没有开启时则只有4.9 CoreMark/MHz。

## 5. Load-Store Unit and Data Cache

In order to maximize RTL re-use, BOOMv1 and BOOMv2 used the L1 data-cache implementation of the Rocket in-order core. However, we observed that this reliance on a L1 data-cache designed for an in-order core incurred substantial performance penalties.

为RTL能够最大可能重用，BOOMv1和BOOMv2使用了Rocket顺序核的L1 data cache的实现。但是，我们观察到，依赖于一个为顺序核设计的L1 data-cache，带来了明显的性能低效。

The interface to Rocket’s L1 data-cache is only 1-wide, limiting throughput. On a wide superscalar core, this limitation is a significant performance bottleneck, blocking the wide fetch and decode pipeline on a narrow load-store pipeline.

Rocket的L1 data cache的接口只是1位宽，吞吐量有限。在一个很宽的超标量核中，这种局限是一个显著的性能瓶颈，很窄的load-store流水线，阻碍了很宽的取指和解码流水线。

Furthermore, Rocket’s L1 data cache cannot perform any operations speculatively, as any cache refill would irrevocably result in a cache eviction and replacement. This results in significant cache pollution from misspeculated accesses, when used in the BOOM core.

而且，Rocket的L1 data cache不能推测的进行任何运算，因为任何cache refill都会不可逆的导致cache逐出和替换。从错误预测的访问中，在使用到BOOM核中时，这导致显著的cache污染。

Finally, Rocket’s L1 data cache blocks load refills on cache eviction. Although the access latency to the L2 is only 14 cycles, the access time as measured from the core was 24 cycles, due to the additional cycles spent evicting the replaced line.

最后，Rocket的L1 data cache在cache逐出时阻碍了load refills。虽然到L2的访问延迟只有14个周期，从核来测量的访问时间是24个周期，因为有额外的周期耗费在逐出替换的行中。

To address these problems, SonicBOOM includes a new load-store unit and L1 data cache, depicted in Figure 4.

为解决这些问题，SonicBOOM包含了一个新的LSU和L1 data cache，如图4所示。

### 5.1 Dual-ported L1 Data Cache

To support dual issue into SonicBOOM’s memory unit, the new design stripes the data cache across two banks. The new L1 data cache supports dual accesses into separate banks, as each bank is still implemented as 1R1W SRAMs.

为给SonicBOOM的内存单元支持双发射，新设计将data cache分割成了两个banks。新的L1 data cache支持对分离的banks的双重访问，因为每个bank都实现为1R1W SRAMs。

While an alternative implementation using 2R1W SRAMs can achieve similar results, we observe that even-odd banking is sufficient in our core to relieve the data-cache bottleneck and enables physical implementation with simple SRAM memories. Common load-heavy code sequences, such as restoring registers from the stack, or regular array accesses, generate loads that evenly access both even and odd pointers.

使用2R1W SRAMs的实现可以得到类似的结果，我们观察到，奇偶banking在我们的核中是足够的，能够缓解data cache的瓶颈，用简单的SRAM内存的物理实现就可以得到。常见的重load代码序列，比如从堆栈中恢复寄存器，或常规的阵列访问，会生成均匀的访问奇偶pointers的loads。

The remaining challenge was to redesign the load-store unit to be dual-issue, matching the L1 data cache. The load-address and store-address CAMs were duplicated, and the TLB was dual-ported. The final SonicBOOM load store unit can issue two loads or one store per cycle, for a total L1 bandwidth of 16 bytes/cycle read, or 8 bytes/cycle write.

剩余的挑战是，重新设计LSU为双发射的，匹配L1 data cache。Load-address和store-address CAMs都进行了复制，TLB也是双端口的。最后的SonicBOOM LSU可以在每个周期内发射两个load或一个store指令，总计的L1带宽是读16 bytes每周期，或写8 bytes每周期。

### 5.2 Improving L1 Performance

In SonicBOOM, a load miss in the L1 data-cache immediately launches a refill request to the L2. The refill data is written into a line-fill buffer, instead of directly into the data-cache. Thus, cache evictions can occur in parallel with cache refills, drastically reducing observed L2 hit times. When cache eviction is completed, the line-fill buffer is flushed into the cache arrays.

在SonicBOOM中，L1 data cache的一个load miss会立刻启动向L2的refill请求。Refill数据写入line-fill buffer，而不是直接写入data cache。因此，cache驱逐与cache refill可以并行进行，极大的降低了L2的hit时间。当cache驱逐完成后，line-fill buffer冲刷到cache阵列中。

The final implementation of SonicBOOM’s non-blocking L1 data cache contains multiple independent state machines. Separate state machines manage cache refills, permission upgrades, writebacks, prefetches, and probes. These state machines operate in parallel, and only synchronize when necessary to maintain memory consistency.

SonicBOOM的非阻塞L1数据缓存的最终实现，包含多个独立的状态机。不同的状态机管理cache refills，permission upgrades，writebacks，prefetches和probes。这些状态机并行运算，只在有必要维护内存一致性的时候进行同步。

We also introduce a small next-line prefetcher between the L1 and the L2. The next-line prefetcher speculatively fetches sequential cache lines after a cache miss into the line fill buffers. Subsequent hits on addresses within the line fill buffers will cause the cache to write the prefetched lines into the data arrays.

我们还在L1和L2 cache之间引入了一个小型的next-line prefetcher。Next-line prefetcher在一个cache miss之后，推测取顺序的cache lines，到line fill buffer中。后续hit到line fill buffer中的地址，会导致将预取的lines写入到数据阵列中。

SonicBOOM’s line-fill buffers can also be modified to provide a small amount of resistance to Spectre-like attacks, which leak information through misspeculated cache refills [8]. The line-fill buffers can be modified to write lines into the L1 cache only if the request for that line was determined to be correctly speculated. Misspeculated cache refills can simply be flushed out of the L1, preventing attacker processes from learning speculated behaviors from L1 data-cache state.

SonicBOOM的line-fill buffers也可以进行修改，对类Spectre的攻击有一定的韧性，这类攻击会通过误推测的cache refills泄露信息。Line-fill buffers可以进行修改，当只有对这个line的请求被确定为推测的正确时，才将这个line写入L1 cache。误预测的cache refills就简单的从L1中冲刷掉，防止攻击过程从L1 data cache状态中学习推测的行为。

## 6. System Support

### 6.1 SoC Integration

SonicBOOM plugs-in within the tile interface of the Rocket Chip SoC generator ecosystem and the Chipyard integrated SoC research and development framework. As such, it integrates with a broad set of open-source heterogeneous SoC components and devices, including UARTs, GPIOs, JTAGs, shared-cache memory systems, and various accelerators. These components include the open-source SiFive inclusive L2 cache, the Hwacha vector accelerator [13], and the Gemmini neural-network accelerator [7]. Within the Chipyard framework, SonicBOOM can be integrated with additional RISC-V cores such as Rocket [3] or Ariane [31] to generate heterogeneous SoCs similar to modern hybrid processor architectures (ARM® big.LITTLE®, Intel® hybrid architectures). Figure 5 depicts how Chipyard generates a complete BOOM-based SoC from a high-level specification.

SonicBOOM作为一个插件，插入到Rocket Chip SoC生成器生态的tile interface中，和Chipyard集成SoC研究和开发框架中。这样，就可以与很多开源异质SoC组成部分和设备都可以集成，包括UARTs，GPIOs，JTAGs，共享缓存的内存系统和各种加速器。这些部分包括，开源的SiFive inclusive L2缓存，Hwacha vector accelerator，Gemmini神经网络加速器。在Chipyard框架中，SonicBOOM可以与其他RISC-V核集成到一起，比如Rocket或Ariane，来生成异质SoCs，与现代混合处理器架构类似(ARM® big.LITTLE®, Intel® 混合架构)。图5展示了，Chipyard怎样从一个高层spec生成一个完整的基于BOOM的SoC。

The Chipyard framework provides SonicBOOM an integrated emulation and VLSI implementation environment which enables continuous performance and efficiency improvements through short iterations of full-system performance evaluation using FPGA-accelerated simulation on FireSim [10], as well as consistent physical design feedback through the Hammer [27] VLSI flow.

Chipyard框架为SonicBOOM提供了一个集成的仿真环境，和VLSI实现环境，这可以通过使用在FireSim上的FPGA加速的仿真，对完整系统的短迭代的性能进行评估，以及通过Hammer VLSI流得到的一致的物理设计反馈，得到持续的性能和效率改进。

### 6.2 Operating System Support

SonicBOOM has been tested to support RV64GC Buildroot and Fedora Linux distributions. As the highest-performance open-source implementation of a RISC-V processor, SonicBOOM enabled the identification of critical data-race bugs in the RISC-V Linux kernel.

SonicBOOM对支持RV64GC Buildroot和Fedora Linux发布版进行了测试。作为RISC-V处理器最高性能的开源实现，SonicBOOM还可以在RISC-V Linux kernel中识别关键的data-race bugs。

The RISC-V kernel page-table initialization code requires careful insertion of FENCE instructions to synchronize the TLB as the kernel constructs the page-table entries. As high-performance cores might speculatively issue a page table walk before a newly constructed page-table-entry has been written to the cache, these FENCE instructions are necessary for maintaining program correctness. SonicBOOM’s aggressive speculation found a section of the kernel initialization code where a missing FENCE caused a stale page-table-entry to enter the TLB, resulting in an unrecoverable kernel page fault. We are working on upstreaming the fix for this issue into the Linux kernel.

RISC-V kernel页表初始化代码，需要仔细的插入FENCE指令，以在kernel构建PTE时，同步TLB。因为高性能核可能在新构建PTE写入cache之前，推测性的发射一个PTW，这些FENCE指令是为维护程序正确性所必须的。SonicBOOM的激进推测，发现kernel初始化代码的一部分，其中缺少FENCE会导致旧的PTE会进入TLB，会导致不可恢复的kernel页错误。我们正在将这个问题上传到Linux kernel中。

### 6.3 Validation

Debugging an out-of-order core is immensely challenging and time-consuming, as many bugs manifest only in extremely specific corner cases after trillions of cycles of simulation. We discuss two methodologies we used to productively debug SonicBOOM’s new features.

调试乱序核是非常有挑战，非常耗时的，因为很多bugs都在非常特定的情况下表现出来的，只会在数十亿个周期的仿真之后出现。我们讨论了两种方法，在调试SonicBOOM的新特征时，非常有成果。

**6.3.1 Unit-testing**. From our experience, the load-store unit and data-cache are the most bug-prone components of an out-of-order core, as they must carefully maintain a memory-consistency model, while hiding the latency of loads, stores, refills, and writebacks as fast as possible. We integrated SonicBOOM’s load/store unit and L1 data-cache with the TraceGen tool in the Rocket Chip generator, which stress-tests the load-store unit with random streams of loads and stores, and validates memory consistency. We additionally developed a new tool memtrace, which analyzes the committed sequence of loads and stores in a single-core device, and checks for sequential consistency. These tools helped resolve several data-cache and load/store-unit bugs that manifested only after trillion of cycles of simulation.

从我们的经验来说，LSU和数据缓存是乱序核中最容易出bug的部分，因为他们必须仔细的保持存储一致模型，同时尽可能快的隐藏loads，stores，refills和writebacks的延迟。我们在Rocket Chip生成器中，将SonicBOOM的LSU和L1数据缓存集成到TraceGen工具，用随机的loads和stores流，来对LSU进行压力测试，并验证存储的一致性。我们另外开发了一个新的工具memtrace，分析单核设备的committed loads和stores序列，检查序列一致性。这些工具帮助解析了几个data cache和LUS的bugs，都是在数十亿个周期的测试之后才表现出来的。

**6.3.2 Fromajo Co-simulation**. SonicBOOM is also integrated with the Dromajo [1] co-simulation tool. Dromajo checks that the committed instruction trace matches the trace generated by a software architectural simulator. Fromajo integrates Dromajo with a FireSim FPGA-accelerated SonicBOOM simulation, enabling co-simulation at over 1 MHz, orders of magnitude faster than a software-only co-simulation system. Fromajo revealed several latent bugs related to interrupt handling and CSR management.

SonicBOOM还与Dromajo co-sim工具集成到了一起。Dromajo检查了committed指令trace与软件架构仿真器生成的trace匹配。Fromajo将Dromajo与一个FPGA加速的FireSim SonicBOOM仿真结合到了一起，可以在超过1MHz的频率进行co-sim，比只有软件的co-sim系统快了几个数量级。Fromajo揭示了几个与中断处理和CSR管理的隐式bugs。

## 7. Evaluation

SonicBOOM was physically synthesized at 1 GHz on a commercial FinFET process, matching the frequency achieved by BOOMv2. We evaluate SonicBOOM on the CoreMark, SPECint 2006 CPU, and SPECint 2017 CPU benchmarks. For all performance evaluations, SonicBOOM was simulated on FireSim [10] using AWS F1 FPGAs. The FireSim simulations ran at 30 MHz on the FPGAs, and modeled the system running at 3.2 GHz. A single-core system was simulated with 32 KB L1I, 32 KB L1D, 512 KB L2, 4 MB simulated L3, and 16 GB DRAM.

SonicBOOM在一个商用FinFET工艺上以1GHz进行了物理综合，与BOOMv2获得的频率是匹配的。我们在CoreMark，SPECint 2006 CPU和SPECint 2017 CPU基准测试上评估了SonicBOOM。对于所有的性能评估，SonicBOOM在FireSim上用AWS F1 FPGAs进行仿真。FireSim仿真以30MHz在FPGAs上进行仿真，对系统运行在3.2GHz进行建模。单核系统以32KB L1I，32KB L1D，512 KB L2 4MB L3和16 GB DRAM上进行了仿真。

### 7.1 SPECintspeed

We compare both SPEC06 and SPEC17 intspeed IPC to existing cores for which data is available. All SPEC benchmarks were compiled with gcc -O3 -falign-X=16, to enable most default compiler optimizations, and to align instructions into the SonicBOOM ICache.

我们与现有的数据可用的核比较了SPEC06和SPEC17 intspeed IPC。所有的SPEC基准测试都是用gcc -03 -falign-X=16编译的，以启动多数默认的编译器优化，并与SonicBOOM ICache的指令对齐。

We compare SPEC17 intspeed IPC against IPC achieved by AWS Graviton and Intel Skylake cores. The Graviton is a 3-wide A72-like ARM-based core, while the Skylake is a 6-wide x86 core. SPEC17 benchmarks were compiled with gcc, with -O3 optimizations.

我们与AWS Graviton和Intel Skylake的核比较SPEC17 intspeed IPC。Graviton是一个3宽度类似A72的基于ARM的核，而Skylake是一个6宽度的x86核。SPEC17基准测试是用gcc编译的，有-03的优化。

The results in Figure 6 show that SonicBOOM is competitive with the Graviton core, and can even match the IPC of the Skylake core on some benchmarks. However, we note that the difference in ISAs between these three systems skews the IPC results.

图6中的结果表明，SonicBOOM与Graviton核性能类似，在一些基准测试中甚至可以与Skylake核相媲美。但是，我们注意到，这三个系统ISAs之间的差异，使IPC结果有些扭曲。

### 7.2 CoreMark

We compare SonicBOOM performance on CoreMark to prior open-source and closed-source cores. While CoreMark is not a good evaluation of out-of-order core performance [19], published results are available for many existing cores. The results in Figure 7 show that SonicBOOM achieves superior CoreMark/MHz compared to any prior open-source core.

我们比较SonicBOOM与其他开源和闭源核的CoreMark性能。CoreMark并不是乱序核性能的好的评估，但对很多已有的核，都有公开的已发表的结果。图7的结果展示了SonicBOOM与任何其他开源核相比，获得了更高的CoreMark/MHz性能。

## 8. What's Next

### 8.1 Vector Execution

Vector (or SIMD) execution remains as an optimization path, as most high-performance architectures provide some set of vector instructions for accelerating trivially data-parallel code. The maturing RISC-V Vector extensions provides a ISA specification for the implementation of an out-of-order vector-execution engine within the BOOM core. We hope to provide an out-of-order implementation of the Vector ISA in a future version of BOOM.

向量执行（或SIMD）仍然是一个优化路径，因为多数高性能架构都给出一些向量指令集合，加速琐碎的数据并行代码。正在成熟的RISC-V向量扩展给出了一个ISA规范，可以在BOOM核中实现一个乱序向量执行引擎。我们希望在BOOM的未来版本中给出向量ISA的一个乱序实现。

### 8.2 Instruction and Data Prefetchers

Both L1 instruction and data cache-misses incur significant performance penalties in an out-of-order core. A L1 ICache miss on the fetch-path inserts at minimum a 10-cycle bubble into the pipeline, equivalent to a branch misprediction. Out-of-order execution may attempt to hide the penalty of a data-cache miss, but the speculation depth in SonicBOOM cannot hide misses beyond L1. As L3 hit-latency is on the order of 50 cycles, BOOM needs to speculate 50-cycles ahead to fully hide the penalty of a L1 miss. On a 4-wide BOOM, 50-cycles ahead is 200 instructions, exhausting the capacity of the reorder buffer.

在乱序核中，L1指令和数据的缓存miss都会带来显著的性能惩罚。在取指令路径上的L1 ICache miss，会在流水线上插入至少10周期的bubble，与分支预测错误等价。乱序执行可能会隐藏data cache miss的惩罚，但SonicBOOM的推测深度不能隐藏L1之外的miss。因为L3 hit延迟是在50个周期的级别了，BOOM需要提前推测50个周期来完全隐藏L1 miss的惩罚。在一个4宽度的BOOM上，提前50个周期就是200条指令，耗尽了ROB的能力。

Thus, both instruction and data prefetchers are vital for maintaining instruction throughput in an out-of-order core. While SonicBOOM provides a small prefetcher to fetch the next cache line after a miss into the L1, a more robust outer-memory prefetcher that can completely hide access time to L3 or DRAM is desired.

因此，指令和数据的prefetchers在乱序核中对于维护指令吞吐量是很关键的。SonicBOOM有一个较小的prefetcher，在L1 miss后取下一个cache line，但是如果有一个更稳健的外部内存prefetcher就更好了，可以完全隐藏到L3的访问时间。

## 9. Conclusion

SonicBOOM represents the next step towards high performance open-source cores. Numerous performance bottlenecks introduced by the physical design improvements for BOOMv2 were resolved, and many new microarchitectural optimizations were implemented. The resulting application-class core is performance-competitive with commercially high-performance cores deployed in datacenters. We hope that SonicBOOM will prove to be a valuable open-source asset for computer architecture research.

SonicBOOM代表了高性能开源核的下一步。BOOMv2的物理设计改进带来的很多性能瓶颈都被解决了，很多新的微架构优化进行了实现。得到了应用级核与数据中心中部署的商用的高性能核性能类似。我们希望SonicBOOM成为计算机架构研究的宝贵的开源资产。