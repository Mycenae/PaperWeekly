# The Rocket Chip Generator

Krste Asanovi´c, et. al., @ University of California Berkeley

## 0. Abstract

Rocket Chip is an open-source Sysem-on-Chip design generator that emits synthesizable RTL. It leverages the Chisel hardware construction language to compose a library of sophisticated generators for cores, caches, and interconnects into an integrated SoC. Rocket Chip generates general-purpose processor cores that use the open RISC-V ISA, and provides both an in-order core generator (Rocket) and an out-of-order core generator (BOOM). For SoC designers interested in utilizing heterogeneous specialization for added efficiency gains, Rocket Chip supports the integration of custom accelerators in the form of instruction set extensions, coprocessors, or fully independent novel cores. Rocket Chip has been taped out (manufactured) eleven times, and yielded functional silicon prototypes capable of booting Linux.

Rocket Chip是一个开源的SoC设计生成器，可以得到可综合的RTL。它利用了Chisel语言，创作了一个复杂的生成器库，有核，缓存，和互联部分，可以得到集成的SoC。Rocket Chip生成的是使用开放的RISC-V ISA的通用目标处理器核，可以给出顺序核生成器(Rocket)，和乱序核生成(BOOM)。对利用异质专长增加性能提升的SoC设计者来说，Rocket Chip支持集成定制加速器，以指令集拓展的形式，或协处理器的形式，或完全独立的新型核。Rocket Chip已经流片了11次，得到的功能型硅原型可以启动Linux。

## 1. Introduction

Systems-on-chip (SoC) leverage integration and customization to deliver improved efficiency. Rocket Chip is an open-source SoC generator developed at UC Berkeley suitable for research and industrial purposes. Rather than being a single instance of an SoC design, Rocket Chip is a design generator, capable of producing many design instances from a single high-level source. It produces design instances consisting of synthesizable RTL, and multiple functional silicon prototypes have been manufactured. Extensive parameterization makes it flexible, enabling easy customization for a particular application. By changing a single configuration, a user can generate SoCs ranging in size from embedded microcontrollers to multi-core server chips. Rocket Chip is open-source and available under a BSD license on Github. For increased modularity, many of the component libraries of Rocket Chip are available as independent repositories, and we use git submodules to track compatible versions. Rocket Chip is stable enough to produce working silicon prototypes, and we continue to expand the space of designs it can express with new functionality. This report briefly catalogues the Rocket Chip features available as of April 2016.

SoC利用集成和定制来给出改进的效率。Rocket Chip是一个开源SoC生成器，在UC Berkeley开发，适用于研究和工业目的。Rocket Chip不是SoC设计的一个实例，而是一个设计生成器，从一个高层源，可以产生很多设计实例。其产生的设计实例，包含可以综合的RTL，和多个功能性硅原型，已经进行生产。广泛的参数化，使其非常灵活，可以对特定的应用很容易的进行定制。改变一个配置，用户可以生成不同大小的SoC，从嵌入式的微控制器，到多核的服务器芯片。Rocket Chip是开源的。对提升模块性，Rocket Chip中的很多组成部分库，都是独立的代码仓库，我们使用git submodules来追踪兼容的版本。Rocket Chip非常稳定，可以生成能工作的硅原型，我们持续扩展设计空间，增加新的功能。本报告将2016.04 Rocket Chip的特征进行简要归类描述。

## 2. Background

Rocket Chip is based on the RISC-V Instruction Set Architecture (ISA) [11]. RISC-V is an ISA developed at UC Berkeley and designed from the ground up to be clean, microarchitecture-agnostic and highly extensible. Most importantly, RISC-V is free and open, which allows it to be used in both commercial and open-source settings [2]. It is under the governance of the RISC-V Foundation and is intended to become an industry standard.

Rocket Chip是基于RISC-V ISA的。RISC-V是一个在UC Berkeley开发的ISA，是从头开始设计的，干净，与微架构无关，高度可扩展。最重要的是，RISC-V是自由开放的，可以商用和开源。它是在RISC-V基金会管理下的，朝着成为工业目标的方向努力。

Using RISC-V as an ISA removes potential licensing restrictions from Rocket Chip and allows the same ISA and infrastructure to be used for a wide range of cores, from high-performance out-of-order designs (Section 5) to small embedded processors (Section 6). RISC-V is flexible due to its modular design, which features a common base of roughly 40 integer instructions (I) that all cores must implement, with ample opcode space left over to support optional extensions, of which the most canonical have already been standardized. Existing extensions include multiply and divide (M), atomics (A), single-precision (F) and double-precision (D) floating point. These common extensions (IMAFD) are collected into the (G) extension that provides a general-purpose, scalar instruction set. RISC-V provides 32-bit, 64-bit, and 128-bit address modes. A compressed (C) extension provides 16-bit instruction formats to reduce static code size. Opcode space is also reserved for non-standard extensions, so designers can easily add new features to their processors that will not conflict with existing software compiled to the standard. RISC-V’s User-level ISA is frozen and described in the official RISC-V Instruction Set Manual [11]. The privileged ISA and platform specification are currently under review, with draft specifications available [10].

使用RISC-V作为ISA，使Rocket Chip没有授权的限制，还使同样的ISA和基础设置可以用于广泛的核中，从高性能乱序设计，到小型嵌入式处理器。RISC-V由于其模块化设计，非常灵活，其共有的基础为大约40条整数指令(I)，所有的核都必须实现，留下了充足的opcode空间，以支持可选的扩展，其中最经典的都已经被标准化了。现有的扩展包括乘除法(M)，原子指令(A)，单精度浮点(F)，和双精度浮点(D)。这些常见扩展(IMAFD)一起称为G扩展，提供了通用目标的标量指令集。RISC-V提供了32-bit，64-bit和128-bit的寻址模式。压缩扩展(C)提供了16-bit的指令格式，以减少静态代码大小。也给非标准扩展保留了opcode空间，设计者可以很容易的增加新特征到处理器中，不会与标准编译得到的现有软件冲突。RISC-V的用户级ISA已经冻结，在官方RISC-V指令集手册中进行了描述。特权级ISA和平台指标目前正在审阅，有可用的草案指标。

A large and growing software ecosystem is available for RISC-V. This includes the GCC and LLVM compilers (and their supporting infrastructure such as binutils and glibc), ports of the Linux and FreeBSD operating systems and a wide range of software through the Yocto Project’s Linux distribution generator, poky. Software simulation is available through QEMU and Spike (homegrown functional simulator).

RISC-V有很大的增长的软件生态。这包括GCC和LLVM编译器（以及支持的基础设施，比如binutils和glibc），Linux和FreeBSD操作系统，和很多软件。用QEMU和Spike可以进行软件模拟。

Rocket Chip itself is implemented in Chisel, an open-source hardware construction language embedded in Scala [3]. While Chisel describes synthesizable circuits directly—and so more closely resembles traditional hardware description languages like Verilog than high-level synthesis systems—it makes the full Scala programming language available for circuit generation, enabling functional and object-oriented descriptions of circuits. Chisel also has additional features not found in Verilog, such as a rich type system with support for structured data, width inference for wires, high-level descriptions of state machines, and bulk wiring operations. Chisel generates synthesizable Verilog code that is compatible FPGA and ASIC design tools. Chisel can also generate a fast, cycle-accurate RTL simulator implemented in C++, which is functionally equivalent to but significantly faster than commercial Verilog simulators and can be used to simulate an entire Rocket Chip instance.

Rocket Chip本身是用Chisel实现的。Chisel描述的是可综合的电路，与传统的硬件描述语言如Verilog很相似，这使Scala编程语言可用于电路生成，使电路可以根据功能进行面向对象的描述。Chisel也有一些额外的特征，这是Verilog所没有的，比如丰富的类型系统，对结构化数据、连线的宽度推理，状态机的高层描述，bulk wiring运算有支持。Chisel生成可综合的Verilog代码，与FPGA和ASIC设计工具兼容。Chisel也可以生成快速的cycle-accurate RTL模拟器，用C++实现的，在功能上与商用的Verilog模拟器一样，但是速度快的多，可以用于仿真完整的Rocket Chip实例。

## 3. Rocket Chip Generator

The Rocket Chip generator is written in Chisel and constructs a RISC-V-based platform. The generator consists of a collection of parameterized chip-building libraries that we can use to generate different SoC variants. By standardizing the interfaces that connect different libraries’ generators to one another, we have created a plug-and-play environment in which it is trivial to swap out substantial design components simply by changing configuration files, leaving the hardware source code untouched. We can also both test the output of individual generators as well as perform integration tests on the whole design. The tests, too, are parameterized so as to provide maximal coverage.

Rocket Chip生成器是用Chisel写的，可以创建一个基于RISC-V的平台。生成器由一系列参数化的构建芯片的库组成，我们可以用以生成不同的SoC变体。我们将不同库生成器的连接的接口进行了标准化，就创建了一种即插即用的环境，替换基本的设计组成部分，就只需要改变配置文件，硬件源代码不需要改变。我们还测试了单个生成器的输出，以及对整个设计进行集成测试。这些测试也进行了参数化，以给出最大的覆盖。

Figure 1 is an example of a Rocket Chip instance. It features two tiles attached to a 4-bank L2 cache that is itself connected to the external I/O and memory systems with an AXI interconnect [1]. Within Tile 1 is an out-of-order BOOM core with an FPU, L1 instruction and data caches, and an accelerator implementing the RoCC interface (Section 4). Tile 2 is similar, but it uses a different core, Rocket, and has different L1 data cache parameters. In general, Rocket Chip is a library of generators that can be parameterized and composed into a wide variety of SoC designs. Here is a summary of the current capabilities of the generators and interfaces:

图1是Rocket Chip实例的一个例子。有两个tiles与一个4-bank L2 cache相连，然后与外部I/O和存储系统，通过AXI相连。在Tile 1中，是一个乱序BOOM核，带有一个FPU，L1指令和数据cache，和一个加速器，实现了RoCC接口。Tile 2是类似的，但使用了不同的核Rocket，而且L1数据缓存参数也是不同的。总体上，Rocket Chip是生成器的库，可以进行参数化，一起组成很多不同的SoC设计。这里是目前的生成器和接口的能力：

• Core: The Rocket scalar core generator and BOOM out-of-order superscalar core generator, both of which can include an optional FPU, configurable functional unit pipelines, and customizable branch predictors. Rocket标量核，和BOOM乱序超标量核生成器，两者都可以包含可选的FPU，可配置的功能单元流水线，可定制的分支预测器。

• Caches: A family of cache and TLB generators with configurable sizes, associativities, and replacement policies. 一族缓存和TLB生成器，大小，相连性和替换策略都可配置。

• RoCC: The Rocket Custom Coprocessor interface, a template for application-specific coprocessors which may expose their own parameters. Rocket定制的协处理器接口，应用专用协处理器的模板，可以有其自己的参数。

• Tile: A tile-generator template for cache-coherent tiles. The number and type of cores and accelerators are configurable, as is the organization of private caches. Tile生成器模板，用于缓存一致的tiles。核和加速器的数量和类型可配置，私有缓存的组织也可以配置。

• TileLink: A generator for networks of cache coherent agents and the associated cache controllers. Configuration options include the number of tiles, the coherence policy, the presence of shared backing storage, and the implementation of underlying physical networks. 缓存一致的代理和相关的缓存控制器的网络的生成器。配置选项包括，tiles的数量，一致性策略，共享备份存储，和潜在的物理网络的实现。

• Peripherals: Generators for AMBA-compatible buses (AXI, AHB-Lite, and APB) and a variety of converters and controllers, including the Z-scale processor. AMBA兼容的总线生成器，一系列转换器和控制器，包含Z-scale处理器。

To support diverse workloads and to improve energy efficiency, Rocket Chip supports heterogeneity. Not only can the SoC be composed of different tiles, but there is also support for adding custom accelerators. Rocket Chip supports three mechanisms for integrating accelerators, depending on how tightly coupled the accelerator is to the core. The easiest and most tightly coupled option is to expose the accelerator by extending RISC-V and attaching the accelerator directly to the core’s pipeline. For more decoupling, the accelerator can act as a coprocessor and receive commands and data from the processor via the RoCC interface. Fully decoupled accelerators can be instantiated in their own tiles and connect coherently to the memory system using TileLink. Furthermore, these techniques can be combined, as in the case of the Hwacha vector-thread accelerator [6], which receives commands via RoCC, but directly attaches to TileLink to bypass the processor’s L1 data cache to obtain greater memory bandwidth.

为支持不同的workloads，改进能耗效率，Rocket Chip支持异质性。SoC可以由不同的tiles构成，也支持定制的加速器。Rocket Chip支持三种机制集成加速器，依赖于加速器与核的耦合紧密程度。最简单的，和最紧密的耦合选项，是通过扩展RISC-V，将加速器直接与核的流水线相连。为更多的解耦合，加速器可以作为一个协处理器，从处理器通过RoCC接口来接收指令和数据。完全解耦合的加速器，可以用其自己的tiles来例化，使用TileLink来与存储系统一致相连。而且，这些技术可以结合在一起，Hwacha矢量线程加速器就是这样的，通过RoCC接收指令，与TileLink直接相连，旁路了处理器的L1数据缓存，以得到更大的内存带宽。

## 4. Rocket Core

Rocket is a 5-stage in-order scalar core generator that implements the RV32G and RV64G ISAs. It has an MMU that supports page-based virtual memory, a non-blocking data cache, and a front-end with branch prediction. Branch prediction is configurable and provided by a branch target buffer (BTB), branch history table (BHT), and a return address stack (RAS). For floating-point, Rocket makes use of Berkeley’s Chisel implementations of floating-point units. Rocket also supports the RISC-V machine, supervisor, and user privilege levels. A number of parameters are exposed, including the optional support of some ISA extensions (M, A, F, D), the number of floating-point pipeline stages, and the cache and TLB sizes.

Rocket是一个5级顺序标量核生成器，实现了RV32G和RV64G ISAs，有一个MMU支持基于页的虚拟存储，一个non-blocking数据缓存，带有分支预测的前端。分支预测是可配置的，有BTB，BHT，RAS。对于浮点，Rocket使用了Berkeley的Chisel实现的浮点单元。Rocket还支持RISC-V三个特权级别，machine, supervisor和user。有几个参数是暴露出来的，包括对一些ISA扩展的支持（M，A，F，D），浮点流水线级数的数量，和缓存和TLB的大小。

Rocket can also be thought of as a library of processor components. Several modules originally designed for Rocket are re-used by other designs, including the functional units, caches, TLBs, the page table walker, and the privileged architecture implementation (i.e., the control and status register file).

Rocket也可以被认为是处理器组成部分的库。为Rocket设计的一些模块，可以被其他设计重用，包括功能单元，cache，TLB，和PTW，以及特权架构实现（即，CSR）。

The Rocket Custom Coprocessor Interface (RoCC) facilitates decoupled communication between a Rocket processor and attached coprocessors. Many such coprocessors have been implemented, including crypto units (e.g., SHA3) and vector processing units (e.g., the Hwacha vector-fetch unit [6]). The RoCC interface accepts coprocessor commands generated by committed instructions executed by the Rocket core. The commands include the instruction word and the values in up to two integer registers, and commands may write an integer register in response. The RoCC interface also allows the attached coprocessor to share the Rocket core’s data cache and page table walker, and provides a facility for the coprocessor to interrupt the core. These mechanisms are sufficient to construct coprocessors that participate in a page-based virtual memory system. Finally, RoCC accelerators may connect to the outer memory system directly over the TileLink interconnect, providing a high-bandwidth but coherent memory port. The Hwacha vector-fetch unit makes use of all of these features and has driven the development of RoCC into a sophisticated coprocessor interface.

RoCC使Rocket处理器和相连的协处理器可以进行解耦合的通信。很多这样的协处理器都已经实现了，包括加密单元（如，SHA3）和向量处理单元（如，Hwacha向量-fetch单元）。RoCC接口接受协处理器指令，由Rocket核通过committed指令执行。命令包括指令字和最多在两个整数寄存器中的值，命令还可以写一个整数寄存器作为响应。RoCC接口还使相连的协处理器共享Rocket核的数据缓存和PTW，为协处理器提供了中断核的设施。这些机制足以参与到基于页的虚拟内存系统的协处理器。最后，RoCC加速器可以通过TileLink直接连接到外部存储系统，提供了高带宽但一致的内存端口。Hwacha向量-fetch单元使用了所有这些特征，使RoCC发展成了一个复杂的协处理器接口。

## 5. Berkeley Out-of-Order (BOOM) Core

BOOM is an out-of-order, superscalar RV64G core generator. The goal of BOOM is to serve as a baseline implementation for education, research, and industry and to enable in-depth exploration of out-of-order micro-architecture. An independent specification available elsewhere elucidates BOOM’s design and rationale.

BOOM是一个乱序超标量RV64G核生成器。BOOM的目标是为教育，研究和工业作为一个基准实现，更深入的探索乱序微架构。有专门的指标阐释了BOOM的设计和基本原理。

BOOM supports full branch speculation using a BTB, RAS, and a parameterizable backing predictor. Some of the available backing predictors that can be instantiated include a gshare predictor and a TAGE-based predictor. BOOM uses an aggressive load/store unit which allows loads to execute out-of-order with respect to stores and other loads. The load/store unit also provides store data forwarding to dependent loads. As shown in Figure 1, a BOOM tile is fully I/O-compatible with a Rocket tile and fits seamlessly into the Rocket Chip memory hierarchy.

BOOM支持完整的分支预测，使用的是BTB，RAS和可参数化的后备预测器。一些可用的后备预测器包括，gshare预测器，和基于TAGE的预测器。BOOM使用一种激进的LSU，可以根据stores和其他loads来载入并执行。LSU还支持store数据forwarding to dependent loads。如图1所示，BOOM tile与Rocket tile是完全I/O兼容的，无缝衔接到Rocket Chip存储层次结构中。

BOOM is written in 10k lines of Chisel code. BOOM is able to accomplish this low line count in part by instantiating many parts from the greater Rocket Chip repository; the front-end, functional units, page table walkers, caches, and floating point units are all instantiated from the Rocket and hardfloat repositories. Chisel has also greatly facilitated making BOOM a true core generator - the functional unit mix is customizable, and the fetch, decode, issue and commit widths of BOOM are all parameterizable.

BOOM用10k行Chisel代码写成。BOOM可以用这么少的代码量就完成，部分是因为，实例化了很多Rocket Chip的很多部件；前端，功能单元，PTW，cache和FPU，都是从Rocket和hardfload库中实例化而来的。Chisel也极大的促进了BOOm成为了一个真正的核生成器，功能单元的混合是可定制的，BOOM的fetch，decode，issue和commit宽度都是可参数化的。

## 6. Z-scale Core

Z-scale is a 32-bit core generator targeting embedded systems and micro-controller applications. It implements the RV32IM ISA and is designed to interface with AHB-Lite buses, making it plug-compatible in a manner analogous to the ARM Cortex-M series. Z-scale uses a 3-stage, single-issue in-order pipeline that supports the machine and user privilege modes. As it interfaces directly with AHB-Lite buses, it is not a direct replacement for a Rocket (or BOOM) application core and it does not fit into the Rocket Chip cache coherent memory hierarchy. Z-scale is available both in Chisel, like the rest of Rocket Chip, and also Verilog.

Z-scale是一个32位核生成器，目标是嵌入式系统和微控制器应用。它实现了RV32IM ISA，设计采用AHB-Lite总线接口，使其与ARM Cortex-M系列的方式类似的plug-compatible。Z-scale使用一个3阶段，单发射，顺序流水线，支持machine和user特权模式。由于其直接采用AHB-Lite总线作为接口，所以并不是Rocket（或BOOM）应用核的直接替代，并不适用于Rocket Chip缓存一致性内存层次结构。Z-scale有Chisel和Verilog版。

## 7. Uncore and TileLink

Rocket Chip includes generators for a shared memory hierarchy of coherent caches interconnected with on-chip networks. Configuration options include the number of tiles, the coherence policy, the presence of a shared L2 cache, the number of memory channels, the number of cache banks per memory channel, and the implementation of the underlying physical networks. These generators are all based around TileLink, a protocol framework for describing a set of cache coherence transactions that implement a particular cache coherence policy.

Rocket Chip包含了共享内存层次结构的生成器，与片上网络相连的是一致性缓存。配置选项包括，tiles的数量，一致性策略，共享L2缓存的存在，内存通道的数量，每个内存通道中cache banks的数量，潜在物理网络的实现。这些生成器都是基于TileLink的，这是一个协议框架，用于描述实现了特定缓存一致性策略的缓存一致性事务的集合。

TileLink’s purpose is to orthogonalize the design of the on-chip network and the implementation of the cache controllers from the design of the coherence protocol itself. This separation of concerns improves the modularity of the HDL description of the memory hierarchy, while also making validation and verification of individual memory system components more tractable. Any cache coherence protocol that conforms to TileLink’s transaction structure can be used interchangeably alongside the physical networks and cache controller generators we provide. By supplying a framework to apply transactional coherence metadata updates throughout the memory hierarchy, TileLink enables simplified expressions of the coherence policies themselves. Conversely, as long as newly designed controllers and networks make certain guarantees about their behavior, system-on-chip designers can be confident that incorporating them into their TileLink-based memory hierarchy will not introduce coherence-protocol-related deadlocks.

TileLink的目的是，将片上网络的设计，缓存控制器的实现，与一致性协议本身的设计分开来。这种考虑的分离，改进了内存层次性HDL描述的模块性，也使单个内存系统组成部分的validation和verification更加可追踪。符合TileLink事务结构的任何缓存一致性协议，都可以互换使用，与我们提供的物理网络和缓存控制器生成器。TileLink提供了一个框架，在内存层次结构各处应用事务性一致元数据更新，这使一致性策略本身有了简化的表述。相反的，只要新设计的控制器和网络对其行为有特定的保证，SoC设计者就可以很自信的表示，将其纳入到其基于TileLink的存储层次结构中，不会带来一致性协议相关的死锁。

TileLink is designed to be extensible, and supports a growing family of custom cache coherence policies implemented on top of it. TileLink also codifies a set of transaction types that are common to all protocols. In particular, it provides a set of transactions to service memory accesses made by agents that do not themselves have caches containing coherence policy metadata. These built-in transactions make TileLink a suitable target for the memory interfaces of accelerators, coprocessors and DMA engines, and allow such agents to automatically participate in a global shared memory space.

TileLink的设计是可拓展的，支持越来越多的定制的缓存一致性协议。TileLink整理了协议类型集合，对所有协议都是共用的。特别是，它提供了服务内存访问的事务集合，这是通过本身没有缓存的agents但包含了一致性策略元数据来得到的。这些内建的事务，使TileLink对于加速器、协处理器和DMA引擎的内存接口非常合适，使这些agents自动的参与到全局共享内存空间中。

Within the framework provided by TileLink, we provide generators for cache controller state machines as well as data and metadata memory arrays. By composing different sets of features using Chisel, we are able to generate outer level caches with reverse-mapped directories as well as broadcast-based snooping networks. While the extant Rocket Chip uses crossbars to connect tiles, alternate physical network designs can be plugged into our framework, and we are working to interface with designs produced by RapidIO [5] and OpenSoC [4].

在TileLink提供的框架中，我们为缓存控制器状态机和数据和元数据内存阵列提供了生成器。我们用Chisel包含了不同的特征集，可以生成外部层级的缓存，带有反向映射的目录，以及基于广播的snooping网络。现有的Rocket Chip使用的是crossbars来连接tiles，其他物理网络设计可以插入到我们的框架中，我们正在尝试与RapidIO和OpenSoC的设计进行接口。

To support memory-mapped IO (MMIO), we provide a TileLink interconnect generator. This generator takes an address map description and uses it to produce a system of routers for directing MMIO requests to the correct memory-mapped peripheral. Rocket Chip also provides converters from TileLink to the AXI4, AHB-Lite, and APB protocols in order to allow interfacing with external third-party peripherals and memory controllers.

为支持内存映射的IO，我们提供了一个TileLink互联生成器。这种生成器以地址映射描述为输入，用其产生一个路由系统，将MMIO请求引导到正确的内存映射的外部设备中。Rocket Chip也提供了从TileLink到AXI4，AHB-Lite和APB协议的转换器，以与外部的第三方外设和内存控制器进行接口。

## 8. Supporting Infrastructure

**FPGA Support**

We have released support for deploying Rocket Chip to a variety of Xilinx Zynq-7000 series FPGAs, such as the Xilinx ZC706 and ZedBoard. This capability is primarily used for fast simulation and typically achieves clockrates of 25-100 MHz depending on the configuration and the FPGA. We provide infrastructure and documentation for three use cases:

我们放出了支持，将Rocket Chip部署到一系列Xilinx Zynq-7000系列FPGAs，比如Xilinx ZC706和ZedBoard中。这种能力主要用于快速仿真，一般可以获得25-100MHz的时钟速度，这依赖于配置和FPGA。我们为三种使用情况提供基础设施和文档：

(1) Deploying pre-built images, for users without Vivado licenses who want to try a RISC-V system on their FPGA; 部署pre-built镜像，这是对没有Vivado许可但想在其FPGA上尝试RISC-V系统的用户的；

(2) Using existing FPGA infrastructure to deploy a custom SoC generated from Rocket Chip, for users making changes to the Rocket Chip RTL or configuration; 使用现有的FPGA基础设施来部署从Rocket Chip生成的定制SoC，这是让用户对Rocket Chip RTL或配置进行改动的；

(3) Building up the complete FPGA infrastructure from scratch, for users making involved changes to I/O or adding additional board support; 从头构建完整的FPGA基础设置，这是用于对IO进行相关改动和增加额外的板级支持的用户的；

The current infrastructure makes use of the hardened core inside the Zynq FPGA to provide support for proxied I/O devices, such as a console and block device. 目前的基础设施在Zynq FPGA中使用硬核，为代理的IO设备提供支持，比如console和块设备。

**Memory System Testing**

In order to test behaviors in our memory hierarchy which are not easy or efficient to test in software, we have designed a set of test circuits called GroundTest. These test circuits fit into the socket given to CPU tiles and directly drive different kinds of memory traffic to the L2 interconnect. The simplest test, MemTest, generates writes followed by reads at fixed strides across the address space. We also have a suite of regression tests which check for specific memory bugs which we have encountered in the past. This is especially useful for replicating ordering-dependent or timing-dependent bugs, since these are very difficult or impossible to reproduce in software. Finally, for more intensive testing, we generate random combinations of different memory operations and record a trace of the events. We then feed these traces into an external tool called Axe [8] in order to check for violations of the memory consistency model.

为测试内存层次结构的行为，这使用软件进行测试很不容易，效率很低，我们设计了一系列测试电路，称为GroundTest。这些测试电路适合于给定的CPU tiles，直接将各种类型的内存流量驱动到L2互联上。最简单的测试，MemTest，在整个地址空间中，在固定的strides生成writes followed by reads。我们还有回归测试包，检查特定的内存bugs，我们在过去遇到过的。这对于复制依赖于排序或依赖于时序的bugs特别有用，因为这在软件中很难复现，或不可能复现。最后，对于更复杂的测试，我们生成了不同内存操作的随机组合，记录了事件的轨迹。我们然后将这些轨迹反馈到外部工具中，称为Axe，为检查内存一致性模型的违反情况。

**Core Testing**

To help stress-test the cores via random testing, we utilize the RISC-V Torture Tester. Torture picks randomly from a library of code sequences and stitches them together to form a RISC-V assembly program. The instructions from each code sequence are interleaved with different code sequences. Each randomly-generated test program is executed both on the core under test and the Spike ISA simulator (RISC-V’s “golden model”). At the end of the test program, the register state is dumped to memory and the output from the core is compared to the output of Spike. If Torture finds a test program that exhibits an error, it regenerates smaller versions of the test program until it can find the smallest test program that still exercises the failure. Torture can be run for hours, creating new tests, and storing any error-finding programs it finds.

为通过随机测试来对核进行压力测试，我们利用了RISC-V Torture Tester。Torture从代码序列库中随机挑选，将其组合到一起，形成RISC-V汇编程序。每个代码序列中的指令与不同的代码序列交织。每个随机生成的测试程序在待测试的核上进行运行，也在Spike ISA模拟器上运行。在每个测试程序最后，寄存器状态都转存到内存中，从核中的输出与Spike的输出进行比较。如果Torture找到了一个表现出错误的测试程序，就重新生成测试程序的更小的版本，直到其找到最小的测试程序，仍然发现这个错误。Torture可以运行几小时，创建新的测试，存储其找到的任意错误寻找程序。

## 9 Silicon Prototypes

Figure 4 presents a timeline of UC Berkeley RISC-V microprocessor tapeouts that were created using earlier versions of the Rocket Chip generator. The 28 nm Raven chips combine a 64-bit RISC-V vector microprocessor with on-chip switched-capacitor DC-DC converters and adaptive clocking. The 45 nm EOS chips feature a 64-bit dual-core RISC-V vector processor with monolithically-integrated silicon photonic links [7, 9]. In total, we have taped out four Raven chips on STMicro-electronics’ 28 nm FD-SOI process, six EOS chips on IBM’s 45 nm SOI process, and one SWERVE chip on TSMC’s 28 nm process. The Rocket Chip generator successfully served as a shared code base for these eleven distinct systems; the best ideas from each design were incorporated back into the code base, ensuring maximal re-use even though the three distinct families of chips were specialized differently to evaluate distinct research ideas.

图4展示了UC Berkeley RISC-V微处理器tapeout的时间线，使用的是更早版本的Rocket Chip生成器创建的。28nm Raven芯片将64-bit RISC-V向量微处理器，与片上switched-capacitor DC-DC转换器和自适应时钟结合到一起。45nm EOS芯片的特色是，64-bit双核RISC-V向量处理器，带有一体集成的硅光子链接。总计，我们在ST微电子的28nm FS-SOI工艺上流片了4个Raven芯片，在IBM的45nm的SOI工艺上流片了6个EOS芯片，在TSMC的28nm工艺上流片了1个SWERVE芯片。Rocket Chip生成器成功的对这些11个不同的系统扮演了共享代码库的角色；每个设计的最好的思想，都纳入到了代码库中，确保了最大的重用，即使这三种芯片的专用性都很不一样，是用于评估不同的研究思想的。