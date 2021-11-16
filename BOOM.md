# The Berkeley Out-of-Order Machine (BOOM): An Industry Competitive, Synthesizable, Parameterized RISC-V Processor

Christopher Celio, David A. Patterson, Krste Asanović @ Berkeley

BOOM is a work-in-progress. Results shown are preliminary and subject to change as of 2015 June. BOOM正在工作过程中。现在给出的结果是初步的，目前是2015.06月的结果，后续会有变动。

## 1. The Berkeley Out-of-Order Machine

BOOM is a synthesizable, parameterized, superscalar out-of-order RISC-V core designed to serve as the prototypical baseline processor for future micro-architectural studies of out-of-order processors. Our goal is to provide a readable, open-source implementation for use in education, research, and industry.

BOOM是一个可综合的，参数化的，超标量乱序RISC-V核，设计作为未来乱序处理的微架构研究的基准原型处理器。我们的目标是给出一个可阅读的开源实现，用于教育，研究和工业。

BOOM is written in roughly 9,000 lines of the hardware construction language Chisel. We leveraged Berkeley’s open-source Rocket-chip SoC generator, allowing us to quickly bring up an entire multi-core processor system (including caches and uncore) by replacing the in-order Rocket core with an out-of-order BOOM core. BOOM supports atomics, IEEE 754-2008 floating-point, and page-based virtual memory. We have demonstrated BOOM running Linux, SPEC CINT2006, and CoreMark.

BOOM用硬件构建语言Chisel写了大约9000行。我们利用了Berkeley的开源Rocket-chip SoC生成器，使我们可以将顺序Rocket核替换为乱序BOOM核后，迅速得到整个多核处理器系统（包括缓存和非核心部分）。BOOM支持原子指令，IEEE 754-2008浮点指令，和基于页的虚拟内存。我们展示了运行Linux，SPEC CINT 2006和CoreMark的BOOM。

BOOM, configured similarly to an ARM Cortex-A9, achieves 3.91 CoreMarks/MHz with a core size of 0.47 mm2 in TSMC 45 nm excluding caches (and 1.1 mm2 with 32 kB L1 caches). The in-order Rocket core has been successfully demonstrated to reach over 1.5 GHz in IBM 45 nm SOI, with the SRAM access being the critical path. As BOOM instantiates the same caches as Rocket, BOOM should be similarly constrained to 1.5 GHz. So far we have not found it necessary to deeply pipeline BOOM to keep the logic faster than the SRAM access. With modest resource sizes matching the synthesizable MIPS32 74K, the the worst case path for BOOM’s logic is ∼2.2 GHz in TSMC 45 nm (∼30 FO4).

BOOM的配置与ARM Cortex-A9类似，在TSMC 45nm工艺下，去除缓存（和1.1mm2的32 kB L1缓存），在核大小0.47 mm2的情况下，获得了3.91 CoreMarks/MHZ的分数。顺序Rocket核在IBM 45nm SOI上展示达到了1.5 GHz的频率，SRAM的访问是关键路径。由于BOOM例化了Rocket一样的缓存，BOOM应当类似的约束在1.5 GHz之下。迄今为止，我们还没有发现加深BOOM的流水线，以保持逻辑比SRAM访问更快的必要。在中等资源大小下，与和综合的MIPS32 74K匹配，在TSMC 45nm (~30 FO4)工艺下，BOOM逻辑的最坏的路径是大约2.2GHz。

## 2. Leveraging New Infrastructure

The feasibility of BOOM is in large part due to the available infrastructure that has been developed in parallel at Berkeley.

BOOM的可行性主要是由于在Berkeley同时开发的可用的基础设施。

BOOM implements the open-source RISC-V ISA, which was designed from the ground-up to enable VLSI-driven computer architecture research. It is clean, realistic, and highly extensible. Available software includes the GCC and LLVM compilers and a port of the Linux operating system.[6]

BOOM实现了开源的RISC-V ISA，这是从头设计以进行VLSI驱动的计算机架构的研究。这个架构非常干净，现实，而且高度可扩展。可用的软件包括GCC和LLVM编译器，和Linux。

BOOM is written in Chisel, an open-source hardware construction language developed to enable advanced hardware design using highly parameterized generators. Chisel allows designers to utilize concepts such as object orientation, functional programming, parameterized types, and type inference. From a single Chisel source, Chisel can generate a cycle-accurate C++ simulator, Verilog targeting FPGA designs, and Verilog targeting ASIC tool-flows.[2]

BOOM是用Chisel写的，这是一种开源的硬件构建语言，开发用于高级硬件设计，使用了高度参数化的生成器。Chisel使设计者可以利用面向对象，函数式编程，参数化类型，和类型推理的概念。单从Chisel源，Chisel可以生成cycle-accurate C++仿真器，面向FPGA设计的Verilog，和面向ASIC工具流的Verilog。

UC Berkeley also provides the open-source Rocket-chip SoC generator, which has been successfully taped out seven times in two different, modern technologies.[6, 10] BOOM makes significant use of Rocket-chip as a library – the caches, the uncore, and functional units all derive from Rocket. In total, over 11,500 lines of code is instantiated by BOOM.

UC Berkeley还提供了开源的Rocket-chip SoC生成器，已经在两种不同的现代技术中成功的流片了7次。BOOM使用了很多Rocket-chip中的库，缓存，非核部分，一些功能单元都是从Rocket中导出的。总计，BOOM例化了超过11500行代码。

## 3. Methodology: What We Plan to Do

The typical methodology for single-core studies, as gathered from an informal sampling of ISCA 2014 papers, is to use CPU2006 coupled with a SimPoints[12]-inspired methodology to choose the most representative section of the reference input set. Each sampling point is typically run for around 10-100 million instructions of detailed software-based simulation.

从ISCA 2014论文的非正式采样看来，单核研究的典型方法论，是使用CPU2006和SimPoints一起，启发得到的方法论，以选择最有代表性的参考输入集。每个采样点一般运行了大约10-100百万命令，是基于纯软件的仿真。

The average CPU2006 benchmark is roughly 2.2 trillion instructions, with many of the benchmarks exhibiting multiple phases of execution.[9] While completely untenable for software simulators, FPGA-based simulators can bring runtimes to within reason – a 50 MHz FPGA simulation can take over 12 hours for a single benchmark. Moreover, we hope to utilize an FPGA cluster to run all the SPEC workloads in parallel.

CPU2006基准测试平均大约2.2 trillion指令，很多基准测试都有多阶段执行。这完全不能用软件模拟，但基于FPGA的模拟器可以使运行时间比较合理，一个50MHz的FPGA仿真可以用12个小时进行单个基准测试。而且，我们希望利用一个FPGA集群来并行运行所有的SPEC workloads。

## 4. Comparison to Commercial Designs

Table 1 shows preliminary results of BOOM and Rocket for the CoreMark EEMBC benchmark (we use CoreMark because ARM does not offer SPEC results for the A9 and A15 cores). Our aim is to be competitive in both performance and area against low-power, embedded out-of-order cores.

表1给出了BOOM和Rocket在CoreMark EEMBC基准测试中的初步结果（我们使用CoreMark，因为ARM没有给出A9和A15核的SPEC结果）。我们的目标是在性能和面积上，与低功耗，嵌入式乱序核相比都有竞争力。

## 5. Related Work

There have been many academic efforts to implement out-of-order cores. The Illinois Verilog Model (IVM) is a 4-issue, out-of-order core designed to study transient faults.[13] The Santa Cruz Out-of-Order RISC Engine (SCOORE) was designed to efficiently target both ASIC and FPGA generation. However, SCOORE lacks a synthesizable fetch unit.

有很多学术工作来实现乱序核。IVM是一个4发射乱序核，设计用于研究瞬态故障。SCOORE设计用于高效的用在ASIC和FPGA生成上。但是，SCOORE缺少可综合的fetch单元。

FabScalar is a tool for composing synthesizable out-of-order cores. It searches through a library of parameterized components of varying width and depth, guided by performance constraints given by the designer. FabScalar has been demonstrated on an FPGA,[8] however, as FabScalar did not implement caches, all memory operations were treated as cache hits. Later work incorporated the OpenSPARC T2 caches in a tape-out of FabScalar.[11]

FabScalar是一个工具，可合成可综合的乱序核。它在不同宽度和深度的参数化的组件库中进行搜索，并以设计者给出的性能约束进行引导。FabScalar在FPGA上进行了证明，但是，由于FabScalar没有实现缓存，所有的内存操作都被认为是cache hits。后续的工作纳入了OpenSPARC T2缓存，对FabScalar进行了流片。

The Sharing Architecture is composed of a two-wide out-of-order core (or “slice”) that can be combined with other slices to form a single, larger out-of-order core. By implementing a slice in RTL, they were able to accurately demonstrate the area costs associated with reconfigurable, virtual cores.[14]

The Sharing Architecture是由two-wide乱序核组成，可以与其他slices结合到一起，以形成单个更大的乱序核。通过用RTL实现一个slice，可以准确的展示相关的可配置的虚拟核的面积代价。

## 6. Lessons Learned

**Single-board FPGAs have gotten more capable of handling mobile processor designs**. Chisel provides a back-end mechanism to generate memories optimized for FPGAs, but requires no changes to the processor’s source code. While some coding patterns map poorly to FPGAs (e.g., large variable shifters), generally techniques that map well to ASICs also map well to FPGAs.

单板FPGAs更加能够处理移动处理器设计了。Chisel给出了一种后端机制，以生成为FPGA优化的内存生成，但不需要处理器的源码有任何变化。一些编码模式向FPGAs的映射很差（如，大型可变shifters），但一般的向ASIC映射很好的技术，也向FPGA映射的很好。

**Re-use is critical**. Some of the most difficult parts of building a processor – for example the cache coherency system, the privileged ISA support, and the FPGA and ASIC flows – came to BOOM “for free” via the Rocket-chip SoC generator. And as the Rocket-chip SoC evolves, BOOM inherits the new improvements.

重用非常关键。构建一个处理器中一些最难的部分，比如缓存一致性系统，特权ISA支持，FPGA和ASIC流，都是通过Rocket-chip SoC生成器免费带到BOOM中的。随着Rocket-chip SoC的演进，BOOM也继承了新的改进。

**Benchmarks are harder to use than they should be**. Benchmarks can be difficult to work with and exhibit poor performance portability across different processors, address modes, and ISAs. Many benchmarks (like CoreMark) are written to target 32-bit addresses, which can cause poor code generation for 64-bit processors. We built a histogram generator into the RISC-V ISA simulator to help direct us to potential problem areas. However, additional compiler optimizations are needed to improve 64-bit RISC-V code generation.

基准测试用起来很难。基准测试很难用，在不同处理器，寻址方式和ISA之间，展现出了很差的移植性。很多基准测试（如CoreMark）是用于32-bit寻址的，这对于64-bit处理器会导致很差的代码生成。我们构建了一个histogram生成器，放到了RISC-V ISA仿真器中，以帮助我们指向可能的问题区域。但是，需要额外的编译器优化，来改进64-bit RISC-V代码生成。

We were also surprised to find that SPECINT contains significant floating point code – an integer core may spend over half its time executing software FP routines. As academic SPEC results are typically reported in terms of CPI, we must be careful to not optimize for the wrong cases. We added hardware FP support to BOOM to address this issue.

我们很惊讶发现，SPECINT包含很多浮点代码，一个整数核会耗费超过一半的时间执行软件的FP子程序。因为学术SPEC结果一般都是以CPI给出的，我们必须小心不要为这些错误的情况进行优化。我们对BOOM加入了硬件浮点支持，以处理这个问题。

Finally, we found SPEC difficult to work with, especially in non-native environments. We created the Speckle wrapper to help facilitate cross-compiling and generating portable directories to run on simulators and FPGAs.[5]

最后，我们发现SPEC很难用，尤其是在非原生环境中。我们创建了Speckle wrapper，以帮助促进跨编译，生成可移植的目录，在模拟器和FPGAs中进行运行。

**Diagnosing bugs that occur billions of cycles into a program is hard**. We mostly rely on a Chisel-generated C++ simulator for debugging, but at roughly 30 KIPS, 1 billion cycles takes 8 hours. A torture-test generator (and a suite of small test codes) is invaluable.

诊断数十亿周期的程序的bugs很难。我们主要依靠Chisel生成的C++仿真器进行debugging，但大约是30 KIPS，十亿周期耗时8个小时。稳定性测试生成器会非常有用。