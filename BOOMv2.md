# BOOM v2 - an open-source out-of-order RISC-V core

Christopher Celio et. al. @ Berkeley

## 0. Abstract

This paper presents BOOM version 2, an updated version of the Berkeley Out-of-Order Machine first presented in [3]. The design exploration was performed through synthesis, place and route using the foundry-provided standard-cell library and the memory compiler in the TSMC 28 nm HPM process (high performance mobile).

本文给出了BOOMv2，BOOM的升级版。通过综合、布局布线和使用代工厂提供的TSMC 28nm HPM工艺的标准单元库，和内存编译器，进行了设计探索。

BOOM is an open-source processor that implements the RV64G RISC-V Instruction Set Architecture (ISA). Like most contemporary high-performance cores, BOOM is superscalar (able to execute multiple instructions per cycle) and out-of-order (able to execute instructions as their dependencies are resolved and not restricted to their program order). BOOM is implemented as a parameterizable generator written using the Chisel hardware construction language [2] that can used to generate synthesizable implementations targeting both FPGAs and ASICs.

BOOM是一个开源处理器，实现了RV64G RISC-V ISA。与多数同时的高性能核类似，BOOM是超标量的（可以在每个周期执行多条指令）和乱序的（可以在解析了其依赖关系下执行指令，不局限在其程序顺序下）。BOOM实现为一个参数化的生成器，使用Chisel语言写成，可以用于生成在FPGAs和ASICs下可综合的实现。

BOOMv2 is an update in which the design effort has been informed by analysis of synthesized, placed and routed data provided by a contemporary industrial tool flow. We also had access to standard single- and dual-ported memory compilers provided by the foundry, allowing us to explore design trade-offs using different SRAM memories and comparing against synthesized flip-flop arrays. The main distinguishing features of BOOMv2 include an updated 3-stage front-end design with a bigger set-associative Branch Target Buffer (BTB); a pipelined register rename stage; split floating point and integer register files; a dedicated floating point pipeline; separate issue windows for floating point, integer, and memory micro-operations; and separate stages for issue-select and register read.

BOOMv2是升级版，其设计努力主要在分析了综合的，布局布线的数据，由同时的工业工具流给出。我们还有代工厂提供的单端双端内存编译器，使我们可以使用不同的SRAM内存探索设计折中，与综合的寄存器阵列进行比较。BOOMv2的主要突出特色是，包含了一个升级的3阶段前端设计，带有更大的组相联BTB；一个流水线的寄存器重命名阶段；分离的浮点和整数寄存器组；一个专用的浮点流水线；对浮点，整数和内存微运算的单独发射窗口；发射选择和寄存器读的分离阶段。

Managing the complexity of the register file was the largest obstacle to improving BOOM’s clock frequency. We spent considerable effort on placing-and-routing a semi-custom 9-port register file to explore the potential improvements over a fully synthesized design, in conjunction with microarchitectural techniques to reduce the size and port count of the register file. BOOMv2 has a 37 fanout-of-four (FO4) inverter delay after synthesis and 50 FO4 after place-and-route, a 24% reduction from BOOMv1’s 65 FO4 after place-and-route. Unfortunately, instruction per cycle (IPC) performance drops up to 20%, mostly due to the extra latency between load instructions and dependent instructions. However, the new BOOMv2 physical design paves the way for IPC recovery later.

寄存器组复杂性的管理，是提升BOOM时钟频率的最大障碍。我们花费了相当大的努力在半定制的9端口寄存器组的布局布线上，以在全综合设计上探索可能的改进，与微架构技术一起，减少寄存器组的大小和端口数量。BOOMv2在综合后有37个FO4 inverter延迟，在布局布线后有50 FO4，比BOOMv1的布局布线后65 FO4有24%的降低。不幸的是，IPC性能降低了20%，主要是因为载入指令和依赖的指令的额外的延迟。但是，新的BOOMv2的物理设计为后续的IPC恢复铺平了道路。

## 1. Introduction

BOOM was inspired initially by the MIPS R10K and Alpha 21264 processors from the 1990s, whose designs teams provided relatively detailed insight into their processors’ microarchitectures [6, 7, 11]. However, both processors relied on custom, dynamic logic which allowed them to achieve very high clock frequencies despite their very short pipelines – the Alpha 21264 has 15 fanout-of-four (FO4) inverter delays [4]. As a comparison, the synthesizable Tensilica’s Xtensa processor, fabricated in a 0.25 micron ASIC process and contemporary with the Alpha 21264, was estimated to have roughly 44 FO4 delays [4].

BOOM最初是受到1990s的MIPS R10K和Alpha 21264处理器启发开始的，这两种设计的团队对其处理器的微架构给出了相对细节的洞见。但是，两种处理器都依赖于定制的动态逻辑，使其可以得到很高的时钟频率，尽管其流水线都很短，Alpha 21264的FO4 inverter延迟为15。比较起来，可综合的Tensilica的Xtensa处理器，在0.25 micron ASIC工艺上制造的，与Alpha 21264属于同时代，其FO4延迟大约是44。

As BOOM is a synthesizable processor, we must rely on microarchitecture-level techniques to address critical paths and add more pipeline stages to trade off instructions per cycle (IPC), cycle time (frequency), and design complexity. The exploration in this work is performed by using only the available single-and dual-ported memory compilers, without custom circuit design. It should be noted that, as process nodes have become smaller, transistor variability has increased, and power-efficiency has become restricting, many of the more aggressive custom techniques have become more difficult and expensive to apply [1]. Modern high-performance processors have largely limited their custom design efforts to more regular structures such as arrays and register files.

由于BOOM是一个可综合的处理器，我们必须依赖于微架构级的技术，来处理关键路径，增加更多的流水线阶段来将IPC，频率和设计复杂度进行折中。本文的探索只使用了单端和双端的内存编译器进行，没有定制的电路设计。必须要指出，由于工艺节点变得更小，晶体管的变化增加了，能量效率变得更加严苛，很多更激进的定制技术的应用变得更难更昂贵。现代高性能处理器将其定制设计努力主要限制在了更常规的结构中，比如阵列和寄存器组。

## 2. BOOMv1

BOOMv1 follows the 6-stage pipeline structure of the MIPS R10K – fetch, decode/rename, issue/register-read, execute, memory, and writeback. During decode, instructions are mapped to micro-operations (uops) and during rename all logical register specifiers are mapped to physical register specifiers. For design simplicity, all uops are placed into a single unified issue window. Likewise, all physical registers (both integer and floating point registers) are located in a single unified physical register file. Execution Units can contain a mix of integer units and floating point units. This greatly simplifies floating point memory instructions and floating point-integer conversion instructions as they can read their mix of integer and floating point operands from the same physical register file. BOOMv1 also utilized a short 2-stage front-end pipeline design. Conditional branch prediction occurs after the branches have been decoded.

BOOMv1的设计与MIPS R10K一样是6级流水线结构-取指，解码/重命名，发射/读寄存器，执行，内存和写回。在解码阶段，指令被映射到微操作(uops)，在重命名阶段，所有逻辑寄存器指示符被映射到物理寄存器指示符上。为了设计上的简单性，所有uops都放在了一个统一的发射窗口中。类似的，所有物理寄存器（包括整数和浮点数寄存器）都放在了单个统一的物理寄存器组中。执行单元包含了整数单元和浮点单元的混合。这极大的简化了浮点内存指令和浮点整型转换指令，因为他们可以从相同的物理寄存器组中读整形和浮点操作数的混合。BOOMv1还利用了一个很短的2级前端流水线设计。条件分支预测在分支被解码后进行。

The design of BOOMv1 was partly informed by using educational technology libraries in conjunction with synthesis tools. While using educational libraries was useful for finding egregious mistakes in control logic signals, it was less useful in informing the organization of the datapaths. Most critically, we lacked access to a commercial memory compiler. Although tools such as Cacti [10] can be used to analytically model the characteristics of memories, Cacti works best for reproducing memories that it has been tuned against such as single-port, cache-sized SRAMs. However, BOOM makes use of a multitude of smaller SRAM arrays for modules such as branch predictor tables, prediction snapshots, and address target buffers.

BOOMv1的设计，部分受益于教育技术库与综合工具的结合。使用教育库对找到控制逻辑信号中极坏的错误非常有用，但对于理清数据通路的组织来说，则没那么有用。最关键的，我们缺少一个商用内存编译器。虽然像Cacti这样的工具可以用于解析的对内存的特性进行建模，但Cacti的效果对于已经调节好对于单端，cache大小的SRAMs效果最好。但是，BOOM使用了多个更小的SRAM阵列，比如分支预测器表格，预测snapshots，和地址目标buffers。

Upon analysis of the timing of BOOMv1 using TSMC 28 nm HPM, the following critical paths were identified: 对BOOMv1使用TSMC 28nm的HPM的时序进行分析，识别出了下列关键路径：

(1) issue select 选择发射

(2) register rename busy table read 寄存器重命名忙表读

(3) conditional branch predictor redirect 条件分支预测期重定向

(4) register file read 寄存器组读

The last path (register-read) only showed up as critical during post-place-and-route analysis. 最后的路径（寄存器读）只在布局布线后的分析中表现为关键路径。

## 3. BOOMV2

BOOMv2 is an update to BOOMv1 based on information collected through synthesis, place, and route using a commercial TSMC 28 nm process. We performed the design space exploration by using standard single- and dual-ported memory compilers provided by the foundry and by hand-crafting a standard-cell-based multi-ported register file.

BOOMv2是BOOMv1的升级，基于通过综合，布局布线收集到的信息，使用的是TSMC商用28nm工艺。我们进行的设计空间探索，使用的是标准的单端和双端内存编译器，由代工厂提供，通过手工设计一个基于标准单元的多端寄存器组。

Work on BOOMv2 took place from April 9th through Aug 9th and included 4,948 additions and 2,377 deleted lines of code (LOC) out of the now 16k LOC code base. The following sections describe some of the major changes that comprise the BOOMv2 design.

BOOMv2的工作是从4月9日到9月9日，包含增加了4948行代码，删除了2377行代码，现在代码库共计16K行代码。下面的小节描述了一些主要变化，涵盖在BOOMv2的设计中。

### 3.1 Frontend (Instruction Fetch)

The purpose of the frontend is to fetch instructions for execution in the backend. Processor performance is best when the frontend provides an uninterrupted stream of instructions. This requires the frontend to utilize branch prediction techniques to predict which path it believes the instruction stream will take long before the branch can be properly resolved. Any mispredictions in the frontend will not be discovered until the branch (or jump-register) instruction is executed later in the backend. In the event of a misprediction, all instructions after the branch must be flushed from the processor and the frontend must be restarted using the correct instruction path.

前端的目的是取指令，以在后端进行执行。当前端给出不间断的指令流时，处理器性能最好。这需要前端利用分支预测技术，在分支被合理解析之前很久，来预测指令流会采取哪条路径。任何前端中的错误预测，都会直到分支（或跳转寄存器）指令后面在后端执行时才会被发现。如果发生了错误预测，分支后的所有指令都需要从处理器中冲刷掉，前端必须使用正确的指令路径重启。

The frontend end relies on a number of different branch prediction techniques to predict the instruction stream, each trading off accuracy, area, critical path cost, and pipeline penalty when making a prediction.

前端依靠几种不同的分支预测技术来预测指令流，在进行预测时，每种都在准确率、面积、关键路径代价、流水线惩罚中有所折中。

**Branch Target Buffer (BTB)**. The BTB maintains a set of tables mapping from instruction addresses (PCs) to branch targets. When a lookup is performed, the look-up address indexes into the BTB and looks for any tag matches. If there is a tag hit, the BTB will make a prediction and may redirect the frontend based on its predicted target address. Some hysteresis bits are used to help guide the taken/not-taken decision of the BTB in the case of a tag hit. The BTB is a very expensive structure – for each BTB entry it must store the tag (anywhere from a partial tag of ≈20 bits to a full 64-bit tag) and the target (a full 64 bit address).

BTB维护了从指令地址PCs到分支目标的映射表集。当进行查找表时，查找的地址索引到BTB中，查找任何tag匹配。如果有一个tag匹配，BTB会进行一次预测，可能会基于其预测的目标地址，来重新引导前端。一些滞后bits用于在tag hit后帮助引导BTB的taken/not taken决定。BTB是一种非常昂贵的结构，因为每个BTB entry都必须存储tag（从一个部分tag大约20bits到完整的64-bit tag）和目标（一个完整的64 bit地址）。

**Return Address Stack (RAS)**. The RAS predicts function returns. Jump-register instructions are otherwise quite difficult to predict, as their target depends on a register value. However, functions are typically entered using a Function Call instruction at address A and return from the function using a Return instruction to address A+1. – the RAS can detect the call, compute and then store the expected return address, and then later provide that predicted target when the Return is encountered. To support multiple nested function calls, the underlying RAS storage structure is a stack.

RAS预测函数的返回地址。跳转寄存器指令是很难预测的，因为其目标依赖于寄存器值。但是，进入函数通常是使用一个Function Call指令，在地址A处，从函数返回，是使用一个Return指令，到地址A+1处，RAS可以检测到这个调用，计算然后存储期望的返回地址，然后当遇到Return时，给出预测的目标地址。为支持多个嵌套的函数调用，RAS的潜在存储结构是一个堆栈。

**Conditional Branch Predictor (BPD)**. The BPD maintains a set of prediction and hysteresis tables to make taken/not taken predictions based on a look-up address. The BPD only makes taken/not-taken predictions – it therefore relies on some other agent to provide information on what instructions are branches and what their targets are. The BPD can either use the BTB for this information or it can wait and decode the instructions themselves once they have been fetched from the instruction cache. Because the BPD does not store the expensive branch targets, it can be much denser and thus make more accurate predictions on the branch directions than the BTB – whereas each BTB entry may be 60 to 128 bits, the BPD may be as few as one or two bits per branch. A common arch-type of BPD is a global history predictor. Global history predictors work by tracking the outcome of the last N branches in the program (“global”) and hashing this history with the look-up address to compute a look-up index into the BPD prediction tables. For a sophisticated BPD, this hashing function can become quite complex. BOOM’s predictor tables are placed into single-ported SRAMs. Although many prediction tables are conceptually “tall and skinny” matrices (thousands of 2- or 4-bit entries), a generator written in Chisel transforms the predictor tables into a square memory structure to best match the SRAMs provided by a memory compiler.

BPD维护了一个预测和滞后表集合，基于查找表地址来进行taken/not taken的预测。BPD只进行taken/not taken的预测，因此它会依赖于其他agent，来给出什么指令是分支，其目标是哪里的信息。BPD可以使用BTB的信息，或其可以等待指令解码后，一旦其从指令缓存中取出。因为BPD并没有存储昂贵的分支目标，因此可以更加密集，可以比BTB在分支方向上进行更准确的预测，每个BTB entry会有60到128 bits那么大，而BPD对每个分支只有1-2 bits大。一种常见的BPD是全局历史预测器(Global History Predictor)。全局历史预测器，通过跟踪程序中最后N个分支（全局的），对这个历史与查找地址进行hash，来计算一个查找索引，存储到BPD预测表中，来进行工作。对于一个复杂的BPD，这种hash函数会变得非常复杂。BOOM的预测器表放到了单端SRAM中。虽然很多预测表在概念上是很高很瘦的矩阵（数千个2或4 bits的entry），用Chisel写的一种生成器，会将这种预测器的表转换成一个方形内存结构，以与内存编译器提供的SRAMs进行匹配。

Figure 3 shows the pipeline organization of the frontend. We found the a critical path in BOOMv1 to be the conditional branch predictor (BPD) making a prediction and redirecting the fetch instruction address in the F2 stage, as the BPD must first decode the newly fetched instructions and compute potential branch targets. For BOOMv2, we provide a full cycle to decode the instructions returning from the instruction cache and target computation (F2) and perform the redirection in the F3 stage. We also provide a full cycle for the hash indexing function, which removes the hashing off the critical path of Next-PC selection.

图3展示了前端的流水线组织。我们在BOOMv1中发现了一条关键路径，是BPD进行了一个预测，并重新引导取指令地址到了F2阶段，因为BPD必须首先解码新取的指令，计算可能的分支目标。对于BOOMv2，我们给出一个完整的周期来解码从指令缓存中返回的指令，进行计算(F2)，然后在F3阶段进行重新引导。我们还对hash索引函数给了一个完整的周期，将hash函数的关键路径移除到了下一个PC的选择上。

We have added the option for the BPD to continue to make predictions in F2 by using the BTB to provide the branch decode and target information. However, we found this path of accessing the prediction tables and redirecting the instruction stream in the same cycle requires a custom array design.

我们对BPD加入了在F2进行预测的选项，即通过使用BTB来给出分支的解码和目标信息。但是，我们发现在同一周期中访问预测表和重新引导指令流的这条路径需要一个定制的阵列设计。

Another critical path in the frontend was through the fully-associative, flip-flop-based BTB. We found roughly 40 entries to be the limit for a fully-associative BTB. We rewrote the BTB to be set-associative and designed to target single-ported memory. We experimented with placing the tags in flip-flop-based memories and in SRAM; the SRAM synthesized at a slower design point but placed-and-routed better.

前端中的另一个关键路径是通过全相联的基于触发器的BTB。我们发现大约40个entries是一个全相联BTB的极限。我们重写了BTB为组相联的，设计为面向单端存储的。我们将基于在触发器的存储和在SRAM中放入tags进行试验；以较低的设计点综合的SRAM但布局布线后的更好一些。

### 3.2 Distributed Issue Windows

The issue window holds all inflight and un-executed micro-ops (uops). Each issue port selects from one of the available ready uops to be issued. Some processors, such as Intel’s Sandy Bridge processor, use a “unified reservation station” where all uops are placed in a single issue window. Other processors provide each functional unit its own issue window with a single issue select port. Each has its benefits and its challenges.

发射窗口保持所有在飞行中的和未执行的微操作(uops)。每个发射端口从这些可用的准备好的uops中选择一个来进行发射。一些处理器，比如Intel的Sandy Bridge处理器，使用了一种统一保留站，其中所有的uops都放在了单个发射窗口中。其他处理器给每个功能单元其自己的发射窗口，带有单个发射选择窗口。每个都有其优势和挑战。

The size of the issue window denotes the number of in-flight, un-executed instructions that can be selected for out-of-order execution. The larger the window, the more instructions the scheduler can attempt to re-order. For BOOM, the issue window is implemented as a collapsing queue to allow the oldest instructions to be compressed towards the top. For issue-select, a cascading priority encoder selects the oldest instruction that is ready to issue. This path is exacerbated either by increasing the number of entries to search across or by increasing the number of issue ports. For BOOMv1, our synthesizable implementation of a 20 entry issue window with three issue ports was found to be aggressive, so we switched to three distributed issue windows with 16 entries each (separate windows for integer, memory, and floating point operations). This removes issue-select from the critical path while also increasing the total number of instructions that can be scheduled. However, to maintain performance of executing two integer ALU instructions and one memory instruction per cycle, a common configuration of BOOM will use two issue-select ports on the integer issue window.

发射窗口的大小表示了在飞行的，未执行的指令的数量，可以被选择进行乱序执行。窗口越大，scheduler试图重新排序的指令越多。对于BOOM，发射窗口实现为一个正在坍缩的队列，使最老的指令朝顶部压缩。对于发射选择，一个级联的优先级编码器选择准备好的最老的指令来进行发射。如果增加要搜索的entries数量，或增加发射端口的数量，那么这条路径就会恶化。对于BOOMv1，带有3个发射端口的20个entry发射窗口的可综合实现，太过激进了，所以我们切换到了3个分布式的发射窗口，每个有16个entries（对整数、内存和浮点操作的分离窗口）。这从关键路径中去除了发射选择，同时增加了可以scheduled的指令的总计数量。但是，为保每个周期持执行两条整数ALU指令和一条内存指令的性能，BOOM的一个常见配置是，在整数发射窗口使用两个发射选择端口。

### 3.3 Register File Design

One of the critical components of an out-of-order processor, and most resistant to synthesis efforts, is the multi-ported register file. As memory is expensive and time-consuming to access, modern processor architectures use registers to temporarily store their working set of data. These registers are aggregated into a register file. Instructions directly access the register file and send the data read out of the registers to the processor’s functional units and the resulting data is then written back to the register file. A modest processor that supports issuing simultaneously to two integer arithmetic units and a memory load/store unit requires 6 read ports and 3 write ports.

乱序处理器的一个关键组成部分，对综合工作最有抵抗力的，是多端口寄存器组。因为内存是昂贵的，访问非常耗时，现在处理器架构都使用寄存器来临时存储工作数据集。这些寄存器聚集在一起，成为寄存器组。指令直接访问这些寄存器组，将从这些寄存器中读出的数据送到处理器的功能单元中，得到的数据写回到寄存器组中。一个中等的处理器，支持同时发射到两个整数算术单元，和一个内存读写单元，需要6个读端口和3个写端口。

The register file in BOOMv1 provided many challenges – reading data out of the register file was a critical path, routing read data to functional units was a challenge for routing tools, and the register file itself failed to synthesize properly without violating the foundry design rules. Both the number of registers and the number of ports further exacerbate the challenges of synthesizing the register file.

BOOMv1中的寄存器组提出了很多挑战，从寄存器组读数据是一条关键路径，读取数据到功能单元的布线，是布线工具的挑战，寄存器组本身不能进行合理的综合，必须要违反代工厂的设计规则。寄存器的数量和端口的数量，进一步家具了综合寄存器组的挑战。

We took two different approaches to improving the register file. The first level was purely microarchitectural. We split apart issue and register read into two separate stages – issue select is now given a full cycle to select and issue uops, and then another full cycle is given to read the operand data out of the register file. We lowered the register count by splitting up the unified physical register file into separate floating point and integer register files. This split also allowed us to reduce the read-port count by moving the three-operand fused-multiply add floating point unit to the smaller floating point register file.

我们选择两种不同的方法来改进寄存器组。第一个层次是纯微架构的。我们将发射和寄存器读分成两个单独的阶段，发射选择现在给了一个完整的周期，来选择和发射uops，另一个完整的周期给定去从寄存器组中读操作数。我们降低了寄存器的计数，将统一的物理寄存器组分成了单独的浮点和整数寄存器组。这种分离也使我们减少了读端口计数，将三操作数混合乘加浮点单元移到了更小的浮点寄存器组中。

The second path to improving the register file involved physical design. A significant problem in placing and routing a register file is the issue of shorts – a geometry violation in which two metal wires that should remain separate are physically attached. These shorts are caused by attempting to route too many wires to a relatively dense regfile array. BOOMv2’s 70 entry integer register file of 6 read ports and 3 write ports comes to 4,480 bits, each needing 18 wires routed into and out of it. There is a mismatch between the synthesized array and the area needed to route all required wires, resulting in shorts.

第二种改进寄存器组的路径涉及到物理设计。在寄存器组的布局布线中，一个显著的问题是短路的问题，这是一个几何规则的违反，其中两个应当保持分离的金属线，在物理上是连在一起的。这些短路是试图对一个相对密集的regfile阵列布线导致的。BOOMv2的70 entry整数寄存器组，有6个读端口，和3个写端口，共有4480 bits，每个需要18条线布线进出。综合的阵列，和需要布线所有需要的线的面积，这之间不匹配，因此导致了短路。

Instead we opted to blackbox the Chisel register file and manually craft a register file bit out of foundry-provided standard cells. We then laid out each register bit in an array and let the placer automatically route the wires to and from each bit. While this fixed the wire shorting problem, the tri-state buffers struggled to drive each read wire across all 70 registers. We therefore implemented hierarchical bitlines; the bits are divided into clusters, tri-states drive the read ports inside of each cluster, and muxes select the read data across clusters.

我们选择将Chisel寄存器组黑箱掉，然后用代工厂提供的标准单元，手工画出了一个寄存器组bit。然后我们将每个寄存器bit排列成一个阵列，让布局器自动布线每个bit。这修正了线短路的问题，三态buffer勉强在所有70个寄存器中驱动每条读线。我们因此实现了层次化的bitlines；这些bits分成了集群，三态驱动每个集群中的读端口，mux在集群中选择读数据。

As a counter-point, the smaller floating point register file (three read ports, two write ports) is fully synthesized with no placement guidance.

作为一个计数点，更小的浮点寄存器组（三个读端口，两个写端口）可以完全综合，不需要布局引导。

## 4. Lessons Learned

The process of taking a register-transfer-level (RTL) design all the way through a modern VLSI tool flow has proven to be a very valuable experience.

从RTL开始设计并经历现代VLSI工具流的过程，是非常宝贵的经验。

Dealing with high port count memories and highly-congested wire routing are likely to require microarchitectural solutions. Dealing with the critical paths created by memories required microarchitectural changes that likely hurt IPC, which in turn motivates further microarchitectural changes. Lacking access to faithful memory models and layout early in the design process was a serious handicap. A manually-crafted cell approach is useful for exploring the design space before committing to a more custom design.

处理很高端口计数内存和highly-congested布线，很可能需要微架构方面的解决方案。处理内存创建的关键路径，需要微架构方面的变化，这很可能会伤害IPC，这又进一步促进了微架构的变化。不能在设计过程的早期访问忠实的内存模型和布局，是一个很严重的缺陷。手工画的单元方法，是在寻求更定制的设计之前，对探索设计空间非常有用的。

Memory timings are sensitive to their aspect ratios; tall, skinny memories do not work. We wrote Chisel generators to automatically translate large aspect ratio memories into rectangular structures by changing the index hashing functions and utilizing bit-masking of reads and writes.

内存时序对其纵横比很敏感；高高瘦瘦的内存不好用。我们写的Chisel生成器将大纵横比的存储翻译成长方形的结构，改变了索引hash函数，利用读和写的bit-masking。

Chasing down and fixing all critical paths can be a fool’s errand. The most dominating critical path was the register file read as measured from post-place and route analysis. Fixing critical paths discovered from post-synthesis analysis may have only served to worsen IPC for little discernible gain in the final chip.

追踪并修改所有的关键路径，是一个愚蠢的错误。最主要的关键路径是寄存器组读，这是从布局布线后的分析测量到的。修正综合后分析发现的关键路径，只会使IPC变得更差，对最终的chip的提升几乎没有。

Describing hardware using generators proved to be a very useful technique; multiple design points could be generated and evaluated, and the final design choices could be committed to later in the design cycle. We could also increase our confidence that particular critical paths were worth pursuing; by removing functional units and register read ports, we could estimate the improvement from microarchitectural techniques that would reduce port counts on the issue windows and register file.

使用生成器描述硬件，是一种非常有用的技术；可以生成并评估多个设计点，最终的设计选择可以在设计周期中后续做出。我们还可以增加我们的信心，特定的关键路径是值得追踪的；通过去除功能单元和寄存器读端口，我们可以从微架构技术估计改进，在发射窗口和寄存器组中降低了端口计数。

Chisel is a wonderfully expressive language. With a proper software engineering of the code base, radical changes to the datapaths can be made very quickly. Splitting up the register files and issue windows was a one week effort, and pipelining the register rename stage was another week. However, physical design is a stumbling block to agile hardware development. Small changes could be reasoned about and executed swiftly, but larger changes could change the physical layout of the chip and dramatically affect critical paths and the associated costs of the new design point.

Chisel是一种非常好的有表达力的语言。对代码库进行合理的软件工程工作，可以对数据通路很快的进行很激进的改动。将寄存器组和发射窗口分开，是一个星期的工作，寄存器重命名阶段的流水线化，是另一个星期。但是，物理设计对于敏捷硬件开发，是一个绊脚石。很小的变化可以推理得到，很迅速的执行，但更大的变化会改变芯片的物理布局，极大的影响关键路径和关联的新设计点的代价。

## 5. What Does it Take to Go Really Fast?

A number of challenges exist to push BOOM below 35 FO4. First the L1 instruction and data caches would likely need to be redesigned. Both caches return data after a single cycle (they can maintain a throughput of one request a cycle to any address that hits in the cache). This path is roughly 35 FO4. A few techniques exist to increase clock frequency but increase the latency of cache accesses.

将BOOM推进到小于35 FO4，存在几个挑战。第一，L1指令和数据缓存，很有可能需要重新设计。两种缓存经过一个单周期就返回数据（对击中缓存的任意地址，可以保持每个周期一个请求的吞吐量）。这条路径大约是35 F04。存在几种技术，可以提升时钟频率，但也增加了缓存访问的延迟。

For this analysis, we used regular threshold voltage (RVT)-based SRAM. However, the BTB is a crucial performance structure typically designed to be accessed and used to make a prediction within a single-cycle and is thus a candidate for additional custom effort. There are many other structures that are often the focus of manual attention: functional units; content-addressable memories (CAMs) are crucial for many structures in out-of-order processors such as the load/store unit or translation lookaside buffers (TLBs) [5]; and the issue-select logic can dictate how large of an issue window can be deployed and ultimately guide how many instructions can be inflight in an out-of-order processor.

对这种分析，我们使用常规的基于阈值电压的SRAM。但是，BTB是一个关键的性能结构，一般设计都是在一个周期以内进行访问和使用，是额外定制努力的一个候选。有很多其他结构，通常是手工注意的焦点：功能单元，content-addressable存储，是乱序处理中很多结构的关键，比如load/store单元，或TLB；在乱序处理器中，发射选择逻辑会决定可以部署多大的发射窗口，最终引导多少指令可以在飞行。

However, any techniques to increase BOOM’s clock frequency will have to be balanced against decreasing the IPC performance. For example, BOOM’s new front-end suffers from additional bubbles even on correct branch predictions. Additional strategies will need to be employed to remove these bubbles when predictors are predicting correctly [9].

但是，任何增加BOOM的时钟频率的技术，会被降低IPC性能所平衡。比如，BOOM的新前端，即使在正确的分支预测中，也存在额外的bubbles。当预测器预测的正确时，需要采用额外的策略来移除这些bubbles。

## 6. Conclusion

Modern out-of-order processors rely on a number of memory macros and arrays of different shapes and sizes, and many of them appear in the critical path. The impact on the actual critical path is hard to assess by using flip-flop-based arrays and academic/educational modeling tools, because they may either yield physically unimplementable designs or they may generate designs with poor performance and power characteristics. Re-architecting the design by relying on a hand-crafted, yet synthesizable register file array and leveraging hardware generators written in Chisel helped us isolate real critical paths from false ones. This methodology narrows down the range of arrays that would eventually have to be handcrafted for a serious production-quality implementation.

现代乱序处理器依赖于几个存储macros和不同形状和大小的阵列，很多都在关键路径中出现。对实际的关键路径的影响，通过使用基于触发器的阵列和学术/教育建模工具，是很难评估的，因为要么得到物理上不能实现的设计，或生成性能和能量特性很差的设计。依赖于手工进行但仍能综合的寄存器组阵列，重新改变设计的架构，利用Chisel写的硬件生成器帮助我们孤立了真实的关键路径，剔除了假的。这种方法论使阵列的范围变窄，最终需要手工设计，得到严肃的生产质量的实现。

BOOM is still a work-in-progress and we can expect further refinements as we collect more data. As BOOMv2 has largely been an effort in improving the critical path of BOOM, there has been an expected drop in instruction per cycle (IPC) performance. Using the Coremark benchmark, we witnessed up to a 20% drop in IPC based on the parameters listed in Table 1. Over half of this performance degradation is due to the increased latency between load instructions and any dependent instructions. There are a number of available techniques to address this that BOOM does not currently employ. However, BOOMv2 adds a number of parameter options that allows the designer to configure the pipeline depth of the register renaming and register-read components, allowing BOOMv2 to recover most of the lost IPC in exchange for an increased clock period.

BOOM仍然在工作过程中，随着我们收集了更多数据，我们可以期待进一步的改进。BOOMv2是改进BOOM的关键路径的主要努力，所以IPC的性能也有一定的降低。使用CoreMark基准测试，我们看到IPC大概降低了20%，参数如表1所示。这种性能下降超过一半是由于load指令和任意依赖的指令的延迟增加。有几种可用的技术来解决这个问题，BOOM暂时还没有采用。但是，BOOMv2加入了一些参数选项，使设计者可以配置寄存器重命名和寄存器读的组成部分的流水线深度，使BOOMv2可以恢复大多数丢失的IPC，以增加时钟周期。