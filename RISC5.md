# RISC5: Implementing the RISC-V ISA in gem5

Alec Roelke, Mircea R. Stan @ University of Virginia

## 0. Abstract

We present an RISC5, an implementation of the RISC-V ISA in the gem5 simulator. Gem5 is a modular, open-source simulation platform that supports several ISAs such as x86 and ARM and includes system-level architecture and processor microarchitecture models. It also has advanced simulation features such as system call emulation and checkpointing that the Chisel C++ simulator lacks, increasing its usefulness for simulating entire RISC-V applications or using phase analysis to estimate system behavior. Gem5 also provides detailed performance data that can be used in power estimation tools such as McPAT, which require fine granularity to provide accurate output. RISC5 is validated against performance data from the Chisel C++ emulator and FPGA softcore and is shown to have less than 10% error on several performance statistics.

我们给出了RISC5，在gem5模拟器中实现的RISC-V ISA。Gem5是一个模块化的，开源仿真平台，支持几种ISA，包括x86，ARM，包括系统级的架构和处理器微架构模型，还包括高级仿真特征，比如系统调用仿真和checkpointing，这是Chisel C++仿真器所缺乏的，这增加了对仿真整个RISC-V应用或使用相位分析来估计系统行为的用处。Gem5还给出了详细的性能数据，可以用在能耗估计工具中，比如McPAT，这需要细粒度来给出准确的输出。RISC5与Chisel C++仿真器和FPGA软核的性能数据进行了对比，在几种性能统计数据中误差都小于10%，展示了其有效性。

**Keywords** RISC-V, gem5, soc simulation, simulation infrastructure, tool flow

## 1. Introduction

As the number of transistors on a chip increases with Moore’s Law, designing and simulating the chip becomes more complex and time-consuming. Additionally, proprietary libraries and architectures can impede collaboration between academic and industry researchers and engineers. In order to address this issue, the gem5 simulator [6] was developed to model a wide range of architectures in a way that is accessible to all researchers. Its modular design allows a researcher to focus on some aspects of a design without having to understand gem5’s implementations of the rest and its open-source license allows easy collaboration.

由于芯片上晶体管的数量随着摩尔定律增加，设计和模拟芯片变得越来越复杂和耗时。另外，专利库和架构阻碍了学术和工业研究者和工程师的合作。为处理这些问题，gem5模拟器的开发来对很多架构进行建模，所有的研究者都可以使用。其模块化的设计使研究者聚焦在设计的某些方面，而不用理解gem5对其他部分的实现，其开源许可也使得合作变得容易。

Even so, many of the instruction sets gem5 supports are proprietary. Licenses for these ISAs can be costly and time-consuming to acquire and difficult to work with due to their complexity [3]. The RISC-V ISA [17] is designed to solve these problems by being simple, extensible, and free to use. Other open ISAs exist, but mistakes in their design have caused them to lose popularity [3]. RISC-V is designed to learn from past mistakes and maintain relevance by being extensible, agnostic of the type of core it runs on (such as in-order, out-of-order, or VLIW), and usable on real hardware running real workloads [17]. This way it can be extended to include future improvements in computer architecture and increases in data size.

即使这样，gem5支持的很多ISA是有专利的。这些ISA的许可很昂贵，获得也很耗时，复杂度很高，很难工作。RISC-V ISA的设计就是用于解决这些问题，很简单，可扩展，免费使用。也存在其他的开放ISA，但其设计中的错误使其失去了流行性。RISC-V的设计学习了过去的过失，而且可以扩展，与运行的核的类型无关（如，顺序，乱序，或VLIW），可以在真实的硬件上运行真实的workloads。这样，其可以拓展包括未来计算机架构的改进，数据大小也可以增加。

Several RISC-V designs [2, 7] are implemented using the hardware construction language Chisel [4], which takes advantage of programming concepts such as polymorphism and inheritance to create hardware generators that facilitate the creation of functional units. It can generate C++ code that models a design and can be used in a simulator for cycle-accurate simulation along with Verilog HDL code that can be mapped to FPGA or used in an ASIC flow.

几种RISC-V设计是使用Chisel实现的，利用一些编程概念，比如多态，继承，创建硬件生成器，促进了功能单元的创建。其可以生成C++代码，对设计进行建模，可以用在模拟器中，进行cycle-accurate的仿真，还包括Verilog代码，可以映射到FPGA中，或在ASIC流中进行使用。

Existing RISC-V simulation tends to fit into two categories: detailed RTL simulation as discussed above or binary translation using emulators like spike or QEMU [5] (Figure 1). Using high-level models of architectural units and memory hierarchy, gem5 is uniquely capable of bridging the gap between these two categories by providing highly accurate simulation at much faster speeds than RTL simulation does. In Section 2, we introduce an RISC5, an implementation of RISC-V in gem5, and discuss its simulation features in Section 3 that allow it to bridge this gap. We validate it in Section 4 against Rocket [2] and compare its performance against the Chisel-generated C++ simulator and an FPGA soft core. Finally, we present future work in Section 5 and conclude in Section 6.

现有的RISC-V仿真一般归为两类：详细的RTL仿真，或用仿真器如spike或QEMU来进行二值翻译（图1）。Gem5使用了架构单元和内存层次结构的高层次模型，填补了这两种类型的空白，可以给出很高精度的仿真，比RTL仿真的速度快的多。在第2部分，我们给出了RISC5，在gem5中实现的RISC-V，在第3部分讨论其仿真特征，为什么可以填补这个空白。在第4部分，与Rocket比较，验证了其性能，与Chisel生成的C++模拟器和FPGA软核进行了性能比较。最后，我们在第5部分给出了未来的工作，在第6部分给出了结论。

## 2. Implementation of RISC-V in gem5

RISC-V is divided into a base integer instruction set, which supports 32- and 64-bit address and data widths, and several extensions that add additional instructions. These extensions include the multiply extension, which adds integer multiply and divide instructions; the atomic extension, which adds instructions to atomically read-modify-write data in memory; and single- and double-precision floating point extensions. Currently RISC5 implements these instruction sets for single-core simulations in system call emulation (SE) mode. Additional nonstandard extensions are available in [3], such as quad-precision floating point arithmetic. Of these, only the compressed instruction set is currently included in RISC5. In this section we present some details about the implementations of these instruction sets.

RISC-V分为一个基础整数指令集，支持32位和64位地址和数据宽度，和几个拓展指令集。这些拓展包含乘法拓展，加入了整数乘法和除法指令；原子拓展，加入了在内存中自动读-修改-写数据的指令；单精度和双精度浮点拓展；目前RISC5对单核在系统调用仿真(SE)模式下实现了这些指令集。另外的非标准拓展可见[3]，比如四精度浮点代数。这些中，RISC5中目前只包括了压缩指令集。在本节中，我们给出了实现这些指令集的一些细节。

### 2.1 Integer and Multiply Instructions

Because MIPS is a RISC instruction set with many analogues to RISC-V [8], much of the code gem5 uses to implement MIPS was adapted to implement RISC-V. All instructions in RV64IM could be implemented either by referring to their definitions in [3] or by referring to their analogues in other ISAs from [8] (usually MIPS) when their implementations required information about gem5’s internal behavior. For example, the RISC-V fence instruction is similar to the MIPS sync instruction, so the implementation of fence in gem5 is based off its implementation of sync. The only instruction not implemented is eret, which returns from a higher privilege level. Privilege levels do not exist in gem5’s SE mode, so this instruction is unnecessary until full-system simulation is implemented.

MIPS是一个RISC指令集，与RISC-V有很多相似性，gem5中用于实现MIPS的代码都修改用于实现RISC-V。RV64IM中的所有指令，都可以参考其在[3]中的定义来实现，或参考其在其他ISA中的类似[8]，得到其在gem5中的内部行为的实现。比如，RISC-V fence指令与MIPS sync指令，所以gem5中fence的实现是基于其对sync的实现。唯一没有实现的指令是eret，是从更高一级的优先级中返回。优先级在gem5中的SE模式中不存在，所以这个指令不是必须的，除非完整系统的仿真要进行实现。

### 2.2 Atomic Instructions

RISC-V follows the release consistency memory model [10], which ensures that the results of a writer’s operations on a memory location are seen by a reader if the reader acquires that location after the writer releases it. To that end, RISC-V provides load-reserved and store-conditional instructions along with atomic read-modify-write (RMW) instructions that can be marked to acquire and/or release a memory location.

RISC-V遵循的是release一致性内存模型，确保了对一个内存位置的写入操作的结果，如果读取操作写入者释放后要求这个位置，则读取操作要看到这个写入操作。为此，RISC-V提供了load-reserved和store-conditional指令，还有原子的read-modify-write (RMW)指令，可以标记获取和/或释放一个内存位置。

Each RMW instruction requires two memory accesses (read and write). Since gem5 does not support multiple memory accesses per instruction when simulating memory with timing, each atomic memory instruction had to be split into two micro-ops: one which would read from memory and one which would write the result back to memory. In order to enable the write micro-op of each atomic memory instruction to keep track of the data loaded from memory by the read micro-op, a new integer register had to be added to RISC-V. This register is only used for storing a value loaded by the first micro-op of an atomic memory instruction. Each micro-op is marked with an ACQUIRE or RELEASE flag to indicate if it is acquiring and/or releasing a memory location. Load-reserved and store-conditional instructions are similarly flagged. Because RISC5 only supports single-threaded simulation, the actual atomicity and memory consistency of the implementations of these instructions cannot be verified.

每个RMW指令需要两个内存访问（读和写）。由于gem5在仿真带有时序的内存时，不支持每条指令的多内存访问，每个原子内存指令必须分成两个微操作：一个是从内存中读取，一个是写回到内存中。为使每个原子内存指令的写微操作，能够追踪读微操作从内存中载入的数据，必须要对RISC-V加入新的整数寄存器。这个寄存器只用于存储由原子内存操作的第一个微操作载入的值。每个微操作都用ACQUIRE或RELEASE flag标记住，来指示是来获得或释放一个内存位置。Load-reserved和store-conditional指令也进行类似的标记。因为RISC5只支持单线程仿真，这些指令的实现的实际的原子和内存一致性不能被验证。

### 2.3 Floating Point Instructions

RISC-V conforms to the IEEE-754 2008 floating-point standard [1], which is slightly different than the standard that the x86 machine used for development conforms to. The IEEE 2008 floating point standard provides five modes for rounding results (roundTowardPositive, roundTowardNegative, roundTowardZero, roundTiesToEven, and roundTiesToAway) while the machine used for development has four

(roundTiesToAway is missing). Attempting to use the missing rounding mode causes the simulation to halt.

RISC-V符合IEEE-754 2008浮点标准，这与x86机器的标准略微不一样。IEEE 2008浮点标准给出了四舍五入结果的5种模式(roundTowardPositive, roundTowardNegetive, roundTowardZero, roundTiesToEven, roundTiesToAway)，用于开发的机器有四种模式（没有了roundTiesToAway）。试图使用缺失的rounding模式，会导致仿真停止。

We verified RISC5’s floating point behavior by comparing its results with those from the RISC-V ISA simulator, spike, and from the Chisel-generated C++ simulator. While Spike emulates RISC-V instructions on a host system, the Chisel-generated C++ simulator makes use of a gate-level model of a RISC-V CPU and so performs its own floating point arithmetic. While the results of the floating-point computation were the same between both simulators, they disagreed in several cases about what exceptions should be generated and sometimes did not conform to the specifications in [17] and [1]. For example, Spike threw a divide-by-zero exception when performing a single-precision divide by zero, but the Chisel model did not. In these cases, gem5 conforms to the specifications without regard to other simulators. In the above example, a divide-by-zero exception will be thrown.

我们验证了RISC5的浮点行为，比较了其结果与RISC-V ISA仿真器spike的结果，和来自Chisel生成的C++仿真器的结果。Spike在一个宿主系统中仿真RISC-V指令，Chisel生成的C++仿真器使用了RISC-V CPU的门级模型，进行其自己的浮点代数。两个仿真器的浮点计算的结果都是一样的，在一些情况中会不一样，关于应该生成什么样的异常，一些时候不符合[17]和[1]中的指标。比如，在进行单精度除以0时，Spike抛出的是divide-by-zero异常，而Chisel模型并没有。在这些情况中，gem5遵循规范，不参考其他模拟器。在上述例子中，会抛出divide-by-zero异常。

## 3. Comparison with Chisel

Prior to RISC-V’s inclusion into gem5, existing simulation tools for RISC-V supported either slow but highly accurate RTL simulation or fast but low-detail binary translation (Figure 1). The former enables debugging and analysis of a design, but is hampered by long simulation times; the latter enables software development due to its high speed, but cannot provide information about an underlying system to inform design. In this section, we discuss gem5’s ability to bridge the gap between the two simulation paradigms high-level architectural and memory models and advanced simulation features that enable it to achieve high accuracy while reducing simulation time. In doing so, we compare its features with those of Chisel and other methods of simulating RISC-V (Table 1), evaluate the compatibility of each system with external tools, and perform an example simulation flow.

将RISC-V纳入到gem5之前，RISC-V现有的仿真工具支持的是很慢但是精确度很高的RTL仿真，或快速但精度很低的二值仿真（图1）。前者可以对一个设计进行debug和分析，但仿真时间过长；后者可以进行软件开发，因为速度高，但是不能给出潜在的系统的信息。本节中，我们讨论gem5弥补这两种仿真模式之间空白的能力，高层架构和内存模型和高级仿真特征，使其可以在降低仿真时间的情况下，获得很高的准确性。这样，我们将其特征与Chisel的进行比较，以及其他仿真RISC-V的方法（表1），评估每个系统与外部工具的兼容性，给出一个仿真流的例子。

### 3.1 Simulation Features

Gem5 provides several features to reduce simulation setup complexity and simulation time. Its system call emulation (SE) mode replaces a program’s own system calls with ones to the host. As a result, the target system can make use of the host’s file system; a kernel and boot device for the target system are not required. This reduces the overhead of booting a kernel and performing system calls, which is useful for measuring workload effects. Gem5 also provides a full-system mode if kernel simulation is desired.

Gem5有几种特征，可以降低仿真设置复杂度和仿真时间。其SE仿真模式将一个程序自己的系统调用替换为宿主自己的。结果是，目标系统可以利用宿主的文件系统；不需要目标系统的kernel和boot设备。这降低了boot一个kernel和进行系统调用的成本，这对度量workload效果是有用的。如果需要kernel仿真的话，Gem5还给出了一个完整系统的模式。

It also contains several CPU models and supports saving simulation state. Four CPU models exist at varying levels of detail and allow exploitation of the tradeoff between simulation accuracy and speed: CPU models can be switched during simulation, allowing for faster simulation until a period of interest is reached followed by slower, but more detailed, simulation of that period. Similarly, at any point during simulation a checkpoint can be saved. Once the simulation is complete, gem5 can resume from any saved checkpoint without having to simulate up to that point first and can use a more detailed CPU model than the checkpoint was taken with. These two features combined allow a user to rapidly simulate a benchmark using a simple CPU model and save checkpoints at the start of each region of interest, then resume from each checkpoint and simulate using a detailed model until the region of interest is complete. This enables compatibility with techniques such as phase analysis [15] so the power and performance of a benchmark can be estimated without having to simulate it in its entirety.

它还包括几种CPU模型，支持保存仿真状态。在不同的细节级别上，有4种CPU模型，可以在仿真精确度和速度之间进行折中探索：在仿真期间可以切换CPU模型，先进行快速仿真，直到到达感兴趣区间，然后是更慢的，但更细节的这个期间的仿真。类似的，在仿真的任何时刻，都可以保存一个checkpoint。一旦完成仿真，gem5可以从任意保存的checkpoint中恢复，不需要再重新仿真到这个点，然后可以使用一个比checkpoint更多细节的CPU模型进行继续仿真。这两个特征合在一起，使一个用户可以用Simple CPU模型迅速仿真一个基准测试，在每个感兴趣区域的开始保存checkpoint，从每个checkpoint恢复，然后使用一个更细节的模型进行仿真，直到这个感兴趣区域结束。这与一些技术是兼容的，比如phase分析，这样一个基准测试的功耗和性能可以进行在不需要对整体进行仿真的情况下进行估计。

Because Chisel only generates a C++ model of a design, it does not have advanced features such as those listed above. The simulation environment is left up to the user, who may decide which features are necessary but must implement them him- or herself. The user may choose to make use of an existing simulation environment such as the one in [2], which does support system call emulation through the use of the RISC-V proxy kernel, but does not support some of gem5’s other features. Unlike gem5, however, the Chisel model can output a trace of the signals in a design for inspection and analysis with a waveform viewer, enabling RTL debugging.

因为Chisel只生成设计的C++模型，所以没有上述的高级特征。仿真环境留给用户，来决定哪种特征是必须的，但必须自己来实现。用户可以选择使用一个现有的仿真环境，比如[2]中使用的，通过使用RISC-V代理核，确实支持系统调用仿真，但并不支持一些gem5的其他特征。但是，与gem5不一样，Chisel模型在用于设计带波形查看器的检查和分析时，可以输出信号的追踪，可以进行RTL debugging。

### 3.2 Compatibility with External Tools

Gem5’s output of performance statistics enables it to be used alongside tools that model other aspects of a design, such as power [14], temperature [12], and voltage noise [19]. These tools often require time traces of their inputs, and gem5 can provide those by outputting performance statistics mid-simulation. By default, Chisel’s C++ model does not keep track of performance information. A design in Chisel can be modified to keep track of that information, but in order to make use of it the program being executed must contain code to read and output it at the desired points.

Gem5输出的性能统计，使其可以与对设计的其他方面进行建模的工具，比如功耗，温度，和电压噪声。这些工具通常需要时间追踪其输入，gem5通过输出性能统计来给出这些。默认的，Chisel的C++模型不能追踪性能信息。用Chisel进行的设计可以进行修改，以追踪这些信息，但为使用，执行的程序必须包含代码来在期望的点进行读取并输出。

Even so, Chisel is able to generate Verilog HDL that can be used in ASIC synthesis or mapped to FPGA [4]. This way a designer can see not only the impact of a new idea on performance but also on power consumption, area, and reliability using existing synthesis tools. Similar simulations using gem5 and other tools can provide estimates, but cannot provide exact information about a particular design.

Chisel可以生成Verilog代码，可以用于ASIC综合，或映射到FPGA中。这样，一个设计者可以不止看到新思想对性能的影响，还包括对功耗，面积和可靠性的影响，只需要使用现有的综合工具。使用gem5和其他工具的类似的仿真，可以提供估计，但不能给出一个特定设计的精确信息。

### 3.3 Example Tool Flow

To illustrate gem5’s ability to simulate power and performance metrics alongside other tools, an example flow is presented in Figure 2. The flow begins by using gem5 in tandem with SimPoint [15] to determine representative phases of the application of interest. Once those phases have been determined, gem5 can simulate them in a detailed fashion and create performance data for power estimation with McPAT [14]. With the performance data and with chip information determined by a designer, McPAT can estimate area and power consumption. Another option for computing power is Strober [13], which can characterize arbitrary RTL for power. Area is input into ArchFP [9], which produces a floorplan that, alongside McPAT’s power calculation, is input into HotSpot [12] to calculate chip temperature. Although we do not include it in our example, VoltSpot [19] can also use McPAT’s power data and ArchFP’s floorplan to estimate voltage droop.

为描述gem5与其他工具一起对功耗和性能进行仿真的能力，图2给出了一个示例流。这个流开始于，用gem5与SimPoint来确定感兴趣应用的代表性阶段。一旦确定了这些阶段，gem5可以很细节的对其进行仿真，生成性能数据，然后用McPAT进行功耗分析。用性能数据，和设计者确定的芯片信息，McPAT可以估计区域和功耗。计算功耗的另一个选项是Strober，可以描述任意RTL的功耗特征。面积输入到ArchFP中，会产生一个平面规划图，与McPAT的功耗计算一起，输入到HotSpot中，计算芯片的温度。VoltSpot可以使用McPAT的功耗数据和ArchFP的平面图来估计其电压下降，但我们没有在例子中进行实现。

We execute this flow using two RISC-V designs: Rocket and BOOM [7]. Using Rocket to guide parameters for gem5 and using SimPoint to find the most representative region, we simulated one million instructions of the libquantum benchmark to get performance data for input into McPAT. Using McPAT’s area estimation, a floorplan was created with ArchFP that was used alongside McPAT’s power estimation to create a temperature map of the system using HotSpot (Figure 3). This process was repeated using the same benchmark and region for BOOM, whose results are shown in Figure 4. The values of parameters from each design can be found in Table 2 and summaries of the calibrated area estimates and power calculations can be found in Table 3.

我们用两种RISC-V的实现来执行这个流：Rocket和BOOM。使用Rocket来引导gem5的参数，使用SimPoint来找到最有代表性的区域，我们仿真了libquantum基准测试的一百万条指令，得到了性能数据，输入到McPAT中。使用McPAT的面积估计，用ArchFP创建了平面规划图，与McPAT的功耗估计一起使用，用HotSpot创建系统的温度图（图3）。对BOOM，使用相同的基准测试和区域来重复这个过程，其结果如图4所示。表2给出了每个设计的参数值，表3给出了校准面积估计和功耗计算。

McPAT has known inaccuracies [18] which affect its power and area calculations. These inaccuracies can be reduced using fixes from [18] and by calibrating results against real data. We applied the fixes and calibrated McPAT’s area estimates using the BOOM’s floorplan from [7], excluding the area labeled “uncore,” which resulted in an area of 1.367 mm2 (Figure 4). To create the floorplan for Rocket, we scaled McPAT’s area estimation using BOOM’s calibrations, which resulted in an area of 4.175 mm2 (Figure 3). Surprisingly, Rocket is much larger than BOOM, which has a larger register file and additional units for out-of-order execution. However, Rocket’s L2 cache has about four times the capacity of BOOM’s L2 cache. Indeed, when comparing the areas of the two chips without L2 cache, BOOM is over twice as large.

McPAT的不准确性是已知的，这影响了其功耗和面积计算。这些不准确性可以使用[18]的修补来降低，用真实数据校准结果。我们应用了这些修复，用[7]中BOOM的平面规划图来校准McPAT的面积估计，排除了标记为uncore的区域，得到了1.367 mm2的面积（图4）。为创建Rocket的平面规划图，我们使用BOOM的校准结果来缩放McPAT的面积估计，得到了4.175 mm2的面积（图3）。令人惊讶的是，Rocket比BOOM要大好多，而BOOM的寄存器组要大好多，而且还有额外的用于乱序执行的单元。但是，Rocket的L2 cache比BOOM的L2 cache要大4倍。确实，当比较没有L2 cache的两种芯片的面积时，BOOM的面积是Rocket的2倍大。

Even though it is smaller, BOOM consumes more power than Rocket. Because BOOM is out-of-order, its expanded register file and additional units add power and area overhead. According to the temperature maps produced by HotSpot, assuming that higher temperature is caused by higher power consumption from higher activity, Rocket has overall lower activity than BOOM. BOOM has multiple pipelines that enable higher core utilization through simultaneous in-flight instructions, increasing power consumption and temperature. Its higher capacity and associativity also consume more energy per instruction than a similar access pattern would in Rocket, leading to high data cache temperature from the frequent memory accesses in the benchmark region. Overall, BOOM is significantly hotter than Rocket; not only is its maximum temperature almost twice as high as Rocket’s, but its minimum temperature is higher than Rocket’s maximum.

即使更小，BOOM的功耗比Rocket要更高。因为BOOM的乱序执行的，其扩展的寄存器组和额外的单元，增加了功耗和面积的代价。根据HotSpot产生的温度图，假设更高的温度是由于更高的功耗导致的，而这是从更高的活动性中导致的，Rocket的总体活动度比BOOM要低很多。BOOM有多条流水线，使其核利用率更高，这是通过同时在飞行中的指令实现的，增加了其能耗和温度。其更高的能力和关联性也使每条指令消耗了更多的能量，这是与Rocket中的类似访问模式比较得到的，基准测试区域中，频繁的内存访问，得到更高的数据缓存温度。总体上，BOOM比Rocket温度要高的多；其最高温度几乎是Rocket的两倍，其最低温度都比Rocket的最高温度要高。

## 4. Validation and Performance

We simulated selected benchmarks from the SPEC CPU2006 suite [11] on gem5, the Chisel-generated C++ simulator, and on a RISC-V soft core on an FPGA. Version information on the tools used is contained in Table 4. In order to keep track of performance statistics in Figure 5, we modified the Chisel code to add additional performance counters and each benchmark’s source to read and print their values. The Chisel code was further modified to pause these counters during system calls to improve the quality of comparison with gem5 and account for the fact that gem5’s SE mode does not count statistics during system calls since it outsources them to the host. We ran gem5 and the C++ simulator on a four-core machine running at 3.7 GHz with 32 kB, 256 kB, and 10 MB L1, L2, and L3 caches, respectively, and 32 GB of main memory. Rocket includes a flow for configuring a Xilinx Zynq FPGA as a RISC-V core with a clock rate of 1 GHz and up to half of the board’s 512 MB of DRAM for the core. We used this flow to create a RISC-V soft core on the Zynq on which we also ran the benchmarks. Figure 5 shows each performance statistic obtained from gem5’s output normalized to the value extracted from the C++ simulator and FPGA.

我们在gem5上，在Chisel生成的C++仿真器上，在FPGA的RISC-V软核上仿真了SPEC CPU2006的一部分基准测试。表4给出工具的版本信息。为在图5中追踪性能统计数据，我们修改了Chisel代码，加入了额外的性能计数器和每个基准测试的源，来读取和打印其值。Chisel代码进行了进一步修改，在系统调用时暂停其计数器，来改进与gem5的比较的质量，考虑了gem5 SE模式下系统调用时不计算统计的事实，因为外包给了宿主。我们在一个4核机器上运行gem5和C++仿真器，主频3.7GHz，L1 L2 L3 cache分别为32kB，256kB和10MB，主存32GB。Rocket有一个配置Xilinx Zynq FPGA为RISC-V核的流，时钟速率为1GHz，板子512MB的DRAM的一半都用于核。我们使用这个流在Zynq上创建了一个RISC-V软核，也运行了基准测试。图5展示了gem5输出每个性能统计，对从C++仿真器和FPGA中得到的数值进行了归一化。

As Figure 5 shows, gem5 is accurate in the number of instructions retired, number of memory operations performed, and number of branch instructions executed. Less accurate is the number of cycles it took to complete each benchmark and number of instructions fetched. This is due to differences between the microarchitectures and memories modeled by gem5 and implemented by Chisel. In particular, the branch prediction may be different, causing a significant difference in the number of instructions fetched and cycles while not affecting the number of retired instructions or memory operations. This also accounts for the slight difference in branch operations.

如图5所示，gem5在退役的指令数量，进行内存操作数量，执行的分支指令数量上是精确的。不太精确的是，完成每个基准测试的周期数，和取到的指令数量。这是因为，在微架构和内存的建模上，gem5和Chisel实现的不一样。特别是，分支预测可能是不同的，导致取到的指令的数量明显不同，周期数量也明显不同，但是不影响推理的指令数量，或内存操作的数量。这也是分支运算略微不同的原因。

The FPGA, on average, took about 26.5 times less time to execute benchmarks than gem5 did while the Chisel simulator took, on average, about 32 times longer to perform each benchmark. As a result, several benchmarks took too long to execute and get data from, so they are excluded from Figure 5 for C++. This slowdown is due to the very high level of detail of the Chisel simulator, which enables the capability for very accurate simulation but adds significant overhead. Gem5’s abstraction of low-level signals using CPU models reduces overhead and allows it to run faster.

进行每个基准测试，平均来说，FPGA执行基准测试的时间比gem5少26.5倍，而Chisel模拟器平均耗时长了32倍。结果是，几个基准测试的时间太长了，不能执行完，得不到数据，所以图中C++的结果少了几个。这种速度慢的原因是因为，Chisel仿真器的细节成都非常高，所以可以进行非常精确的仿真，但增加了显著的代价。Gem5使用CPU模型对底层信号的抽象能力降低了代价，使其运行的更快。

Since all benchmarks except specrand and dealII performed few system calls, the fact that gem5’s SE mode does not simulate them on the target system while Chisel’s C++ simulator does causes no significant slowdown. Of the two with many system calls, only specrand completed in the C++ simulator and had a much higher runtime compared to gem5 than the others (0.0041 slowdown compared to 0.0508 for bzip2, 0.0314 for soplex, and 0.0377 for libquantum) due to the significant overhead of having to simulate the system calls. This is not the case for the FPGA soft core because it ran both user-level code and system call code natively.

由于除了specrand和dealII的所有基准测试进行了很少的系统调用，gem5的SE模式并没有在目标系统上对其仿真，而Chisel的C++仿真器也并没有导致显著的速度降低。对于有很多系统调用的两个，在C++仿真器中只有specrand完成了，与gem5相比有更高的运行时间，因为需要仿真系统调用，所以有更多的代价。在FPGA软核中就不是这个情况，因为原生的运行了用户级和系统调用代码。

## 5. Future Work

There is still some functionality within gem5 that does not yet support RISC-V. The two most important contributions will be support for multithreaded execution in SE mode and for full-system simulation. Additionally, there are some smaller improvements to existing support, such as implementation of roundTiesToAway for floating-point operations, that will improve accuracy of simulation. Finally, it will be useful to characterize the differences between gem5’s models and existing Chisel implementations and modify gem5 to more closely approximate real designs.

在gem5中，仍然有一些功能，不支持RISC-V。最重要的两个贡献，是在SE模式下支持多线程执行，以及完整系统仿真。另外，对现有的支持有一些更小的改进，比如对浮点运算roundTiesToAway的实现，会改进仿真的精确性。最后，描述gem5模型和现有Chisel实现的差异，并修改gem5以更好的近似真实设计，这也是很有用的。

Gem5’s ability to customize modeled hardware introduces a unique opportunity for ISA comparison. Because the system configuration is completely separated from the ISA, it is possible to simulate binaries compiled from the same code for different ISAs on systems that are identical except for the ISA they run. This allows direct comparison between ISAs that is difficult with tools that can only perform RTL simulation or binary translation.

Gem5定制建模硬件的能力，带来了ISA比较的唯一机会。因为系统配置与ISA是完全分开的，对两个系统，除了运行的ISA不同，别的都一样，编译成功后仿真二值文件，这是可能的。这可以直接比较不同的ISAs，这对于只能进行RTL仿真或二值翻译的工具来说，是非常难的。

## 6. Conclusion

We presented RISC5, an implementation of RISC-V in the gem5 simulator, that includes standard integer instructions, integer multiply instructions, atomic memory operations, and floating point instructions for use with gem5’s SE mode simulating a single core. In comparison with Chisel, gem5 has a more robust simulation environment that supports extensive configuration with fast or detailed simulation, storing and resuming from checkpoints, and automatic tracking of performance statistics. On the other hand, Chisel enables fine control over a design and integrates better into a flow while still enabling detailed simulation through a C++ model or FPGA emulation. We then showed an example simulation flow of tools for modeling different aspects of a design, including ArchFP for floorplanning, gem5 for performance, McPAT for power and area estimation, and HotSpot for temperature calculation for two implementations of RISC-V: Rocket and BOOM. Finally, we showed gem5’s accuracy and compared its runtimes for several benchmarks with those of a Chisel-generated FPGA mapping and C++ simulator. RISC5 is available as part of the main gem5 release at http://www.gem5.org.

我们提出了RISC5，在gem5仿真器中实现的RISC-V，包括了标准的整数指令，整数乘法指令，原子内存运算，和浮点指令，使用gem5的SE模式仿真了一个单核。与Chisel比较，gem5有一个更稳健的仿真环境，支持广泛的配置，有快速或更详细的仿真，保存checkpoint或从中恢复，自动跟踪性能统计值。另一方面，Chisel可以对一个设计有精细的控制，可以更好的整合进一个流中，而且仍然可以通过C++模型或FPGA仿真进行详细的仿真。我们展示了一个工具仿真流的例子，对一个设计的不同方面进行建模，包括ArchFP进行平面布局，gem5进行性能建模，McPAT进行功耗和面积估计，和HotSpot进行温度计算，为RISC-V的两个实现Rocket和BOOM进行了建模。最后，我们展示了gem5的准确率，在几个基准测试上，与Chisel生成的FPGA映射和C++仿真器的结果进行了比较。RISC5是gem5的一部分。