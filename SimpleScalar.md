# SimpleScalar: An Infrastructure for Computer System Modeling

## 1. Introduction

To accelerate hardware development, designers often employ software models of the hardware they build. They implement these models in traditional programming languages or hardware description languages, then exercise them with the appropriate workload. Designers can execute programs on these models to validate the performance and correctness of a proposed hardware design.

为加速硬件开发，设计者通常会采用构建硬件的软件模型。他们用传统编程语言或硬件描述语言来实现这些模型，然后用适当的workload进行练习这些模型。设计者可以在这些模型上执行程序，来验证提出的硬件设计的性能和正确性。

Programmers can use software models to develop and test software before the real hardware becomes available. Although software models are slower than their hardware counterparts, programmers can build and test them in minutes or hours rather than in the months needed to build real hardware. This fast mechanism for design and test provides shorter time to market and much higher quality first silicon.

程序员可以在硬件可用之前，就使用软件模型来开发和测试软件。虽然软件模型比其对应的硬件部分更慢，但程序员可以在几分钟或几小时内构建并测试，而构建真实的硬件则需要几个月。这种设计和测试的快速机制，会给出更短的到市场时间，和高的多的一次硅质量。

Three critical requirements drive the implementation of a software model: performance, flexibility, and detail. Performance determines the amount of workload the model can exercise given the machine resources available for simulation. Flexibility indicates how well the model is structured to simplify modification, permitting design variants or even completely different designs to be modeled with ease. Detail defines the level of abstraction used to implement the model’s components. A highly detailed model will faithfully simulate all aspects of machine operation, whether or not a particular aspect is important to any metric being measured.

驱动一个软件模型的实现的三个关键要求：性能，灵活性和细节。性能决定了在给定了用于仿真的机器资源下，可以运行的workload的总量。灵活性指示了模型的结构怎样，可以简化硬件修改，允许设计的变体，甚至是完全不同的设计，可以很简单的进行建模。细节定义了实现模型的组件使用的抽象层次。一个高度细节化的模型会忠实的仿真机器操作的所有方面，不论一个特定的方面对测量的任意度量是否重要。

In practice, optimizing all three model characteristics in tandem is difficult. Thus, most model implementations optimize only one or two of them, which explains why so many software models exist, even for a single product design. Research models tend to optimize performance and flexibility at the expense of detail.

实践中，模型的三种特征一起进行优化是很困难的。因此，多数模型实现只对其中的一个或两个方面进行优化，这也解释了即使是对于一个产品设计，为什么会有这么多软件模型。研究模型通常会以细节为代价，对性能和灵活性进行优化。

The SimpleScalar toolset provides an infrastructure for simulation and architectural modeling. The toolset can model a variety of platforms ranging from simple unpipelined processors to detailed dynamically scheduled microarchitectures with multiple-level memory hierarchies. For users with more individual needs, SimpleScalar offers a documented and well-structured design, which simplifies extending the toolset to accomplish most architectural modeling tasks.

SimpleScalar工具集给出了仿真和架构建模的基础设施。这个工具集可以对多种平台进行建模，从简单的没有流水线的处理器，到详细的动态调度的有多级内存层次结构的微架构。对于有更多单个需求的用户，SimpleScalar给出了有文档的，结构良好的设计，这样拓展这个工具集来完成多数架构建模任务会变得很简单。

SimpleScalar simulators reproduce computing device operations by executing all program instructions using an interpreter. The toolset’s instruction interpreters support several popular instruction sets, including Alpha, Power PC, x86, and ARM.

SimpleScalar模拟器使用了一个解释器来执行所有程序指令，重现了计算设备的操作。这个工具集的指令解释器支持了几个流行的指令集，包括Alpha，Power PC，x86和ARM。

## 2. Modeling Basics

The typical approach to computer system modeling leverages a simple approximate model with good simulation performance and a modular code structure. This simulator style suits researchers and instructors well because the simple model focuses on the design’s primary components, leaving out the minutiae of detail that might otherwise hinder the model’s performance and flexibility. Industrial users, on the other hand, require very detailed models to minimize design risk. Detailed modeling assures that a design has no faulty components or acute performance bottlenecks.

计算机系统建模的典型方法是利用一个简单的近似模型，有着良好的仿真性能，和模块化的代码结构。这种仿真器的类型适合研究者，因为简单的模型聚焦在设计的主要组成部分，省略了细节部分，否则就会对模型的性能和灵活性有影响。另一方面，工业界的用户需要非常细节的模型来让设计风险来最小化。详细的建模确保了一个设计没有含有错误的组成部分，或尖锐的性能瓶颈。

The additional detail necessary to implement these models usually comes at the expense of model performance. In industrial applications, individual model performance often takes a backseat because companies have the resources available to stock large simulation server pools. Hardware simulation tends to be a throughput-bound task: Designers want to simulate multiple design configurations running several benchmarks. Adding more machines to the simulation pool decreases the overall runtime to the maximum runtime of any single experiment. For example, Intel’s Pentium 4 design team used a simulation pool that contained more than 1,000 workstations.

为实现这些模型的额外必要细节，通常会影响模型的性能。在工业应用中，单个模型性能通常没那么重要，因为公司会有很多资源来承载大型仿真服务器池。硬件仿真通常是一个通量受限的任务：设计者希望仿真多个设计配置，运行几个benchmarks。向仿真池中增加更多的机器，会降低总体的运行时间，到任何单个试验的最大运行时间。比如，Intel的Pentium 4设计团队，使用的仿真池，包含的工作站超过了1000台。

In some cases, designers optimize a model for performance and detail at the expense of flexibility. Designers typically employ these models when they need to faithfully represent a device at speeds capable of executing large workloads, but don’t need to change the model.

在一些情况下，设计者优化模型的性能和细节，代价是灵活性。当他们需要忠实的表示一个设备，并能够执行大型的workloads，设计者通常会采用这些模型，但他们不需要去改变这个模型。

Software performance analysis is an example of this type of application. Software developers need accurate depictions of program performance, but rarely require changes to the model because they are only concerned with program performance for a particular processor.

软件性能分析是这种应用的一个例子。软件开发者需要准确的表示程序性能，但很少需要改变模型，因为他们只关心在特定处理器下的程序性能。

The FastSIM simulator microarchitecture uses memoization to record internal simulator states and the actions taken—such as statistical variable update—upon entering those states. This permits microarchitectural models of arbitrarily high detail to quickly process instructions. However, the implementation sacrifices significant flexibility because the approach requires all microarchitecture components to provide internal-state hashing mechanisms and recording of per-cycle actions.

FastSIM处理器微架构仿真器使用记忆化来在进入状态时记录内部仿真器状态和采取的行为，比如统计变量更新。这允许高度细节化的微架构模型可以迅速的处理指令。但是，这些实现牺牲了显著的灵活性，因为这种方法需要所有的微架构组成部分提供内部状态hashing机制，记录每个cycle的行为。

## 3. Modeling with SimpleScalar

SimpleScalar was written in 1992 as part of the Multiscalar project at the University of Wisconsin, under Gurindar Sohi’s direction. In 1995, with Doug Burger’s assistance, the toolset was released as an open source distribution freely available to academic noncommercial users. SimpleScalar LLC now maintains the tool, which is distributed through SimpleScalar’s Web site at http://www.simplescalar.com.

SimpleScalar是在1992年写的，在Gurindar Sohi的指导下，在Wisconsin大学中，作为Multiscalar项目的一部分。在1995年，在Doug Burger的帮助下，该工具集开源了。SimpleScalar LLC现在维护这个工具，这是通过SimpleScalar的网站发布的。

Since its release, SimpleScalar has become popular with researchers and instructors in the computer architecture research community. For example, in 2000 more than one-third of all papers published in top computer architecture conferences used the SimpleScalar tools to evaluate their designs. SimpleScalar provides an infrastructure for computer system modeling that simplifies implementing hardware models capable of simulating complete applications. During simulation, model instrumentation measures the dynamic characteristics of the hardware model and the performance of the software running on it.

自从发布后，SimpleScalar在研究者中很流行。比如，在2000年的计算机架构会议中，超过1/3的文章使用了SimpleScalar工具来评估其设计。SimpleScalar提供了一个计算机系统建模的基础设施，简化了实现硬件模型，可以仿真完整的应用。在仿真过程中，模型的instrumentation测量了硬件模型的动态特性，和在其上运行的软件的性能。

Figure 1 shows a typical SimpleScalar user session, with the persistence-of-vision raytracer (POVray) graphics application running on a detailed Alpha processor microarchitecture model. The simulated graphical display in the screen’s upper-left corner shows the program I/O. The simulator console window in the screen’s upper-right corner displays simulator-generated messages plus stdout/stderr, the simulated program’s standard output. At the bottom of the screen, the graphical pipeline view window provides a graphical representation of the simulated program’s execution on the detailed microarchitecture model.

图1展示了一个典型的SimpleScalar用户会话，带有POVray图形应用，运行在一个详细的Alpha处理器微架构模型上。仿真的图形显示在屏幕的左上角，是程序的I/O。仿真器console窗口在屏幕的右上角，展示了仿真器生成的信息，加上stdout/stderr，仿真程序的标准输出。在屏幕下方，图形化的pipeline显示窗口给出了仿真程序执行在详细的微架构模型上的图形化的表示。

GPV shows the execution of instructions from fetch until retirement, displaying each instruction’s state throughout the pipeline. In the example, the blue lines in the display represent long-latency Icache misses. GPV forms part of SimpleScalar’s visualization infrastructure and provides a useful tool for identifying hardware and software bottlenecks.

GPV展示了执行从fetch到retire的执行，展示了每个指令在整个流程中的状态。在例子中，蓝线表示长延迟的Icache misses。GPV是SimpleScalar的可视化基础设施，是识别硬件和软件瓶颈的有用工具。

SimpleScalar includes several sample models suitable for a variety of common architectural analysis tasks. Table 1 lists the simulator models included with SimpleScalar version 3.0. The simulators range from sim-safe, a minimal SimpleScalar simulator that emulates only the instruction set, to sim-outorder, a detailed microarchitectural model with dynamic scheduling, aggressive speculative execution, and a multilevel memory system.

SimpleScalar包含几个样本模型，适合于几种常见的架构分析任务。表1列出了SimpleScalar 3.0中的仿真器模型。仿真器从sim-safe，一个最小的SimpleScalar仿真器，只仿真指令集，到sim-outorder，一个详细的微架构模型，带有动态调度，激进的预测执行，和多级内存系统。

All the simulators have fairly small code sizes because they leverage SimpleScalar’s infrastructure components, which provide a broad collection of routines to implement many common modeling tasks. These tasks include instruction-set simulation, I/O emulation, discrete-event management, and modeling of common microarchitectural components such as branch predictors, instruction queues, and caches. In general, the more detailed a model becomes, the larger its code size and the slower it runs due to increased processing for each instruction simulated.

所有仿真器都有相对较小的代码量，因为都利用了SimpleScalar的基础设施组件，包含了大量的子程序，实现了很多常见的建模任务。这些任务包含指令集仿真，I/O仿真，离散事件管理，和常见的微架构组成部分的建模，比如分支预测器，指令集队列，和缓存。总体上，一个模型变得越具体，其代码量就越大，运行的就越慢，因为对每条指令仿真处理的内容就越多。

Figure 2 shows the SimpleScalar hardware model’s software architecture. Applications run on the model using a technique called execution-driven simulation, which requires the inclusion of an instruction-set emulator and an I/O emulation module. The instruction-set emulator interprets each instruction, directing the hardware model’s activities through callback interfaces the interpreter provides.

图2展示了SimpleScalar的硬件模型的软件架构。应用运行在模型上，使用的是一种称为execution-driven仿真的技术，这需要包含指令集模拟器和I/O仿真模块。指令集模拟器翻译每条指令，通过解释器给出的回调接口来引导硬件模型的行为。

SimpleScalar includes instruction interpreters for the ARM, x86, PPC, and Alpha instruction sets. The interpreters are written in a target definition language that provides a comprehensive mechanism for describing how instructions modify registers and memory state. A preprocessor uses these machine definitions to synthesize the interpreters, dependence analyzers, and microcode generators that SimpleScalar models need. With a small amount of extra effort, models can support multiple target instruction sets by implementing the full range of callback interfaces the target definition language defines.

SimpleScalar包含对于ARM, x86, PPC和Alpha指令集的指令解释器。解释器是用一种目标定义语言写的，给出了指令怎样修改寄存器和内存状态的的综合机制。一个预处理器使用这些机器定义来综合这些解释器，依赖分析器，和微码生成器，这都是SimpleScalar模型需要的。在额外的努力下，模型可以支持多个目标指令集，只需要实现目标定义语言定义的回调接口的完整范围。

The I/O emulation module provides simulated programs with access to external input and output facilities. SimpleScalar supports several I/O emulation modules, ranging from system-call emulation to full-system simulation. For system-call emulation, the system invokes the I/O module whenever a program attempts to execute a system call in the instruction set interpreter, such as a callpal syscall instruction in the Alpha instruction set. The system emulates the call by translating it to an equivalent host operating-system call and directing the simulator to execute the call on the simulated program’s behalf. For example, if the simulated program attempts to open a file, the I/O emulation module translates the request to a call to open() and returns the resulting file descriptor or error number in the simulated program’s registers.

I/O仿真模块给出的仿真程序可以访问外部输入和输出设施。SimpleScalar支持几种I/O仿真模块，从系统调用仿真到全系统仿真。对于系统调用仿真，系统在程序在指令集解释器中要执行系统调用时调用I/O模块，比如Alpha指令集中的callpal系统指令。系统仿真这个调用，将其翻译成等价的宿主操作系统调用，引导仿真器来代替被仿真的程序来执行这个调用。比如，如果仿真程序试图打开一个文件，I/O仿真模块将这个请求翻译成open()，将得到的文件描述子或错误号返回到仿真程序的寄存器中。

Other I/O targets provide finer-grain emulation of actual hardware devices. For example, the SimpleScalar/ARM release includes an I/O emulator for Compaq IPaq hardware devices. This emulator is detailed enough to boot the ARM Linux operating system. Device-level I/O emulation has the added advantage of analyzing the operating system portion of an application’s execution. This additional fidelity proves especially useful with server applications, where networking and file system services demand much of the workload’s runtime.

其他I/O目标提供了更细粒度的实际硬件设备的仿真。比如，SimpleScalar/ARM版本包含了一个Compaq IPaq硬件设备的I/O仿真器。这个仿真器非常详细，可以启动ARM Linux操作系统。设备层的I/O仿真对分析一个应用的执行的操作系统部分有额外的好处。这些额外的fidelity在服务器应用中是非常有用的，其中网络和文件系统的服务占用了workload的大部分运行时间。

At the center of each model, the simulator core code defines the hardware model organization and instrumentation. Figure 3 lists the code for a hardware timing model of a simple microarchitecture in which all instructions execute in a single cycle except for loads and stores. These instructions execute in two cycles if they hit in the data cache, or in 10 cycles if they miss.

在每个模型的中心，仿真器核心代码定义了硬件模型的组织和instrumentation。图3列出了一种简单微架构的硬件时序模型的代码，其中所有的指令都在一个周期内执行，除了loads和stores。这些指令如果hit了数据cache，那么就在2个周期内执行，如果miss，就在10个周期内执行。

The simulator core defines the simulator’s main loop, which executes one iteration for each instruction of the program until finished. For a timing model, the main loop must account for the progression of execution time, measured in clock cycles for this model. The cycle variable stores the execution time, which counts the total number of clock cycles required to execute the program up to the current instruction. To determine the relative performance of programs, the model compares the total number of cycles to complete their execution. The simulation model increments cycle once for each instruction, once again for loads and stores, and 10 more times for any access that missed in the data cache.

仿真器核心定义了仿真的主循环，对程序的每条指令执行一个循环，直到结束。对于一个时序模型，主循环必须考虑到执行时间的过程，对这个模型来说，是用时钟周期计算的。周期变量存储了执行时间，对执行程序到当前指令所需的时钟周期的总数进行了计数。为确定程序的相对性能，模型比较完成其执行的总计周期数。仿真模型对每条指令递增时钟周期一次，对于loads和stores多1个时钟周期，对于数据cache中miss的访问，再多10个时钟周期。

The cache.c module supplied with the SimpleScalar distribution implements the data cache. Figure 3 shows the relevant portion of the implementation. The cache module uses a hash table to record the cache blocks it contains. If an access address matches an entry in the hash table, the access returns the hit latency. If an access address does not match an entry in the hash table, the system calls the cache miss handler, which returns the number of clock cycles required to service the cache miss. The model builder specifies the miss handler, which may be another cache module or a DRAM memory model. The cache module does not return the value accessed in the cache because this value has no effect on cache access latency. For designs in which the cache contents affect access latency, such as compressed cache designs, the system can configure the cache module to store and return the value accessed.

SimpleScalar中的cache.c模块实现了数据cache。图3展示了实现的相关部分。cache模块用一个hash表来记录其包含的cache块。如果访问地址与hash表中的一个entry匹配上了，这次访问就返回hit延迟。如果访问地址与哈希表中的entry没有匹配，系统会调用cache miss handler，返回cache miss服务所需的时钟周期数量。模型构建者指定了miss handler，可能是另一个cache模块，或DRAM内存模型。cache模块不会返回访问中的值，因为这个值对cache访问的延迟没有任何效果。对于cache内容影响访问延迟的设计，比如压缩cache设计，系统会配置cache模块来存储并返回要访问的值。

In addition to standard component models, SimpleScalar provides a variety of helper modules that implement useful facilities that many models require. These modules include a debugger, program loader, symbol table reader, command line processor, and statistical package.

除了标准的组成部分模型，SimpleScalar还提供了很多helper模块，实现了有用的设备，很多模型都需要。这些模块包括一个debugger，程序载入器，符号表读取器，命令行处理器，和统计包。

The sample code in Figure 3 uses the statistical package to manage model instrumentation. The stat_register() interface registers the simulator instrumentation variables insn and cycle with the statistical module. Once registered, the statistical package tracks updates to statistical counters, producing on request a detailed report of all model instrumentation. The stat_formula() interface allows derived instrumentation to be declared, creating a metric that is a function of other counters. In Figure 3, instruction per cycle (IPC) denotes a derived statistic equal to the number of instructions executed divided by the total execution cycles.

图3中的代码样本使用统计包来管理模型的instrumentation。stat_register()接口用统计模块注册了仿真器的instrumentation变量insn和cycle。一旦注册后，统计包追踪统计计数器的更新，根据需求产生了所有模型instrumentation的详细报告。stat_formula()接口允许声明derived instrumentation，创建一个度量，是其他计数器的函数。在图3中，IPC表示一个derived统计量，等于执行的指令数量，除以总计的执行时钟周期。

Simulators interface to the host machine via the host interface module, a thin layer of code that provides a canonical interface to all host machine data types and operating system interfaces. Simulators that use host interface types and operating system services can be easily run on new hosts by simply porting the host interface module. In Figure 3’s sample code, counter_t and word_t are canonical types, exported by the host interface, that define the maximum-sized unsigned integer and 32-bit signed integers, respectively.

仿真器到宿主机器的接口是通过宿主接口模块，这是很薄一层代码，提供了到所有宿主机器数据类型和操作系统接口的典型接口。仿真器使用宿主接口类型和操作系统服务，可以很容易移植到新的宿主上，只需要将移植宿主接口模块。在图3的代码样本中，counter_t和word_t是经典的类型，由宿主接口导出，分别定义了最大数量的无符号整数和32-bit有符号整数。

## 4. Execution-driven Simulation

All SimpleScalar models use an execution-driven simulation technique that reproduces a device’s internal operation. For computer system models, this process requires reproducing the execution of instructions on the simulated machine. A popular alternative, trace-driven simulation, employs a stream of prerecorded instructions to drive a hardware timing model. This method uses a variety of hardware- and software-based techniques—such as hardware monitoring, binary instrumentation, or trace synthesis—to collect instruction traces.

所有的SimpleScalar模型使用一种执行驱动的仿真技术，复现了一个设备的内部操作。对于计算机系统模型，这个过程需要在仿真的机器上复现指令执行的过程。还有一种流行的替代，即trace驱动的仿真，采用预先录制的指令流来驱动硬件时序模型。这种方法使用多种基于硬件和软件的技术，比如硬件监视，binary instrumentation，或trace合成，来收集指令traces。

### 4.1. Advantages

Execution-driven simulation provides many powerful advantages compared with trace-based techniques. Foremost, the approach provides access to all data produced and consumed during program execution. These values are crucial to the study of optimizations such as value prediction, compressed memory systems, and dynamic power analysis.

执行驱动的仿真与trace驱动的技术比，有很多优势。首先，这种方法可以访问程序执行过程中生成的和消耗的所有数据。这些值对优化的研究非常关键，比如值的预测，压缩存储系统，和动态功耗分析。

In dynamic power analysis, the simulation must monitor the data values sent to all microarchitectural components such as arithmetic logic units and caches to gauge dynamic power requirements. The hamming distance of consecutive data inputs defines the degree to which input bits change, which in turn causes transistor switching that consumes dynamic power.

在动态功耗分析中，仿真必须监控发送到所有微架构组成部分的数据的值，比如ALU和cache，和调整动态功耗要求。连续数据输入的hamming距离定义了输入bits变化的程度，随后导致了晶体管的switching，形成了动态功耗。

Execution-driven simulation also permits greater accuracy in the modeling of speculation, an aggressive performance optimization that runs instructions before they are known to be required by predicting vital program values such as branch directions or load addresses. Speculative execution proceeds at a higher throughput until the simulation finds an incorrect prediction, which flushes the processor pipeline and restarts it with correct program values. Before this recovery occurs, however, misspeculated code will compete for resources with nonspeculative execution, potentially slowing the program. Trace-driven techniques cannot model misspeculated code execution because instruction traces record only correct program execution. Execution-driven approaches, on the other hand, can faithfully reproduce the speculative computation and correctly model its impact on program performance.

执行驱动的仿真，在对推测建模的过程中，还可以得到更高的准确率，这是一种激进的性能优化，会在指令需要运行之前就运行，指令需要运行，是通过预测重要的程序值知道的，比如分支方向，和load地址。推测执行会以更高的通量进行，直到仿真找到一个不正确的预测，这会将处理器的pipeline冲刷掉，用正确的程序值来重启pipeline。在这种恢复发生前，误推测的代码会竞争资源，这是非推测的执行，可能会减缓程序的执行。Trace驱动的技术不能对误推测的代码执行进行建模，因为指令trace只会记录正确程序执行的结果。而执行驱动的方法，会忠实的重现推测计算，将其对程序性能的影响进行正确的建模。

### 4.2. Drawbacks

Execution-driven simulation has two potential drawbacks: increased model complexity and inherent difficulties in reproducing experiments. Model complexity increases because execution-driven models require instruction and I/O emulators to produce program computation, while trace-driven simulators do not. SimpleScalar mitigates this additional complexity through the use of a target definition language and a set of internal tools that synthesize the emulators that SimpleScalar models require. The target definition provides a central mechanism for specifying the complexities of instruction execution, including operational semantics, register and memory side effects, and instruction faulting semantics. The same facility makes it straightforward for SimpleScalar models to support multiple instruction sets.

执行驱动的仿真有两种可能的缺陷：增加模型的复杂度，在重现试验上有内生的困难。模型复杂度的增加，是因为执行驱动的模型需要指令和I/O仿真器来产生程序计算，而trace驱动的仿真器不需要。SimpleScalar通过使用目标定义语言和一系列内部工具来合成SimpleScalar模型需要的仿真器，来缓解这个复杂度。目标定义提供了核心机制来指定指令执行的复杂度，包括运算语义，寄存器和内存的副作用，和指令出错语义。同样的机制使SimpleScalar模型支持多个指令集很容易。

Because execution-driven simulators interface directly to input devices through I/O emulation, reproducing experiments that depend on realworld external events may be difficult. For example, changes in response to network latency and the contents of incoming requests affect a Web server application running on an execution-driven model. Trace-driven experiments do not experience these difficulties because instruction traces record the effects of external inputs within the instruction trace file. To overcome this reproducibility problem, SimpleScalar provides an external input tracing feature. Such traces record external device inputs to a program during its live execution. Replaying traced inputs to the simulated program recreates the original execution. Since these traces only contain external inputs, they are small and can be easily shared with other SimpleScalar users.

因为执行驱动的仿真器通过I/O仿真与输入设备直接接口，重现依赖于真实世界的外部事件的试验，会比较困难。比如，对网络延迟和到来的请求的内容的变化，会影响运行在执行驱动的模型上的网络服务器应用。trace驱动的试验不会有这些困难，因为指令trace会记录这些外部输入的效果在指令trace文件中。为克服这种复现问题，SimpleScalar提供了外部输入trace的功能。这种trace记录了实时执行时外部设备输入到一个程序。Replay traced输入到仿真程序，重新创造了原始的执行。由于这些trace只包含外部输入，所以很小，可以很容易的与其他SimpleScalar用户共享。

## 5. System Model Infrastructure

Our primary impetus for releasing SimpleScalar stemmed from our desire to reduce the effort required to become a productive researcher in the computer architecture research community. By its very nature a quantitative engineering discipline, computer architecture modeling requires access to tools capable of measuring program characteristics and performance. In the past, no such tools were widely available. Thus, much of the early work in computer architecture could be performed only at large universities and corporations where resources were available to develop the necessary tools. Building computer system modeling tools from scratch requires a significant development effort that consists mostly of writing mundane software components such as loaders, debuggers, and interpreters. Constructing these components requires great effort. Making them reliable is even more difficult, taking time that could be better spent on innovation.

我们放出SimpleScalar的主要动力源自于，我们希望在计算机体系结构研究领域成为一个有产出的研究者需要的努力更少一些。计算机体系结构建模需要有能够度量程序性质和性能的工具。在过去是没有这样的工具的。因此，计算机体系结构的早期工作只能在大型大学和公司中进行，其资源足够开发这样的工具。从头构建计算机系统建模工具有很多开发工作，主要是写单调的软件模块，比如loaders，debuggers和interpreters。构建这些组成部分，需要很大的努力。使其可靠，更加困难，所消耗的时间可以放到创新上。

Although SimpleScalar can be thought of as a simulator collection, we view it as an infrastructure for computer system modeling. The differences between a tool and an infrastructure lie in the care taken in designing the internal modules and interfaces. An infrastructure must organize these components to permit their reuse over a wide range of modeling tasks. Moreover, the module interfaces must be expressive and well documented. Computer architecture researchers can use SimpleScalar’s performance-analysis infrastructure to evaluate complex design optimizations by specifying them as changes and comparing their relative impact on baseline models.

虽然SimpleScalar可以认为是仿真器集合，我们将其视为计算机系统建模的基础设施。工具和基础设施的区别，在于设计内部模块和接口的努力。一个基础设施，需要组织这些部分，允许在广泛的建模任务中的重用。而且，模块接口需要描述自己，而且文档齐全。计算机架构研究者可以使用SimpleScalar的性能分析基础设施来评估复杂的设计优化，将其指定为变化，并与基线模型进行比较。

Figure 2 shows the baseline modules that comprise SimpleScalar’s software architecture. These modules export functions ranging from statistical analysis, event handlers, and command-line processing to implementations of modeling components such as branch predictors, caches, and instruction queues.

图2展示了基线模块，这些构成了SimpleScalar的软件架构。这些模块包含的功能包括统计分析，事件处理，命令行处理，到建模组成部分的实现，比如分支预测，缓存，和指令队列。

### 5.1. Open source and academia

Academic noncommercial users can download and use SimpleScalar tools free of charge. In addition, researchers can use SimpleScalar source code to build new tools and release them to other academic noncommercial users. The only restriction regarding redistribution is that the code must include the SimpleScalar academic noncommercial use license.

An open source distribution model gives users maximum flexibility in how they can modify and share the infrastructure. If the interfaces exported within the infrastructure prove insufficient for easily implementing a new model, users can extend the sources as required to implement their ideas. If these additions are generally useful, users can choose to distribute these enhancements to others. The Wattch model (http://citeseer.nj.nec.com/brooks00wattch.html), a framework for analyzing and optimizing microprocessor power dissipation, provides an excellent example of this capability. Wattch required significant changes to the baseline models, including an infrastructure to model physical device characteristics such as area and energy. The “Architecture-Level Power Modeling with Wattch” sidebar describes these challenges and their solutions in detail.

Wattch模型是分析和优化微处理器功耗的框架。Wattch需要对基准模型进行很大改变，包含对物理设备特征的建模的基础设施，比如面积和功耗。下面描述了这些挑战和其解决方案。

An open distribution model has potential drawbacks, however. Once the source code becomes available, the tool is likely to undergo a higher level of inspection than typical for research software. Given such scrutiny, it behooves researchers to make available only their highest quality work.

Many researchers have studied the internals of the SimpleScalar code, including Doug Burger and colleagues, who compared SimpleScalar’s sim-outorder model to a validated Alpha 21264 microarchitecture model. Although this study showed that SimpleScalar successfully predicted the performance trends of programs running on the modeled hardware, it also uncovered several inaccuracies in the sim-outorder model that required code updates.

An open source distribution makes it more difficult to commercialize the effort later, because having access to the source makes it easier for potential customers to use the technology without a license or to recreate a similar product based on the open source.

### 5.2 Architecture-Level Power Modeling with Wattch

Power dissipation and thermal issues have assumed increasing significance in modernprocessor design. As a result, making power and performance tradeoffs more visible to chip architects and even compiler writers has become crucial. To support this work, many researchers have begun developing tools to estimate the energy consumption of architectures and system software.

功耗和热问题在现在处理器设计中正变得越来越重要。结果是，让芯片架构师甚至是编译器开发者更清楚功耗和性能的折中，这变得更加重要。为支持这个工作，很多研究者开始开发工具来估计架构和系统软件的功耗。

Before the late 1990s, most power analysis tools operated below the architecture or microarchitecture levels and achieved high accuracy by calculating power estimates for designs only after developers completed layout or floorplanning. In contrast, architecture-level power modeling seeks to provide reasonably accurate high-level power estimates at useful simulation speeds much earlier in the design process.

在1990s后期之前，多数功耗分析工具都在架构或微架构层次之下运行，在开发者完成layout或floorplanning之后，才计算功耗估计，得到了很高的准确率。比较之下，架构级的功耗建模寻求的是，在更早的设计过程中，得到准确度合理的高层功耗估计，速度也要合理。

**Tracking Data Activity**

With these issues in mind, in 1998 we began to develop Wattch, an architecture-level power modeling framework. Wattch performs power analysis by tracking, on a per-cycle basis, the usage and data activity of microarchitectural structures such as the instruction window, caches, and register files. We use the unit-level usage statistics to scale appropriate power models that correspond to these structures. These fully parameterizable power models are based on capacitance estimates of the major internal nodes within these structures.

有了这个目的，在1998，我们开始开发Wattch，一种架构级的功耗建模框架，Wattch进行功耗分析是在每个时钟周期的基础上，追踪微架构结构中的数据行为和使用，比如指令窗，缓存和寄存器组。我们使用单元级的使用统计来得到合理的功耗模型，对应着这些结构。这些完全参数化的功耗模型，是基于这些结构中的主要内部节点的电容估计。

Wattch can be used for several types of architectural-level studies. Power-performance design tradeoff studies can be performed by simply varying microarchitectural parameters such as issue width, instruction window size, cache size, and so on. Studying the power and performance effect of additions to the microarchitecture can also be explored by modeling performance issues and instantiating power models for the additional hardware structures. Finally, compiler techniques and software energy profiling experiments can be performed.

Wattch可以用于几种架构级的研究。功耗-性能设计折中研究的进行，可以简单的变化微架构参数，比如issue宽度，指令窗的大小，cache大小，等等。研究新加入结构的功耗和性能效果，也可以对新架构的硬件结构进行功耗和性能建模。最后，编译器技术和软件功耗profiling试验也可以进行。

**Choosing an Infrastructure**

When choosing a performance estimation infrastructure on which to base Wattch, we could choose to modify one of a handful of existing infrastructures or write our own. In the end, we found the SimpleScalar toolset attractive because of its parameterizable microarchitecture, wide user support, and well-established code base.

当选择一个性能估计基础设施，作为开发Wattch的基础，我们可以选择修改现有的基础设施中的一个，或写一个我们自己的。最后，我们发现SimpleScalar工具集不错，因为其参数化的微架构，用户支持很好，和代码库基础好。

On the other hand, using SimpleScalar out of the box for power modeling presented some downsides. For example, the Register Update Unit structure is not representative of most modern microarchitectures. We based Wattch on the original RUU version of SimpleScalar so that its differences from the original code base would be minimized. Fortunately, users can fairly easily modify SimpleScalar or Wattch’s microarchitecture to look more appropriate.

另一方面，直接使用SimpleScalar进行功耗建模也有一些问题。比如，Register Update Unit结构并不是多数现代微架构中有代表性的。我们开发Wattch基于SimpleScalar中的原始RUU版本，这样与原始代码库的差异就最小化了。幸运的是，用户可以很容易的修改SimpleScalar或Wattch的微架构，使其更合适。

Wattch was one of the first attempts to demonstrate that power analysis can be performed at the architectural level with reasonable accuracy and speed. It has also provided our group and others with a useful measurement platform for doing power-aware research studies. Perhaps most importantly, Wattch and its first-generation counterpart tools from Penn State and Intel may serve as first steps toward future power-and-performance estimation tools with even better tradeoffs between accuracy and performance.

Wattch第一次证实了，功耗分析可以在架构级进行，而且有着合理的准确率和速度。这也为我们组和其他人提供了一个有用的测量平台，进行功耗相关的研究工作。最重要的可能是，Wattch和第一代类似的工具，是功耗和性能估计工具中的第一步，可以对准确率和性能进行更好的折中。

### 5.3 Proactive user support

## 6. Looking Forward

### 6.1 Embedded system modeling

### 6.2 Enhanced modeling capabilities

### 6.3 Sustainable user-support model