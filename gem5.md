# The gem5 Simulator

Nathan Binkert, et. al.

## 0. Abstract

The gem5 simulation infrastructure is the merger of the best aspects of the M5 [4] and GEMS [9] simulators. M5 provides a highly configurable simulation framework, multiple ISAs, and diverse CPU models. GEMS complements these features with a detailed and flexible memory system, including support for multiple cache coherence protocols and interconnect models. Currently, gem5 supports most commercial ISAs (ARM, ALPHA, MIPS, Power, SPARC, and x86), including booting Linux on three of them (ARM, ALPHA, and x86).

gem5模拟基础设施是M5和GEMS模拟器最好的部分的合并而来。M5给出了一个高度可配置的模拟框架，多ISAs，很多CPU模型。GEMS对这些特征进行了补充，包含了详细的灵活的内存系统，包含了对多缓存一致性协议和互联模型的支持。目前，gem5支持多数商用ISAs (ARM, ALPHA, MIPS, Power, SPARC, and x86)，包括在其中三个上引导启动Linux。

The project is the result of the combined efforts of many academic and industrial institutions, including AMD, ARM, HP, MIPS, Princeton, MIT, and the Universities of Michigan, Texas, and Wisconsin. Over the past ten years, M5 and GEMS have been used in hundreds of publications and have been downloaded tens of thousands of times. The high level of collaboration on the gem5 project, combined with the previous success of the component parts and a liberal BSD-like license, make gem5 a valuable full-system simulation tool.

这个项目是很多学术和工业机构的联合努力的结果，包括AMD, ARM, HP, MIPS, Princeton, MIT和Michigan大学，Texas大学和Wisconsin大学。在过去十年中，M5和GEMS在数百篇文章中进行了应用，已经被下载了数万次。gem5项目上的高层次合作，其组成部分之前的成功，和BSD类的许可，使得gem5成为了一个宝贵的全系统模拟工具。

## 1 Introduction

Computer architecture researchers commonly use software simulation to prototype and evaluate their ideas. As the computer industry continues to advance, the range of designs being considered increases. On one hand, the emergence of multicore systems and deeper cache hierarchies has presented architects with several new dimensions of exploration. On the other hand, researchers need a flexible simulation framework that can evaluate a wide diversity of designs and support rich OS facilities including IO and networking.

计算机架构研究者一般使用软件模拟来使其思想形成原型并进行评估。随着计算机工业持续演进，要考虑的设计的范围逐渐增加。一方面，多核系统和更深的缓存层次结构的出现，使架构师可以有新的探索维度。另一方面，研究者需要一个灵活的模拟框架，能够评估广泛的设计类型，支持丰富的OS功能，包括IO和网络。

Computer architecture researchers also need a simulation framework that allows them to collaborate with their colleagues in both industry and academia. However, a simulator’s licensing terms and code quality can inhibit that collaboration. Some open source software licenses can be too restrictive, especially in an industrial setting, because they require publishing any simulator enhancements. Furthermore, poor code quality and the lack of modularity can make it difficult for new users to understand and modify the code.

计算机架构研究者还需要一个模拟框架，使其可以与工业界与学术界的同事合作。但是，一个模拟器的许可和代码质量会阻碍这样的合作。一些开源软件的许可约束过多，尤其是在工业界的设置中，因为他们需要发表任意的模拟器的强化项。而且，代码质量差，缺少模块性，会使得新用户很难理解和修改代码。

The gem5 simulator overcomes these limitations by providing a flexible, modular simulation system that is capable of evaluating a broad range of systems and is widely available to all researchers. This infrastructure provides flexibility by offering a diverse set of CPU models, system execution modes, and memory system models. A commitment to modularity and clean interfaces allows researchers to focus on a particular aspect of the code without understanding the entire code base. The BSD-based license makes the code available to all researchers without awkward legal restrictions.

gem5模拟器克服了这些限制，给出了灵活的、模块化的模拟系统，可以评估非常多的系统，对所有研究者都可用。这个基础设施给出了多个CPU模型的集合，系统执行模式，和内存系统模型，有很大的灵活性。模块化和干净的接口，使研究者可以聚焦在代码的特定方面，不需要理解整个代码库。基于BSD的许可使代码对所有研究者都可用，不需要奇怪的法律限制。

This paper provides a brief overview of gem5’s goals, philosophy, capabilities, and future work along with pointers to sources of additional information.

本文给出了gem5的目标，哲学，能力的概述，还包括未来的工作，并指出了额外信息源。

## 2 Overall Goals

The overarching goal of the gem5 simulator is to be a community tool focused on architectural modeling. Three key aspects of this goal are flexible modeling to appeal to a broad range of users, wide availability and utility to the community, and high level of developer interaction to foster collaboration.

gem5模拟器的总体目标是成为一个集体工具，聚焦在架构建模上。这个目标的三个关键的方面是，灵活的建模，以吸引广泛的用户，团体内的高可用性和工具性，开发者之间的高层次互动，以促成合作。

### 2.1 Flexibility

Flexibility is a fundamental requirement of any successful simulation infrastructure. For instance, as an idea evolves from a high-level concept to a specific design, architects need a tool that can evaluate systems at various levels of detail, balancing simulation speed and accuracy. Different types of experiments may also require different simulation capabilities. For example, a fine-grain clock gating experiment may require a detailed CPU model, but modeling multiple cores is unnecessary. Meanwhile, a highly scalable interconnect model may require several CPUs, but those CPUs don’t need much detail. Also, by using the same infrastructure over time, an architect will be able to get more done more quickly with less overhead.

灵活性是任何成功的模拟基础设施的基础需求。比如，随着一个想法从高层的概念，到具体的设计，架构师需要工具来在很多细节层次上评估系统，在模拟速度和准确率上进行平衡。不同的试验类型也需要不同的模拟能力。比如，细粒度时钟门试验会需要一个详细的CPU模型，但是对多核进行建模则是没有必要的。同时，高度可扩展的互联模型会需要几个CPU，但那些CPUs不需要很多细节。而且，长时间使用相同的架构，架构师会以很低的代价，来做出更多的事情。

The gem5 simulator provides a wide variety of capabilities and components which give it a lot of flexibility. These vary in multiple dimensions and cover a wide range of speed/accuracy trade offs as shown in Figure 1. The key dimensions of gem5’s capabilities are:

gem5模拟器提供了很多能力和组成部分，这形成了大量灵活性。这在多个维度上有变化，覆盖了很大范围内的速度/准确率的折中，如图1所示。gem5能力的关键维度为：

- CPU Model. The gem5 simulator currently provides four different CPU models, each of which lie at a unique point in the speed-vs.-accuracy spectrum. AtomicSimple is a minimal single IPC CPU model, TimingSimple is similar but also simulates the timing of memory references, InOrder is a pipelined, in-order CPU, and O3 is a pipelined, out-of-order CPU model. Both the O3 and InOrder models are “execute-in-execute” designs [4].

CPU模型。gem5模拟器目前给出了4个不同的CPU模型，每个在速度-准确性的谱中都占据了一个唯一的点。AtomicSimple是一个最小的单IPC (Instructions Per Clock) CPU模型，TimingSimple是类似的，但也模拟了内存参考的时序，InOrder是一个流水线的顺序CPU，O3是一个流水线的乱序CPU模型。O3和InOder模型都是execute-in-excute设计。

- System Mode. Each execution-driven CPU model can operate in either of two modes. System-call Emulation (SE) mode avoids the need to model devices or an operating system (OS) by emulating most system-level services. Meanwhile, Full-System (FS) mode executes both user-level and kernel-level instructions and models a complete system including the OS and devices.

系统模式。每个执行驱动的CPU模型可以在两个模式中的任意一个运行。系统调用仿真(system-call emulation, SE)模式避免了对设备或操作系统进行建模的必要，只仿真多数系统级的服务。同时，全系统(full-system, FS)模式执行用户级的和核级的指令，对整个系统进行建模，包括操作系统和设备。

- Memory System. The gem5 simulator includes two different memory system models, Classic and Ruby. The Classic model (from M5) provides a fast and easily configurable memory system, while the Ruby model (from GEMS) provides a flexible infrastructure capable of accurately simulating a wide variety of cache coherent memory systems.

gem5模拟器包括两个不同的内存系统模型，Classic和Ruby。Classic模型提供了一个快速的、容易配置的内存系统，而Ruby模型提供了一个灵活的基础设施，能够准确的模拟很多缓存一致的内存系统。

The gem5 simulator can also execute workloads in a number of ISAs, including today’s most common ISAs, x86 and ARM. This significantly increases the number of workloads and configurations gem5 can simulate.

gem5模拟器还可以执行几种ISAs的工作，包括现在最常见的ISAs，x86和ARM。这显著增加了gem5可以模拟的工作量和配置的数量。

Section 4 provides a more detailed discussion of these capabilities. 第4部分给出了这些能力的更详细的讨论。

### 2.2 Availability

There are several types of gem5 user; each has different goals and requirements. These include academic and corporate researchers, engineers in industry, and undergraduate and graduate students. We want the gem5 simulator to be broadly available to each of these types of user. The gem5 license (based on BSD) is friendly both to corporate users, since businesses need not fear being forced to reveal proprietary information, and to to academics, since they retain their copyright and thus get credit for their contributions.

有几种类型的gem5用户；每个都有不同的目标和需求。这包括学术研究者，和公司研究者，工业界的工程师，本科生和研究生。我们希望gem5模拟器对每种类型的用户都是可用的。gem5的许可对企业用户和学术界都是友好的，因为生意不需要被迫暴露版权信息，同时还保留了拷贝权，保留了其贡献。

### 2.3 High level of collaboration

Full-system simulators are complex tools. Dozens of person-years of effort have gone into the gem5 simulator, developing both the infrastructure for flexible modeling and the numerous detailed component models. By being an open source, community-led project, we can leverage the work of many researchers, each with different specialties. The gem5 community is very active and leverages a number of collaborative technologies to foster gem5 use and development, including mailing lists, a wiki, web-based patch reviews, and a publicly accessible source repository.

全系统模拟器是复杂的工具。gem5模拟器耗费了几十人年的努力，开发出了灵活建模的基础设施，和大量组成部件模型的细节。这是一个开源的，团体的项目，可以利用很多研究者的工作，每个人都有不同的专长。gem5团体是非常活跃的，利用大量合作的技术来促进gem5的使用和开发，包括邮件列表，wiki，基于网络的patch review，和一个公开可访问的repo源。

## 3 Design Features

This section focuses on a few key aspects of gem5’s implementation: pervasive object orientation, Python integration, domain-specific languages, and use of standard interfaces. While most of these features are simply good software engineering practice, they are all particularly useful for designing simulators.

本节聚焦在gem5实现的几个关键方面：普遍的面向目标，集成了Python，领域专用语言，和使用了标准接口。多数这些特征都是很好的软件工程实践，他们对于设计模拟器都非常有用。 

### 3.1 Pervasive Object-Oriented Design

Flexibility is an important goal of the gem5 simulator and key aspect of its success. Flexibility is primarily achieved through object-oriented design. The ability to construct configurations from independent, composable objects leads naturally to advanced capabilities such as multi-core and multi-system modeling.

灵活性是gem5模拟器的一个重要目标，也是其成功的关键方面。灵活性基本上是通过面向对象的设计得到的。从独立的可组合的目标构建配置的能力，很自然的就会得到很高级的能力，比如多核，和多系统建模。

All major simulation components in the gem5 simulator are SimObjects and share common behaviors for configuration, initialization, statistics, and serialization (checkpointing). SimObjects include models of concrete hardware components such as processor cores, caches, interconnect elements and devices, as well as more abstract entities such as a workload and its associated process context for system-call emulation.

gem5模拟器的所有的主要模拟组成部分是SimObjects，在配置、初始化、统计和序列化（checkpointing）上有着相同的行为。SimObjects包括具体硬件组成部分的模型，比如处理器核心，缓存，互联元素和设备，以及更加抽象的实体，比如一个工作负荷，及其关联的为系统调用仿真的进程上下文。

Every SimObject is represented by two classes, one in Python and one in C++ which derive from SimObject base classes present in each language. The Python class definition specifies the SimObject’s parameters and is used in script-based configuration. The common Python base class provides uniform mechanisms for instantiation, naming, and setting parameter values. The C++ class encompasses the SimObject’s state and remaining behavior, including the performance-critical simulation model.

每个SimObject都由两个类表示，一个是用Python写的，一个是用C++写的，都由相应语言的SimObject基类继承而来。Python类定义指定了SimObject的参数，用在基于脚本的配置中。共同的Python基类提供了统一的机制用于实例化，命名，和设置参数值。C++类包括SimObject的状态和剩下的行为，包括性能关键的模拟模型。

### 3.2 Python Integration

The gem5 simulator derives significant power from tight integration of Python into the simulator. While 85% of the simulator is written in C++, Python pervades all aspects of its operation. As mentioned in Section 3.1, all SimObjects are reflected in both Python and C++. The Python aspect provides initialization, configuration, and simulation control. The simulator begins executing Python code almost immediately on start-up; the standard main() function is written in Python, and all command-line processing and startup code is written in Python.

gem5模拟器将Python集成到了模拟器中，这得到了显著的能量。85%的模拟器是用C++写的，Python非常流行。如3.1节所述，所有的SimObject都用Python和C++进行了编写。Python方面提供了初始化，配置和模拟控制。模拟器几乎在开始后就开始执行Python代码；标准的main()函数是用Python写的，所有的命令行处理和初始化代码都是用Python写的。

### 3.3 Domain-Specific Languages

In situations that require significant flexibility in performing a specialized task, domain-specific languages (DSLs) provide a powerful and concise way to express a variety of solutions by leveraging knowledge and idioms common to that problem space. The gem5 environment provides two domain-specific languages, one for specifying instruction sets (inherited from M5) and one for specifying cache coherence protocols (inherited from GEMS).

在执行专门任务需要显著的灵活性的情况下，领域专用语言(domain-specific language, DSL)提供了强力的简洁的方法来表达很多解决方法，利用那个问题空间的常见知识和习惯用语。gem5环境提供了两种领域专用的语言，一种用于指定指令集（从M5中继承得到），一种用于指定缓存一致性协议（从GEMS中继承得到）。

**ISA DSL**. The gem5 ISA description language unifies the decoding of binary instructions and the specification of their semantics. The gem5 CPU models achieve ISA independence by using a common C++ base class to describe instructions. Derived classes override virtual functions like execute() to implement opcodes, such as add. Instances of these derived classes represent specific machine instructions, such as add r1,r2,r3. Implementing a specific ISA thus requires a set of C++ declarations for these derived classes, plus a function that takes a machine instruction and returns an instance of one of the derived classes that corresponds to that instruction.

gem5 ISA描述语言统一了二值指令的解码，和其语义的规格。gem5 CPU模型通过使用通用的C++基类来描述指令，来获得ISA独立性。继承的类重写了虚函数像execute()，以实现opcodes，比如add。这样的继承类的实例表示具体的机器指令，比如add r1,r2,r3。实现一个特定的ISA，因此需要对这些导出的类的C++声明的集合，加上一个函数，以机器指令为参数，返回一个继承类的实例，对应着这条指令。

The ISA description language allows users to specify this required C++ code compactly. Part of the language allows the specification of class templates (more general than C++ templates) that cover broad categories of instructions, such as register-to-register arithmetic operations. Another portion of the language provides for the specification of a decode tree that concisely combines opcode decoding with the creation of specific derived classes as instances of the previously defined templates.

ISA描述语言使用户可以紧凑的指定这些需要C++代码。这个语言的一部分可以指定类模板（比C++模板更加通用），覆盖了大量指令，比如寄存器到寄存器的代数运算。语言的另一部分可以指定一个解码树，将opcode解码与具体的继承类的创建简洁的组合到一起，作为之前定义的模板的实例。

While the original ISA description language targeted RISC architectures such as the Alpha ISA, it has been significantly extended to cope with complex variable-length ISAs, particularly x86, and ISAs with complex register semantics like SPARC. These extensions include a microcode assembler, a predecoder, and multi-level register index translation. These extensions are discussed in more detail in a recent book chapter [5].

原始的ISA描述语言目标是RISC架构，比如Alpha ISA，但其经过显著的拓展，以应对复杂的变长ISAs，尤其是x86，和带有复杂寄存器语义的ISAs，如SPARC。这些拓展包括了一个微码汇编器，一个预解码器，和多级寄存器索引翻译。这些拓展在最近的一本书中进行了详细的讨论。

**Cache Coherence DSL**. SLICC is a domain-specific language that gives gem5 the flexibility to implement a wide variety of cache coherence protocols. Essentially, SLICC defines the cache, memory, and DMA controllers as individual per-memory-block state machines that together form the overall protocol. By defining the controller logic in a higher-level language, SLICC allows different protocols to incorporate the same underlying state transition mechanisms with minimal programmer effort.

SLICC是一个领域专用语言，使gem5可以灵活的实现很多缓存一致性协议。尤其是，SLICC将缓存，内存和DMA控制器定义为单个的每个内存块的状态机，共同形成了整体的协议。通过在更高层的语言上定义控制器逻辑，SLICC允许不同的协议集成相同的潜在状态迁移机制，而耗费的程序员的工作很少。

The gem5 version of SLICC is very similar to the prior GEMS version of SLICC [9]. Just like the prior version, gem5 SLICC defines protocols as a set of states, events, transitions, and actions. Within the specification files, individual transition statements define the valid combinations and actions within each transition specify the operations that must be performed. Also similar to the previous version, gem5 SLICC ties the state machine-specific logic to protocol-independent components such as cache memories and network ports.

gem5版的SLICC与之前的GEMS版SLICC非常类似。与之前的版本类似，gem5 SLICC将协议定义为状态、事件、迁移和行为的集合。在规格文件内，单个迁移语句定义了在每个迁移的有效的组合和行为，指定了必须执行的操作。与之前的版本类似的是，gem5 SLICC将状态机特定的逻辑与协议独立的组成部分连接到了一起，比如缓存内存和网络端口。

While gem5 SLICC contains several similarities to its predecessor design, the language does include several enhancements. First, the language itself is now implemented in Python rather than C++, making it easier to read and edit. Second, to adhere to the gem5 SimObject structure, all configuration parameters are specified as input parameters and gem5 SLICC automatically generates the appropriate C++ and Python files. Finally, gem5 SLICC allows local variables to simplify programming and improve performance.

gem5 SLICC与其前序设计有很多类似性，但该语言也包括了几个增强部分。第一，语言本身是用Python实现的，而不是C++，使其更易于阅读和编辑。第二，为附着于gem5 SimObject结构上，所有配置参数都指定为输入参数，gem5 SLICC自动的生成了合适的C++和Python文件。最后，gem5 SLICC使局部变量可以简化编程，改进性能。

### 3.4 Standard Interfaces

Standard interfaces are fundamental to object-oriented design. Two central interfaces are the port interface and the message buffer interface.

标准接口对于面向对象的设计是非常基础的。两个核心接口是port接口和信息buffer接口。

Ports are one of the interfaces used to connect two memory objects together in gem5. In the Classic memory system, the ports interface connects all memory objects including CPUs to caches, caches to busses, and busses to devices and memories. Ports support three mechanisms for accessing data (timing, atomic, and functional) and an interface for things like determining topology and debugging. Timing mode is used to model the detailed timing of memory accesses. Requests are made to the memory system by sending messages, and responses are expected to return asynchronously via other messages. Atomic mode is used to get some timing information, but is not message-oriented. When an atomic call is made (via a function call), the state change for the operation is performed synchronously. This has higher performance but gives up some accuracy because message interactions are not modeled. Finally, functional accesses update the simulator state without changing any timing information. These are generally used for debugging, system-call emulation, and initialization.

在gem5中，ports是用于将两个存储objects连接到一起的接口。在Classic存储系统中，ports接口将所有存储目标连接到一起，包括CPUs到缓存，缓存到总线，总线到设备和内存。Ports支持三种访问数据的机制（timing, atomic and functional）和一个接口，用于确定拓扑和调试等。Timing模式用于对内存访问的详细时序进行建模。通过发送信息对内存系统提出要求，响应会通过其他信息异步的返回。Atomic模式用于得到一些时序信息，但并不是面向信息的。当通过函数调用进行了一次atomic调用，该操作的状态变化是同步进行的。这性能会很高，但失去了一些准确性，因为信息交互并没有进行建模。最后，函数访问更新了模拟器的状态，而没有改变任何时序信息。这一般用于调试，系统调用仿真和初始化。

Ruby utilizes the ports interface to connect to CPUs and devices, and adds message buffers to connect to Ruby objects internally. Message buffers are similar to ports in that they provide a standard communication interface. However, message buffers differ in some subtle ways with regards to message typing and storage. In the future, ports and message buffers may evolve into a unified interface.

Ruby利用了ports接口来将CPUs连接到设备，并加入了message buffer，以连接到Ruby objects内部。Message buffers与ports在提供了一个标准通信接口上类似。但是，message buffers有一些微小的不同，包括信息类型和存储。未来，ports和message buffers可能会演进成为一个统一的接口。

## 4 Simulation Capabilities

The gem5 simulator has a wide range of simulation capabilities ranging from the selection of ISA, CPU model, and coherence protocol to the instantiation of interconnection networks, devices and multiple systems. This section describes some of the different options available in these categories.

gem5模拟器模拟能力很强，可以选择不同的ISA，CPU模型，和一致性协议，到互联网络，设备和多系统的实例化。本节描述了在这些类别中一些不同的选项。

**ISAs**. The gem5 simulator currently supports a variety of ISAs including Alpha, ARM, MIPS, Power, SPARC, and x86. The simulator’s modularity allows these different ISAs to plug into the generic CPU models and the memory system without having to specialize one for the other. However, not all possible combinations of ISAs and other components are currently known to work. An up-to-date list can be found on the gem5 website.

gem5模拟器目前支持很多ISAs，包括Alpha, ARM, MIPS, Power, SPARC和x86。模拟器的模块性，可以使这些不同的ISAs插入到通用CPU模型和内存系统中，不需要一个专用于另一个。但是，并不是所有可能的ISAa和其他组成部分的组合都可以工作。最新的列表可以在gem5网站上找到。

**Execution Modes**. The gem5 simulator can operate in two modes: System-call Emulation (SE) and Full-System (FS). In SE mode, gem5 emulates most common system calls (e.g. read()). Whenever the program executes a system call, gem5 traps and emulates the call, often by passing it to the host operating system. There is currently no thread scheduler in SE mode, so threads must be statically mapped to cores, limiting its use with multi-threaded applications. The SPEC CPU benchmarks are often run in SE mode.

gem5模拟器可以在两种模式下运行：系统调用仿真(SE)，和全系统模式(FS)。在SE模式下，gem5仿真所有常见的系统调用（如，read()）。不论何时程序执行一个系统调用，gem5 traps并对这个调用进行仿真，通常是将其传递到宿主操作系统。目前在SE模式中没有线程调度器，所以线程必须静态的映射到核中，在多线程应用中限制了其使用。SPEC CPU基准测试通常以SE模式运行。

In FS mode, gem5 simulates a bare-metal environment suitable for running an OS. This includes support for interrupts, exceptions, privilege levels, I/O devices, etc. Because of the additional complexity and completeness required, not all ISAs current support FS mode. 

在FS模式中，gem5模拟了一个适用于运行OS的bare-metal环境。这包括对中断、异常，特权层次，I/O设备，等。因为需要额外的复杂度和完整性，并不是所有的ISAs目前都支持FS模式。

Compared to SE mode, FS mode improves both the simulation accuracy and variety of workloads that gem5 can execute. While SPEC CPU benchmarks can be run in SE mode, running them in FS mode will provide more realistic interactions with the OS. Workloads that require many OS services or I/O devices may only be run in FS mode. For example, because a web server relies on the kernel’s TCP/IP protocol stack and a network interface to send and receive requests and a web browser requires a X11 server and display adapter to visualize web pages these workloads must be run is FS mode.

与SE模式相比，FS模式改进了gem5可以执行的模拟准确率和工作负载的类型数量。SPEC CPU基准测试可以在SE模式运行，但在FS模式下运行的话，会得到更真实的与OS的互动。需要很多OS服务或I/O设备的工作负载，可能只能在FS模式下运行，因为网页服务器依赖于核心的TCP/IP协议栈和网络接口来发送和接收请求，网页浏览器需要一个X11服务器和display adapter来可视化网页，这些工作负载必须在FS模式下运行。

**CPU Models**. The gem5 simulator supports four different CPU models: AtomicSimple, TimingSimple, In-Order, and O3. AtomicSimple and TimingSimple are non-pipelined CPU models that attempt to fetch, decode, execute and commit a single instruction on every cycle. The AtomicSimple CPU is a minimal, single IPC CPU which completes all memory accesses immediately. This low overhead makes AtomicSimple a good choice for simulation tasks such as fast-forwarding. Correspondingly, the TimingSimple CPU also only allows one outstanding memory request at a time, but the CPU does model the timing of memory accesses.

gem5模拟器支持四种不同的CPU模型：AtomicSimple, TimingSimple, In-Order和O3。AtomicSimple和TimingSimple是非流水线的CPU模型，在每个周期中都试图对一条指令进行取，解码，执行和commit。AtomicSimple CPU是一个最小的单个IPC CPU，立刻完成所有内存访问。这种低消耗使AtomicSimple在模拟像fast-forwarding时，就是一个很好的选择。对应的，TimingSimple CPU也只一次允许一个outstanding内存请求，但CPU确实对内存访问的时序进行了建模。

The InOrder model is an “execute-in-execute” CPU model emphasizing instruction timing and simulation accuracy with an in-order pipeline. InOrder can be configured to model different numbers of pipeline stages, issue width, and numbers of hardware threads.

InOrder模型是一个execute-in-execute CPU模型，强调的是顺序流水线时的指令时序和模拟准确率。InOrder可以配置为对不同数量的流水线级、issue width和硬件线程数量进行建模。

Finally, the O3 CPU is a pipelined, out-of-order model that simulates dependencies between instructions, functional units, memory accesses, and pipeline stages. Parameterizable pipeline resources such as the load/store queue and reorder buffer allow O3 to simulate superscalar architectures and CPUs with multiple hardware threads (SMT). The O3 model is also “execute-in-execute”, meaning that instructions are only executed in the execute stage after all dependencies have been resolved.

最后，O3 CPU是一个流水线的，乱序模型，模拟指令、functional units，内存访问和流水线stages之间的依赖关系。参数化的流水线资源，比如load/store队列，和reorder buffer，使O3可以模拟superscalar架构，和带有多个硬件线程的CPUs。O3模型也是execute-in-execute的，意思是，指令只在执行阶段，在所有的依赖关系都解决好之后，才执行。

**Cache Coherence Protocols**. SLICC enables gem5’s Ruby memory model to implement many different types of invalidation-based cache coherence protocols, from snooping to directory protocols and several points in between. SLICC separates cache coherence logic from the rest of the memory system, providing the necessary abstraction to implement a wide range of protocol logic. Similar to its GEMS predecessor [9], SLICC performs all operations at a cache-block granularity. The word-level granularity required by update-based protocols is not currently supported. This limitation has not been a issue so far because invalidation-based protocols dominate the commercial market. Specifically, gem5 SLICC currently models a broadcast-based protocol based on the AMD Opteron [7], as well as a CMP directory protocol [10].

SLICC使gem5的Ruby内存模型可以实现很多不同类型的基于invalidation的cache coherence protocols，从snooping到directory protocols和之间的几个点。SLICC将cache coherence逻辑与剩下的内存系统分离开来，提供了必须的抽象来实现很多协议逻辑。与其GEMS前序类似，SLICC在一个cache-block的粒度执行很多操作。基于update的协议需要的word-level粒度，现在还不支持。这个局限目前还不是一个问题，因为基于invalidation的协议是商业市场的主流。特别的，gem5 SLICC目前基于AMD Opteron建模了一个基于广播的协议，以及一个CMP directory协议。

Not only is SLICC flexible enough to model different types of protocols, but it also simulates them in sufficient depth to model detailed timing behavior. Specifically, SLICC allows specifying transient states within the individual state machines as cache blocks move from one base state to another. SLICC also includes separate virtual networks (a.k.a. network message classes) so message dependencies and stalls can be properly modeled. Using these virtual networks, the SLICC-generated controllers connect to the interconnection network.

SLICC不仅非常灵活，能够对不同类型的协议进行建模，而且对其进行足够深度的模拟，对详细的时序行为进行建模。特别的，SLICC允许cache block从一个base状态转移到另一个时，在单个状态机中指定瞬态。SLICC还包括了分离的虚拟网络（即，网络信息类别），这样信息依赖关系和stalls可以进行合理的建模。使用这些虚拟网络，SLICC生成的控制器连接到了互联网络中。

**Interconnection Networks**. The Ruby memory model supports a vast array of interconnection topologies and includes two different network models. In essence, Ruby can create any arbitrary topology as long as it is composed of point-to-point links. A simple Python file declares the connections between components and shortest path analysis is used to create the routing tables. Once Ruby creates the links and routing tables, it can implement the resulting network in one of two ways.

Ruby内存模型支持很大的互联拓扑阵列，包括两个不同的网络模型。实质上，Ruby可以创建任意拓扑，只要是由点到点的连接构成的。一个简单的Python文件，声明了组成部分之间的连接，用最短路径分析来创建路由表。一旦Ruby创建了连接和路由表，就可以以两种方式的一种来实现网络。

The first Ruby network model is referred to as the Simple network. The Simple network models link and router latency as well as link bandwidth. However, the Simple network does not model router resource contention and flow control. This model is great for experiments that require Ruby’s detailed protocol modeling but that can sacrifice detailed network modeling for faster simulation.

第一种Ruby网络模型称为Simple网络。Simple网络将连接和路由器延迟，以及连接的带宽进行建模。但是，Simple网络对路由器的资源竞争和流控制并不建模。对于需要Ruby的详细的协议建模，但可以牺牲细节网络建模进行快速模拟的试验来说，这个模型非常合适。

The second Ruby network model is the Garnet network model [1]. Unlike the simple network, Garnet models the router micro-architecture in detail, including all relevant resource contention and flow control timing. This model is suitable for on-chip network studies.

第二种Ruby网络模型是Garnet网络模型。与Simple网络不同的是，Garnet对路由器的微架构进行详细的建模，包括所有相关的资源竞争和流控制时序。模型适用于片上网络的研究。

**Devices**. The gem5 simulator supports several I/O devices ranging from simple timers to complex network interface controllers. Base classes are available that encapsulates common device interfaces such as PCI to avoid code duplication and simplify implementing new devices. Currently implemented models includes NICs, an IDE controller, a frame buffer, DMA engines, UARTs, and interrupt controllers.

gem5模拟器支持几种I/O设备，从简单的时钟，到复杂的网络接口控制器。封装了常用设备接口如PCI的基础类是可用的，以防止代码重复，简化了实现新设备。目前实现的模型包括NICs，一个IDE控制器，一个frame buffer，DMA引擎，UARTs和中断控制器。

**Modeling Multiple Systems**. Because of the simulator’s object oriented design it also supports simulating multiple complete systems. This is done by instantiating another set of objects (CPU, memory, I/O devices, etc.). Generally, the user connects the systems via the network interfaces described above to create a client/server pair that communicate over TCP/IP. Since all the simulated systems are tightly coupled within gem5 the results of multi-system simulation is still deterministic.

因为模拟器的面向对象设计，其还支持模拟多个完整的系统。这是通过实例化另一个对象集合（CPU，内存，I/O设备，等）来完成的。一般来说，用户通过上面描述的网络接口来连接系统，以创建一个客户/服务器对，通过TCP/IP进行通信。由于所有模拟的系统是在gem5中紧密耦合的，因此多系统模拟的结果仍然是确定性的。

## 5 Future Work

While gem5 is a highly capable simulator, there is always a desire for additional features and other improvements. A few of the efforts underway or under consideration include: gem5是一个能力很强的模拟器，但仍然需要额外的特征和改进。正在考虑的一些工作是：

- A first-class power model. While external power models such as Orion [6] and McPAT [8] have been used with GEMS and M5, we are working on a more comprehensive, modular, and integrated power model for gem5. 一流的能量模型。GEMS和M5使用了一些外部能量模型，如Orion和McPAT，我们正在为gem5开发一个更综合的，模块化的和集成的能量模型。

- Full cross-product ISA/CPU/memory system support. The modularity and flexibility of gem5 enables a wide variety of combinations of ISAs, CPU models, and memory systems, as illustrated in Figure 1, each of which can be used in SE or FS mode. Because each component model must support the union of all features required by any ISA in any mode, particular component models do not always work in every conceivable circumstance. We continue to work to eliminate these inconsistencies.

完整的跨产品ISA/CPU/内存系统支持。gem5的模块化和灵活性使得可以使用很多ISAs，CPU模型和内存系统的组合，如图1所示，其中每个都在SE或FS模式下使用。因为每个组成部分必须支持任意ISA在任意模式下的所有特征的合并，特定的组成部分模型并不是在每个情况下都会好用。我们继续工作来消除这些不连续性。

- Parallelization. To address the inherent performance limitations of detailed simulation and leverage the ubiquity of multi-core systems, we have been refactoring gem5’s internal event system to support parallel discrete event simulation [11]. 并行化。为处理内在的详尽模拟的性能限制，利用多核系统的普遍性，我们已经将gem5的内部事件系统分解，以支持并行离散事件模拟。

- Checkpoint import. Although gem5’s simple CPU models are much faster than their detailed counter-parts, they are still considerably slower than binary translation-based emulators such as QEMU [3] and SimNow [2]. Rather than duplicating the enormous effort of developing a binary translation capability within gem5, we plan to enable the transfer of state checkpoints from these emulators into gem5. Users will be able to fast-forward large workloads to interesting points using these high-performance alternatives, then simulate from those points in gem5. Even higher performance may be possible by using a hardware virtual machine environment such as KVM rather than binary translation.

Checkpoint导入。虽然gem5的简单CPU模型比详尽的对应模型要快很多，但比基于二值翻译的仿真器要慢很多，比如QEMU和SimNow。我们不会在gem5中开发一个二值翻译能力，而是计划将这些仿真器的状态checkpoints迁移到gem5中。用户将能够快速的将大型工作负载快进到有趣的点，使用这些高性能的替代品，然后从这些点在gem5中进行模拟。使用一个硬件虚拟机环境，比如KVM，而不是二值翻译，这可能更快。