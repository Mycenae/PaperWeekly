# Complete Computer System Simulation: The SimOS Approach

Mendel Rosenblum et.al. @ Stanford University

## 1. Introduction

The complexity of modern computer systems and the diverse workloads they must support challenge researchers and designers who must understand a system’s behavior. As computer system complexity has increased, software simulation has become the dominant method of system testing, evaluating, and prototyping. Simulation is used at almost every step of building a computer system: from evaluation of research ideas, to verification of the hardware design, to performance tuning once the system has been built. In all these simulations, designers face trade-offs between speed and accuracy. Frequently, they must reduce accuracy to make the simulation run in an acceptable amount of time.

现在计算机系统和它们必须支持的多种workloads的复杂度，在挑战研究者和设计者，它们必须理解一个系统的行为。随着计算机系统的复杂度的提升，软件仿真已经成为系统测试，评估和原型设计的主要方法。在构建一个计算机系统的几乎每个步骤，都在使用仿真：从研究思想的评估，到硬件设计的验证，到构建了系统后的性能调试。在所有这些仿真中，设计者都面临着速度和准确性的折中。在很多情况中，他们都必须降低准确率以使仿真在可接受的时间内运行。

One simplification that reduces simulation time is to model only user-level code and not the machine’s privileged operating system code. Omitting the operating system substantially reduces the work required for simulation. Unfortunately, removing the operating system from the simulation model reduces both the accuracy and the applicability of the simulation environment. Important computing environments, such as database management systems and multiprogrammed, time-shared systems, spend as much as a third of their execution time in the operating system. Ignoring the operating system in modeling these environments can result in incorrect conclusions. Furthermore, these environments use the operating system services heavily, so it is difficult to study such applications in a simulation environment that does not model an operating system. The inability to run OS-intensive applications means that their behavior tends to be poorly understood. Finally, operating system researchers and developers cannot use these simulation environments to study and evaluate their handiwork.

一种减少仿真时间的简化方法，是只对用户级的代码进行建模，而不对机器的特权级操作系统代码进行建模。忽略操作系统极大的减少了仿真需要的工作。不幸的是，从仿真模型中移除操作系统，会降低仿真环境的准确率和应用性。重要的计算环境，比如数据库管理系统，和多编程了，时分系统，会将执行时间的1/3放在操作系统中。在对这些环境建模时，忽略操作系统会导致不正确的结论。而且，这些环境使用了很多操作系统的服务，所以在不对操作系统进行建模的仿真环境中，很难研究这些应用。不能运行OS成分很多的应用，意味着对其行为的理解是很差的。最后，操作系统研究者和开发者不能使用这些仿真环境来研究和评估其工作。

SimOS is a simulation environment capable of modeling complete computer systems, including a full operating system and all application programs that run on top of it. Two features help make this possible. First, SimOS provides an extremely fast simulation of system hardware. Workloads running in the SimOS simulation environment can achieve speeds less than a factor of 10 slower than native execution. At this simulation speed, researchers can boot and interactively use the operating system under study. (We use the term workload to refer to the execution of one or more applications and their associated OS activity.)

SimOS是一个可以建模整个计算机系统的仿真环境，包含一个完整的操作系统和在其上运行的所有应用程序。这个成为可能，主要因为两个特征。第一，SimOS提供了一种非常快速的系统硬件的仿真。在SimOS仿真环境中运行的workload，比在本地执行，速度慢的不会超过10倍。在这种仿真速度下，研究者可以在研究中启动操作系统并进行交互使用。

The other feature that enables SimOS to model the full operating system is its ability to control the level of simulation detail. During the execution of a workload, SimOS can switch among a number of hardware component simulators. These simulators vary in the amount of detail they model and the speed at which they run. Using multiple levels of simulation detail, a researcher can focus on the important parts of a workload while slapping over the less interesting parts. The ability to select the right simulator for the job is very useful. For example, most researchers are interested in the computer system’s running in steady state rather than its behavior while booting and initializing data structures. We typically use SimOS’s high-speed simulators to boot and position the workload and then switch to more detailed levels of simulation. The process is analogous to using the fast forward button on a VCR to position the tape at an interesting section and then examining that section at normal speed or even in slow motion. Additionally, SimOS lets us repeatedly jump into and out of the more detailed levels of simulation. Statistics collected during each of the detailed simulation samples provide a good indication of a workload’s behavior, but with simulation times on a par with the quicker, less detailed models.

使SimOS可以对整个操作系统建模的另一个特征，是其能够控制仿真细节的级别的能力。在workload的执行中，SimOS可以在几种硬件不见仿真器中切换。这些仿真器在建模细节和运行速度上都不一样。使用多种建模细节层次，一个研究者可以聚焦在一个workload的重要部分，而忽略没那么有趣的部分。选择合适的仿真器来进行任务，是一种很有用的能力。比如，多数研究者对计算机系统运行在稳定状态时比较感兴趣，而不是在其启动和初始化数据结构的行为。我们一般使用SimOS的高速仿真器来启动操作系统，并放入workload，然后切换到更详细的仿真。这个过程类似于对VCR使用快进键来将tape放到有趣的部分，然后以正常速度甚至慢速来检查这个部分。另外，SimOS使我们可以重复的跳入跳出更详细的仿真。在详细的仿真样本中收集的统计数据，给出了workload行为的很好的表征，但是其仿真时间与更快的没那么详细的模型则类似。

SimOS allows a system designer to evaluate all the hardware and software performance factors in the context of the actual programs that will run on the machine. Computer architects have used SimOS to study the effects of new processor and memory system organizations on workloads such as large, scientific applications and a commercial database system. OS designers have used SimOS for developing, debugging, and performance-tuning an operating system for a next-generation multiprocessor.

SimOS允许系统设计者在运行在机器上的实际程序的上下文中，评估所有硬件和软件性能因素。计算机体系结构师使用SimOS来研究新的处理器和内存系统组织在大型科学应用和商用数据库系统这样的workload上的效果。OS设计者使用SimOS来对下一代多处理器来开发，调试和性能调试操作系统。

## 2. The SimOS Environment

To boot and run an operating system, a simulator must provide the hardware services expected by the operating system. A modern multiprocessor operating system, such as Unix SVR4, assumes that its underlying hardware contains one or more CPUs for executing instructions. Each CPU has a memory management unit (MMU) that relocates every virtual address generated by the CPU to a location in physical memory or generates an exception if the reference is not permitted (for example, a page fault). An operating system also assumes the existence of a set of I/O devices including a periodic interrupt timer that interrupts the CPU at regular intervals, a block storage device such as a magnetic disk, and devices such as a console or a network connection for access to the outside world.

为启动和运行一个操作系统，一个仿真器必须提供操作系统需要的硬件服务。一个现代多处理器操作系统，比如Unix SVR4，会假设其硬件包含一个或多个CPUs来执行指令。每个CPU都有MMU，将CPU产生的每个虚拟地址重定位到物理内存的位置，如果引用不允许，那么就产生一个异常（比如，是一个页面错误）。一个操作系统也假设存在I/O设备集，包括周期的中断时钟，以定期的intervals中断CPU，一个块存储设备，比如磁盘，还有控制台或网络连接的设备，以访问外部世界。

SimOS, diagrammed in Figure 1, is a simulation layer that runs on top of general-purpose Unix multiprocessors such as the SGI (Silicon Graphics Inc.) Challenge series. It simulates the hardware of an SGI machine in enough detail to support Irix version 5.2, the standard SGI version of Unix SVR4. Application workloads developed on SGI machines run without modification on the simulated system. SimOS can therefore run the large and complex commercial applications available on the SGI platform. Although the current SimOS implementation simulates the SGI platform, previous versions have supported other operating systems, and the techniques SimOS utilizes are applicable to most general-purpose operating systems.

SimOS如图1所示，是一个仿真层，运行在通用目标Unix多处理器上，比如SGI Challenge系列。其仿真的硬件是一个SGI机器，细节足以支持Irix V5.2，这是Unix SVR4的标准SGI版本。在SGI机器上开发的应用workloads在仿真系统上可以不用修改的运行。SimOS因此可以运行SGI平台上可用的大型复杂商业应用。虽然当前的SimOS实现仿真的是SGI平台，之前的版本也支持了其他操作系统，SimOS利用的技术，可以应用于多数通用目标操作系统。

Each simulated hardware component in the SimOS layer has multiple implementations that vary in speed and detail. While all implementations are complete enough to run the full workload (operating system and application programs), we designed the implementations to provide various speed/detail levels useful to computer system researchers and builders. The CPUs, MMUS, and memory system are the most critical components in simulation time, so it is these components that have the most varied implementations.

在SimOS层中，每个仿真的硬件部件都有多个实现，有着不同的速度和详细度。所有实现都是完整的，可以运行完整的workload（操作系统和应用程序），我们设计了实现，以提供各种速度/细节层次，用于计算机系统研究和构建。CPU，MMU和内存系统是仿真时间上最关键的组成部分，所以这些部件的实现变化最大。

By far the fastest simulator of the CPU, MMU, and memory system of an SGI multiprocessor is an SGI multiprocessor. SimOS provides a direct-execution mode that can be used when the host and target architectures are similar. SimOS exploits the similarity between host and target by directly using the underlying machine’s hardware to support the operating system and applications under investigation. Configuring a standard Unix process environment to support operating system execution is tricky but results in extremely fast simulations.

迄今为止，对一个SGI多处理器的CPU，MMU和内存系统的最快的仿真器，就是一个SGI多处理器。当宿主和目标架构是类似的时候，SimOS提供了一种直接模式可以使用。SimOS利用宿主和目标之间的相似性，直接使用机器的硬件来支持研究中的操作系统和应用。配置标准Unix进程环境，以支持操作系统执行，这是很需要技巧的，但是会得到非常快的仿真速度。

The direct-execution mode often executes an operating system and target applications only two times slower than they would run on the host hardware. Because of its speed, researchers frequently use this mode to boot the operating system and to position complex workloads. Operating system developers also use it for testing and debugging new features. Although the direct-execution mode is fast, it provides little information about the workload’s performance or behavior, and thus it is unsuitable for studies requiring accurate hardware modeling. Moreover, this mode requires strong similarities between the simulated architecture and the simulation platform.

直接执行模式在仿真操作系统和目标应用时，比在宿主硬件上运行，只慢了2倍。由于速度的优势，研究者频繁的利用这种模式来启动操作系统，运行复杂的workloads。操作系统开发者也使用这种方式来测试和调试新特征。虽然直接执行模式很快速，但包含很少关于workload性能或行为的信息，因此不适合需要精确硬件建模的研究。而且，这种模式需要被仿真的架构和仿真平台之间的相似度很高。

For users who require only the simulation accuracy of a simple model of a computer’s CPUs and memory system, SimOS provides a binary-translation mode. This mode uses on-the-fly object code translation to dynamically convert target application and operating system code into new code that simulates the original code running on a particular hardware configuration. It provides a notion of simulated time and a breakdown of instructions executed. It operates at a slowdown of under 12 times. It can further provide a simple cache model capable of tracking information about cache contents and hit and miss rates, at a slowdown of less than 35 times. Binary translation is useful for operating system studies as well as simple computer architecture studies.

有的用户只需要对计算机的CPU和内存系统的简单模型进行准确的仿真，对此，SimOS提供了一个binary-translation模式。这种模式使用运行中的目标码，来将目标应用和操作系统代码动态的转换为新的代码，对原始的代码运行在特定的硬件配置上进行仿真。这提供了仿真时间和执行的指令的分解，运行速度减慢了12倍，还可以提供简单的cache模型，可以跟踪cache内容的信息，和hit和miss率，减速达35倍。Binary translation对于操作系统研究，和简单的计算机架构研究，都是有用的。

SimOS also includes a detailed simulate: implemented with standard simulation techniques. The simulator runs in a loop, fetching, decoding, and simulating the effects of instructions on the machine’s register set, caches, and main memory. The simulator includes a pipeline model that records more detailed performance information at the cost of longer simulation time. Different levels of detail in memory system simulation are available, ranging from simple cache-miss counters to accurate models of multiprocessor cache coherence hardware. SimOS’s highly detailed modes have been used for computer architecture studies as well as for performance tuning of critical pieces of an operating system.

SimOS还包含详细的仿真：这是用标准仿真技术实现的。仿真器运行在一个循环中，取指，译码，对指令在机器的寄存器集，缓存和主存中的效果进行仿真。仿真器包含一个流水线模型，记录更详细的性能信息，代价是更长的仿真时间。在内存系统仿真中，不同级别的细节仿真是可用的，从简单的cache-miss计数器，到多处理器cache一致性硬件的精确模型。SimOS高度细节的模式被用于计算机架构研究，以及操作系统关键部分代码的性能调试。

Altogether, SimOS provides hardware simulators ranging from very approximate to highly accurate models. Similarly, the slowdowns resulting from execution on these simulators ranges from well under a factor of 10 to well over a factor of 1,000.

总计，SimOS提供的硬件仿真器，从非常近似的，到高度精确的模型。类似的，在这些仿真器上的运行，其减速程度从10倍到1000倍不等。

## 3. Efficient Simulation by Direct Execution

The greatest challenge faced by the SimOS direct-execution mode is that the environment expected by an operating system is different from that experienced by user-level programs. The operating system expects to have access to privileged CPU resources and to an MMU it can configure for mapping virtual addresses to physical addresses. The SimOS direct-execution mode creates a user-level environment that looks enough like “raw” hardware that an operating system can execute on top of it. Table 1 summarizes the mapping of features in SimOS’s direct-execution mode.

SimOS直接执行模式面临的最大挑战，是操作系统期待的环境，与用户级程序体验的环境是不一样的。操作系统要访问特权CPU资源，以及MMU，MMU可以配置将虚拟地址映射到物理地址。SimOS直接执行模式，创建了一个用户级的环境，与原始的硬件看起来很类似，操作系统可以在其上进行执行。表1总结了在SimOS的直接执行模式中特征的映射。

### 3.1. CPU Instruction Execution

To achieve the fastest possible CPU simulation speed, the direct-execution mode uses the host processor for the bulk of instruction interpretation. It simulates a CPU by using the process abstraction provided by the host operating system. This strategy constrains the target instruction to be binary-compatible with the host instruction set. The operating system runs within a user-level process, and each CPU in the target architecture is modeled by a different host process. Host operating system activity, such as scheduler preemptions and page faults, is transparent to the process and will not perturb the execution of the simulated hardware. CPU simulation using the process abstraction is fast because most of the workload will run at the native CPU’s speed. On multiprocessor hosts, the target CPU processes can execute in parallel, further increasing simulation speed.

为获得最快的CPU仿真速度，直接执行模式使用宿主处理器进行主要的指令翻译。对CPU的仿真，使用的是宿主操作系统提供的进程抽象。这种策略要求目标指令与宿主指令集是二进制兼容的。操作系统运行在用户级的进程中，在目标架构中的每个CPU，是通过不同的宿主进程进行建模。宿主操作系统行为，比如调度器preemptions和页错误，对进程是透明的，不会打扰被仿真的硬件的执行。使用进程抽象的CPU仿真是很快速的，因为workload的多数会以本地CPU的速度运行。在多处理器的宿主上，目标CPU进程可以以并行执行，进一步加速仿真速度。

Because operating systems use CPU features unavailable to user-level processes, it is not possible simply to run the unmodified operating system in a user-level process. Two such features provided by most CPUs are the trap architecture and the execution of privileged instructions. Fortunately, most workloads use these features relatively infrequently, and so SimOS can provide them by means of slower simulation techniques.

因为操作系统会使用用户级进程不可用的CPU特征，所以不可能将未修改的操作系统运行在用户级进程上。多数CPU都会有的两种这样的特征是，trap架构，和特权指令的执行。幸运的是，多数workload对这种特征的使用频率都很低，所以SimOS可以用较慢的仿真技术来运行。

A CPU’s trap architecture allows an operating system to take control of the machine when an exception occurs. An exception interrupts the processor and records information about the cause of the exception in processor-accessible registers. The processor then resumes execution at a special address that contains the code needed to respond to the exception. Common exceptions include page faults, arithmetic overflow, address errors, and device interrupts. To simulate a trap architecture, the process representing a SimOS CPU must be notified when an exceptional event occurs. Fortunately, most modern operating systems have some mechanism for notifying user-level processes that an exceptional event occurred during execution. SimOS uses this process notification mechanism to simulate the target machine’s trap architecture.

CPU的trap架构使操作系统在异常发生时，可以控制机器。异常会中断处理器，将异常的原因信息记录在处理器可以访问的寄存器中。寄存器然后在一个特殊的地址恢复运行，其中的代码对异常进行响应。常见的异常包括页错误，代数overflow，地址错误，和设备中断。为对异常架构进行仿真，代表SimOS CPU的进程，需要在异常事件发生时，通知该进程。幸运的是，多数现代操作系统都有一些机制可以通知用户级进程，在执行的时候发生了一些异常时间。SimOS使用这个进程通知机制，来仿真目标机器的trap架构。

In Unix, user-level processes are notified of exceptional events via the signal mechanism. Unix signals are similar to hardware exceptions in that the host OS interrupts execution of the user process, saves the processor state, and provides information about the cause of the signal. If the user-level process registers a function (known as a signal handler) with the host OS, the host OS will restart the process at the signal handler function, passing it the saved processor state. SimOS’s direct-execution mode registers signal handlers for each exceptional event that can occur. The SimOS signal handlers responsible for trap simulation convert the information provided by the host OS into input for the target OS. For example, upon receiving a floating-point exception signal, the invoked SimOS signal handler converts the signal information into the form expected by the target OS’s trap handlers and transfers control to the target OS’s floating-point-exception-handling code.

在Unix中，用户级的进程通过信号机制来接收异常时间的通知。Unix信号与硬件异常类似，宿主OS会中断用户进程的执行，保存处理器状态，给出信号原因的信息。如果用户级进程在宿主OS中注册了一个函数（信号处理器），宿主OS会在信号处理器函数中重启进程，将保存的处理器状态传入到该进程中。SimOS的直接执行机制对每个可能发生的异常事件都注册了信号处理函数。对trap仿真负责的SimOS信号处理函数，将宿主OS提供的信息，转换成目标OS的输入。比如，在收到浮点异常信号时，调用的SimOS信号处理函数将信号信息转换到目标OS的trap处理器函数期望的形式，然后将控制权交给目标OS的浮点异常处理代码。

In addition to the trap architecture, most CPUs provide privileged instructions that the operating system uses to manipulate a special state in the machine. This special state includes the currently enabled interrupt level and the state of virtual-memory mappings. The privileged instructions that manipulate this state cannot be simulated by directly executing them in a user-level process; at user-level these instructions cause an illegal instruction exception. The host OS notifies the CPU process of this exception by sending it a signal. The direct-execution mode uses such signals to detect privileged instructions, which it then interprets in software. SimOS contains a simple software CPU simulator capable of simulating all the privileged instructions of the CPU’s instruction set. This software is also responsible for maintaining privileged registers such as the processor’s interrupt mask and the MMU registers.

除了trap架构，多数CPUs都有特权指令，操作系统使用这些指令在机器中操作一些特殊状态。这些特殊状态包括，当前使能的中断级别，和虚拟内存映射的状态。操作这些状态的特权指令，不能通过直接在用户级进程执行，来进行仿真；在用户级，这些指令会导致非法指令的异常。宿主OS通过发送信号来将这个异常通知CPU进程。直接执行模式使用这种信号来检测特权指令，然后将其翻译成软件。SimOS包含一个简单的软件CPU仿真器，可以仿真CPU指令集中所有特权指令。这个软件也负责维护特权寄存器，比如寄存器处理器的中断mask，和MMU寄存器。

### 3.2 MMU Simulation

A process is an obvious way to simulate a CPU’s instruction interpretation, but an analog for the memory management unit is not so obvious because a user-level process’s view of memory is very different from the view assumed by an operating system. An operating system believes that it is in complete control of the machine’s physical memory and that it can establish arbitrary mappings of virtual address ranges to physical memory pages. In contrast, user-level processes deal only in virtual addresses. To execute correctly, the target operating system must be able to control the virtual-to-physical address mappings for itself and for the private address spaces of the target user processes.

要仿真CPU的指令翻译，进程是一种很明显的方法，但MMU的类比就没那么明显了，因为内存的用户级进程视角，与操作系统假设的视角是非常不同的。操作系统相信，它对机器的物理内存是有完全控制的，可以将虚拟地址范围，任意的映射到物理内存页。相比之下，用户级进程只处理虚拟地址。为正确的执行，对目标用户进程的私有地址空间，目标操作系统必须能够控制虚拟到物理地址的映射。

The MMU also presents a special simulation challenge because it is used constantly-by each instruction fetch and each data reference. As Figure 2 (next page) shows, we use a single file to represent the physical memory of the target machine. For each valid translation between virtual and physical memory, we use the host operating system to map a page-size chunk of this file into the address space of the CPU-simulating process. The target operating system’s requests for virtual memory mappings appear as privileged instructions, which the direct-execution mode detects and simulates by calling the host system’s file-mapping routines. These calls map or unmap page-size chunks of the physical memory file into or out of the simulated CPU’s address space. If a target application instruction accesses a page of the application’s address space that has no translation entry in the simulated MMU, the instruction will access a page of the CPU simulation process that is unmapped in the host MMU. As discussed earlier, the simulated trap architecture catches the signal generated by this event and converts the access into a page fault for the target operating system.

MMU还有一个特殊的仿真挑战，因为它一直在进行使用，每个取指和每个数据reference都在使用。如图2所示，我们使用单个文件来表示目标机器的物理内存。对每个虚拟和物理内存的有效变换，我们使用宿主操作系统来将本文件的一个页大小的块，映射到CPU仿真进程的地址空间。目标操作系统对虚拟内存映射的请求，是以特权指令的形式出现的，直接执行模式会检测并进行仿真，调用宿主系统的文件映射程序。这些调用，将物理内存文件的页大小的块映射到仿真的CPU的地址空间，或取消其映射。如果目标应用的指令访问这个应用的地址空间，但在仿真的MMU中没有变换entry，这些指令访问的CPU仿真进程的页面，是在宿主MMU中没有映射的。这在之前讨论过，仿真的trap架构会捕捉到这个事件生成的信号，将对这个访问转换成目标操作系统的页面错误。

We simulate the protection provided by an MMU by using the protection capabilities of the file-mapping system calls. For example, mapping a page-size section of the physical memory file without write permission has the effect of installing a read-only translation entry in the MMU. Any target application write attempts to these regions produce signals that are converted into protection faults and sent to the target operating system.

我们将MMU提供的保护，通过文件映射系统调用的保护功能进行仿真。比如，将物理内存文件中页大小没有写权限的部分进行映射，其效果是在MMU中放入一个只读翻译entry。任意目标应用对这些区域的写尝试，产生的信号都会转换成保护错误，送入到目标操作系统中。

In many architectures the operating system resides outside the user’s virtual address space. This causes a problem for the SimOS MMU simulation because the virtual addresses used by the operating system are not normally accessible to the user. We circumvented this problem by relinking the kernel to run at the high end of the user’s virtual address space in a range of addresses accessible from user mode. We also placed the SimOS code itself in this address range. Although this mechanism leaves less space available for the target machine’s user-level address space, most applications are insensitive to this change. Figure 3 illustrates the layout of SimOS in an Irix address space.

在很多架构中，操作系统是在用户的虚拟地址范围之外的。这对SimOS的MMU仿真导致一个问题，因为操作系统使用的虚拟空间不是用户可以正常访问的。我们通过重新链接规避了这个问题，将核运行在用户虚拟地址空间的high end部分，这是用户模式可以访问的地址。我们还将SimOS的代码本身也放入到这个地址范围内。虽然这种机制使目标机器的用户级地址空间的可用部分更少了，但多数应用对这个变化是没有什么感觉的。图3描述了SimOS在Irix地址空间的分布。

### 3.3 Device Simulation

SimOS simulates a large collection of devices supporting the target operating system. These devices include a console, magnetic disks, Ethernet interfaces, periodic interrupt timers, and an interprocessor interrupt controller. SimOS supports interrupts and direct memory access (DMA) from devices, as well as memory-mapped I/O (a method of communicating with devices by using loads and stores to special addresses).

SimOS对支持目标操作系统的大量设备进行仿真。这些设备包括一个console，磁盘，以太网接口，周期性的中断时钟，和处理器间中断控制器。SimOS支持来自设备的中断和DMA，以及内存映射的I/O（使用对特殊地址的loads和stores来与设备进行通信的方法）。

In direct-execution mode, the simulated devices raise interrupts by sending Unix signals to the target CPU processes. As described earlier, SimOS-installed signal handlers convert information from these signals into input for the target operating system. The timer, interprocessor, and disk interrupts are all implemented by this method. We implement DMA by giving the devices access to the physical memory file. By accessing this file, I/O devices can read or write memory to simulate transfers that occur during DMA.

在直接执行模式中，仿真的设备通过向目标CPU进程发送Unix信号，来产生中断。之前描述过，SimOS安装的信号处理函数将这些信号中的信息转换成目标操作系统的输入。时钟中断，处理器间中断，和磁盘中断，都是通过这种方法实现的。我们通过让设备可以访问物理内存文件，来实现DMA。通过访问这个文件，I/O设备可以读写内存，对DMA时发生的行为进行仿真。

To simulate a disk, SimOS uses a file with content corresponding to that of a real disk. The standard file-system build program converts standard files into raw disk format. SimOS uses this program to generate disks containing files copied from the host system. Building disks from host files gives the target OS access to the large volume of programs and data necessary to boot and run large, complex workloads.

为对磁盘进行仿真，SimOS使用了一个文件，内容与真实的磁盘是对应的。标准的文件系统构建程序将标准文件转换成原始的磁盘格式。SimOS使用这个程序来生成包含文件的磁盘，文件是从宿主系统中拷贝过来的。从宿主文件中构建磁盘，使目标OS可以访问大量程序和数据，对启动和运行大型复杂workloads都是必须的。

SimOS contains a simulator of an Ethernet local-area network, which allows simulated machines to communicate with each other and the outside world. The implementation of the network interface hardware in SimOS sends messages to an Ethernet simulator process. Communication with the outside world uses the Ethernet simulator process as a gateway to the local Ethernet. With network connectivity, SimOS users can remotely log in to the simulated machines and transfer files using services such as FTP (file transfer protocol) or NFS (Network File System). For ease of use, we established an Internet subnet for our simulated machines and entered a set of host names into the local name server.

SimOS包含一个以太局域网的仿真器，使被仿真的机器可以互相访问，并访问外面的世界。网络接口硬件在SimOS中的实现，向以太仿真器进程发送信息。与外部世界的通信，将以太网仿真器进程用作局部以太网的网关。有了网络的连接性，SimOS的用户可以远程登录到被仿真的机器，使用FTP或NFS这样的服务来传输文件。为使用简单，我们对被仿真的机器确立了一个Internet子网，输入了一些域名到局部域名服务器中。

## 4. Detailed CPU Simulation

Although the SimOS direct-execution mode runs the target operating system and applications quickly, it does not model any aspect of the simulated system’s timing and may be inappropriate for many studies. Furthermore, it requires compatibility between the host platform and the architecture under investigation. To support more detailed performance evaluation, SimOS provides a hierarchy of models that simulate the CPU and MMU in software for more accurate modeling of the target machine’s CPU and timing. Software-simulated architectures also remove the requirement that the host and target processors be compatible.

虽然SimOS直接执行模式可以很快的运行目标操作系统和应用，但这并没有对被仿真系统的时序进行任何建模，对很多研究可能是不合适的。而且，这还需要宿主平台和被研究的架构的兼容性。为支持更详细的性能评估，SimOS提供了一系列模式，对CPU和MMU用软件进行仿真，更详细的对目标机器的CPU和时序进行仿真。软件仿真的架构不要求宿主和目标处理器是兼容的。

### 4.1 CPU Simulation Via Binary Translation

The first in SimOS’s hierarchy of more detailed simulators is a CPU model that uses binary translation to simulate execution of the target operating system and applications. This software technique allows greater execution control than is possible in direct-execution mode. Rather than executing unaltered workload code as in direct execution, the host processor executes a translation of that code, which is produced at runtime. From a block of application or OS code, the binary translator creates a translation that applies the operations specified by the original code to the state of the simulated architecture. Figure 4 presents an example of this translation and the flexibility it provides.

在SimOS更详细的系列化仿真器中的第一个，是一个CPU模型，使用二进制翻译来仿真目标操作系统和应用的执行。这种软件技术，比在直接执行模式中，可以更好的控制执行。在直接执行中，执行的是未改变的workload代码，在二进制翻译中，宿主处理器执行的是代码的翻译，这是在运行时产生的。从应用或OS的代码中，二进制翻译器创建了一个翻译，将原始代码指定的运算，翻译床被仿真架构的状态。图4给出了这种翻译的一个例子。

Many interesting workloads execute large volumes of code, so the translator must be fast. We amortize the time spent on runtime code generation by storing the code block translations in a large translation cache. Upon entering each basic block, SimOS searches the cache to see if a translation of this code already exists. Basic block translations present in the cache are reexecuted without incurring translation costs.

很多有趣的workloads执行了大量代码，所以翻译器必须快速。运行时代码生成所消耗的时间，分摊到了将代码块翻译存储到一个大型翻译cache中。在进入到每个basic block中时，SimOS搜索这些cache，来看这些代码的翻译是否存在。在cache中存在的basic block的翻译可以重新执行，而不会带来翻译的代价。

As in the direct-execution mode, each CPU in the target machine is simulated by a separate user process. Using separate processes reduces the accuracy of instruction and memory reference interleaving, but it allows the simulated processors to run concurrently on a multiprocessor host. Since the binary-translation mode’s emphasis is execution speed, the efficiency obtained by parallel simulation outweighs the lost accuracy.

如同在直接执行模式中一样，在目标机器中的每个CPU，都由一个单独的用户进程来仿真。使用单独的进程会降低指令和内存reference交叉的精确度，但允许被仿真的处理器在多处理器宿主上并行运行。由于二进制翻译模式强调的是执行速度，并行仿真得到的效率的收益，会优于损失的准确率。

The binary-translation CPU’s ability to dynamically generate code supports on-the-fly changes of the simulator’s level of detail. A faster binary-translation mode instruments its translations only to count the number of instructions executed. A slower binary-translation mode counts memory system events such as cache hits and misses. A highly optimized instruction sequence emitted by the translator performs cache hit checks, quickly determining whether a reference hits in the cache; thus, the full cache simulator is invoked only for cache misses (which are infrequent). Adding cache simulation yields a more accurate picture of a workload’s performance on the simulated architecture.

二进制翻译CPU动态生成代码支持的能力，会改变仿真器的细节程度。更快的二进制翻译模式对其翻译进行增强，对执行的指令数量进行计数。更慢的二进制翻译模式，会对内存系统事件进行计数，比如cache hits和misses。翻译器生成的高度优化的指令序列进行cache hit检查，迅速确定一个reference是否在cache中hits；因此，完整的cache仿真器调用只是为了进行cache misses检查（这是不频繁的）。增加cache仿真，会在被仿真的架构中生成更精确的workload性能的picture。

### 4.2 CPU Simulation Via Detailed Software Interpretation

Although the binary-translation CPU is extremely fast and provides enough detail to derive some memory system behavior, it is not sufficient for detailed multiprocessor studies. Because each simulated CPU executes as a separate Unix process on the host machine, there is no fine-grained control over the interleaving of memory references from the multiple simulated CPUs. To address this deficiency, SimOS includes two more-detailed CPU simulators that provide complete control over the instruction interleaving of multiple processors.

虽然二进制翻译的CPU速度很快，给出了足够的细节，可以推导出一些内存系统的行为，但是仍然不足以进行详细的多处理器研究。因为每个仿真的CPU都在宿主机上执行为分离的Unix进程，所以对多个仿真CPUs的交叉内存reference并没有更细粒度的控制。为处理这个缺陷，SimOS包含了两个更详细的CPU仿真器，对指令交叉在多个处理器内执行提供完全的控制。

The first of these simulators interprets instructions using a straightforward fetch, decode, and execute loop. Because we want precise, cycle-by-cycle interleavings of all CPUs, we simulate them all in a single Unix process. Precise cycle interleavings allow device interrupts to occur with precise timing and allow more accurate cache simulation. The additional accuracy of this mode and its current inability to exploit parallelism on a multiprocessor host result in slowdowns more than an order of magnitude larger than the binary-translation CPU.

这些处理器的第一个，使用一种直接的fetch，decode，和execute循环来解释这个指令。因为我们希望所有CPU的精确的，周期级交叉，我们在一个Unix过程中对其进行仿真。精确的周期交叉使设备中断的发生有精确的时序，允许更精确的cache仿真。这个模式下额外的准确率，以及不能利用多处理器宿主上的并行性，会得到比常规二进制翻译CPU的速度慢了一个数量级。

As its second more-detailed CPU model, SimOS includes a dynamically scheduled CPU similar to many next-generation processors such as the Intel P6, the Mips R10000, and the AMD KS. This model incorporates highly aggressive processor design techniques including multiple instruction issue, out-of-order execution, and hardware branch prediction. The CPU simulator is completely parameterizable and accurately models the pipeline behavior of the advanced processors. This accuracy, combined with the model’s single-process structure, results in extremely time-consuming simulation. We use this CPU model to study the effect of aggressive processor designs on the performance of both the operating system and the applications it supports.

第二个更细节的CPU模型，SimOS包含了一个动态调度的CPU，与很多下一代处理器类似，比如Intel P6，MIPS R10000，和AMD KS。这个模型包含了很激进的处理器设计技巧，包含多指令发射，乱序执行，硬件分支预测。CPU仿真器是完全参数化的，精确的对高级处理器的流水线行为进行建模。这种精确度，与模型的单线程结构结合，会得到非常耗时的仿真。我们使用这个CPU模型来研究激进处理器设计在操作系统和应用上的效果。

### 4.3 Switching Simulators and Sampling

The hierarchy of CPU simulators in SimOS makes accurate studies of complex workloads feasible. The slowdowns at the most detailed level make running entire complex workloads for a reasonable amount of time far too expensive, so we exploit SimOS’s ability to switch modes. We control switching by specifying that a given program be run in a more detailed mode or that we want to sample a workload. Sampling a workload consists of executing it in one CPU simulator for a given number of simulated cycles and then switching execution to another simulator. By toggling simulators, we can obtain most of the information of the more detailed mode at a performance near that of the less detailed mode. Because the different modes share much of the simulated machine’s state, the time required to switch levels of detail is negligible.

SimOS中的CPU仿真器层次使复杂workloads的精确研究成为可能。在最精细级的减速下，使运行完整的复杂workloads，会耗费非常多的时间，所以我们要利用SimOS切换模式的能力。我们指定一个给定的程序运行在更细节的模式下，或我们希望对一个workload采样，这样来控制切换。对一个workload进行采样，是在一个CPU仿真器中执行给定数量的周期，然后切换到另一个仿真器进行执行。通过切换仿真器，我们可以在细节较少的模式下，以该性能得到在更细节的模式下的大部分信息。因为不同的模式共享被仿真机器的大部分状态，切换细节层次所需的时间，是可以忽略的。

Sampling is useful for understanding workload execution because application programs’ execution phases usually display different behavior. For instance, program initialization typically involves file access and data movement, while the main part of the program computation phase may stress the CPU. These phases may be even more dissimilar for multiprocessor workloads, in which the initialization phase may be single-threaded, while the parallel, computation phase may alternate between heavy computation and extensive communication. Thus, capturing an accurate picture of the workload by examining only one portion is not possible. Sampling enables a detailed simulator to examine evenly distributed time slices of an entire workload, allowing accurate workload measurement without the detailed simulator’s large slowdowns.

采样对理解workload执行是有用的，因为应用程序的不同执行阶段通常会展现出不同的行为。比如，程序初始化一般会涉及到文件访问和数据移动，而程序计算阶段的主要部分一般CPU利用率会很高。这些阶段在多处理器workload不相似程度会更高，其中初始化阶段可能是单线程的，而并行的计算阶段，会在很重的计算和通信之间切换。因此，通过检查一部分，来捕获workload的精确picture，这通常是不可能的。采样使详细的仿真器可以检查整个workload的均匀分布的时间片，所以可以在有限的时间内精确的测量workload。

Switching modes is not only useful for sampling between a pair of modes. It is also useful for positioning workloads for study. For example, we usually boot the operating system under the direct-execution mode and then switch into the binary-translation mode to build the state of the system’s caches. Once the caches have been "warmed up" with referenced code and data, we can begin more detailed examination of the workload. We switch to the more detailed CPU model for an accurate examination of the workload’s cache behavior. We present specific performance numbers for the various simulation levels later in the article.

切换模式不仅在几种模式下进行采样是有用的，对于研究中positioning workloads也是有用的。比如，我们通常在直接执行模式下启动操作系统，然后切换到二进制翻译模式下构建系统cache的状态。一旦cache用参考代码和数据进行预热后，我们可以进行更详细的workload检查。我们切换到更详细的CPU模型上，更精确的检查workload的cache行为。本文后面，我们给出各种不同的仿真级下具体的性能数值。

### 4.4 Memory System Simulation

The growing gap between processor speed and memory speed means that the memory system has become a large performance factor in modern computer systems. Recent studies have found that 30 to 5O percent of some multi-programmed workloads’ execution time is spent waiting for the memory system rather than executing instructions. Clearly, an accurate computer system simulation must include a model of these significant delays.

处理器速度和内存速度之间越来越大的差距，意味着在现代计算机系统中，内存系统已经成为了一个很大的性能因素。最近的研究发现，很多多编程workload的执行时间的30%-50%，是在等待内存系统，而不是执行指令。很明显，精确的计算机系统仿真，必须包含这些显著延迟的模型。

To hide the long latency of memory from the CPU, modern computer systems incorporate one or more levels of high-speed cache memories to hold recently accessed memory blocks. Modeling memory system stall time requires simulating these caches to determine which memory references hit in the cache and which require additional latency to access main memory. Since processor caches are frequently controlled by the same hardware module that implements the CPU, SimOS incorporates the caches in the CPU model. This allows the different CPU simulator implementations to model caches at an appropriate level of accuracy.

为向CPU掩盖内存系统的长延迟，现代计算机系统包含了一个或几个级别的高速cache，存储最近访问的内存块。建模存储系统stall时间，需要仿真这些caches，确定哪些内存reference击中cache，哪些需要额外的延迟来访问主存。因为处理器caches是由CPU中实现的硬件模块进行频繁控制的，SimOS在CPU模型中包含了这些caches。这允许不同的CPU仿真器实现在合适的准确度级对cache进行建模。

The direct-execution mode does not model a memory system, so it does not include a cache model. The binary-translation mode can model a single level of cache for its memory system. This modeling includes keeping multiple caches coherent in multiprocessor simulation and ensuring that DMA requests for I/O devices interact properly with caches. Finally, the detailed CPU simulators include a multilevel cache model that we can parameterize to model caches with different organization and timing. It can provide an accurate model of most of today’s computer system caches. As in the CPU models, each level of cache detail increases simulation accuracy as well as execution time.

直接执行模式并没有对存储系统进行建模，所以并没有包含cache模型。二进制翻译模式可以建模一个级别的cache。这个建模包括在多处理器仿真中保持多个caches一致，确保I/O设备的DMA请求与caches合理的互动。最后，详细的CPU仿真器包含多级cache模型，可以参数化对不同组织和时序的cache进行建模。这就可以提供多数现代计算机系统cache的精确模型。就像在CPU模型中一样，每个级别的cache细节增加仿真精确度，以及执行时间。

SimOS also provides multiple levels of detail in modeling the latency of memory references that miss in the caches. In the fastest and simplest of these models, all cache misses experience the same delay. More complex models include modeling contention due to memory banks that can service only one request at a time. This model makes it possible to accurately model the latencies of most modern, bus-based multiprocessors.

SimOS对在caches中miss的内存reference的延迟，还有不同级别细节的建模。在最快和最简单的模型中，所有cache misses都是相同的延迟。更复杂的模型包括由于memory banks导致的建模contention，这些bank一次只能服务一个请求。有了这个模型就可以精确的建模多数现代的，基于bus的多处理器的延迟了。

Finally, SimOS contains a memory system simulator that models the directory-based cache-coherence system used in multiprocessors with distributed shared memory. These machines, such as Stanford’s DASH multiprocessor, have non-uniform memory access (NUMA) times due to the memory distribution. Each processor can access local memory more quickly than remote memory. The simulator also models the increased latency caused by memory requests that require coherency-maintaining activity. This simulator has been useful for examining the effects of the NUMA architecture in modern operating systems.

最后，SimOS包含一个内存系统仿真器，对在带有分布式共享存储的多处理器中基于directory的cache一致性系统进行了建模。这些机器，比如Stanford的DASH多处理器，因为内存分布情况，有非一致的内存访问(NUMA)。每个处理器访问局部存储要比访问远程存储更快。对于需要一致性维护行为的内存请求，会有增加的延迟，仿真器也进行了建模。这个仿真器对于检查现代操作系统中NUMA架构的效果非常有用。

Although we have presented our CPU simulator hierarchy independently from our memory system hierarchy, there are correlations. Certain CPU simulations require certain memory system simulation support. For example, the dynamically scheduled CPU simulator requires a nonblocking cache model to exploit its out-of-order execution. Furthermore, without a memory system that models contention, the timings reported by the processor will be overly optimistic. Likewise, simpler CPU simulation models are best coupled with simpler, faster memory system models.

虽然我们给出了我们的CPU仿真器层次，与存储系统层次独立，它们也有关联。特定的CPU仿真，需要特定的内存系统仿真支持。比如，动态调度的CPU仿真器，需要nonblocking cache模型，以利用其乱序执行。而且，没有对contention进行建模的存储系统，处理器给出的时序会过分乐观。类似的，更简单的CPU仿真模型，与更简单的更快的内存系统模型是匹配的。

## 5. SimOS Performance

The two primary criteria for evaluating a simulation environment are what information it can obtain and how long it takes to obtain this information. Table 2 (next page) compares the simulation speeds of several of the SimOS simulation modes with each other and with execution on the native machine. We used the following workloads in the comparison:

评估一个仿真环境的两个主要原则，是它能够得到什么信息，以及获得这些信息需要多少时间。表2比较了几种SimOS仿真模式的仿真速度，以及在本地机器上执行的速度。我们在比较中使用了下列的workloads：

- SPEC benchmarks: To evaluate uniprocessor speed, we ran three programs selected from the SPEC92 benchmark suite. The performance of these applications has been widely studied, providing a convenient reference point for comparisons to other simulation systems.

为评估单处理器的速度，我们从SPEC92 benchmark包中选择了3个程序运行。这些应用的性能已经广泛研究，提供了一个方便的参考点，可以和其他仿真系统进行比较。

- Multiprogram mix: This workload is typical of a multiprocessor used as a compute server. It consists of two copies of a parallel program (raytrace) from the SPLASH benchmark suite and one of the SPEC benchmarks (eqntott). The mix of parallel and sequential applications is typical of current multiprocessor use. The operating system starts and stops the programs as well as time-sharing the machine among the multiple programs.

这个workload是典型的多处理器用作计算服务器。这包含SPLASH benchmark包中一个并行程序raytrace的两份副本，还包含SPEC benchmark中的一个eqntott。并行和顺序应用的混合，是当前多处理器应用的典型情况。操作系统启动和停止这些程序，并且在多个程序中对机器进行时分应用。

- Pmake: This workload represents a multiprocessor used in a program development environment. It consists of two independent program compilations taken from the compile stage of the Modified Andrew Benchmark. Each program consists of multiple files compiled in parallel on the four processors of the machine, using the SGI pmake utility. This type of workload contains many small, short-lived processes that make heavy use of OS services.

这个workload代表了一个多处理器在程序开发环境中的应用。其中包含了两个独立的程序编译，从Modified Andrew Benchmark的编译阶段中取出。每个程序包含多个文件，在机器的4个处理器中并行编译，使用SGI pmake工具。这种workload包含很多小的，存活时间很短的进程，使用了很多OS的服务。

- Database: This workload represents the use of a multiprocessor as a database server, such as might be found in a bank. We ran a Sybase database server supporting a transaction-processing workload, modeled after TPC-B. The workload contains four processes that comprise the parallel database server plus 20 client programs that submit requests to the database. This workload is particularly stressful on the operating system’s virtual memory subsystem and on its interprocess communication code. It also demonstrates a strength of the SimOS environment: the ability to run large, commercial workloads.

数据库：这种workload代表了将多处理器是用作数据库服务器的情况，比如在银行中就会是这种情况。我们运行一个Sybase数据库服务器，支持事务处理的workload。这个workload包含4个进程，由并行数据库服务器，加上20个客户程序，向数据库提交请求构成。这个workload对操作系统的虚拟内存子系统，和进程间通信代码压力特别大。这也展示了SimOS环境的力量：运行大型，商用workload的能力。

We executed the simulations on a SGI Challenge multiprocessor equipped with four 1 50-MHz R4400 CPUs. We configured the different simulators to behave like this host platform and ran the workloads on top of them. The native execution numbers represent the wallclock time necessary to execute each workload directly on the Challenge machine. The other numbers indicate how much slower the workload ran under simulation. We computed the slowdowns by dividing the simulation wall-clock times by the native execution times. The detailed simulations execute the multiprocessor simulations in a single process.

我们在一个SGI挑战多处理器上运行仿真，带有4个150MHz R4400 CPUs。我们配置了不同的仿真器，以运行workloads。本地运行数字代表在Challenge机器上直接运行每个workload所需的时间。其他数字代表了，在仿真运行的情况下，workload运行的速度慢了多少，即将仿真运行的时间，除以本地运行的时间。详细仿真执行的多处理器仿真是单线程运行的。

The trade-off between CPU speed and the level of modeling detail is readily apparent in Table 2. For the uniprocessor workloads, the highly detailed simulations are more than 100 times slower than the less detailed direct-execution mode and around 200 times slower than the native machine. The binary-translation mode’s moderate accuracy level (instruction counting and simple cache simulation) results in moderate slowdowns of around 5 to 10 times the native machine.

CPU速度和建模细节程度的折中，在表2中很明显。对单处理器的workloads，高度细节的仿真，比细节较少的直接执行模式，慢了100多倍，比本地执行慢了200多倍。二进制翻译模式的中等准确率级别（指令技术和简单缓存仿真），比本地机器运行大约慢了5-10倍。

The trade-off between accuracy and speed becomes even more pronounced for multiprocessor runs. For the relatively simple case of running several applications in the multiprogram mix, the accurate simulations take 500 times longer than the native machine. Since the detailed CPU model does not exploit the underlying machine’s parallelism to speed the simulation, its slowdown scales linearly with the number of CPUs being simulated. This causes a slowdown factor in the thousands when simulating machines with 16 or 32 processors.

准确率和速度的折中，对多处理器运行会变得更加明显。在相对较简单的在multiprogram mix中运行几个应用中，准确的仿真比本地执行的速度慢了500多倍。由于详细的CPU模型并没有利用潜在机器的并行性来加速其仿真，其减速与仿真的CPU数量呈正比关系。这导致在仿真16或32个处理器的时候，速度降低了有上千倍。

The complex nature of the other two multiprocessor workloads, along with heavy use of the operating system, causes all the simulators to run slower. Frequent transitions between kernel and user space, frequent context switches, and poor MMU characteristics degrade the direct-execution mode’s performance until it is worse than the binary-translation model. These troublesome characteristics cannot be handled directly on the host system and constantly invoke the slower software layers of the direct-execution mode.

另外两个多处理器workloads的复杂本质，以及对操作系统的重度使用，导致所有模拟器运行速度会更慢。在kernel和用户空间中的频繁迁移，频繁的上下文切换，和很差的MMU特性，降低了直接执行模式的性能，直到其比二进制翻译模型要更差。这些麻烦的特性不能直接在宿主系统中处理，频繁的调用直接执行模式中更慢的软件层。

The pmake and database workloads cause large slowdowns on the simulators that model caches because the workloads have a much higher cache miss rate than is present in the uniprocessor SPEC benchmarks. Cache simulation for the binary-translation CPU is slower in the multiprocessor case than in the uniprocessor case due to the communication overhead of keeping the multiprocessor caches coherent. Similar complexities in multiprocessor cache simulation add to the execution time of the detailed CPU modes.

pmake和数据库会导致建模cache的仿真器中很大的减速，因为这些workloads中的cache miss rate，比单处理器SPEC benchmarks中要高的多。二进制翻译CPU的Cache仿真，在多处理器中的情况要比单处理器中的情况要慢的多，因为要保持多处理器cache一致的话，要有很多的通信任务。多处理器cache仿真的类似复杂性，也会增加详细CPU模式中的执行时间。

## 6. Experiences with SimOS

SimOS development began in spring 1992 with the simulation of the Sprite network operating system, running on Sparc-based machines. We started development of the Mips-based SimOS described in this article in fall 1993 and have been using it since early 1994. A simulation environment is only as good as the results it produces, and SimOS has proven to be extremely useful in our research. Recent studies in three areas illustrate the SimOS environment’s effectiveness:

SimOS的开发在1992年开始，对Sprite网络操作系统进行仿真，运行在基于Sparc的机器上。我们在1993年开始基于MIPS的SimOS的开发，在1994年开始使用。仿真环境与其产生的结果一样好，SimOS在研究中被证实非常有用。最近在三个领域的研究，证实了SimOS环境的有效性：

- Architectural evaluation: SimOS is playing a large part in the design of Stanford’s Flash, a large-scale NUMA multiprocessor. Researchers have used SimOS’s detailed CPU modes to examine the performance impact of several design decisions. The ability to boot and run realistic workloads such as the Sybase database and to switch to highly accurate machine simulation have been of great value.

架构评估：SimOS在Stanford的Flash的设计中起到了很大作用，这是一个大型NUMA多处理器。研究者使用SimOS详细的CPU模式来检查几种设计决策的性能影响。启动和运行实际workload的能力，比如Sybase数据库，以及能够切换到高精度机器仿真的能力，具有很大作用。

- System software development: In the design of an operating system for Flash, SimOS’s direct-execution and binary-translation modes provide a development and debugging environment. Because SimOS supports full source-level debugging, it is a significantly better debugging environment than the raw hardware. Developers use the detailed CPU models to examine and measure time-critical parts of the software. In addition, SimOS provides the OS development group with Flash “hardware” long before the machine is to be complete.

系统软件开发：在设计Flash的操作系统时，SimOS的直接执行和二进制翻译模式提供了一个开发和调试的环境。因为SimOS支持full source级的调试，这是一个比原始硬件明显更好的调试环境。开发者使用详细CPU模型来检查和测量软件的时间关键部分。此外，SimOS给OS开发组提供了Flash硬件，远比机器完成要早。

- Workload characterization: SimOS’s ability to run complex, realistic workloads including commercial applications enables researchers to characterize workloads that have not been widely studied before. Examples include parallel compilations and the Sybase database.

workload特征刻画：SimOS能运行复杂，真实的workload的能力，包括商业应用，使研究者可以刻画workload的特征，这在之前并没有进行广泛的研究。例子包括并行编译，和Sybase数据库。

Based on our experience with SimOS, we believe that two of its features will become requirements for future simulation environments. These key features are the ability to model complex workloads, including all operating system activity, and the ability to dynamically adjust the level of simulation detail. Modern computer applications, such as database-transaction processing systems, spend a significant amount of execution time in the operating system. Any evaluation of these workloads or the architectures on which they run must include all operating system effects.

基于我们对于SimOS的经验，我们相信它的两个特征会成为未来仿真环境的要求。这些关键特征是，能够建模复杂workload的能力，包括所有操作系统行为，和动态调整仿真细节的能力。现代计算机应用，比如数据库事务处理系统，在操作系统中会耗费很多执行时间。对这些workload或运行在其上的架构的任何评估，，都必须包含所有的操作系统效果。

Simulation of multiple levels of detail that can be adjusted on the fly allows rapid exploration of long-running workloads. A fast simulation mode that allows positioning of long-running workloads is essential for performance studies. Furthermore, as system complexity increases, accurate simulators will be too slow to run entire workloads. Sampling between fast simulators and detailed simulators will be the best way to understand complex workload behavior.

可以随时调整的多层次细节的仿真，就可以迅速探索运行时间很长的workloads。一个快速的仿真模式，允许定位长时间运行的workloads，这是性能研究的基础。而且，随着系统复杂度提升，精确的仿真器运行整个的workloads会非常慢。在快速仿真器和细节仿真器之间采样，会是理解复杂workload行为的最佳方式。