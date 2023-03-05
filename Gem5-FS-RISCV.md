# Supporting RISC-V Full System Simulation in gem5

Peter Yuen Ho Hin et. al. @ Huawei

## 0. Abstract

The RISC-V ISA and ecosystem have been becoming an increasingly popular in both industry and academia. gem5 is a widely used powerful simulation platform for computer architecture research. Previous works have added single-core and multi-core RISC-V support to gem5 but only for system call emulation. The full-system simulation of gem5, on the other hand, provides accurate analysis of systems as an actual system software is loaded and run on the hardware platform modelled in gem5. However, full-system simulation support in gem5 for RISC-V ISA is currently not available.

RISC-V ISA和生态已经在工业界和学术界越来越流行。Gem5是计算机架构研究中广泛使用的很强的仿真平台。之前的工作在gem5中加入了单核和多核的RISC-V支持，但只是对系统调用仿真的模式。另一方面，gem5的全系统仿真模式，将实际的系统软件载入，并在gem5建模的硬件平台中进行运行，给出了系统的精确分析。但是，gem5对RISC-V ISA的全系统仿真仍然是不可用的。

This paper presents our recent work on supporting RISC-V fullsystem simulation in gem5. After describing the implementation details of supporting extensible target system and debugging methodology for overcoming major challenges, we share our experiments of full-system simulations.

本文给出了我们最近在gem5中对RISC-V的全系统仿真的支持的工作。我们先描述了支持可扩展目标系统和调试方法的实现细节，克服了主要的挑战，然后分享了我们的全系统仿真的试验。

## 1. Introduction

RISC-V is a free, open-source ISA [6][5] which has recently gained popularity from both the academia and the industry. RISC-V is designed to be simple, efficient yet future-proof by avoiding the pitfalls of existing ISAs and allowing extension of the instruction set. The RISC-V ISA also adopts a modular approach where vendors can implement any chosen set of the RISC-V ISA extensions. As such, this ISA is friendly to academic research and low volume applications, but powerful enough to be extended to warehousescale applications.

RISCV是一种免费开源的ISA，最近在学术界和工业界都非常流行。RISCV的设计上是非常简单高效的，避免了现有ISAs的缺陷，允许对指令集进行扩充。RISCV ISA采用的是一种模块化的方法，供应商可以实现RISCV ISA的任意扩展集。这样，这种ISA对学术研究和低容量应用都很友好，但是也可以强大到扩展为warehouse级的应用。

gem5 is a powerful open-source simulator [1] [2] widely used in computer architecture research. It strives to achieve a balance between speed, accuracy and development time. Its users can choose between different CPU models, system modes and memory systems to achieve system configurations with desired level of tradeoffs. Such configurations can be easily setup up through gem5’s Python interface, while performance-critical simulation logic is implemented in C++.

gem5是一个强大的开源仿真器，在计算机架构的研究上得到了广泛的应用，在速度、准确率和开发时间上有不错的平衡。其用户可以在不同的CPU模型，系统模式和内存系统中选择，以获得理想的折中水平的系统配置。这样的配置通过gem5的python接口可以很容易的设置，其性能关键的仿真逻辑是用C++实现的。

For each ISA, gem5 offers two modes of simulation: syscall emulation (SE) and full system (FS) simulation. With previous work [3, 4], gem5 is able to support most RISC-V instructions and system calls in SE mode where system calls are emulated. This mode provides a simplified method to run and analyze user-space workloads. FS simulation is needed for accurate analysis of system components and devices as an actual system software (often Linux kernel) is loaded by gem5.

对每个ISA，gem5提供了两种仿真模式：系统调用仿真(SE)和全系统仿真(FS)。有了之前的工作[3,4]，gem5可以在SE模式中支持多数RISCV指令和系统调用，其中系统调用是被模拟的。这个模式提供了一种简化的方法来运行和分析用户空间的workloads。要对系统组成部分和设备进行精确的分析，就需要FS仿真模式，在这种模式中，gem5会载入实际的系统软件（通常是Linux内核）。

Research areas which are made possible by FS simulation include virtual memory, virtualization, distributed systems, storage stack performances and network-related studies. However, the use of RISC-V for those areas was limited due to the lack of FS simulation support. This paper addresses this gap.

用FS仿真可以进行的研究领域包括，虚拟内存，虚拟化，分布式系统，存储栈性能和与网络相关的研究。但是，在这些领域中使用RISCV还是受限的，因为缺少FS仿真支持。这篇文章处理了这个空白。

In this paper, we present our work of adding support for RISC-V full-system simulation, which has been included with a GNU/Linux Busybox distribution with kernel version 5.10 in the official gem5-21.0 release. We describe the target system setup and major challenges in Section 2, followed by the implementation details of new device models and platform in Section 3. Section 4 explains our debugging methodology. Finally, Section 5 presents validation and testing results using the full system setup.

本文中，我们加入了RISCV FS仿真的支持，在官方的gem5-21.0版本中已经包含了，还包括一个GNU/Linux Busybox分布，核的版本为5.10。我们在第2部分描述了目标系统设置和主要的挑战，第3部分描述了新设备模型和平台的实现细节，第4部分解释了我们的调试方法，第5部分使用了FS设置给出了验证和测试结果。

## 2. Development Target and Challenges

In this section, we first show the target system setup and then discuss the challenges to overcome for successful gem5 RISC-V full-system support.

本节中，我们首先给出了目标系统的设置，然后讨论了要克服的挑战，以成功的支持gem5 RISCV FS模式。

### 2.1 Target System

The goal is to build a baseline RISC-V system which can be easily extended based on user needs. This target system has a core set of hardware sources including a minimum set of peripherals. It is able to run the system software, for example, one with bootloader and Linux kernel.

目标是构建一个基准RISCV系统，并且可以基于用户需求很容易的拓展。目标系统有一个硬件源的核心集，包括外设的最小集合。可以运行系统软件，比如，BootLoader和Linux内核。

The hardware configuration of target system is shown in Figure 1, where only modules of interests are presented. Besides bus subsystem, there are two major sub-systems: CPU and HiFive Platform. The blocks represented by orange boxes are the devices newly added in our work for successful booting of system software.

目标系统的硬件配置如图1所示，只给出了感兴趣的模块。除了总线子系统，有两个主要的子系统：CPU和HiFive平台。橙色框表示的模块，是我们的工作新加入的设备，可以成功的启动系统软件。

In CPU sub-system, an extra MMU component, PMA checker, is added. The HiFive platform is based on SiFive’s HiFive series of board and contains the minimal set of critical peripherals. The Core Local Interrupt Controller (CLINT) handles software and timer interrupts via a MMIO interface. The Platform Level Interrupt Controller (PLIC) is responsible for routing interrupts from external sources and peripheral devices to the hardware threads based on a priority scheme. The UART and VirtIOMMIO are not necessary for kernel boot-up but are essential for a usable operating system. UART provides an interactive command line terminal while the VirtIOMMIO provides a copy-on-write root filesystem which contains the workload scripts and operating system binaries.

在CPU子系统中，加入了额外的MMU部件和PMA检查器。HiFive平台是基于SiFive的HiFive系列开发板，包含关键外设的最小集合。CLINT通过一个MMIO接口来处理软件和时钟中断。PLIC负责将外部源和外设的中断，基于一个优先级方案，路由到硬件线程中。UART和VirtIOMMIO对于内核启动并不是必须的，但对于一个可用的操作系统来说是必须的。UART提供了一个交互式的命令行终端，而VirtIOMMIO提供了一个copy-on-write的root文件系统，包含了workload脚本和操作系统的binaries。

Figure 2 shows the stack consisting of several software layers in gem5 full-system simulation. The gem5 (FS) block in the figure contains the hardware modules of system with desired configuration. It also models the interactions between hardware modules. CPU model with RISC-V ISA decoder handles the instructions from OS layer or user applications, which could be at different privileged modes. A gem5 FS simulation starts with parsing Python configuration script and building simulator executable based on configurations. Then, the simulator loads bootloader and Linux kernel to boot up the system. When kernel is up, user applications can be executed in background or via terminal.

图2展示了在gem5 FS仿真中包含了几个软件层的栈。图中的Gem5 FS块包含了期望配置系统的硬件模块，还对硬件模块之间的互动进行了建模。带有RISCV ISA解码器的CPU模型，处理OS层或用户应用的指令，这可能是在不同的特权模式的。一个gem5 FS仿真以解析Python配置脚本，基于配置来构建仿真器可执行文件开始。然后，仿真器载入bootloader和Linux内核以启动系统。当内核运行后，用户程序可以在后台执行或通过terminal执行。

Our gem5 RISC-V FS simulation also supports a hypervisor which doesn’t need hardware-assistant virtualization, i.e., RISC-V H extension. Diosix is such a hypervisor.

我们的gem5 RISCV FS仿真还支持hypervisor，这不需要硬件辅助的虚拟化，即RISCV H扩展。Diosix就是这样一个hypervisor。

### 2.2. Challenges

Compared to the SE mode, the FS simulation has a more complex configuration consisting of aforementioned newly added hardware components and software components such as the kernel and bootloader payloads. During a boot process, a fault can occur in any of the above components, even in interactions with the existing CPU models and memory models due to wrongly implemented privileged instructions and newly added device models.

与SE模式相比，FS仿真的配置要更复杂，包括之前提到的新加入的硬件组件和软件组件，比如kernel和BootLoader的payload。在启动的过程中，上述的任何组件都可能出错，甚至是在与现有的CPU模型和内存模型的交互中，由于错误实现的特权指令和新加入的设备模型导致。

There could be multiple possible reasons for a failure during a FS simulation as below. The DTB configuration errors can cause bootloader run into an erroneous status after parsing device-tree. In FS simulation, interrupt mechanism plays a critical role to keep the system running. The wrong privileged ISA implementation, interrupt triggering mechanism, or the interrupt handling logic within CPU and interrupt controller can lead to faults in simulation. As we found during debugging, in the complex scenarios, there were even errors due to CPU pipeline squashing and memory access of peripheral devices.

在一个FS仿真的过程中，如果有错误的话，可能是下面的几个原因之一。DTB配置错误会导致BootLoader解析设备树运行后进入错误状态。在FS仿真中，中断机制在维持系统运行中扮演了关键的角色。错误的特权ISA实现，中断触发机制，或CPU和中断控制器中的中断处理逻辑，都会导致仿真中的错误。就像我们在调试中发现的，在复杂的场景中，甚至有CPU pipeline squashing和外围设备内存访问导致的错误。

Finding out the root cause of a fault in above mentioned gem5 FS simulation is challenging due to the various reasons and complex scenarios. Identifying the code or instructions within bootloader or kernel that trigger the fault is more challenging because of the code size of them and the entrance into an infinite loop of process where the bootloader or the kernel enters a panic function and ends up in an idle loop when an error occurs. When that happens, millions of assembly instructions could have executed in simulation and it is impossible to trace the root cause manually. Therefore, a method of effective backtrace through millions of assembly instructions to locate the origin of an error is needed.

在上述gem5 FS仿真中，找到错误的根本原因，是很有挑战的，因为会有各种各样的原因，和复杂的场景。找到BootLoader或kernel中触发这个错误的代码或指令，这更加有挑战，因为代码量很大，而且在错误发生时，在BootLoader或kernel进入一个panic函数，最后到达一个空转的循环后，过程会进入无限的循环。当这种情况发生时，可能已经仿真了数百万条汇编指令，基本上不可能追踪到根原因。因此，需要一种方法来有效的回溯数百万条汇编指令，来找到错误的起始位置。

Additionally, gem5 is an event-driven simulator which simulates the desired tick-by-tick behaviour of the hardware. However, this introduces complexities to the debugging as call-stack information is limited to calls within the same simulation tick. Events such as memory read request and response would not be visible under the same call stack. Hence, we would need a method of analyzing beyond the scope of current tick.

此外，gem5是一个事件驱动的仿真器，仿真的是硬件期望的tick-by-tick行为。但是，这带来了调试的复杂性，因为在相同的仿真tick的调用中，调用栈信息是有限的。比如内存读取的请求和回应的事件，在相同的调用栈中，不一定是可见的。因此，我们需要一种方法来分析当前tick范围之外的信息。

To smooth our work of supporting RISC-V FS simulation in gem5, in addition to enhancing traditional remote GDB debugging, we developed a sophisticated tool kit (a set of Python scripts) to analyze the instruction execution trace of a gem5 simulation together the comparison of the corresponding execution trace in Qemu. In Section 4, we will elaborate on our debugging methodology and how to apply it to overcome the above challenges.

为使在gem5中支持RISCV FS仿真中的工作更顺滑，除了增强传统的远程GDB调试，我们还开发了一种复杂的工具集（Python脚本集合），来分析gem5仿真的指令执行轨迹，并与在QEMU中的执行轨迹进行比较。在第4部分中，我们会详述我们的调试方法，怎样应用，以克服上述挑战。。

## 3. Implementation Details

In this section, we present the implementation details of the newly added devices or hardware modules. The UART module is built-in in gem5 while the VirtIOMMIO model is ported over from the ARM setup. The focus is put on the platform and other devices like CLINT, PLIC and PMAChecker.

本节中，我们给出新加入的设备或硬件模块的实现细节。UART模块是gem5中自带的，而VirtIOMMIO模型是从ARM设置中移植过来的。我们主要关注的是平台和其他设备，如CLINT，PLIC和PMAChecker。

We also talk about the other fixes on privileged instructions and CPU models. This section is closed by supports for checkpointing and device tree.

我们还讨论了在特权指令和CPU模型中的其他修正，此外还有对checkpointing和设备树的支持。

### 3.1. HiFive Platform

In gem5, system configurations are organized into container classes called platforms. A Platform class is a parent class with a standardized set of peripherals and utility functions that can be extended in a hierarchical manner to customize the setup to a specific board/system. In ARM, the common Platform class is RealView while in X86, the common Platform class is PC. In RISC-V, we name the platform HiFive, which corresponds to SiFive’s HiFive series of board. The memory map conventions and peripheral addresses are chosen based on the SiFive U54MC SoC datasheet. The HiFive platform contains the minimal set of peripherals upon which other non-critical peripherals can be added to. Such a base configuration is used not only on HiFive boards but also on other SoCs such as the Kendryte K210.

在gem5中，系统配置会被组织到容器类中，称为平台。Platform类是一个父类，有标准的外围设备集，和工具函数，可以以层次化的方式来进行拓展，定制为特定开发板/系统的设置。在ARM中，通用的Platform类是RealView，而在X86中，通用的Platform类是PC。在RISC-V中，我们将平台命名为HiFive，对应着SiFive的HiFive系列开发板。内存映射的约定和外设的地址是基于SiFive U54MC SoC datasheet的。HiFive平台包含了最小外设集，其他非关键的外设也可以加入。这样一个基准配置并不只是在HiFive板上使用，而且也在其他SoCs上，比如Kendryte K210。

The HiFive platform is designed to be easily extendable, with minimal changes needed to be made to port over devices from other ISAs. A PlicIntDevice class is provided to allow easy connection of a peripheral to the PLIC interrupt controller. A set of utility functions within the HiFive platform class also allows users to add new devices to a list and have the necessary connections made automatically.

HiFive平台的设计是可以很容易拓展的，需要进行的修改很少。提供了一个PlicIntDevice来很容易的将外设连接到PLIC中断控制器。HiFive platform类的工具函数集，让用户可以加入新的设备到list上，可以自动进行必要的连接。

### 3.2 CLINT

CLINT, as introduced earlier, handles software and timer interrupts. RISC-V hardware threads can set timer interrupts or send interprocessor interrupts to other threads by writing to memory-mapped registers of CLINT in machine mode.

CLINT处理的是软件和定时器的中断。RISCV硬件线程可以设置定时器中断，或通过在机器模式中写入到内存映射的CLINT寄存器，发送处理器间中断到其他线程中。

In SiFive’s setup, CLINT is often connected to an external RTC clock signal that increments the MTIME register, which then triggers interrupts based on the MTIMECMP register values for each hardware thread. In the current setup, a dummy RTC model is constructed to allows for a configurable clock frequency but is not yet implemented as a MMIO device. Aside from timer interrupts, the MTIME register also supplies the value for the RDTIME instruction.

在SiFive的设置中，CLINT通常是连接到一个外部RTC时钟信号的，对MTIME寄存器进行递增，对每个硬件线程，基于MTIMECMP寄存器值，触发中断。在当前的设置中，构建了一个dummy RTC模型，来允许可配置的时钟频率，但尚未实现为一个MMIO设备。除了时钟的中断，MTIME寄存器还为RDTIME指令提供值。

Software interrupts in CLINT are posted and cleared using an MMIO interface. In machine modes, inter-processor interrupts (IPI) are made possible by allowing access to the MSIP register address of other cores.

在CLINT中的软件中断，是使用MMIO接口来发布和清除的。在机器模式中，通过访问其他核的MSIP寄存器地址，跨处理器中断(IPI)得以实现。

### 3.3 PLIC

PLIC is responsible for routing of external interrupts to different contexts, each of which corresponds to an interrupt pending bit in a certain privilege mode. PLIC can route interrupts from up to 1023 external sources (1 to 1023) to up to 15872 contexts. A context in PLIC can corresponding to a hardware thread, or a (hardware thread, privilege level) tuple depending on the hardware implementation. In the current gem5 implementation, each hardware thread corresponds to two contexts, one for M mode and one for S mode.

PLIC负责将外部中断路由到不同的上下文中，每个都对应着特定特权模式中的一个中断pending位。PLIC可以将最多1023个外部源的中断路由到最多15872个上下文中。PLIC中的一个上下文可以对应着一个硬件线程，或一个（硬件线程，特权级）的对，这依赖于硬件实现。在当前的gem5实现中，每个硬件线程对应着两个上下文，一个是M模式，一个是S模式。

Each source will be assigned a 32-bit priority and each context can enable or disable interrupts from any sources. At the same time, each context can also set a threshold on the minimum interrupt priority to trigger an external interrupt. A simplified working example of PLIC is illustrated in Figure 3. In accordance with the specifications, a 3-cycle delay is simulated between the interrupt sources and the external interrupt signals.

每个源会指定一个32-bit的优先级，每个上下文都可以enable或disable任意源的中断。同时，每个上下文都可以设置最低中断优先级的阈值，来触发外部中断。PLIC的一个简化工作例子如图3所示。

Claiming and completion of external interrupts are also implemented via MMIO. The gem5 implementation ensures that any external interrupt can only be claimed by one context and that a context cannot claim multiple interrupts before completing the last claimed interrupt.

外部中断的claiming和completion也是通过MMIO实现的。Gem5实现确保了任意外部中断都只会被一个上下文claim，一个上下文在完成最后claimed的中断前，不会claim多个中断。

### 3.4 PMAChecker

Aside from the MMIO devices, the RISC-V ISA also requires the implementation of a Physical Memory Attribute (PMA) checking mechanism for checking attributes of the memory address such as atomicity, memory-ordering, coherence, cacheability and idempotency. The RISC-V specification suggests possible memory attributes to check for but does not specify any standards on the implementation on this hardware component.

除了MMIO设备，RISCV ISA还需要实现PMA检查机制，以检查内存地址的属性，如atomicity, memory-ordering, coherence, cacheability和idempotency。RISCV规范建议了要检查的可能的内存属性，但是并没有在这个硬件组件的实现上指定任何标准。

As such, the gem5 implementation of the PMA checker is an abstract components which adds certain flags and attributes to the memory request after address translation. Currently, the PMA checker only checks for uncacheability but further checks can be implemented easily when the need arises. This unit is necessary for the proper functioning of all MMIO peripherals as it ensures that memory requests to these devices will not be cached.

因此，PMA检查器的gem5实现是一个抽象组件，对内存请求，在地址转换后加入了特定的flags和属性。目前，PMA检查器只检查uncacheability，但当需求到来时，可以很容易的加入其他内容的检查。这个单元对于所有MMIO外设的正常工作是必须的，因为这确保了向这些设备的内存请求不会被缓存。

### 3.5 Fixes on Privileged Instructions and CPU Models

Prior to this paper’s work, the RISC-V ISA is already mostly supported in SE mode. However, a few fixes were made by us on the CSR instructions and interrupt handling logic. The current ISA supports still has some minor discrepancy with the RISC-V ISA specifications but these discrepancies should not affect the functionality of the system.

在本文的工作之前，RISC-V ISA在SE模式中已经基本都支持了。但是，我们还是对CSR指令和中断处理逻辑进行了一些修正。目前的ISA支持仍然与RISC-V ISA标准有一些不一致，但是，这些不一致不能影响系统的功能。

In order to support full system booting on MinorCPU and DerivO3CPU, the following changes were made in our work. Firstly, PLIC avoids posting interrupt to hardware threads which are already handling an external interrupt. This should not be necessary in real hardware but was needed for MinorCPU support. MinorCPU lacks the internal logic to prevent receiving interrupt signals while it is inside an interrupt handler. Secondly, a pipeline squash was forced after CSR writes to the SATP register. This is necessary such that AUIPC instructions in the pipeline do not use a wrong PC value when the MMU is activated.

为支持在MinorCPU和DerivO3CPU的FS启动，我们的工作中做出了如下的变化。首先，PLIC避免将中断传递到正在处理外部中断的硬件线程。这在真实硬件上是不必要的，但在MinorCPU支持上是必须的。当已经在处理一个中断的时候，MinorCPU缺少内部逻辑来避免接收到其他的中断信号。第二，在向SATP寄存器进行CSR写的时候，必须要进行流水线squash。这是必须的，只有这样，在MMU激活时，流水线中的AUIPC指令不会使用到错误的PC值。

### 3.6 Checkpointing and Device Tree

Besides a command line interface, another essential feature of a full system we added in gem5 is the ability to store and restore checkpoints. Booting a full system simulation can be very time consuming, especially when multiple peripheral devices are connected to the system. As such, researchers often use a fast CPU model such as AtomicSimpleCPU for the boot process and checkpoint the system after boot-up. Subsequently, researchers can run benchmark using more accurate CPU models by restoring from the checkpoint without the need to boot up the system. In our gem5 RISC-V full system support, we have tested and verified the checkpointing functionality on all CPU models.

除了命令行接口，我们对gem5加入的另一个FS特征是，存储和恢复checkpoints。启动一个FS仿真是非常耗时的，尤其是系统还连接着多个外设的清空。这样，研究者通常对启动过程使用一个快速CPU模型，比如AtomicSimpleCPU，在启动之后对系统进行checkpoint。然后，研究者可以从checkpoint中恢复，而不需要重新启动这个系统，使用更精确的CPU模型来运行benchmark。

Our full system support also comes with devicetree generation functionality. In Linux systems, the device and peripheral setup is made known to the kernel using a devicetree. Since gem5 systems are configured using a Python interface, it is necessary to modify the devicetree binary passed to the kernel everytime the system configuration is modified. In our full system support, we have added devicetree generation feature to each peripheral device such that users can avoid the trouble of having to manually match the devicetree binary with the Python configuration.

我们的FS支持还带有设备树生成的功能。在Linux系统中，核是使用设备树来知道设备和外设的设置的。由于gem5系统是使用Python接口来进行配置的，在每次系统配置进行修改后，都有必要修改传递给kernel的设备树。在我们的FS支持中，我们对每个外设都增加了设备树生成的功能，这样用户就可以不用手动将设备树binary与Python配置进行手动匹配了。

## 4. Debugging Methodology

To overcome the challenges mentioned in Section 2, we enhance remote GDB support and introduce the methodology of trace analysis. In this section, we present details of them and share how they were applied in debugging FS simulation.

为克服第2部分中提到的挑战，我们增强了远程GDB支持，引入了trace分析的方法。本节中，我们给出细节，分享一下在调试FS仿真的时候怎样应用。

### 4.1 Remote GDB and Trace Analysis

A remote GDB stub is commonly attached to the workload running on the simulator. Using breakpoints and stack information, the origin of an error can be quickly traced. Prior to our work, RISC-V remote GDB in gem5 only supports printing values of integer registers. Support for printing values of float-point and CSR registers is added to allow for more efficient checking of privileged instruction implementation and errors.

在仿真器中运行的workload上，通常会附上一个远程GDB stub。使用断点和堆栈信息，错误的源头可以很快的追溯到。我们的工作之前，gem5中的RISCV远程gdb只支持打印整数寄存器的值。增加了打印浮点寄存器和CSR寄存器的值的支持，允许更高效的检查特权指令实现和错误。

Whenever a boot attempt of FS simulation fails, the method of remote GDB cannot be directly applied because it is difficult to identify the instruction causing trouble. Different from the common user workloads, the kernel and bootloader do not exit on an error like a typical workload. Instead, a panic function is called, which eventually goes into an idle loop. Given the size and complexity of the kernel and bootloader, together with the large number of simulation instructions (could be millions), it is not feasible to iteratively backtrack till the origin of the error using remote GDB in a forward-straight way.

当FS仿真的启动尝试失败时，远程GDB的方法就不能直接应用了，因为很难识别到导致错误的指令。与常见的用户workloads不同，kernel和bootloader不会像一个典型的workload一样，以一个错误退出，而是会调用一个panic函数，最终进入到一个空转的循环中。由于kernel和bootloader的代码量都很大，很复杂，要仿真的指令数很大（可能是数百万），要使用远程GDB直接迭代的追溯到错误的源头是不太可行的。

We introduce the method of trace analysis and develop a toolkit which consists of several Python scripts for trace analysis. In order to avoid manual inspection of millions of lines of the output traces, we used QEMU as a reference for the exactly same execution path. Using the same system setup described in a same DTS file, we boot QEMU emulator and gem5 full-system simulator side-by-side and collect both execution traces which are in different formats. We then use the toolkit to parse both traces and perform comparisons on the execution path to find the location of the translation block where the two traces diverges. Subsequently we use aforementioned enhanced remote GDB to automatically insert breakpoints into the blocks to identify where errors come from.

我们引入了trace分析的方法，开发了一个工具集，包含了几个Python脚本进行trace分析。为避免手动分析数百万条输出trace，我们使用QEMU作为同样的执行路径的参考。使用相同的系统设置（即相同的DTS文件），我们启动QEMU仿真器和gem5 FS仿真器，收集两种执行traces，这是不同的格式。然后我们使用这个工具集来解析两种traces，对执行路径来进行比较，找到两个traces产生分歧的translation block的位置。然后，我们使用之前提到的增强型远程GDB，来自动插入断点到blocks中，以识别错误来自的地方。

### 4.2 Experience of Debugging FS Simulation

Using the above debugging methods together with gem5 build-in debugging log, we were able to effectively locate and fix any errors that occur during gem5 FS simulations.

使用上述调试方法，与gem5内建的调试log，我们可以有效的定位和修复在gem5 FS仿真时的任何错误。

An example of such errors is an incorrect return value from a load instruction, which was caused by the incorrect implementation of a peripheral device or the accidentally caching of the MMIO address range. By trace comparison with QEMU, we quickly identified the instruction that caused the panic condition. Looking into the instruction, we figured out the reason for the fault.

这样的错误的一个例子是，一条load指令的错误返回结果，这是由于一个外设的错误实现，或意外的缓存了MMIO地址范围导致的。通过与QEMU进行trace比较，我们迅速的识别出了导致这种panic情况的指令。查看这条指令，我们找到了这个错误的原因。

In some cases, trace analysis helps identify the location but cannot reveal the root cause. We can further leverage the remote GDB functionality to check or write to the interrupt pending and enable bits. Take the debugging of interrupt trigger mechanism for instance. The remote GDB functionality allowed us to efficiently verify the trigger conditions and timing behaviour of interrupt devices such as PLIC and CLINT. It also helped in debugging the interrupt handling logic in gem5 RISC-V CPUs by checking through the register state changes in desired clock cycles.

在一些情况下，trace分析帮助识别了位置，但不能发现根本原因。我们可以进一步利用远程GDB的功能，来检查或写入到中断pending和enable bits。以中断触发机制的调试为例。远程GDB功能使我们可以高效的验证触发条件和中断设备的时序行为，比如PLIC和CLINT。还可以通过检查在期望的时钟周期内的寄存器状态变化，来帮助调试gem5 RISCV CPUs里的中断处理逻辑。

Aside from the above tools trace analysis and remote GDB, gem5’s built-in debug logs were extensively used as well. They are helpful for printing out the internal states of targeted gem5 devices. For example, by enabling the debug logs of DerivO3CPU, we were able to inspect the behaviour of the pipeline stages in each clock cycle and thus identify issues such as accidentally triggered squashes. After fixing CLINT and SATP write side effects, DerivO3CPU is supported in RISC-V FS simulation.

除了上述工具，gem5的内建调试logs也可以使用。它们对于打印出目标gem5设备的内部状态是有帮助的。比如，让DerivO3CPU的调试logs可用，我们可以检查pipeline stages在每个时钟周期的行为，因此识别出意外触发的squash的问题。在修复了CLINT和SATP写入的副作用后，RISCV FS仿真支持了DerivO3CPU。

## 5. Experiments of FS Simulation

### 5.1 Full-System Linux Boot-up

To verify our implementation, we booted up the target system in Section 2 using the Berkeley bootloader together with the Linux kernel v5.10. For simplicity, the system consists of four CPU cores, each with one hardware thread. The filesystem used is a port of the BusyBox disk image.

为验证我们的实现，我们使用bbl+Linux kernel v5.10来启动目标系统。为简单起见，系统包含4个CPU核，每个都有1个硬件线程。使用的文件系统是BusyBox。

The Linux system has been successfully booted under all widely used four CPU models offered by gem5: Atomic Simple, Timing Simple, Minor and DerivO3. We further logged in the system and executed commands using terminal. The commands within BusyBox can be run without errors. Checkpoint and restore functionalities have also been tested with switching CPU models.

Linux系统在gem5提供的4种广泛使用的CPU模型中成功的启动过：Atomic Simple，Timing Simple，Minor和DerivO3。我们进一步登录到系统中，使用terminal执行了命令。BusyBox中的命令可以无错误运行。在切换CPU模型时，也测试了checkpoint和恢复的功能。

This correct functionality of commands shows that the kernel’s process management and scheduler is working. Since the scheduler relies on CLINT’s timer interrupts, we are sure that CLINT’s timing functionality is implemented correctly. The ability to read, write and move files also demonstrates the correct functionality of the file system, which is controller by the VirtIOMMIO device. Furthermore, the proper functioning of the interactive terminal shows that the UART and PLIC models are correctly configured.

命令的正确运行，说明核的进程管理和调度是运行无误的。由于调度器依赖于CLINT的时钟中断，我们确定CLINT的时钟功能是正确实现的。读、写和移动文件的能力，也表明了文件系统的正确功能，这是由VirtIOMMIO设备控制的。而且，交互式terminal的正确运行，表明UART和PLIC的模型也是正确的配置的。

The most important components of Linux OS, including process management, memory management, device drivers and interprocess communication, have been checked to work correctly. We are confident that the target system and FS simulation support have been correctly implemented.

Linux OS最重要的组件，包括进程管理，内存管理，设备驱动和进程间通信，都被证明是正确运行的。我们很自信，目标系统和FS仿真支持都已经得到了正确实现。

### 5.2 Running Multi-threaded Workloads

We created an instance of target system with Symmetric MultiProcessing (SMP) configuration. This configuration has four O3 CPU cores (each with one hardware thread) and 1024M DRAM. We further created a port of the PARSEC benchmark and selected five benchmarks. During experiments, we run multiple multi-threaded workloads on above Linux platform using different numbers of threads to see how this SMP configuration works.

我们创建了目标系统带有SMP配置的一个实例。这个配置有4个O3 CPU核（每个都有1个硬件线程）和1024M DRAM。我们还移植了PARSEC benchmark，选择了5个benchmarks。在试验中，我们在上述Linux平台上，使用不同数量的线程，运行多个多线程workloads，来看看这个SMP配置工作效果如何。

Table 1 shows the simulated run-time taken to complete each execution of benchmarks under different thread counts. As some simulations are still running when this paper is submitted, a few cells are filled with "-" to indicate that the related data could NOT be included in camera ready paper.

表1给出了在不同数量的线程下，完成每个benchmark的仿真运行时间。由于在本文提交时，一些仿真还在运行，一些单元中就填充了-，说明相关的数据还没有获取到。

As shown in the table, when a workload with a same data set to be handled by different numbers of threads, the execution with 2X threads could reduce the simulated run-time, i.e., the execution time of workload in simulation, roughly by half, compared to the one with 1X thread(s). For example, benchmark Blackscholes takes 39ms to finish the execution using 1 thread. After using 4 threads, it only needs 10.1ms to complete. The speed-up is close to 4. For other benchmarks, there are speed-up more or less. Hence, we can conclude that our full-system setup has successful support for running multi-threaded workloads on SMP configuration with multiple hardware threads.

如表格所示，当同样数据集的workload由不同数量的线程处理时，线程数量加倍，会减少仿真的运行时间，大概是一半的样子。比如，benchmark Blackscholes在1个线程时耗时39ms。使用4个线程时，只用大约10.1ms完成。加速大概是4倍。对于其他的benchmarks，也有或多或少的加速。因此，我们得出结论，我们的FS设置可以成功的支持在SMP配置下，在多硬件线程下，多线程workloads的运行。

### 5.3 Running Workloads on Linux as Guest OS of Diosix Hypervisor

Our full-system setup also allows for analysis of more complicated system setups. We created an instance of target system with the hardware configuration of 1 CPU core, which has one hardware thread. This hardware configuration boots up two software configurations: 1) Linux FS; and 2) Diosix (a M mode hypervisor) with Linux as one of its Guest OS. Table 2 shows comparison of the simulated run-time for the blackscholes benchmark under different simulation runs using different numbers of software threads.

我们的FS设置还可以分析更复杂的系统设置。我们以1 CPU核1硬件线程的硬件配置创建了目标系统。这个硬件配置启动两种软件配置：1. Linux FS，2. Diosix，Linux作为其Guest OS。表2展示了blackscholes benchmark在使用不同的软件线程数量时，在不同的仿真运行时的仿真运行时间。

From the third column, we can see that the simulated run-time is increased following the increase of number of threads used to run the benchmark. This shows the thread context switch overhead on a CPU core with single hardware thread.

从第3列可以看出，随着使用的线程数量的增加，仿真的运行时间是增加的。这表明线程上下文切换在一个CPU核上是有代价的。

By comparing the numbers on a same row, we can see that the number on the third column is much less than the one on the fourth column. The extra delay is caused by Diosix for 1) intercepting and running system calls from benchmark; and 2) scheduling other guest OSes running on top of Diosix.

比较同一行的数据，我们可以发现，第3列的数比第4列要小的多。这额外的延迟是由Diosix造成的，用于：1. 拦截并运行benchmark中的系统调用；2. 调度Diosix上其他guest OSes的运行；

## 6. Conclusion and Future Work

In this paper, we have presented our work on adding support for gem5 RISC-V FS simulation. We have implemented a core extensible target system for gem5 FS simulation by adding new devices and fixing errors in privileged instructions and CPU models. We further elaborate on our debugging methodology which is beneficial to other developers as well. At last, we share our experiments of FS simulations in 1) booting up Linux system, 2) running workloads on top of the system and 3) booting up Diosix hypervisor with two guest OSes and running workloads on one of the guest Linux OS.

本文中，我们给出了对gem5加入RISCV FS仿真的支持。我们对gem5 FS仿真实现了一个核心可拓展的目标系统，加入了新的设备，修正了在特权指令和CPU模型中的错误。我们进一步详述了我们的调试方法，对其他开发者这也是有好处的。最后，我们共享了我们在FS仿真上的试验，包括：1. 启动Linux系统，2. 在系统上运行workloads，3. 启动Diosix hypervisor带有2个guest OSes，在其中一个guest Linux OS上运行workloads。

Our gem5 FS simulation support provides the essential infrastructure to a multitude of future research topics on the RISC-V ISA. We give several examples as below.

我们的gem5 FS仿真支持对未来的很多RISCV ISA研究提供了关键的基础设施。我们下面给出几个例子。

gem5 FS simulation is crucial in the analysis of the virtual memory system architecture. The FS simulation allows the modelling of the system with virtual memory that integrates proposals such as page based attributes mechanism and new address translation modes for RISC-V in the OS kernel.

gem5 FS仿真在虚拟内存系统架构的分析上是很关键的。FS仿真允许对带有虚拟内存的系统进行建模，可以集成基于page的属性机制，和在OS核中对RISCV的新的地址翻译模式。

Security research involves multiple components of the system: the cores, the bus, the secure enclave, the external devices, the OS and the firmware. FS simulation can be used to evaluate proposals implemented in gem5 at system level.

安全研究涉及系统的多个组件：核，总线，安全enclave，外设，OS和固件。FS仿真可以用于评估在gem5中实现的系统级的建议。

The hardware-assistant virtualization technique is critical to high-performance processors in HPC and cloud domains. The RISCV H extension and KVM support can be added on top of the gem5 FS simulation so as to enhance the capability of system-level performance evaluation.

硬件辅助的虚拟化技术对HPC和云计算领域的高性能处理器很关键。RISCV H扩展和KVM支持可以加在gem5 FS仿真上，以增加系统级的性能评估的能力。