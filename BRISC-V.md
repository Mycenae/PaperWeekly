# The BRISC-V Platform: A Practical Teaching Approach for Computer Architecture

Rashmi Agrawal et. al. @ Boston University

## 0. Abstract

Computer architecture lies at the intersection of electrical engineering, digital design, compiler design, programming language theory and high-performance computing. It is considered a foundational segment of an electrical and computer engineering education. RISC-V is a new and open ISA that is gaining significant traction in academia. Despite it being used extensively in research, more RISC-V-based tools need to be developed in order for RISC-V to gain greater adoption in computer organization and computer architecture classes. To that end, we present the BRISC-V Platform, a design space exploration tool which offers: (1) a web-based RISC-V simulator, which compiles C and executes assembly within the browser, and (2) a web-based generator of fully-synthesizable, highly-modular and parametrizable hardware systems with support for different types of cores, caches, and network-on-chip topologies. We illustrate how we use these tools in teaching computer organization and computer architecture classes, and describe the structure of these classes.

计算机架构处于电子工程，数字设计，编译器设计，编程语言理论和高性能计算的相交处，是电子工程和计算机工程教育的基础性部分。RISC-V是一种新的开放的ISA，在学术界吸引了很多注意力。尽管在研究中使用很广泛，需要开发更多的基于RISC-V的工具，以使得计算机组成和计算机架构课程更多的采用RISC-V。为此，我们开发出了BRISC-V平台，这是一种设计空间探索工具，可以给出：(1)一种基于web的RISC-V仿真器，可以在浏览器中编译C并运行汇编语言；(2)一种基于web的完全可综合的，高度模块化的，参数化的硬件系统生成器，可以支持不同类型的核，缓存，和NoC拓扑。我们描述了我们怎样在教授计算机组成和计算机架构课程中使用这些工具，描述了这些类的结构。

**Keywords** computer architecture, computer organization, risc-v, simulator, Verilog, generator

## 1. Introduction

Courses such as Computer Organization and Computer Architecture are fundamental for a computer engineering student. Knowledge in these subjects can help students gain a deeper understanding of the concepts in other courses including operating systems, algorithms, programming and many more. But the steep learning curve and large time investment associated with hardware design and development limits the scope and depth of hands-on laboratory exercises of these courses. We believe that the best way to maximize student engagement and educational value of Computer Organization and Computer Architecture classes is via a practical and hands-on approach. The BRISC-V Platform delivers this practical teaching approach for computer architecture by providing an open-source single and multi-core design space exploration platform that eliminates much of the overhead associated with developing a complete processing system.

计算机组成和计算机架构这样的课程，对于计算机工程学生是基础知识。这些主题的知识，可以帮助学生更深的理解在其他课程中的概念，包括操作系统，算法，编程等等。但学习曲线很陡峭，硬件设计和开发要投入很多的时间，限制了这些课程动手试验练习的范围和深度。我们相信，学生参与的最大化，和计算机组成和计算机架构课程的教育价值的最大化，其最佳方式是通过实际的动手方法。BRISC-V平台给出了这个计算机架构的实践教学方法，提供了开源的单核和多核设计空间探索平台，而没有开发完整处理系统的大部分开销。

The BRISC-V Design Space Exploration Platform [1] provides many opportunities for a hands on computer architecture education. The platform consists of (1) a RISC-V simulator to test software independently of any hardware system, (2) a RISC-V toolchain to compile a user’s code for bare-metal execution, (3) a modular, parameterized, synthesizable multi-core RISC-V hardware system written in Verilog, and (4) a hardware system configuration Graphical User Interface (GUI) to visualize and generate single or multi-core hardware systems.

BRISC-V设计空间探索平台为计算机架构动手教育提供了很多机会。这个平台由以下几部分组成，(1)一个RISC-V仿真器，独立于任何硬件系统来测试软件，(2)RISC-V工具链，编译用户的代码用作bare-metal执行；(3)一种模块化的，参数化的，可综合的多核RISC-V硬件系统，用Verilog编写；(4)一种硬件系统配置GUI，可以生成单核或多核硬件系统，并可视化。

In this paper, we describe how the BRISC-V Design Space Exploration Platform can be used to teach an undergraduate level Computer Organization class, a graduate level Computer Architecture class, and a research focused graduate level Hardware Systems Security class. Programming and Assembly labs are supported with a browser based tool named the BRISC-V Simulator for writing, compiling, assembling and executing RISC-V code. Students can use the platform-independent simulator to get started with RISC-V software quickly and easily. The BRISC-V Simulator provides a valuable resource when teaching students about low level concepts, including calling conventions, memory allocation, and the compilation flow.

本文中，我们描述BRISC-V设计空间探索平台怎样用于教授本科级别的计算机组成课程，研究生级别的计算机架构课程，和专注于研究级别的硬件系统安全课程。通过一个基于浏览器的工具，称为BRISC-V仿真器，用于编写，编译和执行RISC-V代码。学生可以使用与平台独立的仿真器，快速容易的开始RISC-V软件。在教授学生低级概念时，BRISC-V仿真器提供了宝贵的资源，包括调用习惯，内存分配和编译流。

The resources provided by the BRISC-V Platform streamline RTL based laboratory exercises by providing functional and tested starting points for hardware systems. The modular and parameterized RISC-V hardware system included with the BRISC-V Platform provides a useful template for students building their first processor. The modular nature of the hardware system allows assignments to be crafted for each stage of the processor: fetch, decode, execute, memory and write-back. More experienced students in a Computer Architecture class can use the hardware system configuration GUI named BRISC-V Explorer to configure baseline systems. Then, students can add micro-architectural features to their highly modular configured hardware system. For example, students can configure a single cycle, single core processor to add pipeline registers to it. Students looking to experiment with more advanced micro-architectural features can configure a complex, multi-core processor. The cache architecture is a more complex feature and we have designed a few assignments around it. Selecting a single or multi-level cache in the BRISC-V Explorer enables students to experiment with different cache size, associativity, replacement policy or custom cache architectures.

BRISC-V平台提供的资源，使基于RTL的实验室试验效率更高，对硬件系统提供了功能和测试的开始点。BRISC-V平台包含了模块化和参数化的RISC-V硬件系统，为学生构建其第一个处理器提供了一个有用的模板。硬件系统模块化的本质，使得可以对处理器的每个阶段布置作业：取指，解码，执行，存储和写回。计算机架构课程上，更有经验的学生可以使用硬件系统配置GUI，名为BRISC-V浏览器，来配置基准系统。然后，学生可以对其高度模块化配置的硬件系统加入微架构特征。比如，学生可以配置一个单周期，单核处理器，对其加入流水线寄存器。想要进行更高级的微架构特征试验的学生，可以配置复杂的多核处理器。缓存架构是更复杂的特征，我们围绕这个设计了几个作业。在BRISC-V浏览器中选择单级cache或多级cache，使学生可以用不同的cache大小，连接性，替换策略或定制的缓存架构进行试验。

The complete BRISC-V Design Space Exploration Platform (including Verilog source code) is open-source and can be downloaded at https://ascslab.org/research/briscv/index.html. 完整的BRISC-V设计空间探索平台是开源的。

## 2. Course Overview

In this section, we illustrate the organization of the (1) Computer Organization and (2) Computer Architecture courses.

本节中，我们描述了计算机组成和计算机架构课程的组织。

The Computer Organization class aims to familiarize students with low-level coding in C and assembly, and provide a high-level view of a processor. The students have no prerequisite programming skills, and are expected to learn the basics of the C programming language and RISC-V assembly. Students will build on their experience with Verilog from the prerequisite Digital Design course.

计算机组成课的目标是让学生熟悉C和汇编的底层代码，给出处理器的高层视角。学生不需要有任何编程技能，可以学到C编程语言和RISC-V汇编的基础。学生会从前置课程数字设计中构建其对Verilog的经验。

They are expected to complete several exercises in C, writing simple programs using functions, recursions, floating point and bitwise operations. These exercises also explore how each datatype they use is actually stored in memory. Next, they analyze how these C programs are compiled to assembly, and learn how to write their own functions and recursions. Finally, as a class project, students are tasked with writing a single-cycle CPU supporting the RV32I instruction set. For both the C and assembly labs, the students make heavy use of the BRISC-V Simulator, presented in Section 3. The simulator allows them to compile C to RISC-V assembly, handwrite assembly, test it, debug it, and view the state of the registers and memory at every instruction of the program. For the single-cycle CPU design project, the teaching assistants use the BRISC-V Explorer (presented in Section 4) to quickly generate a bare-bone single-cycle CPU, and remove all functionality from it while leaving the modules as a project skeleton.

他们要完成几个用C写的练习例子，用函数，递归，浮点和逐位运算写出简单的程序。这些练习还探索了他们使用的每种数据类型是怎样在内存中存储的。下一步，他们分析了这些C程序是怎样编译、汇编的，学习怎样写自己的函数和递归。最后，作为课程项目，学生要写一个单周期CPU，支持RV32I指令集。对于C和汇编实验室，学生要使用BRISC-V仿真器。仿真器可以将C编译到RISC-V汇编，手写汇编，测试，debug，在程序的每条指令处查看寄存器和内存的状态。对于单周期CPU设计项目，TA使用BRISC-V浏览器来迅速生成一个bare-bone单周期CPU，去掉所有功能，将这些模块来作为一个项目骨架。

By the end of the course, the students will have a demystified view of hardware, as they have both programmed a bare-metal CPU and created a simple but fully-synthesizable processor. Even if this is the last hardware course the student may take, they will have a solid footing when exploring topics such as writing high-performance code, using optimizing compilers, or diving deeper into computer architecture.

在课程最后，学生会理解硬件，因为他们对一个bare-metal CPU进行了编程，创建了一个简单但完全可综合的处理器。虽然这是学生接受的最后的硬件课程，但他们已经有了很坚实的基础，尤其是在写高性能代码，使用优化的编译器，或探索更深入的计算机架构。

The second course we describe is Computer Architecture, with Computer Organization as a prerequisite. In this course, students will gain an understanding of a “modern” processor, as concepts such as pipelining, caching, inter-core communication, multiple-issue and out-of-order processors are introduced. The students are expected to be familiar with bare-metal C code, assembly, and Verilog. In order to save time, the students are given a fully-functional and tested single-cycle processor, along with testbenches and assembly programs. The labs are structured in such a way that students can run code on their processors in the very first lab, and the labs only explore modifications to this processor. They are expected to complete several labs focused on the hardware implementation of (1) a 7-stage pipelined processor based on the single-cycle processor, (2) a simple L1 cache and an optimized cache hierarchy, (3) a multi-core processor and (4) an advanced micro-architectural feature covered in the class lectures. In the fourth laboratory exercise, students explore the effectiveness of hardware modifications such as multiple-issue processors by analyzing software binaries using a binary analysis tool such as Intel PIN [4].

我们描述的第二个课程是计算机架构，其前置课程为计算机组成。在本课程中，学生会对现代处理器有所理解，包括介绍流水线，缓存，核间通信，多发射和乱序处理器这样的概念。学生要熟悉bare-metal C代码，汇编和Verilog。为节省时间，学生会给定一个完整功能的测试好的单周期处理器，以及testbenches和汇编程序。试验的组织是这样的，学生可以在第一个实验室中的处理器上运行代码，实验室只对这个处理器进行修正。他们要完成几个实验室，聚焦在下面的硬件实现，(1)一个7级流水处理器，基于单周期处理器；(2)简单的L1缓存，和优化的缓存层次结构；(3)多核处理器；(4)课程中覆盖的高级微架构特征。在第四个实验室练习中，学生探索硬件修正的有效性，比如多发射处理器，使用二进制分析工具如Intel PIN来分析软件二进制。

In this course, the students use the BRISC-V Simulator to write bare-metal code which can run on students’ processors. In the lab exploring caches, students are asked to use the BRISC-V Explorer to find a cache configuration that provides the highest performance on a given task. By the end of the course, students should have a good grasp of major concepts in computer architecture. While this is an architecture class, the students should also walk away with actionable knowledge in writing software, and be able to answer questions such as “why can two algorithms with the same computational complexity have an order-of-magnitude difference in performance”, or “how might the on-chip network topology affect the performance of multi-threaded algorithms”.

在这个课程中，学生使用BRISC-V仿真器来写bare-metal代码，可以在学生的处理器上运行。在实验室探索的缓存中，学生要使用BRISC-V浏览器来找到缓存的配置，对给定的任务得到最高的性能。在课程最后，学生要很好的掌握计算机架构中的主要概念。这是一个架构课程，学生也要熟悉写软件的知识，可以回答一些问题，比如，为什么两个算法相同的计算复杂度但性能差异却是数量级的，或者，片上网络拓扑怎样影响多线程算法的性能。

## 3. The BRISC-V Simulator

The BRISC-V Simulator is a RISC-V simulator targeting the RV32I feature set. It is a single-page web application written in JavaScript, that allows the user to (1) compile C to RISC-V assembly, (2) run or step through assembly, and (3) analyze the state of the processor at every instruction. A web-based implementation provides users with flexibility in terms of running the simulator. There are three ways users may run this application: the user may run the application locally, which requires no installation, but does not come with a compiler, (2) a student or teaching assistant may host the website, which requires installing several python libraries, and (3) the user may access the public version of the simulator from our website which is located at https://ascslab.org/research/briscv/simulator/simulator.html. The simulator executes the code on the client machine. The only computation that happens on the server hosting the simulator is (1) distributing the static website, and (2) the optional compiler support, which allows users to compile their C code from the browser. In large classes, this is advantageous because student machines often may not have privileges to install and run the compiler.

BRISC-V仿真器是一个RISC-V仿真器，面向RV32I特征集。这是一个用JavaScript写的单页面网页应用，用户可以(1)将C编译成RISC-V汇编，(2)运行汇编程序或单步执行，(3)在每条指令处分析处理器的状态。基于网页的实现可以使用户在运行模拟器时很灵活。有三种方式用户会运行这个应用：用户可以在本地运行这个应用，不需要安装，但并不具有编译器的功能，(2)学生或TA会成为网站的宿主，这需要安装几个Python库，(3)用户可以访问仿真器的公开版本，在我们的网站上。仿真器在客户机器上执行代码。在服务器上进行的计算是，(1)静态网页的分发，(2)可选的编译器支持，使用户可以从浏览器中编译C代码。在大的课程中，这是有优势的，因为学生的机器可能不会有特权来安装并运行编译器。

### 3.1 Using The BRISC-V Simulator

Figure 2 shows a screen-shot of the simulator GUI. The page is split into three main columns: the compiler pane 1 on the left, the assembly pane 2 in the center, and the register and memory pane 3 on the right.

图2给出了仿真器GUI的截图。页面分成三个主要的列；左边的编译器面板1，中央的汇编面板2，右边的寄存器和内存面板3。

**Compilation**: in the compiler pane, the user is free to write C code. The user can load C code from the filesystem and compile it using the compiler pane buttons 4. The compiled assembly will be written to the central assembly pane 2, and any standard output from the compiler will be written to the console 5.

编译：在编译器面板中，用户可以自由编写C代码。用户可以从文件系统中载入C代码，用编译器面板上的按钮4来编译。编译的汇编程序会写入到中间的汇编面板2，编译器的任何标准输出，都会写入到主控台5中。

**Executing assembly**: to add the RISC-V assembly to the assembly pane, the user has three options: compile C code, load assembly code from the file system, or load example code. Above the assembly pane are four buttons 6 : load assembly, run, single instruction step, and reset simulator. The load button allows the user to either load their own assembly, or select one of the three example files provided. After loading the assembly program, the simulator wraps the assembly with our kernel code, which consists of both a program prologue and epilogue. The program prologue is tasked with initializing the registers, as well the stack pointer, and the program epilogue traps the simulator in an infinite loop upon program termination. The kernel code has a grey background in the assembly pane to distinguish it from the user code. If the user chooses to load their own assembly, any errors and the result of the parser will be shown in the console 5 . A successful message is presented to the user if no errors are found in the assembly program. However, if at least one error is found, the console window will enumerate each error and its corresponding line number; the last line of the error output will display the total number of errors found. Simulation is also disabled and the simulation buttons will be greyed out as a visual cue. The run button will run the code until the code (1) hits an exit syscall, (2) hits the kernel while loop, (3) hits a breakpoint, or (4) gets trapped in an infinite loop for more than 100000 instructions. The single instruction step button runs only a single instruction, and ignores any breakpoints. The reset button moves the instruction pointer back to the start of the program, and resets the state of the registers and memory.

执行汇编：为将RISC-V汇编加入到汇编面板上，用户有三个选项：编译C代码，从文件系统中载入汇编代码，或载入例子代码。汇编面板上是4个按钮：载入汇编，运行，单指令步骤，或选择给出的三个例子文件中的一个。在载入汇编程序后，仿真器将汇编代码用我们的核代码包裹起来，由程序prologue和epilogue组成。程序prologue的任务是初始化寄存器，以及栈指针，程序epilogue的作用是在程序结束的时候将仿真器陷入到一个无限循环中。核代码在汇编面板上是灰色背景的，以与用户代码区分开来。如果用户选择载入其自己的汇编代码，解析器的任何错误和结果都在console 5上显示。如果在汇编程序中没有找到任何错误，就会向用户展示出成功的信息。但是，如果找到了至少一个错误，console窗口就会把每个错误枚举出来，及其对应的行号；最后一行错误输出，会展示发现的错误的总数量。仿真会禁用掉，按钮会展示灰色。运行按钮会运行代码，直到代码(1)遇到退出的系统调用，(2)在循环的时候遇到了核，(3)遇到了断点，(4)在无限循环中陷入到多于100000条指令中。单指令步骤按钮只运行一条指令，忽视任何断点。复位按钮将指令指针移回到程序的开始，并将寄存器和内存的状态复位。

**Observing the state**: after every single instruction, the user can monitor the state of the registers and memory in the register and memory panes 3 . If an instruction updates the value of a register, that register will be highlighted in red, as shown in Figure 2. To the right of the register pane is the memory pane. It is represented as a descending list with five foldable regions: the stack segment, free segment, heap segment, data segment, and the text segment. Each line of the memory pane represents one word (32 bits). The address is shown on the left in hexadecimal format (light blue), and the value is shown on the right in either hexadecimal, decimal, or binary. The format buttons 7 allow the user to show the memory in their preferred format. Additionally, the breakdown pane 8 lets the user to view the breakdown of the instruction at the current IP.

观察状态：在每条指令后，用户可以在寄存器和内存面板3上，监控寄存器和内存的状态。如果一条指令更新了寄存器的值，寄存器会高亮为红色，如图2所示。寄存器面板的右边是内存面板。表示为下降的列表，有5个可折叠区域：栈，自由段，堆，数据段，和text段。内存面板的每一行表示一个word (32 bits)。地址在左边以16进制格式显示（浅蓝色），值在右边以16进制，10进制或2进制显示。格式按钮7允许用户以喜欢的格式来显示内存。另外，分解面板8可以让用户在当前IP处观察指令的分解。

**Additional Features**: the user may right-click on an instruction, which will open a context menu 9 . If the user selects the “Add breakpoint” option, a pink line will cover that instruction 10 . If the user moves the mouse above any labels, they will see an option to “fold” that region of code. Also, all of the panes are resizable and movable - the user can hide e.g., the memory or the compilation pane, or may stack the console and the instruction breakdown.

其他特征：用户可以右击一条指令，会打开上下文菜单9。如果用户选择了增加断点的选项，会有一条粉色线覆盖那条指令10。如果用户将鼠标移动到任何标签上，会有一个选项，将这个区域的代码折叠起来。同时，所有面板都是可变大小的，可移动的，用户可以隐藏比如内存或编译面板，或可以堆叠console和指令分解。

### 3.2 System Call Support

The BRISC-V Simulator implements support for seven system calls, as shown in Table 1. To call a system call (syscall), the program needs to set the appropriate syscall ID value in the t0 register. Next, any parameters to the syscall should be placed in registers a0 and a1. Finally, the program runs the SCALL or ECALL pseudo-instruction. If the system call returns any value, it will be stored in registers a0 and a1. One of the example assembly files (syscalls.s) provided by the simulator illustrates how the system calls are used.

BRISC-V仿真器实现了7个系统调用的支持，如表1所示。为调用一个系统调用(syscall)，程序需要设置合适的syscall ID值到t0寄存器。下一步，syscall的任何参数都应当放到寄存器a0和a1中。最后，程序运行SCALL或ECALL伪指令。如果系统返回了任何值，会存储到寄存器a0和a1中。仿真器给出了一个例子汇编文件syscalls.s，描述了系统调用怎样使用。

Table 1: System calls supported by the BRISC-V Simulator

Syscall Type | t0 | Description
--- | --- | ---
Print Integer | 1 | Print integer stored in a0
Print Character | 2 | Prints ASCII character stored in a0
Print String | 3 | Prints string stored at address in a0 with length stored in a1
Read Integer | 4 | Reads integer and stores it in a0
Read Character | 5 | Reads an ASCII character and stores it in a0
Read String | 6 | Reads a null-terminated string and stores it at address in a0 with length in a1
Stop Execution | 7 | Stop the program

### 3.3 BRISC-V Simulator Extensibility

Through the BRISC-V Simulator, we aim to provide students with a simple ‘hackable’ tool that they can use in computer organization classes to gain familiarity in assembly and to confirm that their compiled code behaves as expected. In computer architecture classes, the simulator can be used to explore existing ISA extensions (i.e. floating point or vector instructions). Likewise, in a hardware security class, new security specific ISA extensions can be quickly added and tested.

通过BRISC-V仿真器，我们的目标是为学生提供一个简单的hackable工具，可以在计算机组成课程中使用，熟悉汇编代码，确认编译的代码与期望行为一致。在计算机架构课程中，仿真器可以用于探索现有的ISA扩展（如，浮点或向量指令）。类似的，在硬件安全课程中，新的安全特定的ISA扩展可以迅速增加和测试。

As an example, to implement the multiplication operations MUL, MULH, MULHU, MULHSU defined in the RISC-V specification [5], one needs to:

比如，为实现RISC-V spec中定义的乘法运算MUL, MULH, MULHU, MULHSU，需要：

(1) Edit the parser.js file of the BRISC-V Simulator so that these instructions are parsed with the two source registers and a destination register. This requires minimal coding as the registers are already extracted by the parser.

编辑BRISC-V仿真中的parser.js文件，指令用两个源寄存器和一个目的寄存器解析。这需要很少的代码，因为寄存器已经从解析器中提取出来了。

(2) Add a new case condition in the emulator.js file so that MUL* instructions (1) update the IP by 4, and (2) store the correct multiplication result in the correct register. The emulator already has the registers as local variables, so the user just needs to refer to the appropriate ones.

在emulator.js中增加新的情况，这样MUL*指令(1)将IP以4为单位更新，(2)将正确的乘法结果存储于正确的寄存器中。Emulator已经有寄存器作为本地变量，所以用户只需要参考合适的就可以了。

## 4. The BRISC-V Explorer

The BRISC-V Design Space Exploration Platform provides a suite of tools to quickly develop single and multi-core RISC-V processors. The BRISC-V Explorer GUI is used to configure the hardware components of a user’s system. Users can select core, cache, main memory and NoC configuration parameters. Figures 3 and 8 show two views of the BRISC-V Explorer. The Explorer contains several panes with different configuration settings and other information. Figure 3 shows the core configuration settings pane 1 with the downloads pane 2 , console pane 3 and block diagram of the entire configured hardware system 4 . In the core configuration settings pane, users can select between single-cycle, five stage pipelined and seven stage pipelined cores. Pipelined cores can be configured with or without data forwarding logic. The number of cores in the system is can also be selected by the user, and ranges from 1 to 8. The console pane outputs information about invalid configurations and the status of exporting a project. The block diagram pane shows the hardware system that will be exported by clicking download in the download pane. Figure 8 shows the core configuration settings pane, memory hierarchy configuration settings pane and block diagram of the memory hierarchy. Users can select various cache configuration settings in the memory hierarchy pane. Tunable settings include cache associativity, line size, number of lines and depth of the cache hierarchy. Panes in the BRISC-V Explorer GUI can be moved and resized to easily configure settings relevant to the current user.

BRISC-V设计空间探索平台给出了一个工具包，可以迅速的开发单核和多核RISC-V处理器。BRISC-V浏览器GUI用于配置用户系统的硬件组成部分。用户可以选择核，缓存，主存和NoC的配置参数。图3和8展示了BRISC-V浏览器的两个视角。浏览器包含几个面板，有不同的配置设置和其他信息。图3展示了核的配置设置面板1，和下载面板2，console面板3，和配置硬件系统的整体模块图4。在核配置设置面板中，用户可以选择单周期，5级流水线和7级流水线核。流水线核可以配置或不配置数据前向逻辑。用户还可以选择系统中核的数量，范围从1到8。console面板输出的信息是关于无效配置的，和到处一个项目的状态。模块图面板展示的是点击下载到处的硬件系统。图8展示的是核的配置设置面板，内存层次结构配置设置面板，和内存层次结构的模块图。用户可以在内存层次结构面板中选择各种缓存配置设置。可调节的设置包括，缓存连接性，cacheline大小，cacheline个数，和cache层次结构的深度。BRISC-V浏览器GUI中的面板是可以移动的，可以调节大小，很容易的配置与当前用户相关的设置。

After a configuration has been chosen, clicking the "Download Project" button in the downloads pane will output a highly modular Verilog implementation of the system. These advanced multi-core configuration features provide a rich design space for advanced computer architecture classes to explore. Students in these classes can configure a single or multi-core system and add features such as branch prediction, out-of-order execution or more experimental features tied to graduate student research. For students in their first computer organization class, the BRISC-V Explorer can generate a single cycle RV32I core without caches, utilizing a simple dual ported main memory. This simple processor serves as a template for exercises in which students are asked to build their first processor.

在选择了配置后，在下载面板中点击下载工程按钮，会输出系统高度模块化的Verilog实现。这些高级多核配置特征对高级计算机架构课程提供了丰富的设计空间来探索。这些课程中的学生可以配置单核或多核系统，并加入特征，如分支预测，乱序执行，或更多的试验特征，进行研究生的研究。对于第一次计算机组成课的学生，BRISC-V浏览器可以生成单周期RV32I核，不带缓存，利用简单的双端口主存。这种简单的处理器可以作为模板进行练习，学生可以构建其第一个处理器。

## 5. Computer Organization Class

In this section we give an overview of the Computer Organization class, how the BRISC-V Simulator (Section 3) tool we developed is used in it, and how the BRISC-V Explorer (Section 4) is used for generating a template for a single-cycle processor.

本节中，我们概览一下计算机组成的课程，我们开发的BRISC-V仿真器工具是怎样在其中使用的，以及BRISC-V浏览器工具是怎样用于生成单周期处理器模板的。

As seen in Figure 1, the Computer Organization class covers concepts both ‘above’ and ‘below’ the instruction set architecture (ISA). Each class has seven laboratory exercises with an eighth exercise serving as the final project. Two of the exercises are dedicated to C, two to assembly, and the rest are dedicated to implementing a single-cycle processor in Verilog. This class project is further explained in Section 5.3.

如图1所示，计算机组成课程覆盖了指令集架构ISA的上面和下面。每个课程有7个试验练习，第8个练习是最终项目。两个练习是用C写的，两个是汇编，剩下的4个是用Verilog实现一个单周期处理器。这个课程项目在5.3中进一步解释。

### 5.1 C/C++ Exercises

The goal of the C/C++ exercises is to provide a smooth transition into writing RISC-V assembly for students who have no prior experience with assembly language programming. Additionally, students can often be confused by concepts such as pointers or interfacing with the operating system. By providing a machine-level view of these concepts, the C/C++ exercises should clear up any misconceptions they may have.

C/C++练习的目标是为了让之前没有写汇编语言程序的学生，平滑的过度到写RISC-V汇编。另外，学生通常会被指针，或与操作系统的接口的概念弄混淆。通过对这个概念给出机器级的视角，C/C++练习会清除掉他们有的任何错误概念。

**Exercise 1**: In the first exercise, the students should become familiar with common formats such as signed and unsigned integers, conversion between binary, decimal, and hexadecimal formats, using arrays and pointers, and writing recursive functions. For extra points, students are asked to convert a floating point value to its binary representation (without simply using C’s union data type), and the reverse. To complete this exercise, the students can either write code in an editor, and compile and run it from the command line, or they can use the built-in compiler in the BRISC-V Simulator.

练习1：在第一个练习中，学生应当熟悉有符号和无符号整数的常见格式，二进制，十进制和十六进制之间的转换，使用数组和指针，写递归函数。还有一些额外的题目，如学生要将浮点值转换成其二值表示，以及反向转换。为完成这个练习，学生可以在编辑器中写出代码，在命令行中编译和运行，或可以使用BRISC-V仿真器中的内建编译器。

**Exercise 2**: Since the system the students will create can only run bare-metal code, and has no kernel running on top of it, dynamically allocating memory is not possible out-of-the-box. The goal of this exercise is to demystify the workings of heap memory and modern malloc implementations. In this exercise, the students are asked to write a simple library for dynamic memory allocation. The library statically allocates some amount of memory at the start of the program, and provides functions malloc, which takes a size in bytes and returns a pointer to the first contiguous piece of free memory, as well as free, which takes a pointer to the previously allocated block of memory and frees it in the library’s internal data structures.

练习2：由于学生创建的系统只能运行bare-metal代码，没有运行任何内核，所以并不能动态分配内存。本练习的目标，是弄清楚堆内存和现代malloc实现的工作原理。在本练习中，学生要写一个简单的库，进行动态内存分配。这个库在程序开始时静态的分配一些内存，然后给出函数malloc，输入为以bytes为单位的大小，返回第一块该大小连续内存的指针，以及函数free，输入为之前分配的内存块的指针，并以库中的内部数据结构对其进行释放。

### 5.2 Assembly Exercises

In the assembly exercises, the students gain an in-depth understanding of the RISC-V ISA and the inner workings of a processor. At the time of the creation of these exercises, no suitable tool was available that would allow students to execute RISC-V assembly instruction-by-instruction and monitor the state of the registers and memory. Thus, we have created the BRISC-V Simulator (Section 3). For exercises 3 and 4, the students are supposed to load their assembly programs into the simulator and execute them to confirm their correctness or to find bugs. The simulator provides them with the needed tools: the registers and memory are visible right next to the assembly instructions, and as instructions are executed, the updated values are highlighted. Additionally, the simulator allows setting breakpoints, simplifying debugging.

在汇编的练习中，学生会对RISC-V ISA和处理器的内部工作机制有深度的理解。在创建这些练习的时候，并没有合适的工具，能让学生逐条执行RISC-V汇编程序，并监控寄存器和内存的状态。因此，我们创建了BRISC-V仿真器。在练习3和练习4中，学生要将汇编程序载入到仿真器中，执行，以确认其正确性，或找到bugs。仿真器提供了需要的工具：寄存器和内存是可见的，就在汇编指令的右边，随着指令的执行，更新的值会高亮。另外，仿真器允许设置断点，以简化debug的过程。

**Exercise 3**: In this exercise, the students should explore how C code is compiled to assembly, how functions and the stack work, and become familiar with the RISC-V calling convention. To become familiar without having to write assembly right away, we first task students with analyzing how C code is converted to assembly. Here the students are provided with a C program, and are required to (1) compile it in the BRISC-V Simulator’s compiler, and match each line of the C with a sequence of instructions in the assembler. For the second task, we explain the limitations of the RV32I ISA, namely the lack of a multiply instruction. The students are tasked with writing a multiply label in assembly and using it to perform simple calculations. Finally, we expose the students to the RISC-V calling convention, and task them with writing a non-recursive factorial function. This exposes students to basic stack concepts, and how parameters are passed and returned to and from functions.

练习3：在这个练习中，学生要探索C代码是怎样编译成汇编的，函数和堆栈是怎样工作的，并熟悉RISC-V的调用规范。为熟悉这些内容，而不用马上写汇编，我们首先给学生任务，分析一下C代码是怎样转换成汇编的。这里要给学生一段C代码，并要求学生(1)在BRISC-V仿真器的编译器中进行编译，并将每行C代码与一系列汇编指令匹配起来。第二个任务，我们解释RV32I ISA的局限，即缺少乘法指令。学生的任务是，用汇编写一个乘法的标签，用其进行简单的计算。最后，我们让学生接触RISC-V调用规范，给学生一个任务，写一个非递归分解函数。这让学生了解基础堆栈的概念，参数怎样传递和返回到函数。

**Exercise 4**: By this exercise, the students should be familiar with the majority of RISC-V instructions and concepts, and are ready to write more complex programs. First, the students are tasked with writing both non-recursive and recursive versions of a Fibonacci function. Next, they are asked to write a matrix multiplication function, so that they will have to write nested loops. Finally, we expose them to a set of simple system calls built into the BRISC-V Simulator. These system calls allow the students to read and write integers, characters and strings to and from the console, as well as end the program. A list of all system calls can be found in Table 1. This exercise concludes the assembly exercises of the class, and the students can then dig down into the processor microarchitecture.

练习4：在这个练习中，学生要熟悉主要的RISC-V指令和概念，并可以写出更复杂的程序。第一，学生接受一个任务，写出Fibonacci函数的非递归版和递归版。下一步，他们要写出一个矩阵相乘的函数，这样他们需要写出嵌套的循环。最后，我们让其了解一些简单的系统调用，这是在BRISC-V仿真器中内建的。这些系统调用使学生可以读和写整数，字符和字符串，以及终止程序。表1可以看到所有的系统调用。这个练习总结了课程的汇编练习，学生可以继续深入学习处理器的微架构。

### 5.3 Class Project: Building Your First Processor

The hands on experience of building a processor is an indispensable part of any computer organization class. After students have been introduced to the RISC-V ISA, assembly programming, and other fundamental computer organization concepts they can begin to implement their own CPU. A simple single cycle core generated with the BRISC-V Explorer serves as a template for the computer organization class exercises. The modular nature of the single cycle core allows it to be broken up into discrete exercises, guiding students through the process of building a CPU.

上手构建一个处理器的经验，是任何计算机组成课程不可少的部分。学生熟悉了RISC-V ISA，汇编语言和其他基本计算机组成概念后，他们就可以开始实现自己的CPU了。用BRISC-V浏览器生成的单周期核，可以作为计算机组成课练习的模板。单周期核模块化的本质，使其可以分成离散的练习，引导学生完成构建CPU的过程。

The single-cycle core contains separate modules for the fetch, decode, execute, memory and write-back stages. In each exercise, students will build a new stage of the processor. Students are given an interface specification for each module and an empty module template with a port list. Breaking each assignment into discrete modules helps reinforce modular design practices necessary for the complex RTL designs. Providing the template and port list also ensures the modules can be graded with a single test bench. Figure 4 shows how the single cycle core is broken up into modules and how individual exercises are created.

单周期核包含取指，解码，执行，访存，写回阶段。在每个练习中，学生要构建处理器的一个新阶段。学生收到每个模块的接口规范，和一个空的模块模板，带有端口列表。将每个赋值分解到离散的模块中，帮助强化模块化设计的任务，是复杂RTL设计所必须的。给定模板和端口列表，也确保了模块可以用单一的testbench进行打分。图4展示了单周期核怎样分成模块，怎样创建单个的练习。

**Exercise 5**: In this exercise, the students implement an arithemetic logic unit (ALU) for the RISC-V RV32I instruction set. The ISA does not explicitly give an encoding for the ALU control signals, so these are given to students in the exercise description. To keep the first few exercises manageable, students are asked to implement only a subset of the RV32I instructions. Students start by implementing the simple arithmetic and bit-wise logic instructions. Incorporating the remaining instructions is left for future exercises when students have a better understanding of their processor design.

练习5：在这个练习中，学生要实现一个RISC-V RV32I指令集的ALU。ISA并不会显式的给出ALU控制信号的编码，所以这会在练习描述中给学生。为使前几个练习可管理，学生要实现一部分RV32I指令。学生开始的时候实现简单的代数和逐位逻辑指令。加入剩下的指令留作未来的练习，学生要对其处理器设计有更好的理解才可以。

**Exercise 6**: This exercise covers the decode stage of the core. Students build the decode and control logic, as well as the register file. The students are only required to implement decode logic for the same subset of instructions used in the first exercise.

练习6：这个练习覆盖了核的解码阶段。学生构建解码和控制逻辑，以及寄存器组。学生只需要实现第一个练习中使用的指令子集的解码逻辑。

After the decode, register file and ALU modules are complete, students must demonstrate (in simulation) the execution of a simple instruction stream, feeding instructions into the decode module. The decoded instructions are sent to the ALU which computes a result. The ALU output is directly written back to the register file. Figure 5 shows the block diagram used in this exercise.

在解码，寄存器组和ALU模块完成后，学生要在仿真中展示简单指令流的执行，将指令送入到解码模块。解码的指令送入到ALU中，计算得到一个结果。ALU的输出直接写回到寄存器组中。图5展示了这个练习中使用的模块图。

**Exercise 7**: In this exercise, students create the main memory module and the main memory stage of the core. Students must add load and store instructions to their decode module. To support the new load and store instructions, students must add the writeback stage to the processor as well. The writeback stage consists of a single multiplexer to select between the data memory output and the ALU output. The selected value is sent to the register file.

练习7：在这个练习中，学生创建了主存模块，以及核的主要访存阶段。学生需要在解码模块中加入load和store指令。为支持新的load和store指令，学生还要给处理器加入新的写回阶段。写回阶段由一个简单的multiplexer构成，在数据内存输出和ALU输出之间选择。选择的值送到寄存器组中。

The new main memory, memory stage and writeback stage modules must be added to the previous demonstration, so students can demonstrate the execution of a more complex instruction stream. Figure 6 shows a block diagram with the additional modules used for this exercise.

新的主存，访存阶段和写回阶段模块，要加入到前一个展示中，所以学生可以展示更复杂的指令流的执行。图6展示了这个练习所使用的带有额外模块的模块图。

**Exercise 8**: The fetch stage of the single cycle core is relatively simple and does not warrant an entire exercise on its own. Instead, the fetch stage development is combined with the final integration stage of the processor design in the eighth exercise. In addition to completing the fetch module, students add the remaining RV32I instructions not implemented in the previous assignments and verify that their processor can correctly execute a program. This exercise includes the addition of branch and jump instructions to the processor. This integration stage is given as a final project for the computer organization class instead of as a shorter laboratory exercise.

练习8：单周期核的取指阶段是相对简单的，并不能够成为一整个练习。取指阶段的开发与处理器最后的整合阶段结合到一起，成为第8个练习。除了完成取指模块，学生将在之前的作业中没有实现的RV32I指令加入进来，验证处理器可以正确的执行一个程序。这个练习包括，加入分支和跳转指令到处理器中。整合阶段是计算机组成课程的最终项目。

Students are expected to read and understand the relevant sections of the RISC-V Instruction Set Manual [5] to correctly implement instructions including AUIPC (add upper immediate to PC), BEQ (branch if equal) and JALR (jump and link register). These instructions take inputs from or output to the fetch stage, making it easier to incorporate them after students have a semi-complete processor.

学生要阅读理解RISC-V指令集手册的相关章节，以正确的实现包括AUIPC，BEQ，JALR之类的指令。这些指令以取指阶段为输入，或输出到取指阶段，在学生有了半完成的处理器时，更容易将其整合。

Students are encouraged to build the fetch and write-back stages first and integrate the modules for their subset of instructions. Then they can expand their existing modules to include the remaining instructions, testing their additions along the way. Although students performed some integration tests in the previous exercises,inevitably more issues will be discovered as instructions are added and more complete testing is performed. Students are given simple test programs to demonstrate on their processor and must develop additional test programs to provide fuller test coverage. This final integration exercise forces students to devote attention to the correctness of the processor as a whole. Any issues found during integration highlight the challenges of integrating modules in a large project and the need for thorough tests for any RTL design.

学生最好首先构建取指和写回阶段，对指令子集整合这些模块。然后可以扩展现有的模块，以包含剩余的指令，测试加入的内容。虽然学生在之前的练习中进行了一些整合测试，随着指令的加入，进行更完整的测试，不可避免的会发现更多的问题。回给学生一些简单的测试程序，在其处理器上展示，必须开发额外的测试程序来给出更完整的测试覆盖。这个最后的整合练习，迫使学生将注意力放到处理器作为一个整体的正确性上。在整合中发现的任何问题，都会强调将模块整合到一个大型项目中的挑战，和对任何RTL设计进行彻底测试的需要。

Students demonstrate the correct execution of their custom test programs and the provided ones in a simulation. After completing the final exercise, students have completed a functional RISC-V processor compatible with the RISC-V toolchain included with the BRISC-V Design Space Exploration Platform. Optionally, more ambitious students may want to synthesize their design. As the BRISC-V template has strongly encouraged the use of structural Verilog and has provided the students with asynchronous BRAM used for storing instructions and data, we have witnessed few challenges to running the final designs on an FPGA.

学生在仿真中展示其定制测试程序和给定的程序的正确执行。在完成最终练习时，学生完成了一个RISC-V处理器，与BRISC-V设计空间探索平台所包含的RISC-V工具链兼容。想做更多事的学生可能会想综合其设计。因为BRISC-V模板强烈鼓励结构化Verilog的使用，给学生提供了异步BRAM来存储指令和数据，要将最终的设计在FPGA上运行，中间遇到的问题可能会很少。

## 6. Computer Architecture Class

The Computer Architecture class builds on the Computer Organization class to introduce more advanced micro-architecture features, including pipelining, branch prediction, and out-of-order execution. This class is designed for undergraduate seniors and graduate students and consists of four in-depth exercises. In the exercises, students will (1) pipeline a processor, (2) build a cache and optimize a cache hierarchy, (3) develop a multi-core processor and (4) implement an advanced micro-architectural feature covered in the class lectures.

在计算机组成课程的基础上，计算机架构课程介绍了更高级的微架构特征，包括流水线，分支预测，乱序执行。这个课程是面向高年级本科生和研究生的，由4个深入练习构成。在练习中，学生将会(1)将处理器流水线化，(2)构建一个cache并优化cache层次结构，(3)开发一个多核处理器，(4)实现一个课程讲座中的高级微架构特征。

### 6.1 Pipelining a Processor

The first micro-architecture feature covered in this Computer Architecture course is pipelining. To gain first hand experience with processor pipeline logic, students implement a seven stage pipelined RISC-V processor. Students use the BRISC-V Explorer to configure a baseline single cycle processor with a simple synchronous memory. Figure 3 shows the single cycle processor configuration in the BRISC-V Explorer.

在计算机架构课程中，第一个微架构特征是流水线。为获得处理器流水线逻辑的第一手经验，学生会实现一个7级流水线的RISC-V处理器。学生使用BRISC-V浏览器来配置一个基准单周期处理器，带有简单的同步内存。图3展示了在BRISC-V浏览器中的单周期处理器配置。

The synchronous memory used in the processor registers its read port and can be implemented with FPGA BRAM, making larger on-chip main memories practical for synthesized designs. Five of the processor’s seven pipeline stages are placed between each of the fetch, decode, execute, memory, and write-back modules. The last two pipeline registers placed in between the operation issue and receive sides of the fetch and memory stages. Adding a register between the issue side of the memory or instruction fetch interface and the receive side of the interface prevents the need to insert a pipeline bubble while the synchronous memory spends a cycle completing a memory read. A diagram of the pipelined processor is shown in Figure 7. In addition to inserting pipeline registers, students must add the necessary control signals to detect hazards and insert bubbles as needed. Control signals are derived based on in-class examples of hazard resolution logic.

处理器中使用的同步内存登记了其读端口，可以使用FPGA BRAM来实现，使更大的片上主存也可以通过合成的设计来实现。处理器的7级流水线阶段中的5个，放在取指令，解码，执行，访存和写回模块中。最后两级流水线寄存器放在发射操作和取指令和访存阶段的接收端。在访存的发射端，或取指令接口，和接口的接收端之间，加入一个寄存器，防止了插入流水线bubble的需要，同时同步内存花费一个周期完成一个内存读操作。图7展示了流水线处理器的图。除了插入流水线寄存器，学生还要加入必要的控制信号，以检测hazards，插入必须的bubbles。控制信号基于课程内hazard解决逻辑的例子推导出。

### 6.2 Memory Organization

Caches play a vital role in most modern processors. Cache hierarchies with multiple levels are used to overcome the “memory wall". Students in a Computer Architecture class can benefit from designing a processor with caches in the BRISC-V Explorer and analyzing the impact of cache configuration on the performance of a processor.

缓存在多数现代处理器中扮演了重要的角色。多级缓存层次结构用于克服内存墙的问题。计算机架构课程的学生在BRISC-V浏览器中设计一个带缓存的处理器，分析缓存配置对处理器性能的影响，是很有好处的。

**Part 1**: Students are required to build a simple direct mapped cache. While cache size is specified, student are allowed to pick a line size and set count of their choice. Students are encouraged to vary these values and observe the changes in resource usage with a synthesis tool of their choice. The implemented cache should follow the memory interface of the seven stage pipelined RISC-V core implemented using the BRISC-V explorer.

第1部分：学生要构建一个简单的直接映射缓存。指定了缓存大小后，学生可以选择一个cacheline大小，设置选择的个数。学生可以改变这些值，利用综合工具，观察资源利用的变化。实现的缓存应当遵循使用BRISC-V浏览器实现的7级流水线RISC-V核的内存接口。

**Part 2**: Students use the BRISC-V explorer to implement a complete cache hierarchy. The students are provided with binaries for two benchmark programs. They are required to optimize the cache structure to provide the best performance for the benchmark programs. THE First phase of the lab is implementing two different cache hierarchies that will be optimal for each of the programs.

第2部分：学生使用BRISC-V浏览器来实现完整的缓存层次结构。学生收到两个基准测试程序的二进制代码。学生需要优化缓存结构，对基准测试程序给出最好性能。试验的第一阶段，是实现两个不同的缓存层次结构，每个对一个程序是最优的。

The BRISC-V platform includes blocking caches that implement the write back with write allocate policy. The primary caches have one-cycle pipelined access. Secondary caches are based on a configurable cache module that can be used at level 2 or 3 of the hierarchy. The modular design allows most of these properties to be easily modified. The BRISC-V explorer allows the students to vary a multitude of cache hierarchy parameters in their optimization efforts. Some of these parameters are: (i) number of levels in the cache hierarchy, (ii) size of each cache, (iii) associativity, set count, and line width for each cache, and (iv) replacement policy for each cache. Students are allowed to configure any of the tunable parameters as long as the total resource usage for the cache hierarchy remains below a specified threshold.

BRISC-V平台包括blocking cache，实现了带有write allocate策略的写回。主缓存是一周期流水线访问的。次缓存是基于可配置的缓存模块，可以在第2级或第3级中使用。模块化的设计使多数这些性质可以很容易的改变。BRISC-V浏览器使学生可以改变很多cache层次结构的参数，以进行优化。其中一些参数是：(i)缓存层次结构的级数，(ii)每个缓存的大小，(iii)associativity, set count, 每个缓存的cacheline宽度，(iv)每个缓存的替换策略。学生可以配置任何可调节的参数，只要缓存层次结构的总计资源使用在指定的阈值之下。

Next, the students are required to implement a cache hierarchy that is optimized for both the benchmark programs. This lab is graded based on the level of performance achieved by the individual implementations.

下一步，学生需要实现一个缓存层次结构，对基准测试程序是优化的。这个试验的打分，是基于单个实现所获得性能的水平。

### 6.3 Multi-core Architecture

The next step in advanced features labs is implementing a multicore processor. 高级特征试验的下一步是实现一个多核处理器。

**Part 1**: Students use the BRISC-V explorer to implement a dual-core processor. A shared bus based cache hierarchy is used for the dual-core implementation. Students are then required to write a simple program with two threads to be executed on the two cores. Students are allowed to pick a program from a list of programs that include integer search, FFT, counting prime numbers in a given range, and matrix multiplication. Implementing a multi-core program in bare-metal code will give the students an opportunity to appreciate the complexities of implementing parallel programming libraries such as OpenMP and Open MPI.

第1部分：学生使用BRISC-V浏览器实现一个双核处理器。在双核实现中，使用了基于共享总线缓存层次结构。学生需要写一个简单的程序，有两个线程，在双核上执行。学生可以在程序列表中选择一个程序，包括整数搜索，FFT，在给定范围内对素数计数，和矩阵相乘。实现一个bare-metal的多核程序，可以让学生感谢并行程序库，如OpenMP和Open MPI。

**Part 2**: Students implement quad and octa-core processors using the BRISC-V explorer. For this step, a Network-on-Chip (NoC) based architecture is used 10. The NoC router is also fully parameterized. In other words, the number of ports and the number of virtual channels (VCs) per port can be modified, different arbitration schemes, VC allocations, and routing algorithms can be implemented. This gives the students an opportunity to familiarize themselves with different parameters and implementation details of an on-chip network, which is an essential part of current multi-core processors. Next, the students are required to modify the program from the previous step to utilize four or eight cores according to the processor configuration. The students are also required to observe and compare the performance variations with the number of cores.

第2部分：学生用BRISC-V实现四核和八核处理器。这一步中，会使用基于NoC的架构。NoC router也是参数化的。换句话说，端口数和每个端口的虚拟通道(VCs)数都是可以修改的，可以实现不同的仲裁方案，VCs配置，和路由算法。这可以让学生熟悉NoC的不同参数和实现细节，这是当前多核处理器的必不可少的部分。下一步，学生要修改前面一步的程序，以利用配置的四核或八核。学生还要观察并比较性能随着核数量的变化。

### 6.4 Other Advanced Features

For the final project of the Computer Architecture class, students implement an advanced micro-architecture feature covered in class lectures. Students implement the advanced features using the single-core seven stage pipelined processor as a baseline system. Students are free to choose if they want to use a cache hierarchy or not. The advanced micro-architecture feature is selected from a list including branch prediction, vector instructions, very long instruction word (VLIW), hardware multi-threading, floating-point unit, out-of-order execution, and speculative execution, among others.

作为计算机架构课程的最终项目，学生实现本课程讲座中的一个高级微架构特征。学生实现的高级特征，用单核7级流水线处理器作为基准系统。学生可以自由选择是否使用缓存参差结构。高级微架构特征是从列表中选择的，包括分支预测，向量指令，VLIW，硬件多线程，浮点单元，乱序执行，和推测执行，等等。

**Part 1**: Students must instrument an example program with Intel PIN [4] to predict performance improvements and justify their design choices. Instrumented code is executed on lab machines before RTL development of an architectural feature. To support design decisions, students develop software models of their feature and estimate the performance of the RISC-V architecture with and without their new feature.

第1部分：学生要用Intel PIN来分析例子程序，预测性能改进，合理化其设计选择。分析的代码在实验室机器中执行，然后再进行架构特征的RTL开发。为支持设计决定，学生开发特征的软件模型，估计RISC-V架构在有和没有这个新特征的性能差异。

**Part 2**: Students develop the RTL implementation of their micro-architectural feature and incorporate it into the baseline seven stage processor. Students analyze the impact their implemented feature has on the performance of an executed program of their choice. Program runtime is measured with the cycle count control status register included in the RV32I ISA. Students compare their predicted improvements with their actual improvements.

第2部分：学生开发微架构特征的RTL实现，将其整合进基准7级流水线处理器中。学生分析实现的特征对程序执行性能的影响。程序运行时的度量是用周期数CSR，这是包含在RV32I ISA中的。学生比较预测的改进和实际的改进。

## 7. Discussion

**Computer Organization**: Feedback from students in our computer organization class has been positive. Many students have expressed their excitement about completing their first processor design. After completing the computer organization class, several students have continued to work with the BRISC-V Design Space Exploration Platform, contributing to the base hardware system and adding features to the Explorer GUI. Although the single cycle processor students build in the computer organization class is simple, it is compatible with the RISC-V toolchian included in the BRISC-V Platform. With a complete software tool-chain, students can write custom C code for their processor and incorporate it in future projects as a soft core in a more complex design.

计算机组成。我们的计算机组成课程学生的反馈是正面的。很多学生在完成第一个处理器设计时感到很激动。在完成计算机组成课程时，几个学生继续使用BRISC-V设计空间探索平台，对基础硬件系统作出贡献，对浏览器GUI增加特征。虽然学生在计算机组成课程中构建的单周期处理器是简单的，但与BRISC-V中的RISC-V工具链是兼容的。有了一个完整的软件工具链，学生可以给其处理器写定制的C代码，将其整合到未来的项目中，在更复杂的设计中作为一个软核。

**Computer Architecture**: By the time students have completed both our Computer Organization and Computer Architecture classes, they have an in-depth knowledge of the BRISC-V hardware system. This detailed knowledge of the existing code-base makes further study focused on experimental architecture features easier. A firm grasp of the inner workings of the hardware system allows students to quickly add custom features.

计算机架构。在学生完成了计算机组成和计算机架构课程时，他们对BRISC-V硬件系统有了深入的了解。对现有代码库的详细了解，会使得后续的试验架构特征的研究更容易。对硬件系统的内部工作原理掌握的牢靠，会让学生可以迅速的加入定制特征。

Using the BRISC-V Explorer to implement the baseline system allows the students to focus on the advanced architecture features. The BRISC-V platform also includes the RTL implementations of several advanced architecture features such as Network-on-Chip and caches. Students have the opportunity to look at concrete implementations of these advanced features as opposed to learning about a feature from lecture notes or a textbook. This provides the students with deeper insight regarding certain tradeoffs involved in a real implementation.

使用BRISC-V浏览器来实现基准系统，使学生可以聚焦在高级架构特征中。BRISC-V平台还包括几个高级架构特征的RTL实现，如NoC和缓存。学生可以观察这些高级特征的具体实现，而不是从讲座或教科书上学习一个特征。这使得学生可以有更深入的洞见。

**BRISC-V platform in other courses**: The use of the BRISC-V platform goes beyond the canonical computer architecture courses. It can be used or adopted for any course where the students require an RTL code base of a processor to be used as the starting point of laboratory exercises of class projects. Our graduate level Hardware and Systems Security class is one such course. It focuses on in-depth analysis of hardware security’s role in cybersecurity, and the computer hardware related attacks and defenses in computing systems. Students have been able to use the BRISC-V Platform to further their research while working on the class project, which requires the students to implement a security feature on a baseline hardware system configured with the BRISC-V Explorer. The BRISC-V platform allows the students to quickly implement a working processor and focus on implementing their security extension.

在其他课程中的BRISC-V平台。BRISC-V平台的使用不仅是在经典的计算机架构课程中。在学生需要处理器的RTL代码的任何课程中都可以使用。我们的研究生级别的硬件和系统安全课就是这样一个课程。这个课程聚焦在硬件安全在网络安全中角色的深入分析，还有计算机硬件相关的攻击和防护。学生可以使用BRISC-V平台进一步研究，同时在课程项目中工作，这需要学生在一个基准硬件系统中实现一个安全特征，这个基准硬件系统可以用BRISC-V浏览器配置。BRISC-V平台可以使学生快速实现一个可以工作的处理器，专心实现其安全拓展。

One successful project implemented hardware multi-threading (HMT) on the seven stage core available in the BRISC-V Explorer. A cache hierarchy was connected to the HMT core and a cache side channel was demonstrated. This side-channel demonstration served as the baseline for future research in adaptive cache architectures to mitigate such side-channels. Another project implemented a multi-core processor with a hardware-isolated core, which provides secure execution capability. Other successful projects have developed micro-architectural support for control flow obfuscation [2] and hardware based Return-Oriented-Programming mitigation techniques [3]. These projects were presented at the 2019 Boston Area Architecture Workshop.

一个成功的项目，在BRISC-V浏览器中可用的7级流水线中实现了硬件多线程(HMT)。HMT上连接了一个缓存层次结构，展示了一个缓存side通道。这个side-channel的展示，是未来研究的基线，缓解side-channel的自适应缓存架构。另一个项目实现了一个多核处理器，带有硬件孤立的核，具有安全执行的能力。其他成功的项目开发了control flow obfuscation的微架构支持，和基于硬件的Return-Oriented-Programming mitigation技术。这些项目在2019 Boston Area Architecture Workshop上进行了展示。

## 8. Conclusion

In this work, we introduce the RISC-V-based BRISC-V Simulator and BRISC-V Explorer. The web-based BRISC-V Simulator enables students and teachers to quickly write software for RISC-V systems. The BRISC-V Emulator allows users to rapidly design a single- or multi-core RISC-V RTL processor to go with their software.

在本文中，我们介绍了基于RISC-V的BRISC-V仿真器和浏览器。基于网页的BRISC-V仿真器使学生和老师可以迅速写出RISC-V系统的软件。BRISC-V Emulator使用户可以迅速设计单核或多核RISC-V RTL处理器。

Together, these tools and the rest of the BRISC-V platform provide a wealth of RISC-V based resources for computer architecture education, streamlining software and hardware based laboratory exercises. Our experience with the BRISC-V Platform has been positive and our students have appreciated the support provided by a complete platform with the hardware system, compiler toolchain, software simulator and hardware configuration GUI.

这些工具和BRISC-V平台的其他部分，为计算机架构教育提供了基于RISC-V的资源，让基于软件和硬件的实验室练习更流畅。我们在BRISC-V平台上的经验是正面的，学生很感谢由这样一个完整的硬件系统，编译器工具链，软件仿真器和硬件配置GUI的平台的支持。