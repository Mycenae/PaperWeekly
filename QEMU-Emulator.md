# QEMU, a Fast and Portable Dynamic Translator

Fabrice Bellard

## 0. Abstract

We present the internals of QEMU, a fast machine emulator using an original portable dynamic translator. It emulates several CPUs (x86, PowerPC, ARM and Sparc) on several hosts (x86, PowerPC, ARM, Sparc, Alpha and MIPS). QEMU supports full system emulation in which a complete and unmodified operating system is run in a virtual machine and Linux user mode emulation where a Linux process compiled for one target CPU can be run on another CPU.

QEMU是一个快速机器仿真器，使用了一个原始的可移动动态翻译器，我们给出其内部原理。QEMU可以在几种宿主上(x86, PowerPC, ARM, Sparc, Alpha and MIPS)上对几种CPU(x86, PowerPC, ARM and Sparc)进行仿真。QEMU支持完整系统仿真和Linux用户模式仿真，在完整系统仿真中，会在虚拟机器中运行一个完整的、未修改的操作系统，在Linux用户模式中，在一个目标CPU上编译的Linux进行，会在另一种CPU上运行。

## 1. Introduction

QEMU is a machine emulator: it can run an unmodified target operating system (such as Windows or Linux) and all its applications in a virtual machine. QEMU itself runs on several host operating systems such as Linux, Windows and Mac OS X. The host and target CPUs can be different.

QEMU是一种机器仿真器，可以在一个虚拟机器上运行一个未修改的目标操作系统（如Windows或Linux）及其所有应用。QEMU本身可以在几种宿主操作系统上运行，如Linux，Windows和Mac OS X上。其宿主和目标CPUs可以是不一样的。

The primary usage of QEMU is to run one operating system on another, such as Windows on Linux or Linux on Windows. Another usage is debugging because the virtual machine can be easily stopped, and its state can be inspected, saved and restored. Moreover, specific embedded devices can be simulated by adding new machine descriptions and new emulated devices.

QEMU的主要用途是在一种操作系统上运行另一种操作系统，如在Linux上运行Windows，或在Windows上运行Linux。另一种用途是调试，因为虚拟机器可以很容易停止，其状态可以进行检查，保存和恢复。而且，通过添加新的机器描述和新的被仿真设备，可以模拟指定的嵌入式设备。

QEMU also integrates a Linux specific user mode emulator. It is a subset of the machine emulator which runs Linux processes for one target CPU on another CPU. It is mainly used to test the result of cross compilers or to test the CPU emulator without having to start a complete virtual machine.

QEMU还集成了一个Linux特定的用户模式仿真器。这是机器仿真器的一个子集，在另一个CPU上运行对一个目标CPU编译的Linux进程。其主要用于测试跨编译器的结果，或在不需要开启完整虚拟机器的情况下测试CPU仿真器。

QEMU is made of several subsystems: QEMU由几个子系统组成：

- CPU emulator (currently x86, PowerPC, ARM and Sparc)

CPU仿真器（目前包括x86, PowerPC, ARM和Sparc）

- Emulated devices (e.g. VGA display, 16450 serial port, PS/2 mouse and keyboard, IDE hard disk, NE2000 network card, ...)

被仿真的设备（如，VGA显示，16450串口，PS/2鼠标和键盘，IDE硬盘，NE2000网卡，等）

- Generic devices (e.g. block devices, character devices, network devices) used to connect the emulated devices to the corresponding host devices

通用设备（如，块设备，字符设备，网络设备），用于将被仿真的设备连接到对应的宿主设备上

- Machine descriptions (e.g. PC, PowerMac, Sun4m) instantiating the emulated devices

机器描述（如，PC，PowerMac，Sun4m），例化被仿真的设备

- Debugger 调试器
- User interface 用户界面

This article examines the implementation of the dynamic translator used by QEMU. The dynamic translator performs a runtime conversion of the target CPU instructions into the host instruction set. The resulting binary code is stored in a translation cache so that it can be reused. The advantage compared to an interpreter is that the target instructions are fetched and decoded only once.

本文检查了QEMU使用的动态翻译器的实现。动态翻译器将目标CPU指令在运行时转换到宿主的指令集。得到的二进制代码保存到翻译缓存中，这样可以进行重用。与解释器相比，其优势是，目标指令的取指和译码只有一次。

Usually dynamic translators are difficult to port from one host to another because the whole code generator must be rewritten. It represents about the same amount of work as adding a new target to a C compiler. QEMU is much simpler because it just concatenates pieces of machine code generated off line by the GNU C Compiler [5].

通常动态翻译器从一个宿主移植到另一个上是很困难的，因为整个代码生成器都需要进行重写，这与对一个C编译器添加一个新的目标的工作量一样。QEMU更简单，因为它只是将GNU C Compiler离线生成的机器代码片段拼接到一起。

A CPU emulator also faces other more classical but difficult [2] problems: 一个CPU仿真器还面临着其他更经典但更困难的问题：

- Management of the translated code cache 翻译的代码缓存的管理
- Register allocation 寄存器分配
- Condition code optimizations 条件码优化
- Direct block chaining 直接块链接
- Memory management 内存管理
- Self-modifying code support 支持自修改的代码
- Exception support 支持异常
- Hardware interrupts 硬件中断
- User mode emulation 用户模式仿真

## 2 Portable dynamic translation

### 2.1 Description

The first step is to split each target CPU instruction into fewer simpler instructions called micro operations. Each micro operation is implemented by a small piece of C code. This small C source code is compiled by GCC to an object file. The micro operations are chosen so that their number is much smaller (typically a few hundreds) than all the combinations of instructions and operands of the target CPU. The translation from target CPU instructions to micro operations is done entirely with hand coded code. The source code is optimized for readability and compactness because the speed of this stage is less critical than in an interpreter.

第一步是将每条目标CPU指令分裂成少数更简单的指令，称为微操作。每个微操作都由一小段C代码来实现。这种小型C源代码由GCC进行编译，形成目标文件。这些微操作是经过选择的，这样其数量与目标CPU上指令和操作数的所有组合相比会很少（通常是几百）。从目标CPU指令到微操作的翻译，完全是用手写代码进行的。源代码对可读性和紧凑性进行了优化，因为与在解释器中相比，这个阶段的速度并不是那么关键。

A compile time tool called dyngen uses the object file containing the micro operations as input to generate a dynamic code generator. This dynamic code generator is invoked at runtime to generate a complete host function which concatenates several micro operations.

一种称为dyngen的编译时工具，使用包含有微操作的目标文件作为输入，来生成动态代码生成器。这个动态代码生成器在运行时调用，以生成完整的宿主函数，即将几个微操作拼接到一起。

The process is similar to [1], but more work is done at compile time to get better performance. In particular, a key idea is that in QEMU constant parameters can be given to micro operations. For that purpose, dummy code relocations are generated with GCC for each constant parameter. This enables the dyngen tool to locate the relocations and generate the appropriate C code to resolve them when building the dynamic code. Relocations are also supported to enable references to static data and to other functions in the micro operations.

这个过程与[1]类似，但在编译时做了更多的工作，以得到更好的性能。特别是，一个关键思想是，在QEMU中，常数参数可以给到微操作中。为此，对每个常数参数用GCC生成的dummy code relocation。这让dyngen工具可以找到relocation，在构建动态代码时可以生成合适的C代码，以进行解析。支持relocation，还可以在微操作中参考静态数据和其他函数。

### 2.2 Example

Consider the case where we must translate the following PowerPC instruction to x86 code: 考虑下面的情况，我们需要将下面的PowerPC指令翻译到x86代码：

```
addi r1,r1,-16 # r1 = r1 - 16
```

The following micro operations are generated by the PowerPC code translator: 由PowerPC代码翻译器生成下面的微操作：

```
movl_T0_r1 # T0 = r1
addl_T0_im -16 # T0 = T0 - 16
movl_r1_T0 # r1 = T0
```

The number of micro operations is minimized without impacting the quality of the generated code much. For example, instead of generating every possible move between every 32 PowerPC registers, we just generate moves to and from a few temporary registers. These registers T0, T1, T2 are typically stored in host registers by using the GCC static register variable extension.

微操作的数量进行了最小化，而不太影响生成代码的质量。比如，我们没有对32个PowerPC寄存器之间的每个可能的移动都进行生成，而只对一些临时寄存器的移入和移出进行了生成。这些寄存器T0，T1和T2一般就存储在宿主寄存器中，使用了GCC静态寄存器变量扩展。

The micro operation movl_T0_r1 is typically coded as: 微操作movl_T0_r1一般编码为：

```
void op_movl_T0_r1(void)
{
    T0 = env->regs[1];
}
```

env is a structure containing the target CPU state. The 32 PowerPC registers are stored in the array env->regs[32]. env是一个结构，包含目标CPU的状态。32个PowerPC寄存器存储在阵列env->regs[32]中。

addl_T0_im is more interesting because it uses a constant parameter whose value is determined at runtime: addl_T0_im更有趣，因为其用的常数参数其值是在运行时决定的：

```
extern int __op_param1;
void op_addl_T0_im(void)
{
    T0 = T0 + ((long)(&__op_param1));
}
```

The code generator generated by dyngen takes a micro operation stream pointed by opc_ptr and outputs the host code at position gen_code_ptr. Micro operation parameters are pointed by opparam_ptr: dygen生成的代码生成器，以opc_ptr指向的微操作流为输入，输出的宿主代码在位置gen_code_ptr。微操作参数的指针为opparam_ptr：

```
[...]
for(;;) {
    switch(*opc_ptr++) {
        [...]

        case INDEX_op_movl_T0_r1:
        {
            extern void op_movl_T0_r1();
            memcpy(gen_code_ptr, (char *)&op_movl_T0_r1+0, 3);
            gen_code_ptr += 3;
            break;  
        }

        case INDEX_op_addl_T0_im:
        {
            long param1;
            extern void op_addl_T0_im();
            memcpy(gen_code_ptr, (char *)&op_addl_T0_im+0, 6);
            param1 = *opparam_ptr++;
            *(uint32_t *)(gen_code_ptr + 2) = param1;
            gen_code_ptr += 6;
            break; 
        }
        [...]
        }
    }
[...]
```

For most micro operations such as movl_T0_r1, the host code generated by GCC is just copied. When constant parameters are used, dyngen uses the fact that relocations to _op_param1 are generated by GCC to patch the generated code with the runtime parameter (here it is called param1).

对于多数微操作，比如movl_T0_r1，GCC生成的宿主代码就是复制的。当使用了常数参数，dyngen使用的事实是，GCC生成的relocation到_op_param1，以用运行时参数对生成的代码进行patch（这里，参数称为param1）。

When the code generator is run, the following host code is output: 当运行代码生成器，就会输出下面的宿主代码：

```
# movl_T0_r1
# ebx = env->regs[1]
mov 0x4(%ebp),%ebx
# addl_T0_im -16
# ebx = ebx - 16
add $0xfffffff0,%ebx
# movl_r1_T0
# env->regs[1] = ebx
mov %ebx,0x4(%ebp)
```

On x86, T0 is mapped to the ebx register and the CPU state context to the ebp register. 在x86上，T0映射到ebx寄存器，CPU状态上下文映射到ebp寄存器上。

### 2.3 Dyngen implementation

The dyngen tool is the key of the QEMU translation process. The following tasks are carried out when running it on an object file containing micro operations:

dyngen工具是QEMU翻译过程的关键。当在包含微操作的目标文件上运行的时候，会进行下面的任务：

- The object file is parsed to get its symbol table, its relocations entries and its code section. This pass depends on the host object file format (dyngen supports ELF (Linux), PE-COFF (Windows) and MACH-O (Mac OS X)).

解析目标文件，得到其符号表，其relocation entries和其code section。这些工作依赖于宿主目标文件格式（dyngen支持elf-linux, PE-COFF-windows, 和MACH-O-Mac OS X）。

- The micro operations are located in the code section using the symbol table. A host specific method is executed to get the start and the end of the copied code. Typically, the function prologue and epilogue are skipped.

使用符号表在代码段中定位微操作。执行宿主中特定的方法，得到复制代码的开始和结束。一般来说，函数的prologue和epilogue会被跳过。

- The relocations of each micro operations are examined to get the number of constant parameters. The constant parameter relocations are detected by the fact they use the specific symbol name _op_paramN.

检查每个微操作的relocation，以得到常量参数的数量。常量参数relocation的检测，是通过它们使用特定的符号名称_op_paramN的事实。

- A memory copy in C is generated to copy the micro operation code. The relocations of the code of each micro operation are used to patch the copied code so that it is properly relocated. The relocation patches are host specific.

生成C中的memcp，以复制微操作的代码。每个微操作代码的relocation，用于对复制代码进行patch，这样就得到了合适的relocate。这些relocation patches是随着宿主特定的。

- For some hosts such as ARM, constants must be stored near the generated code because they are accessed with PC relative loads with a small displacement. A host specific pass is done to relocate these constants in the generated code.

对于一些宿主，比如ARM，常量必须存储在生成的代码附近，因为是用PC相对的loads进行访问的，有一个小的偏移。进行宿主特定的动作，以将这些常量relocate在生成代码的附近。

When compiling the micro operation code, a set of GCC flags is used to manipulate the generation of function prologue and epilogue code into a form that is easy to parse. A dummy assembly macro forces GCC to always terminate the function corresponding to each micro operation with a single return instruction. Code concatenation would not work if several return instructions were generated in a single micro operation.

当编译微操作码时，用了一系列GCC标志来操作函数prologue和epilogue代码的生成，使其成为容易解析的形式。用了一个dummy汇编宏，迫使GCC用单个返回指令来停止对应每个微操作的函数。如果在一个微操作中，生成了几个返回指令，则代码拼接就不会起作用了。

## 3. Implementation details

### 3.1 Translated Blocks and Translation Cache

When QEMU first encounters a piece of target code, it translates it to host code up to the next jump or instruction modifying the static CPU state in a way that cannot be deduced at translation time. We call these basic blocks Translated Blocks (TBs).

当QEMU遇到了目标代码的片段时，它将其翻译成宿主代码，直到下一条跳转指令，或到达无法在翻译时推断的修正静态CPU状态的指令。我们称这些basic blocks为翻译块(Translated Blocks)。

A 16 MByte cache holds the most recently used TBs. For simplicity, it is completely flushed when it is full.

用了一个16MB的缓存，保存了最近使用的TBs。为简化起见，当满的时候，就完全冲刷掉。

The static CPU state is defined as the part of the CPU state that is considered as known at translation time when entering the TB. For example, the program counter (PC) is known at translation time on all targets. On x86, the static CPU state includes more data to be able to generate better code. It is important for example to know if the CPU is in protected or real mode, in user or kernel mode, or if the default operand size is 16 or 32 bits.

静态CPU状态定义为，当进入TB时，在翻译时，是已知的那部分CPU状态。比如，在所有目标上，在翻译时，程序计数器是已知的。在x86上，静态CPU状态包括更多数据，可以生成更好的代码。比如，要知道CPU是处于protected还是real模式，在用户模式还是kernel模式，或默认的操作数大小是16 bits还是32 bits，这很重要。

### 3.2 Register allocation

QEMU uses a fixed register allocation. This means that each target CPU register is mapped to a fixed host register or memory address. On most hosts, we simply map all the target registers to memory and only store a few temporary variables in host registers. The allocation of the temporary variables is hard coded in each target CPU description. The advantage of this method is simplicity and portability.

QEMU使用固定的寄存器分配。这意味着每个目标CPU寄存器都映射到固定的宿主寄存器或内存地址。在多数宿主中，我们只是将所有的目标寄存器映射到内存中，在宿主寄存器中只存储几个临时变量。临时变量的分配，是在每个目标CPU描述中硬编码的。这种方法的优势是简单和可移植。

The future versions of QEMU will use a dynamic temporary register allocator to eliminate some unnecessary moves in the case where the target registers are directly stored in host registers.

QEMU的未来版本会使用一个动态临时寄存器分配器，在目标寄存器直接存储在宿主寄存器中时，消除一些不必要的移动。

### 3.3 Condition code optimizations

Good CPU condition code emulation (eflags register on x86) is a critical point to get good performances. QEMU uses lazy condition code evaluation: instead of computing the condition codes after each x86 instruction, it just stores one operand (called CC_SRC), the result (called CC_DST) and the type of operation (called CC_OP). For a 32 bit addition such as R = A + B, we have:

好的CPU条件码（x86系统中的eflags寄存器）仿真，是得到好性能的关键点。QEMU使用懒条件码评估：没有在每条x86指令后计算条件码，而只是存储一个操作数(称为CC_SRC)，结果(称为CC_DST)和操作类型(称为CC_OP)。对于一个32 bit加法，比如R=A+B，我们有：

```
CC_SRC=A
CC_DST=R
CC_OP=CC_OP_ADDL
```

Knowing that we had a 32 bit addition from the constant stored in CC_OP, we can recover A, B and R from CC_SRC and CC_DST. Then all the corresponding condition codes such as zero result (ZF), non-positive result (SF), carry (CF) or overflow (OF) can be recovered if they are needed by the next instructions.

从存储在CC_OP中的常数，可以知道了我们有一个32 bit加法，我们就可以从CC_SRC和CC_DST中恢复A，B和R。那么，所有对应的条件码，如ZF，SF，CF或OF，如果下一条指令需要的话，都可以恢复出来。

The condition code evaluation is further optimized at translation time by using the fact that the code of a complete TB is generated at a time. A backward pass is done on the generated code to see if CC_OP, CC_SRC or CC_DST are not used by the following code. At the end of TB we consider that these variables are used. Then we delete the assignments whose value is not used in the following code.

条件码评估在翻译时进一步进行了优化，因为一个完整TB的代码是一次性生成的。对生成的代码进行反向过程，以查看CC_OP, CC_SRC或CC_DST没有被下面的代码所用。在TB的最后，我们考虑这些变量被使用了。然后，我们删除这些值在后续代码中不进行使用的赋值。

### 3.4 Direct block chaining

After each TB is executed, QEMU uses the simulated Program Counter (PC) and the other information of the static CPU state to find the next TB using a hash table. If the next TB has not been already translated, then a new translation is launched. Otherwise, a jump to the next TB is done.

在每个TB执行后，QEMU使用仿真PC和其他静态CPU状态信息，使用一个哈希表来寻找下一个TB。如果下一个TB还没有被翻译，那么就启动一个新的翻译。否则，就完成了到下一个TB的跳跃。

In order to accelerate the most common case where the new simulated PC is known (for example after a conditional jump), QEMU can patch a TB so that it jumps directly to the next one.

最常见的情况是，下一个仿真PC是已知的（比如，在一个条件跳转之后），为加速这种情况，QEMU可以对一个TB进行打补丁，让其直接跳转到下一个。

The most portable code uses an indirect jump. On some hosts (such as x86 or PowerPC), a branch instruction is directly patched so that the block chaining has no overhead.

最可移植的代码使用一个间接跳转。在一些宿主上（比如x86或PowerPC），分支指令可以直接打补丁，让模块链接没有任何代价。

### 3.5 Memory management

For system emulation, QEMU uses the mmap() system call to emulate the target MMU. It works as long as the emulated OS does not use an area reserved by the host OS.

对于系统仿真，QEMU使用mmap()系统调用来仿真目标MMU。只要仿真OS不使用宿主OS的保留区域，就没有问题。

In order to be able to launch any OS, QEMU also supports a software MMU. In that mode, the MMU virtual to physical address translation is done at every memory access. QEMU uses an address translation cache to speed up the translation.

为能够启动任何OS，QEMU还支持软件MMU。在这个模式下，MMU虚拟到物理地址转换在每个内存访问时都会进行。QEMU使用一个地址转换缓存来加速这个转换。

To avoid flushing the translated code each time the MMU mappings change, QEMU uses a physically indexed translation cache. It means that each TB is indexed with its physical address.

为避免在每次MMU映射变化的时候，都要冲刷掉翻译的代码，QEMU使用了一个物理索引的翻译缓存。这意味着每个TB都是由其物理地址进行索引的。

When MMU mappings change, the chaining of the TBs is reset (i.e. a TB can no longer jump directly to another one) because the physical address of the jump targets may change.

当MMU映射变化时，TBs的链接就重置了（即，一个TB不能直接跳转到另一个TB），因为跳转目标的物理地址也可能变化。

### 3.6 Self-modifying code and translated code invalidation

On most CPUs, self-modifying code is easy to handle because a specific code cache invalidation instruction is executed to signal that code has been modified. It suffices to invalidate the corresponding translated code.

在多数CPUs中，自我修改的代码都很容易处理，因为会执行一个特定的代码缓存无效指令，以发出信号，说明代码已经被修改了。这足以使对应的翻译代码无效。

However on CPUs such as the x86, where no instruction cache invalidation is signaled by the application when code is modified, self-modifying code is a special challenge.

但是，在类似x86这样的CPUs上，当代码被修改时，应用不能传递指令缓存无效的信息，自我修改的代码就是一个特别的挑战。

When translated code is generated for a TB, the corresponding host page is write protected if it is not already read-only. If a write access is made to the page, then QEMU invalidates all the translated code in it and reenables write accesses to it.

当对一个TB生成了翻译的代码，对应的宿主页面如果还没有变成只读，那就一定是写保护的。如果对这个页面有了写的访问，那么QEMU就将所有的翻译代码无效化，然后重新开启写访问权限。

Correct translated code invalidation is done efficiently by maintaining a linked list of every translated block contained in a given page. Other linked lists are also maintained to undo direct block chaining.

通过维护给定页面中包含的每个TB的连接列表，可以很高效的进行正确的翻译代码无效化。也需要维护其他连接列表，以撤销直接模块链接。

When using a software MMU, the code invalidation is more efficient: if a given code page is invalidated too often because of write accesses, then a bitmap representing all the code inside the page is built. Every store into that page checks the bitmap to see if the code really needs to be invalidated. It avoids invalidating the code when only data is modified in the page.

当使用软件MMU时，代码无效化是更高效的：如果由于写访问，一个给定的代码页面被无效化了过多次，那么就构建一个bitmap，表示在这个页面中的所有代码。对这个页面中的每次存储，都会检查这个bitmap，看看这个代码是否真的需要进行无效化。在只有页面中的数据得到修改时，就避免了对代码的无效化。

### 3.7 Exception support

longjmp() is used to jump to the exception handling code when an exception such as division by zero is encountered. When not using the software MMU, host signal handlers are used to catch the invalid memory accesses.

longjmp()是在遇到异常时，如除以0，用于跳转到异常处理的代码。在不使用软件MMU时，宿主信号处理者用于捕获无效的内存访问。

QEMU supports precise exceptions in the sense that it is always able to retrieve the exact target CPU state at the time the exception occurred. Nothing has to be done for most of the target CPU state because it is explicitly stored and modified by the translated code. The target CPU state S which is not explicitly stored (for example the current Program Counter) is retrieved by re-translating the TB where the exception occurred in a mode where S is recorded before each translated target instruction. The host program counter where the exception was raised is used to find the corresponding target instruction and the state S.

QEMU支持精确的异常，这个意思是在异常发生时，总能够获取到精确的目标CPU状态。对于多数目标CPU状态，不需要做任何事，因为这由翻译代码显式的存储和修改。没有显式存储的目标CPU状态S，比如当前PC，通过重新翻译TB取出，其中异常发生的mode中，S在每条翻译的目标指令之前记录下来。发生异常的宿主PC，用于找到对应的目标指令和状态S。

### 3.8 Hardware interrupts

In order to be faster, QEMU does not check at every TB if an hardware interrupt is pending. Instead, the user must asynchronously call a specific function to tell that an interrupt is pending. This function resets the chaining of the currently executing TB. It ensures that the execution will return soon in the main loop of the CPU emulator. Then the main loop tests if an interrupt is pending and handles it.

为了让QEMU更快，QEMU并没有在每个TB处检查是否有一个硬件中断在挂起。而是用户必须异步的调用一个特定的函数，来告诉一个异常在挂起。这个函数重设了当前执行的TB的链接。这确保了，执行会很快回到CPU仿真器的主循环，然后主循环测试是否有中断在挂起，然后处理之。

### 3.9 User mode emulation

QEMU supports user mode emulation in order to run a Linux process compiled for one target CPU on another CPU.

QEMU支持用户模式仿真，以在一个CPU上运行为另一个目标CPU编译的Linux进程。

At the CPU level, user mode emulation is just a subset of the full system emulation. No MMU simulation is done because QEMU supposes the user memory mappings are handled by the host OS. QEMU includes a generic Linux system call converter to handle endianness issues and 32/64 bit conversions. Because QEMU supports exceptions, it emulates the target signals exactly. Each target thread is run in one host thread.

在CPU级别上，用户模式仿真只是完整系统仿真的一个子集。不用进行MMU模拟，因为QEMU假设用户内存映射是由宿主OS进行处理的。QEMU包含了一个通用Linux系统调用转换器，来处理endianness问题，和32/64 bit的转换。因为QEMU支持异常，可以对目标进行进行精确仿真。每个目标进程都在一个宿主线程上运行。

## 4. Porting work

In order to port QEMU to a new host CPU, the following must be done: 为将QEMU移植到一个新的宿主CPU上，需要做以下工作：

- dyngen must be ported (see section 2.2). 必须要移植dyngen

- The temporary variables used by the micro operations may be mapped to host specific registers in order to optimize performance. 微操作使用的临时变量，可以映射到宿主的特定寄存器，以优化性能。

- Most host CPUs need specific instructions in order to maintain coherency between the instruction cache and the memory. 多数宿主CPUs需要特定的指令，以保持指令cache和内存之间的一致性。

- If direct block chaining is implemented with patched branch instructions, some specific assembly macros must be provided. 如果实现了直接block链接，带有补丁的分支指令，必须要提供一些特定的汇编宏。

The overall porting complexity of QEMU is estimated to be the same as the one of a dynamic linker. QEMU的总体移植复杂度，估计与动态链接器类似。

## 5. Performance

In order to measure the overhead due to emulation, we compared the performance of the BYTEmark benchmark for Linux [7] on a x86 host in native mode, and then under the x86 target user mode emulation.

为度量仿真的代价，我们比较了BYTEmark基准测试在一个x86宿主上的性能，和在x86目标用户模式仿真下的性能。

User mode QEMU (version 0.4.2) was measured to be about 4 times slower than native code on integer code. On floating point code, it is 10 times slower. This can be understood as a result of the lack of the x86 FPU stack pointer in the static CPU state. In full system emulation, the cost of the software MMU induces a slowdown of a factor of 2.

在整数代码上，用户模式QEMU（版本0.4.2）比本地代码要慢4倍。在浮点代码上，要慢10倍。这可以理解为，在静态CPU状态下缺少x86 FPU stack pointer的结果。在完整系统仿真下，软件MMU的代码会带来2倍的减速。

In full system emulation, QEMU is approximately 30 times faster than Bochs [4]. 在完整系统仿真中，QEMU大约比Bochs快30倍。

User mode QEMU is 1.2 times faster than valgrind --skin=none version 1.9.6 [6], a hand coded x86 to x86 dynamic translator normally used to debug programs. The --skin=none option ensures that Valgrind does not generate debug code.

用户模式QEMU比valgrind --skin=none版本1.9.6快1.2倍。--skin=none选项确保了，valgrind不会生成debug代码。

## 6. Conclusion and Future Work

QEMU has reached the point where it is usable in everyday work, in particular for the emulation of commercial x86 OSes such as Windows. The PowerPC target is close to launch Mac OS X and the Sparc one begins to launch Linux. No other dynamic translator to date has supported so many targets on so many hosts, mainly because the porting complexity was underestimated. The QEMU approach seems a good compromise between performance and complexity.

QEMU已经达到了可以在日常工作中使用的程度，特别是仿真商用x86 OS比如Windows。PowerPC目标已经接近于启动Mac OS X，Sparc目标已经接近启动Linux。其他动态翻译器都没有在这么多宿主上支持这么多目标，主要是因为移植的复杂度被低估了。QEMU方法似乎是性能和复杂度之间一个好的折中。

The following points still need to be addressed in the future: 未来仍然需要处理下面的点：

- Porting: QEMU is well supported on PowerPC and x86 hosts. The other ports on Sparc, Alpha, ARM and MIPS need to be polished. QEMU also depends very much on the exact GCC version used to compile the micro operations definitions.

移植：QEMU在PowerPC和x86宿主上支持的很好。在Sparc，Alpha，ARM和MIPS上的支持仍然需要跟进。QEMU还非常需要依赖于GCC的确切版本，这是用于编译微操作的定义。

- Full system emulation: ARM and MIPS targets need to be added. 完整系统的仿真：需要支持ARM和MIPS的目标。

- Performance: the software MMU performance can be increased. Some critical micro operations can also be hand coded in assembler without much modifications in the current translation framework. The CPU main loop can also be hand coded in assembler.

性能：软件MMU性能可以进行提升。一些关键的微操作也可以用汇编手写代码，不会对目前的翻译框架进行很大改动。CPU主循环也可以用汇编进行手工编码。

- Virtualization: when the host and target are the same, it is possible to run most of the code as is. The simplest implementation is to emulate the target kernel code as usual but to run the target user code as is.

虚拟化：当宿主和目标一样时，可以运行大多数的代码。最简单的实现是，正常仿真目标的核代码，并运行目标用户代码。

- Debugging: cache simulation and cycle counters could be added to make a debugger as in SIMICS [3].

调试：可以加入缓存模拟和周期计数器，以成为一个调试器，就像在SIMICS中一样。