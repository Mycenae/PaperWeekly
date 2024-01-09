# Chapter 2 Background on GPU architecture

In this chapter we cover the necessary background on GPU architecture. 本章中，我们讨论一下GPU架构的必须背景知识。

GPUs are parallel processors designed to accelerate computer graphics applications. They have faster memory and faster arithmetic capabilities than CPUs, which makes them also attractive for
applications other than graphics. Using these capabilities, however, requires a large degree of parallelism. A serial code, for example, typically runs faster on a CPU, not a GPU. As a result,
GPUs are never used alone but only as an extension of a CPU-based host system. The main program is executed on the CPU and only small code fragments called kernels are executed on the
GPU. If well-optimized, these kernels run at higher rates performing the bulk of the work. In this text we are exclusively interested in the performance of GPU kernels.

GPUs是并行处理器，设计用于加速计算机图形应用。GPU比CPU有着更快的内存速度，和更快的算术运算能力，这使其也可以应用于图形之外的应用。但是，使用这些能力需要很大的并行度。比如，顺序代码在CPU上的运行速度更快，GPU会慢一些。所以，GPUs一般不会单独使用，而是作为基于CPU的宿主系统的扩展。主程序运行在CPU，只有一些小型代码片段，称为kernel，会运行在GPU上。如果优化的好，这些kernel会以很高的速度运行，执行主要的任务。在本文中，我们只对GPU kernel的性能感兴趣。

In the following, we describe GPU architecture at a level sufficient to understand the remaining text. For another entry-level introduction to GPU architecture see Nickolls and Kirk [2009]. For
an introduction oriented towards kernel development see CUDA C programming guide [NVIDIA 2015]. Differences between different generations of GPU architecture are detailed in the respective vendor’s whitepapers, such as those listed in Table 2.1. Additional details not published elsewhere are found in NVIDIA patents; some of them are cited below.

下面，我们描述的GPU架构层次足以理解本文余下的部分。GPU架构的入门，可以参考Nickolls and Kirk [2009]。Kernel开发的介绍性工作见CUDA C编程指南。不同代GPU架构的差异，在相应的白皮书中有详述，比如在表2.1中所列的。还有一些细节可以在NVIDIA的专利中找到，下面我们会引述一些。

## 2.1 GPUs used in this work

The scope of this study is limited to NVIDIA GPUs. NVIDIA is the leading GPU vendor and also pioneered better support for general-purpose computing on GPUs. Their solution, called CUDA, allows programming NVIDIA GPUs using an extension of C language unburdened by graphics semantics. We use CUDA throughout this work, but the conclusions likely apply to kernels implemented using other frameworks, such as OpenCL [Howes 2015], as long as a similar hardware is used.

本文只研究NVIDIA的GPU。NVIDIA是领先的GPU供货商，也是在GPUs上提供通用目标计算的先驱。它们的解决方案称为CUDA，使用C语言的扩展来对NVIDIA GPUs进行编程。我们在本文中使用CUDA，只要使用类似的硬件，结论应当也适用于其他框架开发的kernel，比如OpenCL。

Six generations of CUDA-capable GPUs were released so far: G80, GT200, Fermi, Kepler, Maxwell and Pascal. In this study we consider one GPU in each of these generations except Pascal; these GPUs as listed in Table 2.1. Each of them was a flagship product at the time of its release: it offered the leading performance and was used to promote the respective architecture generation. We often refer to these particular GPUs by their generation name. For example, “the Kepler GPU” refers to the GeForce GTX 680 GPU that we use.

目前发布了6代具备CUDA能力的GPUs：G80, GT200, Fermi, Kepler, Maxwell和Pascal。在本文中，我们每代选取了一种GPU，列在表2.1中。其中的每个都是其发布时的旗舰产品：性能领先，用于推动相应的每代架构。我们通常用其代名称呼这些特定的GPUs。比如，Kepler GPU是指GeForce GTX 680 GPU。

Table 2.1: GPUs used in this study. All of them are GeForce GPUs.

| GPU | Year | Generation | Related publications and whitepapers by vendor |
| --- | ---- | ---------- | ---------------------------------------------- |
| 8800GTX | 2006 | G80 | NVIDIA [2006] |
| GTX280  | 2008 | GT200 | NVIDIA [2008], Lindholm et al. [2008] |
| GTX480  | 2010 | Fermi | NVIDIA [2009], NVIDIA [2010b], Wittenbrink et al. [2011] |
| GTX680  | 2012 | Kepler | NVIDIA [2012a], NVIDIA [2012b] |
| GTX980  | 2014 | Maxwell | NVIDIA [2014a], NVIDIA [2014b] |

## 2.2 SIMD architecture

One of the key features of GPU architecture is wide SIMD width and a novel hardware support for treating SIMD divergence.

GPU架构的关键特征之一，是很宽的SIMD宽度，对SIMD divergence有新的硬件支持。

SIMD stands for Single Instruction Multiple Data. All instructions on NVIDIA GPUs are SIMD instructions, i.e. operate on vectors, not scalars. The length of the vector is called SIMD width and equals 32 on all CUDA-capable GPUs released so far. For example, an add instruction adds element-wise 32 numbers in one register to 32 numbers in another register and writes the 32 results into a third register. Each register, therefore, is an array of 32 elements. Each element is 32-bit; dealing with wider data types such as double precision numbers and 64-bit pointers requires using two or more registers.

SIMD代表单指令多数据。在NVIDIA GPUs上，所有指令都是SIMD指令，即，是在矢量上运算的，而不是标量。矢量的长度，称为SIMD宽度，在目前具备CUDA功能的GPUs上，就是32。比如，一条加法add指令，将一组寄存器中的32个数字，与另一组寄存器中的32个数字相加，将32个结果写入第三组寄存器。每组寄存器都是32个元素的阵列。每个元素都是32-bit的；对更宽的数据类型，比如双精度数和64-bit的指针，需要使用两个或更多的寄存器。

The GPU programming model conceals SIMD operations by exposing each physical thread as a number of logical threads – as many logical threads as the SIMD width. The GPU program is then written as operating on scalars, not vectors, and many logical threads are created to both populate the SIMD width and run multiple physical threads.

GPU编程模型将每个物理线程都暴露为多个逻辑线程，数量与SIMD宽度相同，这样就可以将SIMD运算隐藏起来。GPU程序可以写为在标量上运算，而不是在矢量上运算，会创建很多逻辑线程来传播SIMD宽度，运行多个物理线程。

To differentiate the two types of threads, physical threads are called warps and the term “thread” is reserved for logical threads only. In this notation there are 32 threads in 1 warp.

为区分两种线程，物理线程我们称之为warps，而thread则代表逻辑线程。在这种表示下，1个warp中有32个线程。

## 2.3 Single Instruction Multiple Threads (SIMT)

The GPU programming model permits different threads in the same warp to take different paths in the program. But as the underlying architecture is SIMD, these different paths must be mapped into a single path for the warp to be executed. A key feature of GPUs is that this mapping is done during execution by the hardware itself. SIMD processors that implement this novel capability are said to have SIMT architecture; SIMT stands for Single Instruction Multiple Threads. SIMT architecture is a flavor of SIMD architecture.

GPU编程模型允许程序中相同warp的不同线程有不同的执行路径。但由于底层的架构是SIMD，这些不同的路径必须映射到单路径上，以让warp去执行。GPUs的一个关键特征是，这种映射是硬件本身在执行的时候完成的。实现了这种新能力的SIMD处理器，被称为具有SIMT架构；SIMT代表单指令多线程。SIMT架构是SIMD架构的一种风格。

For a basic understanding of SIMT operation consider execution of an “if … else …” statement, such as: 为对SIMT运算有一个基本的理解，考虑一个if...else...语句的执行，比如：

```
if(condition) { a=a+1; b=b+2; c=c+3; }
else { d=d*4; e=e*5; f=f*6; }
```

The corresponding assembly code with no SIMD semantics is shown in Table 2.2, left. In the code, @!P0 BRA is a predicated branch instruction – the branch is taken if predicate !P0 is 1, i.e. if P0 is 0. P0 is a predicate register that corresponds to variable condition above.

对应的没有SIMD语义的汇编代码，如表2.2左所示。代码中，@!P0 BRA是一个谓词分支指令，如果谓词!P0是1，即P0为0，则执行分支。P0是一个谓词寄存器，对应着上面的变量条件。

Suppose this code is executed in more than one thread. Different threads may branch differently: some may take the if branch, when others take the else branch. This may seem to be incompatible with SIMD execution: if all of these threads are in the same warp, they all must follow the same instruction sequence – branching differently is not possible.

假设这个代码在多线程中执行。不同的线程可能会有不同的分支：一些会取if分支，其他的会取else分支。这似乎与SIMD的执行会有些不兼容：如果所有这些线程都在相同warp中，它们必须沿着相同的指令顺序执行，有不同的分支是不可能的。

Yet, a SIMT processor can execute this code as is, despite the potentially diverging paths. When the processor finds that thread paths diverge, it disables some of the threads and executes instructions in one path; then it disables the other threads and executes instructions in the other path. Similar execution is possible on SIMD processors that don’t have SIMT architecture but only if this operation is implemented explicitly in the assembly code.

但是，SIMT处理器可以执行这些代码，尽管可能会有分叉的路径。当处理器发现线程路径发散时，会将一些线程disable掉，执行一条路径中的指令；然后disable掉其他的线程，执行另一条路径中的指令。在没有SIMT架构的SIMD处理器中，如果这些运算在汇编代码中显示的写出来了，也可以进行类似的执行。

Although the presented code can be executed on a SIMT processor as is, its performance may be lacking. A better SIMT code must specify not only where execution paths diverge – which is at branch instructions – but also where they converge back.

虽然给出的代码可以在SIMT处理器上执行，但其性能可能会比较差。更好的SIMT代码必须指定执行路径在哪里分化（一般是在分支指令处），还要指定在哪里指令再次汇合。

The better code is shown in Table 2.2, right. It has two new features. First, the program address where the threads converge is set prior to their divergence using instruction SSY. Second, the last instruction in each of the diverged paths is tagged with suffix “.S”. (“S” stands for “synchronization”.)

更好的代码如表2.2右所示。有2个新特征。第一，线程汇合的程序地址，在分叉之前就设置好了，使用的是指令SSY。第二，每条分叉路径的最后一条指令，都加上了后缀.S（S代表同步，synchronization）。

## 2.4 SIMT execution

Below we detail how the code in Table 2.2, right, is executed on a SIMT processor with two cases of input data. For the sake of example, we assume there are only four threads, all in one warp. In Case 1 the inputs in predicates P0 are set respectively to 0, 1, 1, 0 across the threads, which means that threads 1 and 2 take the if branch and threads 0 and 3 take the else branch. This case is detailed in Table 2.3. In Case 2 the predicates are set to 1, 1, 1, 1: all threads take the if branch. This case is detailed in Table 2.4.

下面我们详细描述一下表2.2右的代码，在两种输入数据的情况下，是怎样在SIMT处理器上执行的。为举例，我们假设只有4个线程，都在一个warp中。在第1种情况下，谓词P0的输入设为0，1，1，0，即线程1和2执行if分支，而线程0和3执行else分支。表2.3给出了详细情况。在第2种情况中，谓词设为1，1，1，1，即所有的线程都执行if分支。这在表2.4中进行了详述。

Column “Threads” in the tables lists the logical execution order for each thread as defined by the program. Column “Warp” lists the physical execution order for the warp as executed on a SIMT processor. Also, this column lists the thread masks, which are explained below.

表中的列Threads，列出了每个线程的逻辑执行顺序，这是程序所定义的。列warp列出了SIMT处理器的warp的物理执行顺序。这个列还列出了thread masks，下面进行解释。

As SIMT architecture is a flavor of SIMD architecture, executing an instruction in one thread implies executing it also in all other threads of the same warp – unless some of them are disabled. A SIMT processor keeps track of which threads are currently disabled using a special bit mask called active mask. For example, executing a SIMD instruction when active mask is 0110 implies that the instruction is effective only for threads 1 and 2 in the warp. Active mask is a part of warp state, similarly to program counter. Below we refer to pair (program counter, active mask) as current execution state.

由于SIMT架构是一种风格的SIMD架构，在一个线程中执行一条指令，意味着同warp中的所有其他线程也会执行，除非有些线程disable掉了。SIMT处理器使用特殊的bit mask称为active mask来追踪目前哪些线程是被disable掉的。比如，当active mask为0110时，执行SIMD指令意味着只有对warp中的线程1，2指令是有效的。Active mask是warp状态的一部分，与pc类似。下面，我们将pc与active mask的对，称为当前的执行状态。

To keep track of the paths taken by different threads, SIMT processor employs another warp state called divergence stack. When a branch instruction diverges, one of the branches is taken by the warp immediately, and the other is taken later; the data necessary to take it later is put on the stack. This data is a similar pair of a program counter address and an active mask.

为追踪不同线程的执行路径，SIMT处理器利用了另一个warp状态，称为divergence stack。当一个分支指令分叉时，warp会立刻执行其中一个分支，后面会执行其他的分支；后面执行其他分支所必须的数据，是放在stack上的。这个数据是类似的pc地址和active mask对。

Additional entries are put on the stack by the SSY instructions. These entries include the address of the convergence point as specified in the instruction and the active mask at the time of executing the instruction, i.e. before the divergence. In both of our examples, this entry is (ENDIF, 1111).

SSY指令会将额外的entries放到stack上。这些entries包括指令中指定的汇合点的地址，以及在执行指令的时候的active mask，即，在分叉之前的时候。在我们的两个例子中，这个entry都是(ENDIF, 1111)。

Entries are popped from the stack upon executing the .S instructions. The taken entry is then assigned to the current execution state.

在执行.S指令时，从stack上将entries弹出来。选定的entry就指定为当前的执行状态。

In Case 1, the branch instruction diverges into paths (IF, 0110) and (ELSE, 1001). Suppose the else branch is taken first. Then the current execution state is set to (ELSE, 1001) and pair (IF, 0110) is put on the stack. The warp state after executing the branch instruction then is:

在case 1中，分支指令分叉为两个路径，(IF 0110)和(ELSE 1001)。假设首先选定else分支执行。那么当前的执行状态就设置为(ELSE, 1001)，(IF, 0110)就放到stack上。执行了分支指令后的warp状态为：

```
Current execution state: (ELSE, 1001)
Divergence stack: (IF, 0110)
(ENDIF, 1111)
```

As a result, the else branch is executed with mask 1001 first, then the if branch is executed with mask 0110, and then execution is transferred to label ENDIF with mask 1111; the transfers are at .S instructions. The effect is the same as if threads 1 and 2 took the if branch and threads 0 and 3 took the else branch. Note that different threads took different branches, but the warp took both.

结果是，首先执行else分支，mask为1001，然后执行if分支，mask为0110，然后执行转移到标签ENDIF上，mask为1111；转移是在.S指令上发生的。执行的效果就像是线程1，2执行了if分支，线程0，3执行了else分支。 注意，不同的线程执行了不同的分支，但是warp会执行两个分支。

In Case 2, the branch options are (IF, 1111) and (ELSE, 0000). The second pair has zero mask, which means that the branch is not diverging. The pair with zero mask is discarded and the other pair is set as the current execution state. The warp state after executing the branch instruction then is:

在case 2中，分支选项为(IF, 1111)和(ELSE, 0000)。第二对的mask数为0，这意味着该分支并没有分叉。0个mask的对被抛弃，另一对设为当前的执行状态。在执行分支指令后的warp状态为：

```
Current execution state: (IF, 1111)
Divergence stack: (ENDIF, 1111)
```

Then, execution proceeds at the if branch with mask 1111, but only up to the .S instruction. After executing the .S instruction, the control is transferred to label ENDIF with the same mask. In effect all threads take the if branch, and the warp also takes only the if branch; instructions in the else branch are not executed.

然后，继续执行if分支，mask为1111，但只执行到.S指令。在执行了.S指令后，控制被转移到标签ENDIF，mask相同。实际上，所有线程都执行了if分支，warp也只执行了if分支；else分支中的指令并没有执行。

This illustrates the basic operation of a SIMT processor. Further details can be found in NVIDIA patents such as Coon et al. [2008], Coon et al. [2009], Coon et al. [2010], and Coon et al. [2011]. In particular, there is an additional detail needed to handle different types of control divergence, such as divergence at branch instructions, divergence at return instructions, and divergence at break instructions.

这描述了SIMT处理器的基本操作。更详细的内容可以在NVIDIA专利中找到。特别的，需要额外的细节来处理不同类型的控制分叉，比如分支指令的分叉，return指令的分叉，和break指令的分叉。

Thread divergence and SIMT processing are orthogonal to this work as they only determine in which order SIMD instructions are executed and with what masks.

线程分化和SIMT处理与本文内容互补，因为它们只决定SIMD指令的执行顺序，和用什么masks来执行。

## 2.5 Coalescing memory accesses

Similarly to independently taking branches, each GPU thread can also independently access memory. Given the SIMD width, this amounts to 32 independent memory accesses per instruction. Each access is typically to a four-byte data. The memory system, on the other hand, processes requests at a coarser granularity of 32 to 128 byte continuous and aligned blocks. Requests for such blocks are called memory transactions. When different thread accesses in the same SIMD instruction fit in same block, they may be “coalesced” and served with one transaction. Coalescing is done in load-store units.

与独立的执行分支类似，每个GPU线程也可以独立的访问内存。给定SIMD宽度，这意味着每条指令需要有32个独立的内存访问。每个访问一般是4字节的数据。另一方面，存储系统处理的请求，则是更粗的粒度，即32到128 byte连续和对齐的数据块。对这些块的请求，称为内存事务。当相同的SIMD指令的不同的线程访问可以合并到相同的数据块时，它们可以被合并，用一个事务进行。合并是在LSU中完成的。

Below we consider a few examples of memory coalescing; in all cases we assume there are no cache hits. For the purpose of illustration we assume there are only four threads per warp and only four words per transaction. Also, we assume that memory is word-addressed and each thread accesses a single word. Each warp access in this case can be described with a tuple (a, b, c, d), where a, b, c, and d are the memory addresses used in the respective thread accesses. Each transaction can be described with a similar tuple but under additional constraints: the addresses must be sequential and aligned as in (4k, 4k+1, 4k+2, 4k+3) for some integer k.

下面，我们考虑内存访问合并的几个例子；在所有的情况中，我们假设没有cache hits。为进行描述，我们假设每个warp只有4个线程，每个事务只有4个words。而且，我们假设内存是按word寻址的，每个线程访问一个word。在这种情况中，每个warp访问可以用tuple (a,b,c,d)表示，其中a,b,c,d是在相应的线程访问中所用到的内存地址。每个事务可以用类似的tuple描述，但有额外的约束：地址必须是顺序的，对齐的，可以表示成某个整数k的表示，即(4k, 4k+1, 4k+2, 4k+3)。

The simplest case of memory coalescing is when a warp access directly corresponds to a valid memory transaction, such as accesses (0,1,2,3) and (4,5,6,7). This is also the only case when accesses are coalesced on early CUDA-capable GPUs such as G80. This trivial case is shown in Figure 2.1, left.

内存访问合并的最简单的情况是，一个warp访问直接对应着一个有效的内存事务，比如访问(0,1,2,3)和(4,5,6,7)。在早期的具备CUDA功能的GPUs上，比如G80，这也是唯一的内存访问合并的情况，如图2.1左所示。

When individual accesses in a SIMD instruction are similarly lined up but together are not aligned with transaction boundaries, an additional transaction is needed. For example, warp access (1,2,3,4) translates into two transactions: (0,1,2,3) and (4,5,6,7), as shown in Figure 2.1, center left. The extra transaction poses an additional stress on the memory system and requires an additional time for processing.

当一个SIMD指令的单个访问是类似排布的，但是放在一起与事务边缘并没有对齐，就需要额外的事务。比如，warp访问(1,2,3,4)就需要2个事务：(0,1,2,3)和(4,5,6,7)，如图2.1左中所示。额外的事务就给内存系统更多的压力，需要额外的时间来进行处理。

If individual accesses are lined up but with a stride larger than one, proportionally more transactions are needed. For example, stride-2 access (0,2,4,6) requires two transactions: (0,1,2,3) and (4,5,6,7), and stride-3 access (0,3,6,9) requires three transactions: (0,1,2,3), (4,5,6,7), and (8,9,10,11). (As above, lack of alignment may further increase the transaction count.) An example with stride-3 access is shown in Figure 2.1, center right; two accesses are coalesced into a single transaction, but two others are not.

如果额外的访问是排布好的，但是步长超过1，那么就相应的需要更多的事务。比如，步长为2的访问(0,2,4,6)需要2个事务：(0,1,2,3)和(4,5,6,7)，步长为3的访问(0,3,6,9)需要3个事务：(0,1,2,3), (4,5,6,7), 和(8,9,10,11)。（如果没有对齐，则可能需要更多的事务数量。）一个步长为3的访问例子如图2.1中右所示；两个访问合并为一个，但另外两个则没有合并。

In the worst case, a separate transaction is needed for each individual thread access. This is the case, for example, with warp access (2,4,13,11). No coalescing is possible in this case, as shown in Figure 2.1, right.

在最坏的情况下，每个单独的线程内存访问都需要一个单独的事务。比如，warp访问(2,4,13,11)就是这种情况。在这种情况下就不可能有合并，如图2.1右所示。

Memory accesses on earlier GPUs may involve additional intricacies. For example, when different accesses in a warp fit in transaction boundaries but come in a wrong order, they are coalesced on GT200 and newer GPUs, but not on G80. In the G80-type coalescing rules, warp accesses (3,2,1,0) and (1,1,1,1) both allow no coalescing and are both translated to four identical transactions: (0,1,2,3), (0,1,2,3), (0,1,2,3), and (0,1,2,3).

在早期的GPUs中的内存访问可能有更多的复杂情况。比如，当一个warp中的不同访问在事务边界内，但是顺序错误，在GT200和更新的GPUs中是合并的，但是在G80中则没有合并。在G80的合并规则中，warp访问(3,2,1,0)和(1,1,1,1)都可以不合并，都会被翻译成4个相同的事务：(0,1,2,3), (0,1,2,3), (0,1,2,3), 和(0,1,2,3)。

GPUs in the GT200 generation have another feature: they may use transactions of different sizes when processing the same instruction. Suppose that in our case both 4-word and 2-word transactions are available. Then warp access (1,2,3,4) translates into transactions (0,1,2,3) and (4,5) instead of transactions (0,1,2,3) and (4,5,6,7). This may reduce the stress on the memory system.

GT200代的GPUs有另一个特性：当处理相同的指令时，可以使用不同大小的事务。假设在我们的情况中，4-word事务和2-word事务都是可用的。warp访问(1,2,3,4)可以翻译成事务(0,1,2,3)和(4,5)，而不是事务(0,1,2,3)和(4,5,6,7)。这可能会降低内存系统的压力。

Another feature of the G80 and the GT200 GPUs is that they process warp accesses one half-warp at a time. In terms of our example, warp access (1,2,3,4) is first split into half-warp accesses (1,2) and (3,4), which are then separately translated into transactions: (0,1,2,3) in the first case and (2,3) and (4,5) in the second.

G80和GT200 GPUs的另一个特征是，它们一次处理半个warp的内存访问。在我们的例子中，warp访问(1,2,3,4)首先分解成半warp访问(1,2)和(3,4)，然后分别翻译成事务：在第一个情况中是(0,1,2,3)，第二个情况中是(2,3)和(4,5)。

This illustrates some of the basic options in coalescing memory accesses. The practical reference on coalescing rules is the CUDA C Programming Guide [NVIDIA 2010a; NVIDIA 2015].

这就描述了合并内存访问的几个基本选项。合并规则的实际参考见CUDA C编程指南。

## 2.6 Shared memory bank conflicts

GPUs have a special fast memory for sharing the current working set among threads executed at the same time. This fast memory is called shared memory. The “usual” memory, such as discussed above, is called global memory where it is necessary to differentiate. There are also local memory, constant memory and texture memory, which do not take a substantial part in this work.

GPUs有一个特殊的快速内存，用于在同时执行的线程中共享当前的working set。这个快速内存叫shared memory。通常的内存，就像上面所述的，称为global memory，这必须要进行区别。还有local memory，constant memory和texture memory，本文中不进行深入讨论。

Accesses to shared memory are processed differently. For example, they are not coalesced into larger blocks. Instead, the stored data is sliced into a number of banks, and each bank serves accesses independently, one access at a time. Ideally, different threads access different data in different banks or the same data in the same bank; in the latter case, the data is accessed once and then broadcast. Otherwise, some banks perform several data accesses, which takes an additional time; it is said, then, that the respective warp access causes bank conflicts.

对shared memory的访问是进行了不同处理的。比如，它们并没有合并成更大的blocks。相反，存储的数据被划分到几个banks中，每个bank都对访问独立进行响应，一次一个访问。理想的，不同的线程访问不同banks中的不同数据，或者同一bank中的相同数据；在后面的情况中，数据访问一次，然后进行广播。否则，一些banks进行几个数据访问，这需要额外的时间；那么，相应的warp访问被称为导致了bank冲突。

For an illustration, assume there are, again, four threads, addressing is word-based, and each thread accesses a single word. Also, assume there are only four banks. Then, the stored data is distributed in the following manner: words 0, 4, 8, … are in bank 0, words 1, 5, 9, … are in bank 1, and so on (Figure 2.2).

为了进行描述，我们再次假设有4个线程，寻址是基于word的，每个线程访问是一个word。同时，假设只有4个banks。那么，存储的书是按照下面的方式分布的：words 0, 4, 8, … 是在bank 0的，words 1, 5, 9, … 是在bank 1的，等等（图2.2）。

Any stride-1 access in this case incurs no bank conflicts; a special alignment, other than by word boundary, is not necessary. For example, accesses (0,1,2,3) and (1,2,3,4) are both processed equally fast (Figure 2.2, top left and top center).

在这种情况下，步长为1的任何访问，都不会导致bank冲突；不需要进行特殊的对齐，除了word的边界对齐。比如，访问(0,1,2,3)和(1,2,3,4)的处理是一样快的（图2.2，上左和上中）。

In contrast, stride-n accesses with even n – but not odd n – cause bank conflicts and require additional processing time. For example, warp access (0,2,4,6) is stride-2 and incurs a 2-way bank conflicts in banks 0 and 2: these two banks must serve all four accesses alone, two accesses per bank, as shown in Figure 2.2, bottom left. There are no bank conflicts if the access is stride-3, as in (0,3,6,9); this case is shown in Figure 2.2, top right. A stride-4 access causes a 4-way bank conflict; an example is (0,4,8,12), as shown in Figure 2.2, bottom center: all four thread accesses are served by bank 0. In practice, up to 32-way bank conflicts are possible.

对比起来，当n为偶数时，步长为n的访问会导致bank冲突，需要额外的处理时间。比如，warp访问(0,2,4,6)的步长为2，在banks 0和2中导致2-way bank冲突：这两个banks需要相应所有的4个访问，每个bank 2个访问，如图2.2下左。如果访问步长为3，则不会有bank冲突，如(0,3,6,9)；这种情况如图2.2上右所示。步长为4的访问，会导致4-way bank冲突；一个例子是(0,4,8,12)，如图2.2下中所示：所有4个线程访问都是由bank 0响应的。在实际中，最多可能有32 way的bank冲突。

Examples of a broadcast are warp accesses (1,1,1,1) and (1,1,1,2). They involve no bank conflicts: bank 1 serves many thread accesses, but fetches the requested data only once.

warp访问(1,1,1,1)和(1,1,1,2)，就是广播的例子。它们没有bank冲突，bank 1响应了很多线程访问，但是只取一次数据。

When different banks experience conflicts of different degrees, the largest degree determines the execution time. For example, access (1,3,5,8), shown in Figure 2.2, bottom right, incurs a 2-way bank conflict when accessing bank 1, and no conflicts when accessing banks 3 and 0. Execution time is then bound by the 2-way bank conflict.

当不同的banks有不同程度的冲突，最大的冲突程度决定了执行时间。比如，如图2.2下右的访问，(1,3,5,8)，在访问bank 1时会导致2-way bank冲突，而访问bank 3和0时则没有冲突。那么执行时间就由2-way bank冲突所限制。

Parameters of shared memory units, which process shared memory accesses, vary across GPU generations, as shown in Table 2.5. Earlier GPUs have 16 banks in each unit and process warp accesses one half-warp at a time (a half-warp is 16 threads). Newer GPUs have 32 banks per unit. Bank width is 32 bits on all but Kepler GPUs, which have 64-bit banks; these wider banks also imply more complicated conflict rules when accesses are 32-bit. (Bank width is how much data can be accessed at a time.) Each bank can finish one access every cycle on Kepler and newer GPUs, but only one access every two cycles on Fermi and earlier GPUs.

不同的GPU代，其shared memory单元的参数不同，如表2.5所示。早期的GPUs在每个单元中有16 banks，每次处理半个warp的内存访问（半个warp是16个线程）。更新的GPUs每个单元有32 banks。除了Kepler GPUs，其他GPUs的bank宽度都是32 bits；这些更宽的banks也意味着，在访问是32-bit时，会有更复杂的冲突规则。（Bank宽度是指一次可以访问多少数据。）在Kepler和更新的GPUs中，每个bank在一个周期中可以完成一次访问，在Fermi和更早的GPUs中，每2个周期才可以完成一次访问。

The reference on shared memory access performance and shared memory bank conflicts is the CUDA C programming guide [NVIDIA 2010a; NVIDIA 2015].

Shared memory访问性能和shared memory bank冲突的参考，见CUDA C编程指南。

## 2.7 Other functional units and SMs

GPUs have a number of different functional units that participate in execution (Figure 2.3). We have already cited load-store units, which take part in processing global memory accesses, and shared memory units, which process shared memory accesses. There are also CUDA cores, SFU units, double precision units, warp schedulers, memory controllers and caches.

GPUs有数个不同的功能单元，会参与运算（图2.3）。我们已经提到了LSU，会参与处理global memory访问，以及shared memory单元，会处理shared memory访问。还有CUDA cores，SFU单元，双精度单元，warp调度器，内存控制器和缓存。

CUDA cores execute basic integer and single precision floating-point operations; each core is capable of finishing one such operation per cycle. 32 such operations are needed to fully execute an instruction.

CUDA cores执行基本的整型和单精度浮点运算；每个核每个时钟周期可以完成一个这样的运算。需要32个这样的运算来完整的执行一条指令。

Special Function Units, or SFUs, execute more complicated arithmetic operations, such as reciprocations, reciprocal square roots, and transcendental functions – also only in single precision. In most cases, each SFU unit is similarly capable of finishing one such operation per cycle.

SFU执行的是更复杂的代数运算，比如倒数，倒数平方根，超越函数，也都是单精度的。在多数情况下，每个SFU也类似的可以在每个周期内完成一个这样的运算。

Double precision units execute double precision operations; we leave them out of scope of this work.

双精度单元执行双精度的运算；在本文中，我们不进行更深入的讨论。

Warp schedulers are responsible for instruction issue. Each issue cycle each scheduler may issue one or two instructions selecting from a number of warps assigned to it; a different warp can be selected each time. This important mechanism is discussed in more detail shortly below. A number of these units are assembled into one larger block – a tile – which is replicated across processor chip. The tile is called “streaming” multiprocessor, or SM. The number of units per SM varies with GPU generation (in some cases also within generation), and the number of SMs per GPU varies with the price segment the GPU is in. Table 2.6 lists these numbers for the GPUs used in this work; all of them are in the high-end segment. The number of shared memory units per SM is always one.

Warp调度器负责指令发射。每个发射周期，每个调度器可能发射1或2条指令，从几个warps中选择并指定到其上；每次可以选择一个不同的warp。这个重要的机制下面会更详细的讨论。几个这样的单元会组合成更大的block，也就是一个tile，并不断在处理器芯片中进行复制。这个tile称为流式多处理器，或SM。每个SM的单元数量，在每代GPU中都不一样（在一些情况下，每代中也可能不一样），每个GPU中的SM数量随着价格的不同而不同。表2.6列出了本文中使用的GPUs中的数量；所有GPU都是高端GPU。每个SM中的shared memory单元的数量永远是1。

Additional processing units are found in the memory system. GPU memory system starts with load-store units, which are part of each SM. These units process memory access instructions and emit the necessary memory transactions. The data is stored in the DRAM devices, which are implemented as separate chips. Access to these devices is managed by memory controllers. Each transaction is delivered to the appropriate memory controller via the interconnection network. The controllers accumulate the incoming transactions and serve them in the same or a different order for a better DRAM operating efficiency. Reordering goals include, for example, improving row buffer hit rates and reducing the number of switches between reads and writes.

内存系统中还有更多的处理单元。GPU内存系统从LSU开始，这是SM的一部分。这些单元处理内存访问指令，发射必要的内存事务。数据存储在DRAM设备中，这是独立的芯片。对这些设备的管理，由内存控制器管理。每个事务通过互联网络发送到合适的内存控制器。控制器累积到来的事务，以相同或不同的顺序进行服务，以得到更好的DRAM操作效率。比如，重排序的目标包括改进row buffer的击中率，降低读写之间切换的数量。

The memory system may also include caches: L1 cache in each multiprocessor and L2 cache in each memory controller. There are also texture caches, caches for address translation (TLBs), and, on some GPUs, read-only caches for global memory accesses.

内存系统也包括缓存：在每个多处理器中的L1缓存，和在每个内存控制器中的L2缓存。还有texture缓存，地址翻译的缓存(TLBs)，在一些GPUs上，还有global memory访问的只读缓存。

## 2.8 Clock rates

For future reference, Table 2.7 lists processor clock rates and memory pin bandwidths for the GPUs used in this work. Memory pin bandwidth is the product of memory clock rate – which is different than processor clock rate –, the number of memory data pins, and the data rate per pin. The rate per pin is two bits per cycle for the double data rate (DDR) used in graphics DRAM.

表2.7列出了本文中使用的GPUs的处理器时钟频率，和内存接口带宽。内存接口带宽是内存时钟频率（与处理器时钟频率），内存数据pins的数量，和每个pin的数据速率的乘积。每个pin的速率，对于图形DRAM中使用的DDR，是每时钟2 bits。

Newer GPUs support dynamic frequency scaling, in which case some of these clock rates may vary during execution. For example, processor clock rate may be automatically decreased when there is little arithmetic but much memory activity, or when processor temperature exceeds a certain threshold. The numbers listed in the table for such GPUs represent the top of the allowed range. To leave dynamic frequency scaling out of scope, we ensure that only these rates are used in the reported measurements. The particular technique used to achieve this is explained in §5.4.

更新的GPUs支持动态频率调节，即在执行中其时钟频率可能会变化。比如，当代数运算较少，但是内存访问行为很多时，或处理器温度超过了特定的阈值后，可能会自动降低处理器时钟频率。表中列出的这种GPUs的数字，是允许范围的最高值。我们会确保这些速率在本文中进行使用，避开动态频率调节的功能。在5.4节中会讲解使用的这种特殊技术。

## 2.9 Fine-grained multithreading

Another distinctive feature of GPU architecture is multithreaded issue, also called fine-grained multithreading. Each instruction issue unit on GPU, i.e. each warp scheduler, may choose between a large number of warps when issuing an instruction and may issue from a different warp each cycle (Figure 2.4).

GPU架构的另一个突出特征是多线程的发射，也称为细粒度多线程。GPU上的每个指令发射单元，即，每个warp调度器，在发射指令时，会在大量warp中进行选择，每个周期都可能会从不同的warp中issue（图2.4）。

Multithreaded issue is used to satisfy the concurrency needs of pipelined execution: the execution is divided into a number of pipeline stages, where each instruction (or transaction, etc.) occupies one stage at a time and moves to the next stage when the next stage is available. A number of instructions are thus executed at the same time, each at a different stage. This improves instruction execution rate without improving execution times of individual instructions.

多线程发射用于满足流水线执行的并发需求：执行是被划分为几个流水线阶段，其中每个指令（或事务，等）一次占据一个阶段，当下一个阶段可用时，就挪到下一个阶段。因此，同时会执行多条指令，每个都在不同的阶段。这会改进指令执行速度，而不用改进单条指令的执行时间。

Two instructions can be executed at the same time only if they are independent, i.e. if output of one is not an input of another. One warp can supply a limited number of independent instructions at a time. More independent instructions are provided by executing a number of warps simultaneously.

两条指令如果相互独立的话，即，一个的输出不是另一个的输入，那么就可以同时执行。一个warp一次只能提供数条独立的指令。同时执行多个warps，就可以提供更多的独立指令。

Each warp scheduler is assigned a number of warps. All of these warps are current, i.e. the state of each is equally available for instruction issue. Context switches in terms of saving and restoring warp states are therefore not needed and don’t occur in a regular execution. A similar mechanism with saving and restoring warp contexts, however, may still be used to implement advanced functionality, such as dynamic parallelism [NVIDIA 2012c] and debugging. The term context switching is also used for application contexts, such as when switching between using graphics and computational kernels; see, for example, whitepapers on the Fermi architecture [NVIDIA 2009; NVIDIA 2010b]. On all GPUs until the recent Pascal generation such switching does not involve saving warp contexts to DRAM [NVIDIA 2016].

每个warp调度器都指定了多个warps。所有这些warps都是当前可用的，即，每个的状态对于指令发射是同样可用的。上下文切换，即存储和恢复warp状态，因此是不需要的，在常规执行时不会发生。但是，类似的保存和回复warp状态的机制，仍然可以用于实现高级的功能，比如动态并行性和调试。上下文切换这个术语还用于应用上下文，比如，在使用图形核和计算核之间切换；比如，可以参考Fermi架构的白皮书。在所有GPUs上，直到最近的Pascal代，这样的切换都不涉及到将warp上下文保存到DRAM中。

The scheduler keeps track of which warps have an instruction ready for issue. If next instruction in a warp is not ready for issue, such as due to dependencies, the warp is stalled – instructions in the same warp are issued only in-order. One of the warps that are not stalled, if any, is selected each issue cycle and used to issue an instruction or two.

调度器追踪哪些warps有准备好的指令待发射。如果一个warp中的下一条指令还没有准备好发射，比如因为依赖关系，那么这个warp就stall了，同样warp中的指令都是顺序发射的。如果有的话，就把那些没有stall的warps中的一个，选择出来在每个发射周期中用于发射1或2条指令。

For example, at cycle 0 the scheduler may issue from warp 0, at cycle 1, from warp 1, etc. – not necessarily in a round-robin manner. A different warp is used independently of whether the last used warp is stalled for a long time, such as on memory access, or for a short time, such as on arithmetic operation. Moreover, there is evidence that schedulers on many GPUs cannot issue from the same warp again for a short time after each issue, even if the warp is not stalled on register dependency. We discuss this finer point in some additional detail in §3.7.

比如，在周期0，调度器可能会从warp 0发射，在周期1，从warp 1中发射，等等，不一定需要是round-robin的方式。使用不同的warp的时候，与最后使用的warp是否被stall了很长时间，比如因为等待内存访问，还是stall了很短时间，比如因为代数运算，是无关的。而且，有证据显示，很多GPUs上的调度器在每次发射后短时间内，不能再从同一warp中发射，即使这个warp没有因为寄存器依赖关系而stall。我们在3.7中更详细的讨论这一点。

Table 2.8 lists some of the basic parameters of warp schedulers for different generations of GPU architecture. The maximum number of warps per scheduler is 16 to 32. Issue is done either each processor cycle, or once in two cycles – faster on newer GPUs. Dual-issue, i.e. the capability to issue two independent instructions at a time, at the same warp scheduler, is available since the later versions of the Fermi architecture – the GPUs with “compute capability 2.1” [NVIDIA 2015]. It is not available on the Fermi GPU used in this work. Vendor’s publications disagree on whether dual-issue is available on Maxwell GPUs: the respective whitepapers suggest it is [NVIDIA 2014a; NVIDIA 2014b], but the CUDA C programming guide suggests it is not [NVIDIA 2015; Ch. G.5.1]. Another reason to believe that it is available is that vendor’s utility cuobjdump marks up instruction pairs in assembly codes for Maxwell GPUs.

表2.8列出了不同代GPU架构的warp调度器的基础参数。每个调度器的最大warps数为16到32。在每个处理器周期内，或每两个周期一次，都会进行一次发射，在更新的GPUs上会更快。双发射，即同一warp调度器每次发射两条独立指令的能力，自从Fermi架构的后期版本就有了，但是本文使用的Fermi GPU中还没有。在Maxwell GPUs上是否具有双发射的功能，不同文档给出了不同的结果：白皮书说是有的，但CUDA C编程指南说没有。但供应商的cuobjdump工具对Maxwell GPUs在汇编代码中标记了指令对，说明很可能还是具有这个功能的。

It is not uncommon to assume that issue rate on G80 and GT200 processors is limited by one instruction per 4 cycles. However, this limit is effective only for CUDA core instructions, as bound by 8 CUDA cores per scheduler and SIMD width 32; the schedulers per se can issue twice as fast [Lindholm et al. 2008, p.45].

假设在G80和GT200上，指令发射速度不大于每4周期1条指令，这是很正常的。但是，这个限制只是对CUDA core指令是有效的，因为受到每个调度器8个CUDA cores和SIMD宽度32的限制；每个se的调度器的发射速度可以达到2倍。

## 2.10 Thread blocks

As warps are executed asynchronously, an additional synchronization is needed to exchange data between warps using shared memory. To address this difficulty, GPUs provide a lightweight synchronization primitive: a barrier instruction.

因为warps是异步执行的，要在warps间用shared memory交换数据，就需要额外的同步措施。为处理这个困难，GPUs提供了一个轻量级的同步原语：barrier指令。

Suppose that there are two warps: A and B, each writes its data to shared memory and then each reads the data written by the other warp. The read in warp A, then, must be processed after the write in warp B, which is not necessarily the case if the warps are executed independently. To ensure the intended access order, the reads and writes in each warp must be separated with a barrier instruction, as shown below:

假设有2个warps：A和B，每个warp都将其数据写入到shared memory中，然后读另一个warp写入的数据。那么warp A中的读，就必须在warp B的写之后进行处理，如果warps的执行是独立的话，就不一定是这个情况。为确保是期望的访问顺序，每个warp的读和写都必须用barrier指令进行分隔，如下所示：

| warp A: | warp B: |
| ------- | ------- |
| write A data | write B data | 
| barrier | barrier |
| read B data | read A data |

When a barrier instruction is executed in one warp, the warp is stalled until a barrier instruction is executed also in another warp. This ensures that both reads are executed after both writes.

当一个warp执行了一条barrier指令，这个warp就stall了，直到在另一个warp中也执行了一条barrier指令。这确保了两条读指令是在两条写指令之后执行的。

More generally, each warp is stalled until a barrier instruction is executed in every other warp of the same thread block, where thread block is a group of warps designated at the kernel launch time. All executed warps are divided into thread blocks; warps in the same thread block are executed on the same multiprocessor, access the same shared memory unit, and can be inexpensively synchronized. Different thread blocks are assigned to the same or different multiprocessors, but in either case access separate shared memory partitions.

更一般的，每个warp都stall了，直到同一thread block中的每个其他warp都执行了barrier指令，这里thread block是一组warps，在kernel启动的时候指定的。所有执行的warps都分为thread blocks；在相同thread block中的warps在相同的多处理器上执行，访问相同的shared memory单元，可以很容易的同步。不同的thread blocks指定到相同的或不同的多处理器上，在不管在哪种情况下，访问的都是不同的shared memory partitions。

## 2.11 Occupancy

Execution on a GPU starts with a kernel launch, where a GPU program (i.e. kernel) and its launch configuration are specified. The launch configuration includes the number of thread blocks to be executed, the number of warps per thread block, and the size of shared memory partition per thread block.

在GPU上的执行以kernel启动开始，这里要指定一个GPU程序（即kernel）及其启动配置。启动配置包括要执行的thread blocks数量，每个thread block中的warps数量，每个thread block中的shared memory partition的大小。

It is a common and recommended practice to launch many more thread blocks than can be executed at the same time. When all warps in one thread block complete, the next thread block in line is started in its place. The number of thread blocks executed at the same time depends on how much shared memory is allocated per thread block, and how many registers are used in each warp.

推荐启动的thread blocks数量，要比可以同时执行的数量要多很多，这是很常见的情况。当一个thread block中的所有warps完成后，队列中的下一个thread block就启动了。同时执行的thread blocks数量，依赖于每个thread block分配了多少shared memory，以及每个warp中使用了多少寄存器。

The number of warps executed at the same time is one of the key metrics characterizing execution on GPU; this metric is called occupancy in this text. Occupancy is also the number of warps executed at the same time divided by the maximum number of warps that can be executed at the same time. The second definition is prevalent elsewhere. We use occupancy in both meanings. The latter occupancy is quoted in percent and is, therefore, relative occupancy. The former occupancy is quoted in warps per SM, or warps per scheduler, and is absolute occupancy.

同时执行的warps数量，是GPU的关键度量；这个度量在本文中称为占用率occupancy。Occupancy同时也是，同时执行的warps数量，除以同时执行的warps的最大数量。第二个定义在其他地方更常用。这两种意义我们都使用。后面的occupancy以百分比为单位，因此是相对占用率。前面的occupancy以每个SM中的warps数量为计，或者每个调度器的warps，是绝对占用率。

Table 2.9 lists a few examples of the correspondence between relative and absolute occupancies for different GPU generations. For example, occupancy 25% on a Maxwell GPU is the same as occupancies 16 warps/SM and 4 warps/scheduler; occupancy 25% on a G80 GPU is the same as occupancies 6 warps/SM and 6 warps/scheduler.

表2.9列出了不同GPU中的绝对占用率与相对占用率之间的关系。比如，在Maxwell GPU上占用率25%，就是16 warps/SM，4 warps/scheduler；在G80 GPU上的25%占用率，就是6 warps/SM，6 warps/scheduler。