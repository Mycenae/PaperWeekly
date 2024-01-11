# Chapter 3 Modeling latency hiding

The contributions of this chapter are the following. In §3.3 we point to an important oversight in prior work on GPU performance modeling, which is understanding the difference between occupancy and instruction concurrency. In §3.5 we suggest a simple performance modeling framework that doesn’t have this oversight. In §3.9 we discuss what it means to have latency “hidden”, correcting the view implied in vendors’ programming guides. Finally, in §3.10 we show how to use Little’s law to make quick intuitive judgements on GPU performance, such as on how the occupancy needed to attain a better throughput depends on various practical factors.

本章贡献如下。3.3节中，我们指出了之前GPU性能建模中的一个重要疏忽，即对占用率和指令并发率的差异的理解。在3.5中，我们提出了一个简单的性能建模框架，解决了这个问题。在3.9中，我们讨论了将延迟隐藏掉是什么意思，纠正了供应商编程指南中的观点。最后，在3.10中，我们展示了怎样利用Little定律来对GPU性能进行快速的直觉的判断，比如在各种实际因素下，需要怎样的占用率才能得到更好的吞吐量。

We start with introducing basic concepts, such as latency, throughput and Little’s law, which are then developed into instruction latency, instruction throughput, warp latency, and warp throughput.

我们开始先介绍一下基本的概念，比如，延迟，吞吐量，和Little定律，然后演化出指令延迟，指令吞吐量，warp延迟，和warp吞吐量。

## 3.1 Latency, throughput and Little’s law

In serial processing, the total execution time of a program can be broken up into a sum of execution times of individual instructions. In concurrent processing, in contrast, execution times of individual instructions may overlap and not sum to the total execution time. To address this difficulty, it is convenient to use other concepts than execution time, such as latency, throughput and concurrency. 

在顺序处理中，一个程序的总计执行时间，可以分解为单条指令的执行时间之和。对比起来，在并发处理中，单条指令的执行时间可能会重叠，加起来就不是总计执行时间了。为处理这个问题，就需要使用其他的概念了，比如延迟，吞吐量和并发，而不是执行时间了。

Consider a generic concurrent activity, such as shown in Figure 3.1. It is comprised of a number of items, which can be instructions, memory transactions, warps, etc. Each item is defined with two numbers: its start time, and its end time. Separately given is the time interval during which the activity occurs, which is between cycle 0 and cycle 10 in this case. To characterize the entire process, we may use the following aggregate metrics.

考虑一个通用的并发行为，如图3.1所示。它由数个items组成，可以是指令，内存事务，warps，等等。每个item由两个数定义：其起始时间，和结束时间。同时还给出了这些行为发生的时间间隔，在这个情况中，是在周期0和周期10之间。为刻画整个过程，我们可以使用下列的总计的度量。

The first metric is mean latency. This is the average of latencies of individual items, where latency of an individual item is the difference between its start time and its end time. In the figure, individual latencies are 1, 2, or 3 cycles, and mean latency is 2 cycles.

第一个度量是平均延迟。这是单个item的延迟平均，单个item的延迟是其结束时间与起始时间之差。在这个图中，单个的延迟是1，2，或3个周期，平均延迟是2个周期。

Another metric is throughput, or processing rate. It is defined as the number of items that fall within the given time interval divided by the duration of the interval. In this figure, it is 10 items over 10 cycles, or 1 item per cycle. We also often quote reciprocal throughput, which is 1 cycle per item in this case.

另一个度量是吞吐量，或处理速度。这定义为，在给定时间间隔内的items的数量，除以间隔的时长。在这个图中，就是10个items除以10个周期，或每周期1个item。我们还通常引用倒数吞吐量，在这个情况中，就是每个item耗时1个周期。

Concurrency is the number of items processed at the same time. It is defined for each particular moment of time and also as an average over the interval. In this example, concurrency varies between 1 and 3 items, and averages to 2 items.

并发度是同时处理的items数量。这个定义包括对每个特定时刻的，和在整个区间的平均。在这个例子中，并发度在1 item和3 items之间变化，平均为2 items。

These metrics are related via Little’s law (see, for example, Little [2011]): 这些度量通过Little定律关联起来：

mean concurrency = mean latency × throughput. 平均并发度 = 平均延迟 × 吞吐量

For this relation to hold, it is important that all of these metrics are derived from the same set of start and end times, the same time interval is used to define both throughput and mean concurrency, and all items fall entirely within the interval. Little’s law does not necessarily hold if some of the items do not fully fall within the interval – see Little and Graves [2008] for a discussion. But even in that case the relation may still apply, at least approximately, if the fraction of such items is small. Little’s law is also used with infinite processes, for which case it was proven in Little [1961].

这个关系要成立，所有这些度量要都从相同的开始时间和结束时间之间推导出来，相同的时间区间用于定义吞吐量和平均并发度，所有的items都要在区间之内，见Little and Graves [2008]中的讨论。但即使在这些情况中，这些关系仍然可能成立，至少是近似的，如果这些items的部分是很小的。Little定律也用于无限过程，这种情况下在Little[1961]中得到了证明。

## 3.2 Instruction latency and throughput

Little’s law is often used to estimate concurrency that is needed to attain a given throughput in executing any particular kind of instruction, such as arithmetic instructions or memory instructions(Table 3.1).

Little定律经常用于估计并发度，在执行任意类型指令时（如算术运算指令，或内存指令，表3.1），要得到给定的吞吐率，就需要并发度。

Consider a concurrent activity where items are instructions. Latency, throughput and concurrency in this case correspond to, respectively, instruction latency, instruction throughput and the number of instructions executed at the same time. Instruction latency is usually understood as register dependency latency. This is the part of the total instruction processing time that starts when an instruction is issued and ends when a subsequent register-dependent instruction can be issued. Register-dependent instructions are defined as instructions that use register output of a given instruction as input. This defines the start and end times for instruction execution, and thus also defines the respective throughput and concurrency metrics.

考虑一个并发行为，其中的items就是指令。延迟，吞吐率和并发度，在这种情况下就分别对应着，指令延迟，指令吞吐率，和同时执行的指令数量。指令延迟通常理解为寄存器依赖关系延迟。这就是指令的总计处理时间，从一条指令发射开始，到后续的依赖于寄存器的指令可以发射为止。依赖于寄存器的指令定义为，使用指定指令的寄存器输入作为输入的指令。这定义了指令执行的开始和结束时间，因此也定义了相应的吞吐率和并发度度量。

Little’s law in this case reads, if reordering the terms: 在这种情况下的Little定律如下所示：

instruction throughput = instructions in execution / instruction latency.

For example, the latency of basic arithmetic instructions on the Maxwell GPU is 6 cycles. Executing 1 such instruction at a time results in throughput of 1/6 instructions per cycle (IPC), executing 2 such instructions at a time, in throughput of 1/3 IPC, executing 3 such instructions at time, in throughput of 1/2 IPC, etc. We also may consider executing 0.5 such instructions at a time on average, such as if executing 1 such instruction at a time half of the time and 0, another half of the time. This would result in throughput equal 1/12 IPC. In all of these examples we assume that latency itself is constant and does not depend on concurrency.

比如，在Maxwell GPU上，基础算术指令的延迟为6周期。一次运行一条这样的指令，得到的吞吐率为1/6 IPC，一次运行两条这样的指令，吞吐率为1/3 IPC，一次运行3条这样的指令，得到吞吐率为1/2 IPC，等等。我们还可以考虑，平均每次执行0.5条这样的指令，比如在一半的时间内一次执行一条这样的指令，剩下的一半时间内不执行这样的指令。这样得到的吞吐率为1/12 IPC。在所有这些例子中，我们假设延迟本身都是常数，并不依赖于并发度。

Instruction throughput is limited by the capability of the hardware. For example, Maxwell GPUs have only 128 CUDA cores per SM and, as a result, can execute arithmetic instructions not faster than 4 IPC per SM – since 1 instruction does 32 arithmetic operations. This bound is called peak throughput. Instruction latency and the respective peak throughput are basic hardware parameters that characterize instruction execution. They depend on instruction type, but may also depend on bank conflicts, memory coalescing and other factors.

指令吞吐率受硬件能力限制。比如，Maxwell GPUs每个SM只有128个CUDA cores，由于1条指令要进行32个算术运算，所以执行算术指令时，不会快过每SM 4 IPC。这个限制称为峰值吞吐率。指令延迟和对应的峰值吞吐率，是基本的硬件参数，刻画了指令执行。它们依赖于指令类型，但可能也依赖于bank冲突，内存访问合并，和其他因素。

It is easy to see that attaining peak throughput in this example requires executing 24 instructions per SM at the same time, where all instructions are implied to be the basic arithmetic instructions executed by CUDA cores. This result can be formally found by using Little’s law: we multiply instruction latency, which is 6 cycles, by peak throughput, which is 4 IPC per SM, and get 24 instructions per SM. Note that by doing so we assume that this mean latency and this throughput are sustained at the same time.

可以很容易看出，在这个例子中要得到峰值吞吐率，需要同时在每个SM上执行24条指令，其中所有指令都要是基本算术指令，由CUDA cores执行。这个结果可以使用Little定律得到：我们将指令延迟，即6周期，乘以峰值吞吐率，即每SM 4 IPC，得到每个SM 24条指令。注意，在这样做时，我们假设这个平均延迟和这个吞吐率是同时得到的。

A similar estimate can be also used for memory accesses [Bailey 1997]. Latency of memory access instructions on the Maxwell GPU is about 368 cycles – this is if accesses miss cache, are 32-bit, are fully coalesced and only one or a few such instructions are executed at the same time. For peak throughput it is common to use pin bandwidth, which is 224 GB/s on this processor. This rate corresponds to instruction throughput equal to 0.086 IPC per SM: to convert we divide by the clock rate (1.266 GHz), the number of SMs (16) and the number of bytes requested per instruction (128). Substituting this peak throughput and the above latency into Little’s law suggests that 33 such memory access instructions must be executed at the same time per SM to attain the peak.

对内存访问可以得到类似的估计。Maxwell GPU上内存访问指令的延迟大约为368周期，这是假设访问cache miss，是32-bit的，是完全合并的，同时只有1个或几个这样的指令在执行。对于峰值吞吐率，通常使用pin带宽，在这个处理器上，是224GB/s。这个速率对应到指令吞吐率为每SM 0.086 IPC：我们将224GB/s，除以时钟速率1.266GHz，除以SM数量16，除以每条指令请求的bytes数量128，就可以转换得到。将这个峰值吞吐率和上面的延迟替换到Little定律中，说明要得到这样的峰值吞吐率，每个SM上要同时执行33条这样的内存访问指令。

Complexity of the memory system creates a few extra difficulties. First, pin bandwidth is only an approximation of peak memory throughput – the best throughput that can be sustained in practice is noticeably less, such as by 3 to 20%, depending on GPU, according to our findings in Chapter 6. For this reason, we don’t use pin bandwidths in this work and instead use peak throughputs found experimentally. On the Maxwell GPU, this throughput is found to be 211 GB/s, which corresponds to 0.081 IPC per SM. Substituting this number into the concurrency estimate above produces a new estimate equal 30 instructions per SM.

内存系统的复杂性，会得到一些额外的复杂度。首先，pin带宽只是峰值内存吞吐率的一个近似，在实际中可以持续的最佳吞吐率要少的多，根据我们在第6章的发现，大约在3%-20%，不同的GPU有不同的结果。因此，我们在本文中不使用pin带宽，而使用通过试验得到的峰值吞吐率。在Maxwell GPU上，我们发现这个吞吐率为211GB/s，这对应着每SM 0.081 IPC。将这个数字替换到并发度计算公式中，会得到每SM大约30条并发指令的结果。

Another difficulty is with memory latency. As we show in Chapter 6, memory latency is not constant, but varies from one individual instruction to another, and its average, i.e. mean latency, tends to be larger when throughput is larger. Using unloaded latency and peak memory throughput in Little’s law at the same time, therefore, does not correspond to a realistic execution scenario and serves only as an approximation. We keep our estimates approximate in this respect, assuming, for simplicity, that memory latency is constant relative to concurrency. This is a common practice.

另一个困难是内存访问延迟。我们在第6章中会展示，内存访问延迟并非常数，而是不同的指令有不同的延迟，其平均，即平均延迟，在吞吐率大的时候也会更大。在Little定律中使用无负载的延迟，和峰值内存吞吐率，因此并不对应着实际的执行场景，只能作为一个近似。我们保持我们的估计近似，也假设内存访问延迟与并发度没有相关性。这是常用的假设。

Latency and peak throughput of memory access instructions depend on the memory access pattern. If accesses are not coalesced, instruction latency is larger due to issuing additional transactions, and peak instruction throughput is smaller due to transferring more data through the memory system.

内存访问指令的延迟和峰值吞吐率依赖于内存访问的模式。如果内存访问没有合并的话，那么指令延迟就会更大，因为要发射额外的事务，峰值指令吞吐率就会更小，因为要通过内存系统传递更多数据。

## 3.3 Instruction concurrency vs occupancy

Throughput and concurrency estimates similar to the above are found in many prior GPU performance models – but with one important difference: the concurrency is understood, with some reservations, as the number of concurrently executed warps, not the number of concurrently executed instructions. This prior work includes CUDA C programming guide [NVIDIA 2015], and performance models by Hong and Kim [2009], Baghsorkhi et al. [2010], Sim et al. [2012] and Song et al. [2013].

与上述类似的吞吐率和并发度估计，在很多之前的GPU性能估计模型中都能找到，但有一个重要的区别：并发度理解为并发执行的warps的数量，而不是并发执行的指令的数量。这些之前的工作包括CUDA C编程模型，和Hong and Kim [2009], Baghsorkhi et al. [2010], Sim et al. [2012] and Song et al. [2013]的性能模型。

There are at least two important reasons why the number of concurrently executed warps may be different than the number of concurrently executed instructions. These are instruction-level parallelism, which is well understood in prior work, and instruction mix, which is not.

并发执行的warps数量与并发执行的指令数量是不同的，这至少有2个重要的原因。这是指令级的并行，这在之前的工作中理解的很好，还有指令混合，这个则没有很好的理解。

Having instruction-level parallelism, or ILP, in a code means that the code permits executing more than one instruction at the same time from the same warp. In particular, this refers to register dependencies: two instructions may be executed at the same time only if register output of one instruction is not used as an input in the other instruction. Instruction-level parallelism is common in practice, and we consider a simple example that includes it – vector add kernel – later in this chapter.

代码中有指令级并行，即ILP，意味着同一warp中同时可以执行多于一条指令。特别的，这是指寄存器依赖关系：两条指令可以同时执行，只有在一条指令的输出并不会用作另一条指令的输入。指令级并行在实践中很常见，我们本章稍后考虑一个简单的例子，矢量加法kernel。

ILP allows attaining the same instruction concurrency using fewer warps. For example, if on average 2 memory access instructions are executed in each warp at the same time, instruction concurrency equal to 30 memory instructions per SM may be attained by using 15 warps per SM. This effect is recognized in performance models by Sim et al. [2012] and Baghsorkhi et al. [2010], and is also noted in the CUDA C programming guide [NVIDIA 2015; Ch. 5.2.3]. Volkov [2010] presents a number of illustrations of this effect and its use in code optimization.

ILP使得可以使用更少的warps得到相同的指令并发度。比如，如果每个warp同时平均能执行2条内存访问指令，指令并发度为每SM 30条内存访问指令，可以通过每SM 15 warps得到。这种效果在Sim et al. [2012] and Baghsorkhi et al. [2010]的性能模型中得到承认，在CUDA C编程指南中也提到。Volkov [2010]提出了这种效果的几个描述，以及在代码优化中的用处。

Another factor responsible for mapping between warps and instructions is instruction mix. It refers to the proportion in which different instruction types are present in the code. When several instruction types are present in the same code, which is the common case, executing an instruction of one type in a warp typically implies not executing instructions of other types in the same warp at the same time. In result, the number of concurrently executed instructions of any particular type may be smaller than the number of concurrently executed warps. This effect is overlooked in all previously published work that we know. This includes the occupancy estimate in Song et al. [2013], Section IV.E, a similar metric MWP_peak_bw in models by Hong and Kim [2009] and Sim et al. [2012], and throughput estimates in Baghsorkhi et al. [2010], Zhang and Owens [2011], and Sim et al. [2012]. A detailed study of prior performance models is presented later in this text.

对warps和指令的映射有责任的另一个因素是，指令混合。这是指不同类型的指令都在代码中。当几种指令类型都在同一代码中，这是常见的情况，在一个warp中执行一种类型的一条指令，通常意味着在同一warp中同时不会执行其他类型的指令。结果是，任何类型的并发执行的指令数量，通常都会小于并发执行的warps数量。在之前发表的工作中，都忽略了这个效果。这包括Song et al. [2013]的使用率估计，Hong and Kim [2009] and Sim et al. [2012]，Baghsorkhi et al. [2010], Zhang and Owens [2011], and Sim et al. [2012]等。本文后面会给出详细的之前的性能模型。

Consider an example shown in Figure 3.2. It includes instructions of two types: arithmetic instructions and memory instructions. The instructions are present in such proportions that each warp has one arithmetic instruction in execution half of the time and one memory instruction in execution another half of the time. Thus, given 3 warps, the number of arithmetic instructions in execution is only 1.5 on average, and the number of memory instructions in execution is also only 1.5 on average. Attaining the concurrency equal 30 memory instructions in this case requires executing 60 warps at the same time – compared to 15 warps in the example with ILP. Even a stronger effect may be expected if kernel includes more than two instruction types, such as four. These may be, for example, CUDA core instructions, SFU instructions, global memory access instructions and shared memory access instructions – all executed in different functional units and having different latencies and peak throughputs.

考虑图3.2所示的例子。这包括两种类型的指令：算术指令和内存访问指令。指令的比例为，每个warp的一半时间来执行算术指令，另一半来执行内存访问指令。因此，给定3个warps，平均只有1.5条在执行的算术指令，在执行的内存访问指令的数量也是平均1.5条。要得到30条内存访问指令的并发度，需要同时执行60个warps。相比起来，在有ILP的情况下，只需要15个warps。如果kernel包含的指令类型数更多，那么就会有更强的效果。比如，这可能是，CUDA core指令，SFU指令，global memory访问指令，shared memory访问指令，这都在不同的功能单元中执行，有着不同的延迟和峰值吞吐率。

To summarize, the number of warps needed to attain a peak throughput – unlike the needed number of instructions – depends not only on hardware parameters, such as instruction latencies and peak throughputs, but also on executed code, such as register dependencies and the instruction mix.

总结起来，要达到峰值吞吐率所需的warps数量，不仅依赖于硬件参数，比如指令延迟和峰值吞吐率，还依赖于被执行的代码，比如寄存器依赖关系和指令混合。

## 3.4 Warp latency and throughput

To establish a connection between occupancy and kernel performance, we introduce a few intermediate metrics, which are directly connected with occupancy via Little’s law. These are warp latency and warp throughput.

为确定占用率和kernel性能之间的关系，我们引入几个中间度量，通过Little定律来与占用率直接产生关系。这就是warp延迟和warp吞吐率。

Consider a concurrent process where items are warps. Each item in this case is defined by start time and termination time of the respective warp. The concurrency metric, then, corresponds to the number of warps executed at the same time, which is occupancy. The throughput metric corresponds to the total number of executed warps divided by the total execution time, which is warp throughput. The difference between start time and termination time of an individual warp corresponds to warp latency. The metrics are connected via Little’s law as:

考虑一个并发进程，其中的items是warps。这种情况下，每个item的定义是对应warp的开始时间和结束时间。并发度度量对应着同时执行的warp数量，也就是占用率。吞吐率度量对应着执行的warps总数，除以总计执行时间，这就是warp吞吐率。单个warp的开始时间与结束时间之差，就是warp延迟。这些度量通过Little定律联系到一起：

mean occupancy = mean warp latency × warp throughput. 平均占用率 = 平均warp延迟 × warp吞吐率

An example of such a concurrent process is shown in Figure 3.3. The shown data corresponds to an execution of a particular kernel on the Kepler GPU and is recorded as explained in §5.5. Individual warps are shown as horizontal lines. Length of a line corresponds to the respective warp latency, and the number of lines crossing a vertical section, to occupancy. Examples of these metrics are shown in the figure. Plotted are only a part of the total execution time, and only the warps that were executed on a particular SM.

这样一个并发过程的例子如图3.3所示。展示的数据对应着一个特定的kernel在Kepler GPU上的执行，在5.5节中进行了解释。单个warp如横线所示。线段长度对应着相应的warp延迟，跨越一个垂直截面的线段数量，对应着占用率。这些度量的例子如图所示。画出来的是总计执行时间的一部分，并且是在特定SM上执行的warps。

According to the figure, latencies of individual warps may differ. This is despite executing the same number of instructions in each warp, which is the case here. A possible explanation for the difference is the variance in latencies of individual memory access instructions; we study this variance in more detail in §6.6.

根据图示，单个warps的延迟是不一样的。在这个例子中，每个warp执行的是相同数量的指令，但也会这样。一个可能的解释是，单个内存访问指令的延迟是不一样的；我们在6.6中研究了这种变化。

Also noticeable in the figure is a variance in occupancy. This is a result of the differences in warp latencies: warps in the same thread block terminate at different times, but new warps may be assigned in their place only simultaneously. There are also other factors contributing to occupancy variance; we review some of them in Chapter 5. Occupancy variance is usually not taken into account in performance modeling, and we also consider it only to a limited degree.

这个图中还值得注意的是，占用率的变化。这是warp延迟变化的结果：在相同thread block中的warps会在不同的时刻结束，但新的warps会同时被指定到其中。还有其他的因素对占用率变化有贡献；我们在第5章中进行了部分研究。占用率变化通常不会在性能建模中考虑，我们也只在一定程序上进行了考虑。

With these metrics, we formulate the problem of performance modeling as finding warp throughput given occupancy and other parameters. Having found warp throughput, other integral performance metrics are also easily found. For example, the overall instruction throughput is found as the average number of instructions executed per warp multiplied by warp throughput, and the total execution time, as the total number of executed warps divided by warp throughput.

有了这些度量，我们将性能建模的问题表述为，在给定占用率和其他参数的情况下，找到warp吞吐率。找到了warp吞吐率，其他性能度量也可以很容易找到。比如，整体指令吞吐率，就是每warp执行的平均指令数量，乘以warp吞吐率，总计执行时间，就是总计执行的warps数量，除以warp吞吐率。

## 3.5 Two basic performance modes

We divide the problem of estimating warp throughput into two independent subproblems. First, we find a bound on warp throughput that is due to processor throughput limits. Second, we find a bound on warp latency that is due to processor latencies. The bounds are then merged using Little’s law to produce the estimate. The two bounds correspond to two distinct performance modes.

我们将估计warp吞吐率的问题，分解成两个独立的子问题。第一，我们找到一个受处理器吞吐率限制影响的warp吞吐率的界限。第二，我们找到一个受处理器延迟影响的warp延迟的界限。这些界限使用Little定律合并，以得到估计。这两个界限对应着两种不同的性能模式。

We write the bound on warp throughput in a generic form: 我们用通用形式写出warp吞吐率的界限：

warp throughput ≤ throughput bound. 

This bound is due to hardware limits on throughput: the number of CUDA cores and other functional units, peak issue throughput, peak throughput of memory system, etc. Unlike the bounds on instruction throughputs, this bound also depends on executed code. Its estimation is discussed in §3.6.

这个界限是硬件上对吞吐率的限制：CUDA cores的数量，和其他功能单元的数量，峰值发射吞吐率，内存系统峰值吞吐率，等等。与指令吞吐率的界限不同，这个界限还依赖于被执行的代码。其估计在3.6节中进行讨论。

Another factor that may contribute to execution time is processor latencies. It imposes a lower bound on latency of each individual warp and, similarly, a lower bound on mean warp latency:

另一个对执行时间有贡献的因素，就是处理器延迟。这对每个warp的延迟加上了一个下限，以及平均warp延迟的下限：

mean warp latency ≥ latency bound.

For example, register dependency latencies limit how soon a dependent instruction can be issued from the same warp; another latency parameter limits how soon an independent instruction can be issued from the same warp. There are latencies specific for memory access instructions and a latency associated with replacing a retired thread block. We detail how they contribute to the bound on warp latency in §3.7.

比如，寄存器依赖关系延迟限制了，一个有依赖关系的指令从相同的warp中要多久才能发射；另一个延迟参数限制了，一个独立的指令从相同的warp要多久才能发射。这就是用于内存访问指令特定的延迟，和与替代退休的thread block相关的延迟。我们在3.7中详细讨论它们怎样对界限进行贡献。

We reduce the bound on mean warp latency to a bound on warp throughput by using Little’s law. This produces: 我们使用Little定律，将对平均warp延迟的界限，转化为对warp吞吐率的界限，这就得到了：

warp throughput ≤ occupancy / latency bound .

Note that a similar reduction is not possible with a bound on individual warp latencies: the averaging is important. 注意，对单个warp延迟的界限则不能这样估计：平均是很重要的。

Putting the two bounds together we get: 将这两个界限放到一起，我们得到：

warp throughput ≤ min ( occupancy / latency bound , throughput bound ) .

If there were no other factors contributing to kernel execution time, we would expect this combined bound to be an accurate estimate of warp throughput. But otherwise we expect it to be an approximate estimate. For the performance estimate we therefore use:

如果没有其他因素对kernel执行时间贡献了，我们就会认为，这个界限是warp吞吐率的一个精确估计。但否则，我们就认为是一个粗略的估计。对于性能估计，我们因此使用：

warp throughput ≈ min ( occupancy / latency bound , throughput bound ) ,

which summarizes our performance model at a high level. 这在较高的层次上总结了我们的性能模型。

This model has two modes shown in Figure 3.4. When occupancy is small, throughput is linear in occupancy and depends on processor latencies: this is latency-bound mode. When occupancy is large, throughput is fixed and depends on processor throughput limits: this is throughput-bound mode.

这个模型有两种模式，如图3.4所示。当占用率较小时，吞吐率与占用率是线性关系，依赖于处理器延迟：这是延迟界限的模式。当占用率很大时，吞吐率是固定的，依赖于处理器的吞吐率限制：这是吞吐率界限的模式。

In practice, throughput usually doesn’t grow linearly all the way to a maximum. This is due to interference between warps, which is not included in the bounds. For example, when two warps are executed at the same scheduler, both warps may have an instruction ready for issue, but only one can be selected at a time. Another warp, therefore, is delayed, which increases its latency and increases mean warp latency. For a similar reason, mean memory latency tends to increase when memory concurrency increases. Due to these additional delays, which are not part of the model, we might expect warp throughput to fall somewhat short of the estimate in both latency-bound mode and throughput-bound mode. This blurs the boundary between the modes so that one gradually translates into another, as schematically shown in the same figure. We call this “the gradual saturation effect”. We consider it the next level of accuracy and leave it out of scope until §6.4.

在实践中，吞吐率通常不会一直线性增长直到最大值。这是因为warp之间的干扰，这并没有包括在界限中。比如，当两个warps在相同的调度器上执行，两个warps可能都有一条指令准备好了可以发射，但一次只能选择一个。因此，另一个warp的发射就被推迟了，这就增加了其延迟，以及平均warp延迟。由于类似的原因，当内存访问并发度增加时，平均内存访问延迟也会增加。由于这些额外的推迟（这些并不是模型的一部分），我们可能会期望warp吞吐率会比估计稍微低一些，在延迟界限模式，和吞吐率界限模式中都会这样。这使得两种模式的边界变模糊，一种模式逐渐编程另一种，在图中也有展示。我们称之为逐渐饱和效果。我们认为这是下一级的准确率，将其放到6.4中讨论。

The model suggests that throughput may never decrease when occupancy increases. This is not always the case; it may decrease, for example, if cache hit rate decreases. This framework can be extended to cover such less common cases, but instead we choose to focus on more basic behaviors, which are yet not sufficiently well understood in prior work.

模型说明，随着占用率增加，吞吐率永远不会降低。并不永远是这种情况；比如，如果cache hit率降低，就可能会降低。这个框架可以拓展，以覆盖这种不太常见的情况，但我们选择聚焦在更基础的行为上，这在之前的工作中尚未充分理解。

Similar bounds are discussed in classical work on concurrent systems such as Lazowska et al. [1984], Chapter 5.2, and Jain [1991], Chapter 33.6. Hong and Kim [2009] also separately consider what we call latency-bound and throughput-bound modes, but disregard arithmetic latency. This latency is important as we show later in Chapter 4. Sim et al. [2012] improves Hong and Kim model to include arithmetic latency and other factors, but the improved model has other limitations and is not similar to our approach. Huang et al. [2014] suggest a model that includes a similar latency-bound solution, but doesn’t include similar throughput bounds. Saavedra-Barrera et al. [1990], Chen and Aamodt [2009], and Huang et al. [2014] model the gradual saturation effect but not some of the other factors that are important on GPUs. The Roofline model [Williams et al. 2009] includes a number of throughput bounds but doesn’t provide a specific solution for the latency-bound case. Limiter theory [Shebanow 2008] considers similar throughput bounds, but does not consider latency bounds, at least not explicitly.

并发系统的经典工作也讨论了类似的界限，如Lazowska et al. [1984], Chapter 5.2, and Jain [1991], Chapter 33.6。Hong and Kim [2009]也分别考虑了我们称为延迟界限和吞吐率界限的模式，但没有考虑算术延迟。这个延迟是很重要的，我们在第4章进行了展示。Sim et al. [2012] 改进了 Hong and Kim的模型，包含了算术延迟和其他因素，但改进的模型有其他的限制，与我们的方法不太类似。Huang et al. [2014]提出了一个模型，包含了类似的延迟界限方法，但并没有包含类似的吞吐率界限。Saavedra-Barrera et al. [1990], Chen and Aamodt [2009], and Huang et al. [2014]对渐进饱和效果进行了建模，但对一些在GPU上很重要的因素并没有包含。Roofline模型包含了几个吞吐率界限，但并没有为延迟界限的情况提供方法。[Shebanow 2008]的Limiter理论考虑了类似的吞吐率界限，但并没有考虑延迟界限。

## 3.6 Throughput bound

To find the throughput bound we consider all hardware throughput limits one by one and convert each to a respective bound on warp throughput by taking into account instruction mix.

为找到吞吐率界限，我们一个一个考虑了所有的硬件吞吐率限制，将每个都转换到对warp吞吐率分别的界限，还考虑了指令混合。

Consider an example detailed in Tables 3.2, 3.3 and 3.4. Table 3.2 details instruction mix of a hypothetical kernel; the numbers are averages across executed warps. Table 3.3 details hardware throughput limits; they correspond to the Maxwell GPU. Table 3.4 details the resulting bounds on warp throughput and the involved cycle-counting.

考虑一个例子，如表3.2，3.3，3.4所示。表3.2详述了一个假设的kernel的指令混合情况；这些数值是在执行的warps上平均的。表3.3详述了硬件吞吐率的限制；这对应着Maxwell GPU。表3.4详述了得到的warp吞吐率限制，和涉及到的周期数计数。

The first throughput limit we consider is due to the limited number of CUDA cores. There are 128 CUDA cores per SM, which limits throughput of CUDA core instructions to 4 IPC per SM, or 0.25 cycles per instruction at each SM. The kernel has 100 such instructions per warp, which therefore limits warp throughput to 25 cycles per warp at each SM.

我们考虑的第一个吞吐率限制，是因为CUDA cores数量的限制。每个SM有128个CUDA cores，这限制了CUDA core指令的吞吐率为每SM 4 IPC，或在每个SM上每条指令0.25周期。Kernel在每个warp上有100条这样的指令，因此限制了warp吞吐率为每SM上每warp为25个周期。

Similarly, given 32 SFU units per SM and 5 SFU instructions per warp, the bound on instruction throughput is 1 cycle per SFU instruction, and the respective bound on warp throughput is 5 cycles per warp at each SM.

类似的，每个SM有32个SFU单元，每个warp有5条SFU指令，指令吞吐率的限制为每个SFU指令1周期，对应的warp吞吐率限制为每个SM每warp 5周期。

The calculations are slightly more complicated if instruction throughput limit depends on other factors, such as memory access pattern. Consider the bound due to the limited number of shared memory banks. There are 32 of them per SM. We model them as a single resource that is exclusively used for 1 cycle by each shared memory access instruction with no bank conflicts, and for 2 cycles by each shared memory access instruction with 2-way bank conflicts. There are 10 instructions of both kinds per warp in this kernel, which sum up to 30 cycles of exclusive use. The respective throughput bound, therefore, is 30 cycles per warp at each SM.

如果指令吞吐率限制依赖于其他因素，比如内存访问模式，那么这个计算就更加复杂。考虑由于shared memory banks数量限制造成的界限。每个SM有32个shared memory banks。我们将其建模为单个资源，由shared memory访问指令在1个周期内进行独占性的访问，没有bank冲突，对于有2-way bank冲突的shared memory 访问指令，需要2个周期。在这个kernel中，每个warp有10条这样的指令，对独占性使用，共计30个周期。因此，对应吞吐率界限，是每个SM每warp 30个周期。

Global memory accesses, both coalesced and non-coalesced, are considered in a similar fashion. It is given that peak throughput of memory system is 10.4 B/cycle per SM. We model it as a single resource independently available at each SM that is exclusively used for 1/10.4 cycles when transferring each byte. Each fully coalesced access instruction in this case, assuming cache misses, involves transferring 128 bytes and thus exclusively uses this resource for 12.3 cycles. Each memory access instruction that has stride-2 access pattern involves transferring twice as much data and therefore exclusively uses this resource for 24.6 cycles. As there are 5 instructions of each kind per warp, the total resource use is 184.5 cycles. The throughput bound therefore is also 184.5 cycles per warp at each SM.

Global memory访问，包括合并的和未合并的，以类似的方式考虑。给定的内存系统峰值吞吐率为每SM 10.4B每周期。我们将其建模为单个资源，对每个SM独占性使用，在传输每个byte时的速率为1/10.4 周期。每个完全合并的访问指令，假设cache misses，会将128 bytes数据传输，因此会独占性使用这个资源12.3周期。步长为2的访问模式，每个内存访问指令会传输两倍的数据，因此独占性的使用这个资源24.6周期。因为每个warp有5条这样的指令，总计的资源使用是184.5周期。吞吐率界限因此是每SM每个warp 184.5周期。

The last throughput bound we consider is due to instruction issue. It involves such additional details as dual-issue and reissue. For the sake of example, we assume that each SFU instruction is dual-issued with a CUDA core instruction, and all instructions that involve non-coalesced accesses or bank conflicts are reissued once. Then, for 135 executed instructions per warp we have 5 dual issues, 125 single issues, and 15 reissues, which constitute 145 issue events in total. Given 4 warp schedulers per SM, the respective throughput limit is 0.25 cycles per issue at each SM, and the resulting bound on warp throughput is 36.25 cycles per warp at each SM.

我们考虑的最后一个吞吐率界限，是因为指令发射。这还涉及到额外的细节，包括双发射，和重发射。为举例方便，我们假设每个SFU指令是和CUDA core指令一起双发射，涉及到非对齐的内存访问或bank冲突的所有指令都进行一次重新发射。那么，对每个warp中135条执行的指令，我们有5个双发射的，125个单发射的，和15个重发射的，这总共构成了145个发射事件。每个SM有4个warp调度器，相应的吞吐率限制是每个SM中每次发射0.25周期，得到的warp吞吐率界限是每SM每warp 36.25周期。

The tightest of these bounds is 184.5 cycles per warp at each SM, or 0.00542 warps per cycle per SM. This is the best bound on warp throughput given the data. The more such hardware throughput limits we consider, and in more detail, the more accurate a bound we may construct.

最近的这些界限是每个SM每warp 184.5周期，或每个SM每周期0.00542 warps。给定这些数据，这是warp吞吐率最好的界限。我们考虑的硬件吞吐率限制越多，越详细，我们得到的界限就越精确。

## 3.7 Latency bound

A lower bound on mean warp latency is found by averaging lower bounds on individual warp latencies. A lower bound on each individual warp latency is found by simulating instruction issue in the respective warp assuming no throughput limits and no inter-warp interference. This allows finding instruction issue times in an inexpensive manner, using only processor latencies and executed code. Some interference between warps still has to be taken into account to model barrier synchronization.

将单个warp延迟的下界进行平均，就得到了平均warp延迟的下界。通过对相应的warp进行模拟指令发射，假设没有吞吐率限制，没有warp间干扰，就可以得到每个单个warp延迟的下界。这就可以只用处理器延迟和被执行的代码，很轻松的就得到指令发射时间。Warps间的一些干扰仍然需要进行考虑，以对barrier同步进行建模。

The most widely recognized processor latency is register dependency latency; we discussed it before. If an instruction is issued at cycle X, and its register dependency latency is Y cycles, then its register output is ready at cycle X + Y. An instruction that uses this output cannot be issued sooner than then. An example is shown in Listing 3.1. Here, all instructions are back-to-back dependent. Issue time of the first instruction is cycle 0. Issue time of each following instruction is the sum of register dependency latencies of all instructions before it.

最广泛承认的处理器延迟是寄存器依赖关系延迟；我们之前讨论过。如果一条指令在周期X被发射，其寄存器依赖关系延迟是Y周期，那么其寄存器输出在周期X+Y准备好。一条使用这个输出的指令，其发射不能早于这个时候。一个例子如列表3.1所示。这里，所有的指令都是背对背依赖的。第一条指令的发射时间为周期0。每条后续指令的发射时间是其之前的所有指令的寄存器依赖关系延迟之和。

Another, less well known latency, is the latency to issue the next independent instruction from the same warp. We call it ILP latency for instruction-level parallelism. If an instruction is issued at cycle X, and ILP latency is Z cycles, next instruction from the same warp can be issued not sooner than at cycle X + Z. This is even if the next instruction does not depend on the recently issued instructions. An example is shown in Listing 3.2, left. This is a sequence of CUDA core instructions as executed on the Fermi GPU. All instructions are independent and issued 6 cycles after one another, where 6 cycles is the ILP latency. We find nontrivial ILP latencies on all five GPU architectures used in this study except Maxwell. Similar latency on a GT200 GPU was earlier noted by Lai and Seznec [2012].

另一个不那么有名的延迟，是从相同warp发射下一条独立指令的延迟。我们称之为ILP延迟，因为是指令级并行。如果一条指令在周期X发射，ILP延迟为Z周期，那么同一warp中下一条指令的发射不会早于周期X+Z。即使下一条指令并不依赖于最近发射的指令，也是这样。一个例子如列表3.2左所示。这是一个CUDA core指令序列，在Fermi GPU上执行。所有指令都没有相互依赖，每6个周期发射一次，这6个周期就是ILP延迟。我们本研究中的所有GPU中，除了Maxwell，其ILP延迟都不可忽视。Lai and Seznec [2012]指出了GT200 GPU上类似的延迟。

Global memory access instructions may incur a larger ILP latency. For example, store instructions on the Fermi GPU, according to our study, cannot be issued more often than once in 34 cycles, as shown in Listing 3.2, center. This is if one warp is executed at a time, and all accesses are fully coalesced. ILP latencies for load instructions may even be larger and have a complicated structure, as shown in the same listing, right. The first five instructions in the example are issued 6 cycles after one another, except that the fourth instruction is issued 26 cycles after the third. What is more notable, is that the sixth instruction is issued 482 cycles after the fifth. This delay is about as long as memory latency itself, which equals approximately 513 cycles on this GPU (as we find in Chapter 6, Table 6.2). The delay may be due to a limitation on the number of concurrently processed memory loads allowed for each individual warp. This limitation was studied earlier by Nugteren et al. [2014].

Global memory访问指令可能会带来更大的ILP延迟。比如，根据我们的研究，在Fermi GPU上的store指令，最多只能每34个周期发射一次，如列表3.2中间所示。这是一次执行一个warp，所有访问都完全合并的情况。Load指令的ILP延迟会更大，而且结构更复杂，如列表3.2右所示。例子中前5条指令每6个周期发射一次，除了第4条指令在第3条指令后26周期发射。更值得注意的是，第6条指令在第5条指令发射后482周期才发射。这个延迟大约等于内存访问延迟本身，在这个GPU上大约是513周期。这个延迟可能是因为每个单独的warp的并发处理的内存load访问的数量限制。这个限制Nugteren et al. [2014]也进行了研究。

Similar long sequences of independent instructions are common in highly optimized kernels such as matrix multiply [Volkov and Demmel 2008; Gray 2014], FFT [Volkov and Kazian 2008] and radix sort [Merrill and Grimshaw 2010]. They are also not unusual in other codes. For example, lbm kernel in the Parboil benchmarking suite [Stratton et al. 2012] includes a sequence of 20 independent loads and a sequence of 19 stores.

这样长的独立指令序列，在高度优化的kernels里是很常见的，如矩阵乘法，FFT和radix排序。其他代码中也是较为常见的。比如，Parboil基准测试包中的lbm kernel，包含了20个独立loads序列，和19个stores的序列。

Another potential contribution to minimum warp latency is due to barrier synchronization. When a barrier instruction is executed in one warp, the warp is stalled until all other warps in the same thread block also execute a barrier instruction. The stall times can easily be found if issue times of the barrier instructions are known. To find these issue times we schedule all instructions in the participating warps up to the barrier instruction, inclusively. In doing so we assume that each warp starts at cycle 0 and is executed alone. Then we take the largest of these times, add pipeline latency of barrier instruction if known, and use the result as issue time for the next instruction in each warp.

另一个对最小warp延迟可能有贡献的是barrier同步。当一个barrier指令在一个warp上执行时，这个warp就stall了，直到同一thread block中的所有其他warps也执行了这个barrier指令。如果知道barrier指令的发射时间已知，stall时间就很容易知道。为找到这些发射时间，我们将参与的warps中的所有指令都调度到barrier指令上。这样做，我们假设每个warp都从周期0开始执行，并单独执行。那么我们取这些时间的最大值，如果知道的话，就把barrier指令的流水线延迟加上去，在每个warp的下一条指令的发射时使用这个结果。

An example for the Fermi GPU is shown in Listing 3.3. There are two warps. One warp takes a longer path that includes an additional stall on memory access. The warps are synchronized at a barrier instruction, which is denoted as BAR.RED.POPC. Both warps start at the same time but arrive to the barrier instruction at different times: cycle 546 in one case and cycle 32 in another. Assuming that pipeline latency of barrier instruction is 34 cycles, issue time of the next instruction in both warps is cycle 580.

Fermi GPU的一个例子如列表3.3所示。有2个warps。一个warp执行了较长了的路径，包含了内存访问指令，有额外的stall。这两个warps在barrier指令处进行了同步，即BAR.RED.POPC指令。两个warps在同样的时间开始，但是到达barrier指令的时间不同：在warp 1中是546周期，warp 2中是32周期。假设barrier指令的流水线延迟是34周期，两个warps中下一条指令的发射时间都在580周期。

Also used in the example is the latency of a branch instruction. We find it equals 32 cycles if the branch is taken and 28 cycles if it is not. These and other latencies for the Fermi GPU are found by timing similar instruction sequences using clock register on the GPU. The instruction sequences were created using a custom assembler similar to asfermi. There are other similar latencies, which may be used to refine the bound.

这个例子中还使用了分支指令的延迟。我们发现，如果执行了分支，则是32周期，如果没有执行，则是28周期。Fermi GPU上这些延迟和其他延迟的发现，是对类似的指令序列，使用GPU上的时钟寄存器计时发现的。指令序列是用定制的汇编器创建的，类似于asfermi。还有其他类似的延迟，可以用于细化界限。

As discussed earlier, terminated warps are not replaced immediately, which results in occupancy variance. We can compensate for this effect by considering the time it takes to replace a warp as a part of the warp latency itself. There are two such delays that can be easily taken into account. The first is the time until other warps in the same thread block also terminate. This may be taken into account by rounding up each warp latency to the largest warp latency in the same thread block. The second is the time to assign a new thread block after previous thread block terminates. This latency can be found by timing execution of an empty kernel such as

前面讨论过，执行结束的warps并不是立刻就替换掉的，这就会导致占用率的变化。我们将替换warp所需的时间，认为是warp延迟本身的一部分，来补偿这种效果。有两种这样的delay，可以很容易的纳入考虑。第一是同一thread block中的其他warps也停止的时间。将每个warp延迟变为同一thread block中的最大warp延迟，就将这种效果进行了考虑。第二是在之前的thread block停止之后，指定新的thread block的时间。这可以通过执行一个空的kernel，比如下面的kernel，来计时得到

```
__global__ void empty() { }
```

when it is launched with a large number of thread blocks. We find this latency equal to 150 to 300 cycles, depending on GPU. This number is then added to the latency of each individual warp.

当与大量thread blocks一起启动时。我们发现这个延迟等于150到300周期，不同的GPU会不一样。这个数值会加入到每个单独的warp的延迟之上。

## 3.8 An example

Consider an application of the presented methodology to performance estimation of a simple kernel such as vector add. The kernel is shown in Listing 3.4. It is written in CUDA; for a quick but sufficient introduction to CUDA see Nickolls et al. [2008]. CUDA code resembles C code but includes a few additional features. For example, the first line in the kernel computes global thread index using thread index in a thread block and index of the thread block.

将上述的方法论应用到一个简单的kernel的性能估计上，比如矢量加。这个kernel如列表3.4所示。这是用CUDA写的，CUDA的介绍可以见Nickolls et al. [2008]。CUDA代码与C代码比较像，但有一些额外的特征。比如，kernel的第一行，使用thread block的index和thread block中的thread index来计算global thread index。

We consider execution of this kernel on the Kepler GPU. We choose this, less recent GPU because this results in a slightly shorter, by two instructions, assembly code than if using the newer Maxwell GPU and thus allows keeping the following presentation a little less cumbersome.

我们考虑这个kernel在Kepler GPU上的运行。我们选择这个没那么新的GPU，因为生成的汇编代码会比使用Maxwell GPU少2行，下面的描述会不那么冗繁。

The respective assembly code is shown in Listing 3.5. The code is easy to understand. The first instruction is redundant; according to Greg Smith, of NVIDIA, it sets the stack pointer. The next two instructions read thread index and thread block index from special registers. The indices are then combined into a global thread index using integer multiply-add instruction, IMAD. Instructions ISCADD are used to compute memory addresses in the arrays, and instructions LD and ST are used to access the arrays. FADD is a floating-point add instruction. Operands such as c[0x0][0x44] refer to the data stored in constant memory, such as kernel arguments and thread block size. Operands R0, R1, etc. are registers. Bars on the left denote the dual-issued instruction pairs. The assumed pairing rule is explained below.

对应的汇编代码如列表3.5所示，很容易理解。第一条指令是冗余的，设置了stack pointer。后面两条指令是从特殊寄存器中读取thread index和thread block index。这两个index然后通过整数乘加指令IMAD计算得到global thread index。指令ISCADD用于计算阵列中的内存地址，LD和ST则是访存指令。FADD是浮点加法指令。操作数如c[0x0][0x44]是存储在constant内存中的数据，比如kernel参数和thread block大小。操作数R0，R1等是寄存器。左边的竖条是爽发射的指令对。配对的规则下面进行解释。

To find a bound on individual warp latencies we represent all related latency constraints with a graph. The graph is shown in Figure 3.5. The nodes correspond to executed instructions, and the edges, to processor latencies. The graph is directed. Each edge points down the execution order and is annotated with the respective latency. Thick edges connect instructions in program order and represent ILP latencies and the warp termination latency. Dashed edges correspond to register dependency latencies, for which we use the numbers found in Chapter 6. These are 301 cycles for memory loads and 9 cycles for floating-point adds. All other instructions that are not memory accesses are assumed to be executed using CUDA cores and have similar 9 cycle latency. ILP latency is separately found to be 3 cycles. Dual-issue is modeled with zero ILP latency. We assume that load instructions are not paired with other load instructions, which leaves four dual-issue opportunities in the code as shown in the listing and the figure. Latency to replace a terminated thread block is found to be 201 cycles. This graph is similar to the work flow graph introduced in Baghsorkhi et al. [2010] but with substantially different edge assignments.

为找到单个warp延迟的界限，我们用一个图来表示所有相关的延迟约束，如图3.5所示。节点对应着执行的指令，边对应着处理器的延迟。这个图是有向的。每个边都指向执行顺序，用分别的延迟进行了标注。厚的边以程序顺序来连接指令，表示ILP延迟和warp停止延迟。虚线边缘对应着寄存器依赖关系延迟，我们使用第6章中的数字，对于内存load，是301周期，对于浮点加法是9周期。所有其他的非访存指令都假设用CUDA cores执行，延迟为9周期。ILP延迟为3周期。双发射建模为0 ILP延迟。我们假设load指令不与其他load指令成对，这就有4次双发射的机会。替换一个结束的thread block的延迟为201周期。本图与Baghsorkhi et al. [2010]的图类似，但边的赋值是不同的。

As edges represent latency constraints, length of a longest path in the graph corresponds to a lower bound on warp latency. The longest path in this graph is 544 cycles long; it is highlighted in the figure. Since all warps are similar, this is both a bound on individual warp latency and a bound on mean warp latency. Thus:

因为边表示延迟约束，图中最长路径的长度对应着warp延迟的下界。这个图中最长的路径为544周期，在图中进行了高亮。由于所有的warps都是类似的，这是单个warp延迟的界限，也是平均warp延迟的界限。因此：

mean warp latency ≥ 544 cycles.

For a bound on warp throughput we consider three bottlenecks: instruction issue, CUDA cores, and memory system. Given 12 instructions and 4 dual-issues, there are 8 issues per warp. Peak instruction issue rate is 1 issue per cycle per warp scheduler. The bound on warp throughput is thus 8 cycles per warp at each scheduler. Given 4 warp schedulers per SM, this is 0.5 warps / (cycle × SM).

对于warp吞吐率，我们考虑3个瓶颈：指令发射，CUDA cores，和内存系统。给定12条指令，和4个双发射，每个warp有8次发射。峰值指令发射速度是每个warp调度器每周期1个发射。因此warp吞吐率的界限是，每个warp每个调度器8个周期。给定每个SM 4个warp调度器，这就是0.5 warps / (cycle × SM)。

Out of these 8 issue events, 5 include one CUDA core instruction and 2 include two CUDA core instructions. Given 48 CUDA cores per scheduler, at least one CUDA core instruction can be finished every cycle at each scheduler, and two can be finished every second cycle. Therefore, finishing all CUDA core instructions in this kernel, when executing many warps, requires at least 7 cycles per warp at each warp scheduler. This bound is less tight than the bound above.

 在这8个发射事件中，5个包含了一条CUDA core指令，2个包含了2条CUDA core指令。每个调度器对应48个CUDA cores，在每个调度器上每周期至少可以完成1条CUDA core指令，每2个周期可以完成2条。因此，在执行很多warps的时候，完成这个kernel中所有的CUDA core指令，在每个warp中每个warp调度器上至少需要7个周期。这个界限比上个界限还是要宽松的。

Peak throughput of the memory system is found in Chapter 6: it is 154 GB/s or 17.1 B / (cycle × SM). This is if all accesses are reads and miss cache. The amount of data transferred by the memory system is 3 × 128 B / warp, and, therefore, warp throughput is bound by 0.0445 warps / (cycle × SM).This is if we don’t differentiate read traffic and write traffic.

内存系统的峰值吞吐率在第6章中找到：是154 GB/s或17.1 B / (cycle × SM)，这是在所有访问都是读而且cache miss的情况下。内存系统传递的数据量为3 × 128 B / warp，因此，warp吞吐率的界限是 0.0445 warps / (cycle × SM)，这里我们假设不区分读流量和写流量。

The tightest bound is the one due to the memory system. So: 最紧的界限是因为内存系统的。所以：

warp throughput ≤ 0.0445 warps / (cycle × SM).

Putting the two bounds together we get the following estimate: 将两个界限放在一起，我们得到下面的估计：

warp throughput ≈ min( n / 544, 0.0445 ),

where warp throughput is in warps per cycle per SM, and n is occupancy in warps per SM.

It is convenient to express this result as memory throughput in GB/s. In order to do so we multiply it by the number of bytes transferred per warp (3 × 128 B), the number of SMs (8) and the clock rate (1.124 GHz). The result is:

将这个结果表达为GB/s的内存吞吐率是很方便的。为此，我们将其乘以每warp传输的bytes数量(3 × 128 B)，SM的数量（8），和时钟频率(1.124 GHz)。结果是：

memory throughput ≈ min( n × 6.35 GB/s, 154 GB/s ).

It is plotted in Figure 3.6 as a thick grey line. Also plotted, as dots, are the throughputs attained in experiment when executing this kernel on large arrays. Multiple dots at each occupancy correspond to attaining the same occupancy with different thread block sizes; details of the experimental setup are given in Chapter 5. The figure shows that the model is highly accurate at both large and small occupancies but is less accurate otherwise due to the gradual saturation effect.

这如图3.6中的灰线所示。图中的黑点，是通过试验得出的吞吐率，在大型阵列上执行这个kernel。在每个占用率上的多个点，是在不同的thread block大小上得到了相同的占用率；第5章给出了试验设置的细节。如图所示，该模型在占用率很大或很小时都是非常精确的，但中间情况时由于渐进饱和效应则没那么精确。

## 3.9 Latency hiding

The ability to attain a better throughput by executing many warps at the same time is often called “latency hiding”. However, the use of this term requires a clarification, in which case we can use the above example as an illustration.

通过同时执行很多warps，来得到更好的吞吐率的能力，称为延迟隐藏。但是，使用这个术语需要进行澄清，在哪个情况下，我们可以使用上面的例子进行描述。

The widely used introductory materials on GPU programming, such as CUDA C Programming Guide [NVIDIA 2015; Ch. 5.2.3] and similar manuals for AMD GPUs [AMD 2015; Ch. 2.6.1], suggest considering latency “hidden” when attaining full utilization of functional units on a multiprocessor if on NVIDIA GPUs, and when keeping compute units busy if on AMD GPUs. This is in line with how latency hiding – also called latency tolerance – is understood in classical work, such as on coarse-grained multithreaded processors and prefetching: its purpose was to maximize processor utilization by avoiding stalls on memory accesses. We briefly review this prior work later, in §4.9.

GPU编程广泛使用的入门材料中，如CUDA C编程指南，以及AMD GPU的类似手册，完全利用NVIDIA GPUs上multiprocessor的功能单元，或者让AMD GPUs上的compute units一直繁忙，就可以得到延迟隐藏的效果。这与经典工作中对延迟隐藏的理解是一致的，比如粗粒度多线程处理器和预取：其目的是通过避免内存访问的stalls，来最大化处理器利用。我们在4.9中简要回顾这个之前的工作。

However, this understanding of latency hiding leads to surprising results if applied to GPUs. For example, we find that in this case latency is not hidden in the example above and, moreover, cannot be hidden even theoretically. Indeed, the best attained throughput as per Figure 3.6 is 156 GB/s, which corresponds to the overall instruction throughput of 0.54 IPC/SM, which is a small fraction of both peak issue throughput (4 to 8 IPC/SM), and peak arithmetic throughput (6 IPC/SM). Functional units on multiprocessors are thus poorly utilized, and, in the above understanding, latency is not hidden. Yet, execution times at large occupancies are as small as if latencies were nil: the attained memory throughput equals peak memory throughput and therefore would not improve even if processor latencies were smaller. In this, more intuitive sense, latencies are hidden.

但是，这种延迟隐藏的理解，应用到GPUs上，会得到令人惊奇的结果。比如，我们发现在这种情况下，在上面的例子中延迟并没有得到隐藏，而且，在理论上也不能得到隐藏。确实，在图3.6中，得到的最好的吞吐率为156 GB/s，这对应着整体指令吞吐率为0.54 IPC/SM，这是峰值发射吞吐率 (4 to 8 IPC/SM)，和峰值算术吞吐率(6 IPC/SM)的一小部分。在multiprocessor上功能单元的利用因此是很差的，在上面的理解中，延迟并没有隐藏。但是，在高占用率上的执行时间也是很小的，仿佛延迟为0：得到的内存吞吐率等于峰值内存吞吐率，因此即使处理器延迟更小，也不能改进了。这样，延迟就被隐藏了。

It is therefore useful to differentiate hiding latency and avoiding stalls. Stalls in instruction issue can be roughly divided into two types: latency stalls and bandwidth stalls. Latency stalls are the stall cycles that would be eliminated if some of the processor latencies were shorter. Bandwidth stalls are the stall cycles that would be eliminated if some of the processor throughput limits were looser. Latency hiding techniques help avoiding only latency stalls. A similar classification is suggested in Burger et al. [1996], except in application to memory stalls only. The two types of stall cycles correspond to the two execution modes discussed in §3.5: latency-bound mode and throughput-bound mode. Hiding latency, thus, means attaining throughput-bound mode.

因此，区分隐藏延迟和避免stalls是有用的。指令发射中的stall可以大致分为两类：延迟stall和带宽stall。延迟stall中，如果一些处理器延迟更小的话，就可以被消除掉。带宽stall中，如果一些处理器吞吐率限制比较宽松的话，就会被消除掉。延迟隐藏技术只能避免延迟stall。Burger et al. [1996]也提出了类似的分类，但只应用到了memory stalls。这两种类型的stall周期，对应着3.5中讨论的两种执行模式：延迟界限模式，和吞吐率界限模式。因此，延迟隐藏意思是得到吞吐率界限模式。

To summarize, we understand latency hiding as synonymous to attaining any hardware throughput limit. This may be a limit due to a full utilization of functional units on a multiprocessor, or due to a full utilization of processing units elsewhere, such as in memory system. We use this understanding later, in Chapter 4.

总结起来，我们理解延迟隐藏与得到任意的硬件吞吐率限制是相同意思。这可能是因为完全利用了multiprocessor上的功能单元，或因为完全利用了其他地方的处理单元，比如内存系统。我们之后在第4章中利用了这个理解。

## 3.10 Back-of-the-envelope analysis

An important property of each GPU kernel is the occupancy needed to minimize its execution time or, what is the same, to attain a maximum throughput. In our model it is found as the product of a lower bound on mean warp latency and an upper bound on warp throughput. For short, below we also refer to these bounds as simply warp latency and warp throughput. Thus we can write:

每个GPU kernel的一个重要属性，是最小化其执行时间所需的占用率，或者说，得到最大吞吐率。在我们的模型中，这就是平均warp延迟的下界，和warp吞吐率上界的乘积。下面我们将其简称为warp延迟和warp吞吐率。因此，我们可以写为：

needed occupancy ≈ warp latency × warp throughput.

This is similar to how the needed instruction concurrency equals the product of instruction latency and peak instruction throughput, and is also a consequence of Little’s law.

指令并发率等于指令延迟和峰值指令吞吐率，这与其类似，也是Little定律得到的结果。

This simple relation, at least in some cases, provides an intuitive aid for understanding how the needed occupancy is affected by changes in executed code or input data. Below we consider a few examples of such back-of-the-envelope analyses.

在一些情况下，这个简单的关系为理解执行的代码或输入数据怎样影响需要的占用率，提供了一个直观的帮助。下面，我们考虑这种粗糙的分析的几个例子。

**Coalesced versus non-coalesced accesses**. In the first example we consider how needed occupancy may depend on memory access pattern used in a kernel.

第一个例子中，我们考虑需要的占用率是怎样依赖于kernel中使用的访存模式。

Consider the kernel shown in Listing 3.6. It performs a permutation of the contents of an array as described by assignment a[i] := b[c[i]], where i = 0, …, m – 1 and m is the size of each array. Of the given three arrays, two are always accessed in a coalesced manner, and one, array b, is accessed in either a coalesced or a non-coalesced manner, depending on the contents of array c.

考虑列表3.6中所示的kernel，它进行的是阵列的排列，a[i] := b[c[i]]，其中i = 0, …, m – 1，m是阵列的大小。在给定的3个阵列中，a,c是以合并的方式进行访存的，而阵列b的访存方式要依赖于阵列c中的内容，可以是合并的或非合并的方式。

We compare the occupancies needed in the following two cases. In the first, the base case, the permutation is trivial as defined by c[i] = i for all i. All accesses to array b are then fully coalesced. In the second case, the permutation is random with repetition: each c[i] is set to a random number between 0 and m–1 so that all or virtually all accesses to array b are fully diverging.

我们考虑下面两种情况下的所需的占用率。第一种情况，基本情况，c[i] = i。这样阵列b的访存都是完全合并的。在第二种情况下，排列是随机的：每个c[i]设置为0到m-1之间的随机数，这样阵列b几乎所有的访问都是发散的。

What we show below is that the needed occupancy can be easily guessed to be substantially smaller in the second case, because the drop in warp throughput caused by memory divergence is much more severe than the associated increase in warp latency.

下面所展示的需要的占用率，是可以很容易猜测到的，在第二种情况中，会小很多，因为因为内存访问发散导致的warp吞吐率的下降，比warp延迟的增加要严重的多。

It is well known that memory throughput dramatically deteriorates when memory accesses are diverging. In part this is due to the increase in data traffic. Suppose that all thread accesses are 4 bytes wide. Then a fully coalesced access instruction requires transferring 128 bytes. A fully diverging access instruction, on the other hand, results in transferring 32 transactions with 32 or 128 bytes per transaction [NVIDIA 2010a; NVIDIA 2015]. The respective increase in data traffic, thus, is 8x or 32x. Similarly, the number of bytes transferred in the permutation kernel is 3×128 per warp in the base case and 2×128+32×32 or 2×128+32×128 in the second case; the respective increases in the overall data traffic are 3.3x or 11x. Warp throughput may be expected to drop by a similar factor.

大家都知道，当访存发散时，内存吞吐率会急剧下降。这部分是因为数据traffic的增加。假设所有的线程访问是4字节宽。那么一个完全合并的访存指令需要传输128 bytes。完全发散的访存指令，则需要传输32个事务，每个事务32或128 bytes。相应的数据traffic的增加，是8x或32x。类似的，在排列kernel中，需要传输的bytes数量，在基础情况中，是3×128每warp，在第二种情况中，则是2×128+32×32 or 2×128+32×128；整体数据traffic的增加相应的为3.3x或11x。Warp吞吐率会以相似的隐私降低。

In practice, however, memory divergence often causes a larger throughput deterioration than can be explained by data traffic alone. In Chapter 6 we find that on the five GPUs used in this study peak instruction throughput in fully diverging accesses is smaller than peak instruction throughput in fully coalesced accesses by either a factor of 28 to 33 or a factor of 56. (These numbers correspond to the particular array sizes used here and in §6.7.) This is substantially larger than the factor of 8 expected with 32-byte transactions and is similar to as if all transactions were not shorter than about 128 bytes. Assuming that this observation also applies in execution of the permutation kernel, the deterioration of warp throughput in it must be around 11x or larger on each of the five GPUs.

但是，在实际中，访问发散导致的吞吐率下降，通常要比数据traffic单独导致的要更大。在第6章中，我们发现，在我们研究中所用的5个GPUs上，在完全发散访问情况下的峰值指令吞吐率，比完全合并访问时的峰值指令吞吐率，要小28到33倍，或小56倍。（这些数值是6.7节中和这里使用的阵列大小的情况下的值。）在32-byte的事务下，期待会有8倍的下降，这个下降因子要大的多，似乎所有的事务都不短于128 bytes。假设这个观察也适用于排列kernel的执行，warp吞吐率的下降一定在11x或更大。

The respective increase in warp latency, in contrast, is relatively small. Indeed, issuing each additional transaction requires only a small additional delay since all of the transactions are independent and can be processed at the same time. In Chapter 6 we find that this increment is around 6 to 8 cycles per transaction on some GPUs and about 33 cycles per transaction on other GPUs. In the latter case, given SIMD width 32, the increments may yet add up to about a thousand cycles, but even then warp latency only about doubles.

对比之下，warp延迟的增加是相对较小的。发射每个额外的事务，需要的额外delay是较小的，因为所有这些事务都是独立的，可以同时进行处理。在第6章中，我们发现这个增加，在一些GPUs上大约是每个事务6到8周期，在另外的GPUs上大约是每事务33周期。在后者的情况下，在SIMD宽度32的情况下，增加的幅度可能会达到1000周期，但即使是这样，warp延迟也只是加倍了。

To summarize, warp latency increases only moderately, but warp throughput decreases dramatically. Therefore, the needed occupancy, which equals their product, must also substantially decrease.

总结起来，warp延迟的增加不是很大，但warp吞吐率下降的很严重。因此，需要的占用率，等于其乘积，也会有很大的下降。

To compare this expectation with experiment, we run the kernel on all five GPUs. The code is slightly adapted to fit each GPU better. For the G80 and GT200 GPUs we replace integer multiplication done to compute i with the __mul24 intrinsic function, and for the G80, GT200 and Fermi GPUs we organize thread blocks into a two-dimensional grid. Both are common practices described in the CUDA C programming guide [NVIDIA 2010a]. The array sizes are large and are the same as selected in §6.7, where non-coalesced accesses are studied in more detail. Otherwise the experimental setup is described in Chapter 5.

将这个期望与试验进行比较，我们在所有5个GPUs上运行了这个kernel。代码略微进行了调整，以适应各个GPU。对于G80和GT200 GPUs，我们将计算i用到的整数相乘替换成了__mul24内部函数，而对于G80，GT200和Fermi GPUs，我们将thread blocks组织成二维网格。这都是CUDA C编程指南中的常见操作。阵列规模很大，与6.7中的一样，在6.7中非合并的访问研究的更加深入。其他的试验设置在第5章中进行了描述。

The result is shown in Figures 3.7 and 3.8. Throughput is reported in GB/s, counting each thread as 12 bytes. As expected, the needed occupancy is substantially smaller if the permutation is random: only about 6% to 15%, as shown with vertical bars (right column). This is compared to over 50% to 100% occupancy needed when the permutation is trivial (left column).

结果如图3.7和3.8所示。吞吐率单位为GB/s，每个线程12 bytes。就像期望的一样，如果排列是随机的，那么所需的占用率就会小的多，大概只有6%到15%，如右列的竖直条所示。当排列是基础情况时，所需的占用率为50%到100%（左列）。

**Reads versus writes**. In the second example, we consider how the needed occupancy may depend on execution path taken in a kernel.

在第二个例子中，我们考察的是kernel中执行路径的不同如果影响所需的占用率。

The kernel is shown in Listing 3.7. It computes the absolute value of array elements. As above, we slightly modify the code to fit each GPU better, such as use the __mul24 intrinsic function and two-dimensional grid of thread blocks on earlier GPUs. The array is assumed to be large. When finding memory throughput, each thread is counted as 4 or 8 bytes.

列表3.7是所示的kernel，计算的是阵列元素的绝对值。像上面一样，我们会略微修改代码，以适配各个GPU，比如在早期GPUs上使用__mul24内部函数和二维网格thread blocks。阵列规模应当很大。在计算访存吞吐率的时候，每个线程是4或8 bytes。

We compare the occupancies needed in the following two cases. In the first case, all array elements are positive, so that data is read but not written back as no change in sign is required. In the second case, all array elements are negative; execution takes a different path, so that the data is both read and written. In both cases, memory is accessed in a fully coalesced manner.

我们在下面两种情况下比较所需的占用率。在第一种情况下，所有的阵列元素都是正的，这样数据读出了，但是写回。在第二种情况中，所有阵列元素都是负的；执行会到不同的路径上，这样数据既读出来了，还要写回去。在两种情况中，访存都是完全合并的形式。

By using Little’s law, we can easily see that occupancy needed in the second case is smaller. Indeed, the additional writes don’t substantially increase warp latency, but do substantially reduce warp throughput. Therefore, they also reduce the product of warp latency and warp throughput. This product is the estimate for the needed occupancy.

使用Little定律，我们可以很容易看到，在第二种情况下所需的占用率是更小的。确实，额外的写操作不会显著增加warp延迟，但是会极大降低warp吞吐率。因此，也会降低warp延迟和warp吞吐率的乘积，这个乘积就是所需的占用率。

Warp latency is not substantially affected because execution paths taken in the two cases differ by only a few instructions, none of which require a substantial stall. These are the store instruction and the arithmetic instruction that changes sign. Unlike loads, stores don’t incur dependency stalls and thus add only a small number of cycles to warp latency. Latency of the arithmetic instruction is also small. The drop in warp throughput, in contrast, is substantial: the kernel is memory-bound, warp throughput is bound by data traffic, and twice as much data is transferred in the second case. Transferring twice as much data at about the same latency saturates memory system twice as fast and at half the occupancy.

Warp延迟没有受到很大影响，因为两个情况中执行的不同路径只影响少数指令，它们都不需要进行大型stall。这就是store指令和改变符号的算术指令。与loads指令不同，store并不会导致依赖关系stalls，因此只会对warp延迟增加小数量的周期。代数指令的延迟也是很小的。对比起来，warp吞吐率的下降则是很大的：kernel是受访存限制的，warp吞吐率是受到数据traffic约束的，在第二种情况中，传输的数据量是两倍之多。以相同的延迟，传输两倍的数据，会让内存系统以两倍的速度饱和，占用率减半。

This is similar to what we see in practice on the Maxwell GPU, as shown in Figure 3.10, bottom. Throughput equal to 190 GB/s, which is close to the maximum, is attained at occupancy 60 warps per SM if all data is positive, and at 40 warps per SM if all data is negative. The difference is somewhat less than the expected factor of 2. There are a few reasons for this: the small but nonzero increase in warp latency, the gradual saturation effect, and a slightly lower peak throughput in the case when half of the accesses are writes (about 200 GB/s) compared to the case when all accesses are reads (about 210 GB/s).

这与我们在Maxwell GPU上看到的类似，如图3.10所示。吞吐率为190GB/s，接近于最大值，如果所有数据为正，在60 warps/SM的占用率时达到，如果所有数据为负值，则以40 warps/SM的占用率达到。差别比期望的2倍要小。有几个原因：warp延迟的较小增加，渐进饱和效果，在一半的访存为写时，峰值吞吐率会比所有访存为读时要略低。

On the other GPUs, as shown in the same figure and Figure 3.9, the same pattern manifests itself in a different fashion. When all data is positive, the needed occupancy is so large that it, apparently exceeds 100% (left column). Peak throughput, in this case, is never attained and the needed occupancies cannot be compared directly. Instead, we compare the occupancies needed to achieve a smaller throughput: the largest throughput attained in the all-positive case. It is attained at 100% or similar occupancy if all data is positive, and at 47% to 63% occupancy if all data is negative, as shown with vertical bars in the figure. This is a similar difference of about a factor of 2.

在其他GPU上，如图3.9所示，有同样的结论。当所有数据是正的时候，需要的占用率非常大，明显超过了100%（左列）。峰值吞吐率一直都没有达到，不能进行直接对比。我们比较了要达到一个更小吞吐率所需的占用率：在所有都为正的情况下所达到的最大吞吐率。在数据都为正的情况下，基本上需要100%的占用率，而在所有数据为负的情况下，占用率为47%到63%，如图中的竖直条所示，这就是大概为2倍的差异。

**A larger instruction count**. In the last example, we consider how the needed occupancy may depend on compilation options.

在最后一个例子中，我们考虑编译选项怎样影响所需的占用率。

We consider the Black-Scholes kernel in the CUDA SDK [Podlozhnyuk 2007] as executed in our setup. In one case, it is compiled with option -use_fast_math; in the other, using no such option. Using the option reduces instruction count for the cost of reduced accuracy.

我们采用CUDA SDK中的Black-Scholes kernel，在我们的设置中执行。在第一种情况中，使用了-use_fast_math的选项编译；在另一种情况中，没有使用这个选项。使用这个选项会减少指令数量，代价是降低准确率。

Our interest is to find how the needed occupancy is affected by using this option. We assume it is known that the kernel is memory-bound in both cases, when the instruction count is larger and not. By being memory-bound we understand that overall throughput is bound by memory system, not other throughput limits.

我们感兴趣的是使用这个选项怎样影响所需的占用率。我们假设知道在两种情况中kernel都是存储受限的，不论指令数是多还是少。内存受限的意思是，整体吞吐率是受到内存系统约束的，而不是其他吞吐率限制。

It is easy to see that occupancy requirements are smaller when using the option. Again, we break down the needed occupancy into a product of warp latency and warp throughput. Warp throughput is the same in both cases, as the kernel is memory-bound, and using the option does not affect memory accesses. On the other hand, the reduced instruction count does results in a smaller warp latency. The required occupancy, therefore, is also smaller.

很容易看出来，当使用了这个选项后，占用率需求更小了。我们再次将需要的占用率分解为warp延迟和warp吞吐率的乘积。在两种情况下，warp吞吐率是一样的，因为kernel是内存受限的，使用这个选项并不影响内存访问。另一方面，减少了指令数，确实会导致warp延迟会更小。因此，需要的占用率也就更小了。

This is what we see on the Maxwell GPU, as shown in Figure 3.11. Close to peak throughput, 190 GB/s, is attained at occupancy equal about 44 or 48 warps per SM if not using the option, and at about 18 warps per SM if using the option. A similar result is found on the Kepler GPU: peak memory throughput is attained at about 50% occupancy if using the option, and no bound is visibly attained at even 100% occupancy if not using the option. In the latter case, however, it is not clear if peak memory throughput can be attained at even larger than 100% occupancies (if such were supported) – the code may be not memory-bound. On the Fermi GPU the code is clearly not memory-bound when instruction count is larger; the presented analysis does not apply. The compiler option does not substantially affect code generation on the G80 and GT200 GPUs, and we don’t consider them here.

我们在Maxwell GPU上也看到了这个结果，如图3.11所示。如果不使用这个选项，会在44-48 warps/SM的占用率下达到峰值吞吐率，190GB/s；而使用这个选项，则会在18 warps/SM时达到。Kepler GPU也会得到类似的结果：如果使用这个选项，达到峰值内存吞吐率只需要50%的占用率，如果不使用这个选项，在100%的占用率下也没有达到这个界限。在后者的情况下，即使占用率超过100%，也不知道是否能达到峰值内存吞吐率。在Fermi GPU上，当指令数更多时，代码明显不是内存受限的；给出了分析也不使用。在G80和GT200 GPU上，编译选项并不会显著影响代码生成，这里我们不进行考虑。