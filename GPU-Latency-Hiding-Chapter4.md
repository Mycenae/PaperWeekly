# Chapter 4 Understanding latency hiding

In this chapter we make the following contributions. 本章中，我们有以下贡献。

In §4.1 we present a simple workload that brings into focus some of the basic properties of GPU operation. We use it in §§4.5–4.7 to study how the occupancy needed to hide latency depends on arithmetic intensity of executed code. Surprisingly, the dependence, at least in this simplistic workload, is found to be distinctly non-monotonic: the needed occupancy increases with arithmetic intensity when arithmetic intensity is small and decreases otherwise.

在4.1节中，我们给出了一个简单的workload，关注GPU运算的一些基本属性。我们在4.5节到4.7节中使用，研究隐藏延迟所需的占用率，是怎样依赖于执行代码中的算术运算的密度。令人惊讶的是，在这个最简单的workload中，这个依赖性很显然是非单调的：当算数运算密度很低时，需要的占用率随着算数运算密度的增加而递增，但在算数运算密度高时，就开始递减了。

Prior to that, in §4.3, we consider a simpler question: how the occupancy needed to hide arithmetic latency compares with the occupancy needed to hide memory latency. The first case corresponds to the execution of a workload where all instructions are arithmetic instructions, and the second case, where all instructions are memory instructions. Contrary to what is often assumed, the occupancies are found to be similar. In §4.4 we discuss a connection between Little’s law and the “memory wall”, which partially explains the finding.

在这之前，在4.3节中，我们考虑一个更简单的问题：隐藏算数运算的延迟所需的占用率，与隐藏内存延迟所需的占用率，相比起来会怎样。第一种情况对应于，workload中所有的指令都是算数运算指令，第二种情况下，所有的指令都是内存访问指令。与通常假设的情况相反，两个占用率是类似的。在4.4节中，我们讨论了Little定律和内存墙之间的关系，这部分解释了这个发现。

In §§ 4.8–4.13 we discuss prior approaches to GPU performance modeling and identify a number of pitfalls. In particular, we show that some prior models are limited due to ignoring arithmetic latency, others are limited due to ignoring memory bandwidth, and yet others are limited despite including all relevant latencies and throughput limits into consideration – due to considering execution of different instruction types in isolation.

在4.8到4.13节中，我们讨论了之前的GPU性能建模的方法，发现了一些缺点。特别是，我们展示了一些之前的模型因为忽略了算数运算的延迟，或者忽略了内存访问的带宽，或者虽然考虑了所有相关的延迟和吞吐率，但是孤立的考虑不同指令类型的执行，而都受到了限制。

## 4.1 The setup and the model

We consider a kernel reduced to its most basic features: a mix of arithmetic and memory instructions, where all arithmetic instructions are identical and all memory instructions are also identical. The only variable parameter in the kernel is the proportion in which the two instructions are mixed. The instructions are selected to approximate the common case: memory accesses are coalesced and miss in the cache, arithmetic instructions are some of the more common CUDA core instructions – floating-point adds. The instructions are evenly interleaved and back-to-back dependent. The proportion in which instructions are mixed is called arithmetic intensity, or α: it is α arithmetic instructions per 1 memory access instruction, as shown below (to the left is the pseudo-code):

我们考虑一个kernel，只包含最基本的特征：算数指令和内存访问指令的混合，其中所有的算术运算指令都是一样的，所有的内存访问指令也是一样的。Kernel中唯一的变量参数是两种指令混合的比例。选择的指令要对常见的情况进行近似：内存访问要进行合并，在cache中miss，算术指令是一些很常见的CUDA core指令，浮点加法。这些指令均匀交替，相互有依赖关系。指令混合的比例称为算术运算密度，或α：每1条内存访问指令，就有α条算术运算指令，如下图所示：

```
a = memory[a] |  LD R1, [R1]
a=a+b         |  FADD R1, R1, R2
a=a+b         |  FADD R1, R1, R2
...           |  ... 
a=a+b         |  FADD R1, R1, R2
a = memory[a] |  LD R1, [R1]
a=a+b         |  FADD R1, R1, R2
...           |  ... 
```

This instruction sequence is executed in each warp. 这个指令序列在每个warp中都进行执行。

For the purpose of performance modeling, we assume that the instruction sequence is infinite and includes no other instructions than these two. In practice, a small number of additional instructions still have to be used: a loop structure to avoid thrashing the instruction cache, an initialization code, timing, etc. A finer detail is using floating-point arithmetic on memory pointers. To avoid ruining the pointers, the added values are always zeros; yet, this does not suffice in floating-point semantics and we implement a few additional precautions. Note that adding zeroes does not mean that all loads in the same warp are from the same memory location. The locations are different: the address of each location is fetched by the previous load, so that the resulting memory access pattern depends on the contents of memory, as in the traditional pointer-chasing benchmark. The experimental setup is described in more detail in Chapter 5.

为了进行性能建模，我们假设指令序列是无限的，且没有其他的指令。在实践中，仍然需要使用少量其他指令：一个循环的结构，以避免指令cache中的颠簸，初始化代码，计时，等等。更具体的说，是对内存指针使用浮点算术运算。为避免破坏指针，被加的值永远是0；但是，这并不符合浮点语义，我们实现了几个额外的预防措施。注意，加上0，并不意味着在所有warp中的所有load，都是从相同的内存位置的。这些位置是不同的：之前的load拿到每个位置的地址，所以得到的内存访问模式，依赖于内存中的内容，就像在传统的指针追逐基准测试中一样。在第5章中更详细的描述试验设置。

The kernel is inspired by the Roofline model [Williams et al. 2009] and a discussion in the CUDA C programming guide, Ch. 5.2.3 [NVIDIA 2015]. 这个kernel是受roofline模型和CUDA C编程指南5.2.3节的讨论启发得到的。

The following is the solution according to our model. First, we find a latency bound. Suppose that latency of arithmetic instructions is alu_lat cycles and latency of memory access instructions is mem_lat cycles. Warp latency per a repeating group of α + 1 instructions, when only one warp is executed at a time, then is

根据我们的模型，下面是得到的解决方案。首先，我们发现了一个延迟界限。假设算术指令的延迟是alu_lat周期，内存访问指令的延迟为mem_lat周期。那么当一次只执行一个warp时，每组重复的α + 1条指令的warp延迟就是

Latency = mem_lat + α ✖️ alu_lat.

For convenience, we consider the throughput sustained in executing memory access instructions. When executing 1 warp at a time, it equals 1 / Latency instructions per cycle (IPC). When executing n warps at a time, it is at most n times larger:

为方便起见，我们考虑执行内存访问指令时的通量。当一次执行一个warp时，等于1/latency IPC。当一次执行n个warp时，最多是n倍：

Memory throughput ≤ n / Latency

We usually deal with throughput and occupancy metrics as given per SM, in which case n is in warps per SM and throughput is in IPC per SM. 我们通常对每个SM处理吞吐率和占用率的度量，在这种情况下，n就是每个SM中的warps数量，吞吐率是每个SM的IPC。

Next step is finding a throughput bound. We consider three bottlenecks: CUDA cores, memory system and instruction issue. For the issue bottleneck, we assume that each executed instruction is issued only once and consumes a single issue cycle. Suppose that the throughputs sustained in each of these bottlenecks are bound by constants alu_thru, mem_thru and issue_thru respectively, i.e.:

下一步是找到吞吐率的界限。我们考虑3个瓶颈：CUDA cores，内存系统和指令发射。对于发射的瓶颈，我们假设每条执行的指令只发射一次，消耗单个发射周期。假设与这些瓶颈相关的吞吐率受下面的常数约束：

Arithmetic throughput ≤ alu_thru, 
Memory throughput ≤ mem_thru, 
Instruction throughput ≤ issue_thru.

The throughputs are related by the fraction of arithmetic instructions in the mix. Since α arithmetic instructions and α+1 overall instructions are executed for each memory instruction, the relationship is: 吞吐率与在混合中算术指令的比例有关。对每条内存访问指令，有α条算术运算指令，和共计α+1条指令，则关系为：

Arithmetic throughput = α ✖️ Memory throughput, 
Instruction throughput = (α+1) ✖️ Memory throughput.

This allows reducing the bounds on different throughputs to a bound on one throughput, such as memory throughput. We get: 这就可以将多个吞吐率的界限缩减为一个吞吐率的界限，比如，访存的吞吐率。我们得到：

Memory throughput ≤ min ( mem_thru, alu_thru/α, issue_thru/α+1 ). 

This concludes the derivation of throughput bound. Adding in the latency bound, we get an overall bound on throughput, which we treat as the expected value. The expected memory throughput, thus, is: 这就得到了吞吐率界限的推导的结论。加上延迟的界限，我们得到一个吞吐率的总体界限，我们当作期望的值。因此，期望的访存吞吐率就是：

Memory throughput ≈ min ( n/Latency, mem_thru, alu_thru/α, issue_thru/α+1). 

This solution can be somewhat simplified if we note that only two of the three throughput limits matter on any particular GPU. Indeed, on the G80 and GT200 GPUs issue throughput is overprovided and therefore the respective bound can be omitted. The solution, then, reduces to:

我们注意到，在任意特定的GPU上，这3个吞吐率限制只有2个有用，因此可以被简化。在G80和GT200 GPU上，发射吞吐率是overprovided，因此相应的界限可以被忽略：

Memory throughput = min ( n/Latency, mem_thru, alu_thru/α).

On the Fermi and later GPUs, on the other hand, we may omit the throughput limit due to CUDA cores: it is the same or looser than the limit due to instruction issue unless dual-issue is possible, which is not the case here as all instructions are back-to-back dependent. The resulting solution is:

在Fermi和后面的GPU上，我们可以忽略由于CUDA cores导致的吞吐率：这与指令发射的限制相同，或更宽松，除非多发射是可能的，但这里所有的指令都是相关的，所以也不会出现这种情况。得到的解是：

Memory throughput = min ( n/Latency, mem_thru, issue_thru/α+1 ).

These are our final solutions for memory throughput. We get the respective arithmetic throughputs in adds per cycle per SM by multiplying them by 32 ✖️ α. 这是我们对访存吞吐率的最终解。相应的算术运算吞吐率，每SM每周期的加法，就是将其乘以32 ✖️ α。

Input parameters for the GPUs used in our study are listed Table 4.1. Issue throughputs are taken from vendor’s documentation (as quoted in Tables 2.6 and 2.8), but otherwise the numbers are found experimentally as described in Chapter 6. For example, peak memory throughputs are taken from Table 6.4 and translated to IPC/SM units by dividing them by the number of SMs (Table 2.6), the clock rate (Table 2.7), and the number of bytes transferred per instruction (128 bytes). For convenience below, single-issue is factored into both issue and arithmetic throughput limits. For example, arithmetic throughput on the Kepler GPU is limited to 6 IPC/SM if taking into account CUDA cores only, and to 4 IPC/SM if also taking into account single-issue.

我们的研究中使用的GPUs的输入参数如表4.1所示。发射吞吐率是供应商的文档中得到的，其他的数值都是通过试验得到的，如第6章所述。比如，峰值访存吞吐率在表6.4中，将其除以SM数量，时钟速率，每条指令传输的bytes数量（128），就得到IPC/SM的单位。为了下面的方便，单发射分解为发射吞吐率限制和算术运算吞吐率限制。比如，在Kepler GPU上的算术运算吞吐率，如果只考虑CUDA cores，界限为6 IPC/SM，如果还考虑到单发射，就是4 IPC/SM。

For convenience, Table 4.2 lists the resulting solutions with numerical values of constant parameters substituted in. 为方便，表4.2列出了得到的解。

## 4.2 Limit cases

At limits α = 0 and α = ∞, the kernel reduces to the most basic forms, performance in which is well-understood. At α = ∞, all instructions are adds: 加上两种极限情况，kernel会缩减到最基本的形态，其性能已经很好理解。在α = ∞时，所有的指令都是加法：

```
a=a+b         |  FADD R1, R1, R2
a=a+b         |  FADD R1, R1, R2
a=a+b         |  FADD R1, R1, R2
...           |  ... 
```

and the model becomes: 模型变成：

Arithmetic throughput = min ( n/alu_lat, alu_thru ).

This model matches experiment well; Figure 4.1, left, shows an example for the Maxwell GPU. The thick grey line in the figure is the model, the dots are the experiment. This case is the best understood. As we discuss later, a number of prior performance models use this model as a building block. 这个模型与试验非常匹配；图4.1左是在Maxwell GPU上的例子。厚的灰线是模型，点是试验结果。这个情况是理解的最好的。我们后面会讨论到，几个之前的性能模型使用这个模型作为基础。

In the case α = 0, all instructions are memory accesses: 在α = 0的情况中，所有的指令都是访存：

```
a = memory[a]  | LD R1, [R1]
a = memory[a]  | LD R1, [R1]
...            | ...
```

and the model reduces to: 模型缩减为：

Memory throughput = min ( n/mem_lat, mem_thru).

This case is shown in the same figure, right. This time the difference with experiment is noticeable: in the experiment the latency bound smoothly translates into the throughput bound, whereas the model has a sharp knee. This difference is a basic limitation of our model.

这种情况在图右。这次与试验的差别是很明显的：在试验中，延迟界限平滑的变化到吞吐率界限，而模型则有很尖锐的地方。这个差异是我们模型的基本限制。

The focus in the discussion below is on the numbers of warps needed to hide latency, i.e. attain a maximum throughput. In these two cases the model suggests the following two solutions: 下面讨论的焦点，是要隐藏延迟所需的warps数量，即，得到最大吞吐率所需的warps数量。在这两种情况中，模型建议的解如下：

warps needed (arithmetic) = alu_lat ✖️ alu_thru,
warps needed (memory) = mem_lat ✖️ mem_thru.

These relations can be recognized as applications of Little’s law. Note that in the case of memory instructions the solution is approximate. 这些关系就是Little定律的应用。注意，在访存指令的情况下，这个解是近似的。

These solutions for occupancies are well-known. The case of hiding arithmetic latency (α = ∞) is considered in the CUDA C programming guide, Ch. 5.2.3 [NVIDIA 2015]. The case of hiding memory latency is considered in Bailey [1997], and the result can be recognized as MWP_peak_BW in Hong and Kim [2009] and Sim et al. [2012]. A minor difference in our approach is that we treat arithmetic and memory instructions in a similar manner.

占用率的这些解大家都知道。隐藏算术延迟的情况，(α = ∞)，在CUDA C编程指南中提到了。隐藏访存延迟在Bailey [1997]进行了研究，结果就是Hong and Kim [2009] 和 Sim et al. [2012]中的MWP_peak_BW。我们方法中的一个微小差异是，我们对算术和访存指令进行了类似的处理。

## 4.3 Does hiding memory latency require more warps?

Substituting the hardware parameters listed in Table 4.1 into the solutions above suggests that hiding arithmetic latency requires 24 warps per SM and hiding memory latency requires 30 warps per SM if on the Maxwell GPU – which are similar numbers. This is despite the dramatic difference in latencies – 6 cycles in one case and 368 cycles in another – and in contrast with the prevailing common wisdom.

将表4.1中的硬件参数替换到上面的解中表明，如果在Maxwell GPU上，隐藏算术运算的延迟需要每个SM 24 warps，隐藏访存延迟需要每SM 30 warps，这个数值是类似的。在第一种情况下的延迟为6个周期，在另一种情况下是368个周期，这个差异还是很大的。

Indeed, latency hiding on GPUs is almost universally associated with the longer memory latency. This is how GPU’s massive multithreading was originally introduced in vendor’s publications, such as Lindholm et al. [2008], Nickolls et al. [2008], and Nickolls and Dally [2010]; this view is still dominant today. Even publications that do recognize arithmetic latency, such as Gebhart et al. [2011], tend to downplay it as requiring “a much smaller pool of warps”. Our findings are contrary: the required pools of warps are similar in both cases, at least on recent GPUs.

确实，在GPU上的延迟隐藏，几乎都是与更长的访存延迟是相关的。这是GPU的大量多线程在供应商的文章中引入的方式，比如Lindholm et al. [2008], Nickolls et al. [2008], and Nickolls 和 Dally [2010]；这个观点今天仍然占主流。承认有算术运算延迟的文章，比如Gebhart et al. [2011]，也会认为其需要的warp pool要小的多。我们的发现是相反的：需要的warp pool在两种情况下是类似的，至少在最近的GPUs上是这样的。

The result must not be surprising given Little’s law: the required concurrency is a product of latency and throughput, not a function of latency alone. Memory latency is indeed larger than arithmetic latency but, at the same time, memory throughput is smaller than arithmetic throughput, so that the respective latency-throughput products are similar. Putting it in other words, accessing memory one access at a time is slow, but accessing memory is slow anyway, so that processing too many accesses at the same time is not needed and not helpful.

考虑Little定律，这个结果就并不惊讶了：需要的并发度是延迟和吞吐率的乘积，而不是延迟本身的函数。访存延迟确实比算术延迟要大很多，但是同时，访存的吞吐率比算术运算的吞吐率要小很多，所以对应的延迟-吞吐率乘积是类似的。换句话说，一次访存是很慢的，但是访存反正就是很慢，所以同时处理很多访存是没有必要的，也没有多少帮助。

The comparison between hiding memory and arithmetic latencies is made more complicated by the gradual saturation effect, which is found with memory accesses but not arithmetic operations: we find that only about 80% of peak memory throughput is typically attained at the occupancies suggested by the model. Therefore, for a more complete account we also find the occupancies where 90 and 95% of the peaks are sustained in practice. For the Maxwell GPU, such as shown in Figure 4.1, these are 40 and 46 warps per SM respectively – the latter is about 1.5 times larger than the estimate.

隐藏访存延迟，和隐藏算术运算的延迟的比较，因为渐进饱和效果，会变得更加复杂，这种效果只在访存的时候才有，算术运算并没有这种效果：我们发现，在模型建议的占用率下，只会得到80%的峰值访存吞吐率。因此，为了更完整的说明，我们在实践中会找到90%和95%峰值时候的占用率。对于Maxwell GPU，这个结果如图4.1所示，这分别是每SM 40和46 warps，后者比估计的要大1.5倍。

We collected similar numbers for all five GPUs used in the study; they are plotted in Figure 4.2. The collection of data is detailed in Chapter 6, and the numbers are listed in Tables 6.3 and 6.4. The figure shows that the difference in occupancies has nearly monotonic dynamics: it is usually smaller in newer GPUs. If we divide the occupancy needed to sustain 90% of memory peak by the occupancy needed to sustain arithmetic peak, we see that this ratio decreases with every GPU generation except the last, as shown in the following table:

我们在研究中对5种GPUs收集了类似的数字，如图4.2所示。数据的收集在第6章中详述，数字如表6.3和6.4所列。图中表明，占用率的差异几乎是单调变化的：在更新的GPUs上会更小。如果我们将维持90%峰值访存所需的占用率，除以维持算术峰值所需的占用率，我们可以看到，这个比率在下降，除了最新的一代，如下面的表所示：

｜ Generation ｜ G80  ｜ GT200 ｜ Fermi ｜ Kepler ｜ Maxwell ｜
｜ ---------- ｜ ---— ｜ ----- ｜ ----- ｜ ------ ｜ ------- ｜ 
｜ Year       ｜ 2006 ｜ 2008  ｜ 2010  ｜ 2012   ｜ 2014    ｜
｜ Ratio      ｜ 4x   ｜ 2.7x  ｜ 2.3x  ｜ 1.6x   ｜ 1.7x    ｜

In the last generation the ratio increases back from 1.6x to 1.7x, but this is not a substantial reversal. The 4x difference in year 2006 is partially a feature of our setup; if we used multiply-add instructions instead of add instructions, this ratio would be 3.3x with other data unchanged. Also, we wouldn’t see a different trend if were not limited by single-issue: dual-issue doesn’t change the numbers of warps needed – as both instructions in the issued pairs come from the same warp – but does result in a larger number of concurrent arithmetic instructions on the Kepler GPU. Thus, if we considered similar ratios for instruction concurrency, the ratio for the Kepler GPU and this GPU alone would be smaller.

在最新一代，这个比率从1.6x增长到1.7x，但这并不是显著性的逆转。在2006年，4x的差异部分是因为我们的设置的原因；如果我们使用乘加指令，而不是加法指令，这个比率将会是3.3x，其他的数字则不会变。同时，如果不是单发射，我们也不会看到不同的趋势：双发射不会改变需要的warps数量，因为发射的对中的两条指令都是从相同的warps中发射出的，但在Kepler GPU上并不会导致并发的算术指令更大。因此，如果我们对指令并发性考虑类似的比率，对Kepler GPU和这个GPU的比率会更小。

Note that the five GPUs we use are comparable: all are high-end GPUs used to promote each respective new GPU generation; each offered the leading performance at the time of its release and therefore represents the best effort in scaling GPU architecture.

注意，我们使用的5种GPUs是可比的：所有都是高端GPUs；每种都是那时的最高性能，因此代表了GPU架构的最好努力。

Occupancy required to hide memory latency depends on the memory access pattern. Our primary focus is on the common access patterns where all accesses are fully coalesced and miss in the cache. If we consider accesses that are not so well coalesced but similarly miss in the cache, we find they require lower occupancy, not higher. This is because each instruction in this case is served with a larger number of transactions, so that the memory system is saturated with fewer concurrent instructions and fewer concurrent warps.

隐藏访存延迟所需的占用率，依赖于访存的类型。我们主要关注的是常见的访存模式，即所有的访问都是完全合并的，并且是cache miss的。如果我们考虑没有完全合并的访存，也是cache miss的，我们发现会需要更低的占用率，而不是更高。这是因为在这种情况下，每条指令需要用更多的事务数量，所以存储系统在更少的并发指令下，更少的并发warps下，就会饱和。

Figure 4.3 shows two such examples for the Maxwell GPU. The graph on the left shows the throughput attained when using stride-2 access pattern. Peak throughput is 2x smaller – only 0.041 IPC/SM – whereas memory latency is only a little larger: 376 cycles. As a result, the required occupancy is 2x lower: 20 warps per SM for 90% of the peak. If the accesses are random, i.e. unstructured and fully divergent, latency increases less than 2x compared to the base level – to 534 cycles – whereas throughput drops by more than an order of magnitude, down to 0.0029 IPC/SM. As a result, only 3 warps per SM are needed to attain 90% of the peak. This case is shown in Figure 4.3, right. Also shown in the figure, in a thick grey line, is our model plotted using the quoted metrics. In §6.7 we find that in the case of unstructured accesses the same pattern also applies to other GPUs in the study – see Table 6.3.

图4.3展示了Maxwell GPU下的2个这样的例子。图左是在步长2访存模式下得到的吞吐率结果。峰值吞吐率小了2倍，只有0.041 IPC/SM，而访存延迟只大了一点点：376周期。结果是，需要的占用率低了2倍：对90%峰值为每SM 20 warps。如果访问是随机的，即无结构的，完全发散，与基础相比，延迟增加了不到2倍，为534周期，而吞吐率下降了一个数量级还要多，为0.0029 IPC/SM。结果是，只需要每SM 3 warps，来得到90%的峰值性能，这个如图4.3右所示。图中还有一条灰色的厚线，是我们模型的结果。在6.7中，我们发现无结构访存的情况下，在其他GPUs上也发现了这样的模式，见表6.3。

These findings suggest that arithmetic latency must not be considered a second-class citizen in GPU performance modeling; yet, some of relatively recent models, such as Song et al. [2013], entirely omit it. A number of GPU performance models do consider arithmetic latency but with limited success, as we discuss later.

这些发现说明，算术运算延迟不能在GPU性能建模中被忽视；一些最近的模型则完全忽略了这个，如Song et al. [2013]。几个GPU性能模型确实考虑了算术延迟，但成果有限，后面会进行讨论。

## 4.4 Latency hiding and the “memory wall”

To understand the forces behind the dynamics shown in Figure 4.2 and to speculate on its future development, it is instructive to factor the ratio between the occupancies into ratios between the latencies and the throughputs, i.e. as:

为理解图4.2背后的动力，推测其未来的发展，可以将占用率分解为延迟的，和吞吐率的：

warps needed (memory) / warps needed (arithmetic) = memory latency/arithmetic latency ∙ memory throughput/arithmetic throughput

The last ratio in this product – the ratio between memory and arithmetic throughput limits of a processor – has an important historical trend: it has been long observed that it tends to decrease, i.e. that improvements in memory throughput tend to lag behind improvements in arithmetic throughput. This trend is a part of what is known as “memory wall” – see, for example, Burger et al. [1996] and McKee [2004]. The trend is not exclusive to GPUs but is observed on GPUs as well.

上面乘积的后面的比率，处理器的访存吞吐率和算术运算吞吐率的比率，有一个重要的历史性趋势：这个比率一直在下降，即访存的吞吐率没有算术运算吞吐率改变的快。这个趋势就是内存墙，如Burger et al. [1996] 和 McKee [2004]。这并不是GPU独有的，CPU中也有这个趋势。

To qualify the trend better, in Figure 4.4 we plot the ratios between memory and arithmetic throughputs versus the respective release dates for CUDA-capable GeForce GPUs; GeForce is NVIDIA’s primary GPU product line. The vertical axis in the graph is pin memory bandwidth divided by the throughput of CUDA cores, where the latter is found as the number of CUDA cores multiplied by the processor clock rate. The numbers are taken from vendor’s website www.geforce.com. When two processor clock rates are given, we use the larger, “boost” clock rate. The horizontal axis in the graph is the release date found as the earliest date the respective product was first available at the popular online store amazon.com. The published online data, in both cases, includes a few obvious errors, which were corrected. A few GPU versions listed in vendor’s website were not found in the store and are not plotted. Also excluded are laptop and OEM products. Highlighted in the figure are the GPUs used in this work. These are the key products associated with releases of new architecture generations.

为更好的展现这个趋势，图4.4中我们画出了访存和算术运算吞吐率之比，与对应的具有CUDA能力的Geforce GPUs上发布时间；GeForce是NVIDIA的主要GPU产品线。图中的竖轴访存带宽除以CUDA cores的吞吐率，后者是CUDA cores的数量乘以处理器时钟频率，这些数是从供应商网站上获得的。当给出2个处理器时钟频率时，我们使用更大的加速的时钟频率。横轴是发布时间。图中强调的点，是本文中使用的GPUs。这些是新架构代的关键产品。

The data is plotted in log-linear coordinates to highlight the well-defined exponential trend. The best fit (not shown) suggests that improvements in GPU’s arithmetic throughput outpace improvements in GPU’s memory throughput by a factor of about 1.24 per year on average. The trend survived transition from GDDR3 memory technology (G80, GT200) to GDDR5 technology (Fermi and newer); some of the GPUs in the figure use DDR2 technology. The most recent generation of GPU architecture, Pascal, features even newer memory technology: HBM2 [NVIDIA 2016]. We leave finding if this affects the trend to future work.

数据是用对数线性坐标系画的，以强调明显的指数趋势。拟合结果表明，GPU算术运算吞吐率的改进，每年比访存吞吐率的改进，平均要高1.24倍。这个趋势从GDDR3到GDDR5都符合，一些GPUs使用的是DDR2技术。最近的GPU架构代，Pascal，有更新的内存技术：HBM2。这个是否影响这个趋势，我们放在未来的工作中。

As provision of arithmetic throughput increases, so do the respective concurrency requirements. They could have already exceeded the concurrency requirements for memory accesses if not the improvements in arithmetic latency. Latency of floating-point add instructions, for example, was reduced from 24 cycles in the GT200 generation to 18 cycles in the Fermi generation to 9 cycles in the Kepler generation to 6 cycles in the Maxwell generation. Memory latency, in contrast, did not change as substantially (Table 4.1).

随着算术运算吞吐率的增加，相应的并发度需求也增加了。如果不是算术运算延迟的改进，就已经超过了访存需求的并发度了。比如，浮点加法指令的延迟，从GT200中的24周期，降低到了Fermi中的18周期，到Kepler中的9周期，到Maxwell中的6周期。比较起来，访存延迟则没有显著变化（表4.1）。

If the trend is to continue, we may eventually find that hiding arithmetic latency requires more warps than hiding memory latency, which is already the case with non-coalesced memory accesses.

如果这个趋势持续下去，我们最后会发现，隐藏算术运算延迟所需的warps数量，比隐藏访存延迟的要更多，这在非合并访存的情况下已经是这样了。

## 4.5 Cusp behavior

We return to the consideration of instruction mixes. Again, our interest is how many warps are needed to hide latency, i.e. to attain one of the peak throughputs. Given the model for throughput, the solution for occupancy in the general case is easily found as

我们返回到考虑指令混合。我们的兴趣还是，需要多少warps来隐藏延迟，即，获得峰值吞吐率。给定吞吐率的模型，在一般情况下，占用率的解可以很容易的得到：

n = Latency ∙ min (mem_thru, alu_thru/α , issue_thru/α+1 ).  

The solution has a distinct property: it depends on arithmetic intensity in a substantially non- monotonic fashion. Indeed, when throughput is bound by the memory system, i.e. at small arithmetic intensities, it reduces to

这个解有一个独特的属性：这以非单调的模式依赖于算术运算强度。确实，当吞吐率受限于存储系统时，即，在小强度算术运算强度时，就缩减为

n (small α) = mem_lat ∙ mem_thru + α ∙ alu_lat ∙ mem_thru,

which increases with arithmetic intensity. (We used that Latency equals mem_lat + α ∙ alu_lat.) In the opposite case, i.e. when α is large and throughput is bound by either instruction issue or arithmetic capability, the needed number of warps decreases with arithmetic intensity. For example, on GPUs where throughput is bound by arithmetic capability, not instruction issue, the solution is

这是随着算术运算强度的增加而增加的。（我们使用的延迟等于mem_lat + α ∙ alu_lat。）在相反的情况中，即，当α很大，吞吐率受指令发射能力或算术运算能力时，需要的warps数量随着算术运算的强度的增加而减少。比如，在吞吐率受限于算术运算强度，而不是指令发射的GPUs上，解就是：

n (large α) = mem_lat ∙ alu_thru/α + alu_lat ∙ alu_thru. 

On other GPUs the number is slightly smaller. 在其他GPUs上，这个数值就略小。

The largest occupancy is required where the two branches meet, i.e. where the code is both compute- and memory-bound. Again, if we assume that throughput is bound by arithmetic capability, not instruction issue, the largest occupancy is required at α = alu_thru / mem_thru and equals

在两个分支汇合的时候，需要的占用率最大，即当代码既是compute-bound，也是memory-bound的时候。再次，如果我们假设，吞吐率受到算术运算能力限制，而不是指令发射，在α = alu_thru / mem_thru的时候，需要的最大占用率为

n (maximum) = alu_lat ∙ alu_thru + mem_lat ∙ mem_thru.

This number can be recognized as sum of the occupancies required at limits α = 0 and α = ∞. The sum is substantially larger than either occupancy (at most by a factor of 2) and the solution is substantially non-monotonic only if the occupancies are comparable – but they are comparable, as we found earlier in this chapter.

这是在极限情况α = 0和α = ∞时的占用率的和。这个和比任意一个占用率都高很多（几乎是2倍），这个解只有在占用率是可比的时候，才会是非单调的，但是它们确实是可比的，我们在本章前面部分已经得到了这个结论。

The solution for the Maxwell GPU is plotted in Figure 4.5; the two branches are easily recognized in the figure. The values attained on the left- and the right-hand sides of the graph are approximately 30 and 24 warps found previously for limits α = 0 and α = ∞. In the middle of the graph (α ≈ 49) the solution achieves approximately 54 warps, which is the sum of the two values. We call this pattern “cusp behavior” by the look of the graph.

对Maxwell GPU的解画在图4.5中；在图中很容易的就看到两个分支。图中左右两边的值大约是30和24个warps，这是在α = 0和α = ∞时的值。在图中间(α ≈ 49)，解大约是54个warps，是这两个值之和。我们称这个模式尖点行为。

In Figure 4.6 the model is compared with experiment. The model is shown as a dashed line, the experiment is shown as dots, and another model, explained below, is shown as a thick grey line. The values plotted in each case are the smallest occupancies where 90% of one of the peak throughputs is attained. No experimental data is plotted if the target throughput of 90% of peak is not attained at any occupancy, such as in cases where more warps are needed than physically supported.

在图4.6中，模型与试验进行了比较。模型为虚线，试验为点线，另一个模型是厚的灰线。每种情况中画出的值，是在得到90%的峰值吞吐率时的最小占用率。在任何占用率时都无法得到目标吞吐率的的90%时，则不画出试验数据，比如需要的warps数量比物理上存在的数量还要多的时候。

According to the figure, cusp behavior is experimentally observed on each of the five GPUs, as predicted. This qualitative accuracy is a unique feature of our model. Later, in §7.10, we show that prior performance models suggest no similar behavior.
Quantitatively, the accuracy of our model is limited, which is due to the unaccounted gradual saturation effect. In §6.5, we describe a refined model that partially addresses the gradual saturation effect by taking into account memory contention. The respective new estimates are shown in the same figure as thick grey lines. They closely follow the experimental data except when using large α on the Kepler GPU.

根据图示，尖点行为在5种GPUs上都通过试验发现了。这种定性的准确率是我们模型的唯一特征。后面，在7.10节中，我们展示了之前的性能模型没有得到类似的行为。定量的来说，我们模型的准确率是有限的，这是因为没有计入的渐进饱和效果。在6.5节中，我们描述了一种更精细的模型，部分的顾及到了渐进饱和效果，考虑了访存竞争效果。图中的灰色厚线就是这种模型的顾及。这很好的拟合了试验数据，除了在Kepler GPU在α很大的情况下。

## 4.6 Concurrent instructions are not concurrent warps

For a better understanding of cusp behavior we also find the numbers of arithmetic and memory instructions executed at the same time. They are different than the number of warps executed at the same time.

为更好的理解尖点行为，我们还找到了同时执行的算术运算指令和访存指令的数量。它们与同时执行的warps数量是不一样的。

Consider latency hiding when bound by arithmetic throughput. In this case, when latency is hidden, arithmetic instructions are executed at rate alu_thru, and their latency equals alu_lat. The average number of arithmetic instructions executed at the same time then can be found by using Little’s law as the product of these numbers:

考虑在算术运算吞吐率受限的隐藏延迟情况。在这种情况下，当延迟被隐藏时，算术运算指令的执行速率为alu_thru，其延迟为alu_lat。同时执行的算术运算指令的平均数量，可以用Little定律，将这两个数字相乘得到：

arithmetic instructions in execution = alu_lat ∙ alu_thru.

The number of memory instructions executed at the same time is found similarly. It is the product of the rate sustained in executing memory instructions, which is alu_thru / α, and the latency of these instructions, which is mem_lat:

可以类似的得到同时执行的访存指令数量。即，执行访存指令的速率，alu_thru / α，和这些指令的延迟，mem_lat，的乘积：

memory instructions in execution = mem_lat ∙ alu_thru/α. 

Since all instructions are back-to-back dependent, one warp can have only one instruction in execution at a time, such as one arithmetic instruction or one memory instruction, but not both. Therefore, the required number of warps is the sum of these numbers:

由于所有指令都是前后相互依赖的，一个warp只能一次执行一条指令，比如一条算术运算指令，或一条访存指令，而不是两者。因此，需要的warps数量，是这两个数之和：

warps needed = arithmetic instructions in execution + memory instructions in execution.

This produces the solution found previously, i.e. 这就得到了之前发现的解，即：

warps needed = mem_lat ∙ alu_thru/α + alu_lat ∙ alu_thru, 

but also suggests an insight into the origins of the terms. 这也说明了这些项的起源的洞见。

As arithmetic intensity increases, memory access rate decreases and the number of memory instructions executed at the same time also decreases. The number of arithmetic instructions executed at the same time, on the other hand, does not change because the same number is needed to sustain the same peak throughput. As a result, fewer warps are needed at larger arithmetic intensities.

随着算术运算强度增加，访存率下降，同时执行的访存指令数也下降了。另一方面，同时执行的算术运算指令数量不变，因为需要相同的数量来维持峰值吞吐率。结果是，在更大的算术运算强度下，需要的warps数量变少了。

The case when it is the memory system that is the bottleneck is considered similarly. The number of concurrently executed memory access instructions in this case is constant: it is the number needed to sustain peak memory throughput. The number of concurrently executed arithmetic instructions, on the other hand, increases when α increases – because arithmetic throughput increases. The needed occupancy then increases too.

当访存系统是瓶颈时的情况，也可以类似的考虑。在这种情况下，并发执行的访存指令的数量是常数：这是维持峰值访存吞吐率所需的数量。另一方面，并发执行的算术运算指令数量，当α增加时则会增加，因为算术运算吞吐率增加了。需要的占用率也会增加。

The difference between concurrent instructions and concurrent warps is illustrated in Figure 4.7. There are α = 4 arithmetic instructions per memory instruction and n = 6 warps being executed at the same time. Instruction issue times are shown with circles, filled for arithmetic instructions and empty for memory instructions. Shown with dashed lines are the execution times of arithmetic instructions. Arithmetic latency is alu_lat = 3 cycles, memory latency is mem_lat = 12 cycles. Peak arithmetic throughput is alu_thru = 1 IPC, which is sustained in the example.

并发执行指令和并发warps的区别，如图4.7所示。同时执行n=6个warps，每个访存指令有α = 4条算术运算指令。指令发射的时间用圆圈表示，对于算术运算指令，是充满的，对于访存指令，是空的圆圈。虚线表示的是算术运算指令的执行时间。算术运算延迟是alu_lat = 3周期，访存延迟是mem_lat = 12周期。峰值算术运算指令的吞吐率是alu_thru = 1 IPC，在例子中可以达到这个水平。

For 6 concurrently executed warps, there are only 3 concurrently executed arithmetic instructions and only 3 concurrently executed memory access instructions at any moment of time. This can be found in the figure or by using Little’s law. Indeed, the product of arithmetic latency and arithmetic throughput is 3 cycles multiplied by 1 IPC, which equals 3 instructions. The product of memory latency and sustained memory throughput is 12 cycles multiplied by 0.25 IPC, which also equals 3 instructions. The sustained memory throughput, 0.25 IPC, is found as 1 / α multiplied by arithmetic throughput.

对于6个并发执行的warps，在任意时刻同时只有3个并发执行的算术运算指令，和3个并发执行的访存指令。使用Little定律，在图中就能得到这个结论。确实，算术运算延迟和算术运算吞吐率的乘积是3周期乘以1 IPC，等于3条指令。访存延迟和访存吞吐率的乘积是，12周期乘以0.25 IPC，也等于3条指令。维持的访存吞吐率，0.25 IPC，是算术运算吞吐率乘以1/α。

Executing 5 or fewer warps in this example would result in attaining less than peak arithmetic throughput, which is in contrast to only 3 warps being sufficient to attain peak throughput when no memory instructions are present.

在这个例子中，如果执行5个或者更少的warps，则不能得到峰值算术运算吞吐率，对比起来，当没有访存指令时，则只需要3个warps就可以得到峰值吞吐率。

## 4.7 Cusp behavior with other instruction types

Cusp behavior is also found with other instruction mixes. We briefly consider two of them: a mix of adds and reciprocal square roots: 尖点行为在其他的指令混合中也可以找到。我们简要的考虑两种情况：一种是加法和平方根逆的混合：

```
a=1/sqrt(a)   |  MUFU.RSQ R1, R1
a=a+b         |  FADD R1, R1, R2
a=a+b         |  FADD R1, R1, R2
...           |  ... 
```

and a mix of adds and shared memory accesses: 与加法和共享内存访问的混合：

```
a=share memory[a]   |  MUFU.RSQ R1, R1
a=a+b               |  FADD R1, R1, R2
a=a+b               |  FADD R1, R1, R2
...                 |  ... 
```

In the latter case the accesses are set up to cause no bank conflicts. The second mix is not feasible and is not considered on the G80 and the GT200 GPUs. On these GPUs shared memory is addressed using special registers, which requires using additional instructions. In both mixes the instructions are evenly interleaved and all are back-to-back dependent. There are α add instructions per 1 other instruction, or, if α is less than one, 1 add instruction per 1 / α other instructions.

在后者的情况下，这些访问设置成不会造成bank冲突。第二种混合在G80和GT200 GPUs上是不可行的，所以也不考虑。早这些GPUs上，共享内存是使用特殊的寄存器构成的，这需要额外的指令。在这两种混合中，指令平均混合，所有指令都是相互依赖的。每一条其他指令，有α条加法指令，如果α小于1，那么1条加法指令对应1/α条其他指令。

The performance model is derived similarly as before. Instruction latencies are used to find the latency bound, and peak throughputs are used to find the throughput bound. There are two latencies and three bottlenecks; the difference with the prior case, therefore, is only in the metrics used. In place of latency and throughput of global memory access instructions, we use similar metrics for SFU and shared memory access instructions. The needed metrics are found in Chapter 6, Tables 6.1 and 6.2.

性能模型就像以前进行类似的推导。指令延迟用于发现延迟界限，峰值吞吐率用于找到吞吐率界限。有2个延迟，3个瓶颈；因此，与之前情况的不同，只是在于使用的度量。我们使用SFU和共享内存访问指令的延迟和吞吐率，替换掉了全局内存访问指令，需要的度量在第6章中的表6.1和6.2中可以找到。

In the case of the Maxwell GPU we consider an additional detail: we note that some of the register dependency latencies on this processor depend on the type of the dependent instruction. Specifically, they are shorter if dependent instruction is a CUDA core instruction. For example, register dependency latency of an SFU instruction is 13 cycles if the dependent instruction is also an SFU instruction, and 9 cycles if the dependent instruction is a CUDA core instruction. Similarly, latency of a shared memory access instruction is 24 cycles if the dependent instruction is also a shared memory access instruction, and 22 cycles if the dependent instruction is a CUDA core instruction. We use these details when computing term Latency in the model.

在Maxwell GPU的情况中，我们考虑一个额外的细节：我们注意到，在这个处理器中，一些寄存器依赖关系延迟依赖于依赖指令的类型。具体的，如果依赖的指令是一个CUDA core指令，那么就会短一些。比如，一条SFU指令的寄存器依赖延迟，如果其依赖的指令也是一条SFU指令，就是13周期，如果依赖的指令是一条CUDA core指令，就是9周期。类似的，对于共享内存访问指令，如果其依赖的指令也是一条共享内存访问指令，其延迟就是24周期，如果依赖的指令是一条CUDA core指令，就是22周期。我们在模型中的延迟项中，会使用这些细节。

The results are compared with experiment in Figures 4.8 and 4.9. The thick grey line is where 90% of one of the peak throughputs is attained in the model, and the dots are where the same throughput is attained in experiment. Cusp behavior is pronounced in a number of cases such as with SFU instructions on the G80 and GT200 GPUs, and with shared memory instructions on the Fermi and Kepler GPUs. This suggests that cusp behavior is not uncommon and not specific to global memory accesses.

结果与图4.8，4.9中的试验进行了比较。灰色厚线是取得90%的模型中峰值吞吐率的地方，点是在试验中得到相同吞吐率的地方。尖点行为在一些情况中是很明显的，如G80和GT200的SFU指令中，和Fermi和Kepler GPUs中的共享内存指令中。这说明，尖点行为并不是罕见的，页并不是只对于全局内存访问指令的。

In the case of the Maxwell GPU, the model is noticeably less accurate. This may be due to unaccounted issue costs. We assumed that issuing each instruction takes a single issue cycle, but this is not necessarily the case. Indeed, throughput in the experiment is found to saturate at lower values than suggested by the model; two such examples are shown in the same figures, bottom right graphs. The discrepancy is larger when the code is issue-bound or nearly so and SFU or shared memory instructions are more frequent, which suggests that these instructions consume an additional issue throughput. If we assume that issuing each SFU instruction takes 2 issue cycles, and issuing each shared memory instruction takes 3 issue cycles, the estimated saturated throughputs in the two examples match the experiment well. We leave further investigation to future work.

在Maxwell GPU的情况中，模型明显没有那么精确。这可能是因为没有计入的发射代价。我们假设，发射每条指令消耗一个发射周期，但这并不是这个情况。在试验中，吞吐率在比模型建议的值更小的地方就饱和了；在同样的图中，右下角处就有2个例子。在代码是发射受限，并且SFU或共享内存指令更频繁的时候，这个差异就会更大，这说明这些指令会消耗额外的发射吞吐率。如果我们假设发射每条SFU指令消耗2个发射周期，发射每条共享内存指令消耗3个发射周期，在这两个例子中估计的饱和吞吐率，就和试验吻合的很好。我们在未来的工作中会进行更近一步的调查。

## 4.8 Solution in the CUDA C Programming Guide

In the remainder of this chapter we review several approaches taken in prior work. The purpose is to illustrate challenges in modeling latency hiding and to discuss common pitfalls. This review is done at a basic, conceptual level. A more “to the letter” review follows in Chapter 7.

在本章剩下的部分，我们回顾一下之前的工作中采取的几种方法。目的是描述对隐藏延迟进行建模的挑战，讨论常见的pitfalls。这个回顾是基本的，概念性的。更详细的回顾在第7章。

We start with the CUDA C programming guide [NVIDIA 2015], which briefly discusses a similar problem: given arithmetic intensity of a kernel, find the occupancy needed to attain the best possible throughput. The discussion is found in Chapter 5.2.3 of the programming guide starting with version 3.0 [NVIDIA 2010a]. It is based on considering a few numerical examples; in the following, we use the numbers cited in version 5.5 of the guide [NVIDIA 2013]. The discussion is simplistic but important, as a similar pattern recurs in a number of other, more elaborate models.

我们从CUDA C编程指南开始，其中也简要的讨论了类似的问题：给定一个kernel的算术运算强度，找到要获得最佳可能的吞吐率所需的占用率。在3.0版的编程指南的5.2.3章中进行了讨论。其中考虑了几个数值例子；下面，我们使用5.5版指南中的数字。这个讨论是简化的，但是很重要，因为在几个其他更复杂的模型中，都重现了这种模式。

Suppose memory latency is mem_lat = 600 cycles, peak instruction throughput is instruction time = 4 CPI, and there are α = 30 arithmetic instructions per memory access instruction. If one warp stalls at a memory access, but instruction issue continues at the peak rate – which means that latency is hidden – then 150 instructions are issued by the time the stalled warp may continue. These instructions can only be issued from other warps. As only 30 arithmetic instructions can be issued from any particular warp before it similarly stalls on a memory access, “about 5 warps” are needed. This result can be summarized as:

假设访存延迟是mem_lat = 600周期，峰值指令吞吐率是instruction time = 4 CPI，每条访存指令有α = 30条算术运算指令。如果一个warp在一条访存指令处停住了，但指令发射以峰值速度继续，这意味着延迟被隐藏了，那么到停止的warp继续之前，发射了150条指令。这些指令只能从其他warps被发射。由于任意特定的warp在类似的停止于访存指令时，只能发射30条算术指令，所以大约需要5个warps。这个结果可以总结为：

warps needed (CUDA guide) ≈ mem_lat / (instruction time ∙ α)

This is the result presented in the programming guide. There are a few trivial elaborations it does not include. First, we didn’t count the stalled warp – the total warp count therefore is 6. Second, we did not include memory instructions – in total there are 31 instructions issued from each warp per memory stall. With these refinements, we get:

这就是编程指南中给出的结果。有几个细节并没有包含。首先，我们没有对停止的warp进行计数，总计的warp数量是6。第二，我们并没有包含访存指令，每次访存停止，总计有31条指令发射。有了这些修正，我们得到：

warps needed (refined) = mem_lat / (instruction time ∙ (α + 1)) + 1. 

However, this introduces only a little difference, and we use the original version below. 但是，这只是引入了很小的差异，我们下面使用原始的版本。

The substantial factor missing in this analysis is arithmetic latency. This is in line with the popular understanding of GPUs as processors that use multithreading to hide the long memory latency, in which case the shorter arithmetic latency is often deemphasized. This omission results in a substantial error.

在这个分析中缺失的重大因素是算术运算延迟。流行的GPUs理解是处理器使用多线程来隐藏长的访存延迟，在这种情况中，算术运算延迟通常就会忽视了。这种忽略会产生显著的误差。

The error can be found by rewriting the estimate as 我们重写这个估计，得到误差为

warps needed (CUDA guide) ≈ mem_lat ∙ alu_thru / α .

Here, we use the fact that most instructions in the example are arithmetic, so that peak instruction throughput approximately equals peak arithmetic throughput. We, therefore, replace instruction time with 1 / alu_thru and get the above.

这里，我们使用的事实是，例子中的多数指令都是算术运算指令，这样峰值指令吞吐率近似的等于峰值算术运算吞吐率。因此，我们将instruction time替换为1 / alu_thru，得到上面的式子。

In this form, the estimate may be easily recognized as the number of memory instructions executed at the same time, which was found in §4.6. We also found that the number of warps executed at the same time is larger and equals

以这种形式，可以很容易得到这个估计，就是同时执行的访存指令数量，这可以在4.6节中找到。我们还可以发现，同时执行的warps数量更大一些，等于

warps needed = mem_lat ∙ alu_thru / α + alu_lat ∙ alu_thru. 

The difference is the additional term alu_lat ∙ alu_thru that accounts for hiding arithmetic latency. 差别就是额外的alu_lat ∙ alu_thru项，对应的是隐藏算术运算延迟。

To highlight the importance of this term, we compare the estimate with experiment. We choose similar arithmetic intensities to those used in the programming guide: large enough to keep the kernel compute-bound but small otherwise. Indeed, the estimate is not intended for cases where α is too small – throughput is then bound by memory bandwidth, which is not considered in the analysis. Also, the estimate is obviously inaccurate when α is very large: it converges to zero at α = ∞, where alu_lat ∙ alu_thru warps are needed in practice. We, therefore, set arithmetic intensities to the values approximately equal but somewhat larger than ratios alu_thru / mem_thru. These are larger values for newer GPUs, as these ratios on newer GPUs are similarly larger – this trend was discussed in §4.4. A similar trend is present in the programming guide: the considered arithmetic intensity is 10 in version 3.0, 15 in version 3.2 and 30 in version 5.0 – these are larger values in newer versions covering newer GPUs. The arithmetic intensities we choose are listed in Table 4.3.

为强调这一项的重要性，我们将估计与试验进行了比较。我们选择的算术运算强度与编程指南中使用的例子相似：足够大，可以使kernel是compute-bound，但是其他地方就比较小。确实，这个估计并不是为α较小的情况，那种情况下，吞吐率是受到带宽约束的，并不在本分析考虑的范围内。同时，当α很大时，这个估计明显就是不准确的。在α = ∞时，收敛到了0，这里在实践中需要alu_lat ∙ alu_thru个warps。因此，我们将算术运算强度设置为大致等于但又略大于alu_thru / mem_thru。对更新的GPUs，这些值更大，因为这个比率在更新的GPUs都是略大的，这个趋势在4.4中进行了讨论。在编程指南中，给出了类似的趋势：考虑的算术运算强度，在3.0版中是10，在3.2版中是15，在5.0版中是30，在更新的版本中这些值越大。我们选择的算术运算强度如表4.3所示。

For other input parameters – memory latencies and peak arithmetic throughputs – we use the same values as previously in this chapter. They are not always the same as the numbers quoted in the programming guide, but are more appropriate: they are found experimentally, tuned for the specific GPUs and the specific setup we use (e.g. assume no dual-issue) and are better consistent with prior work, such as with the latencies reported in Volkov and Demmel [2008], Hong and Kim [2009] and Wong et al. [2010]. These parameters and the resulting estimates are also listed in Table 4.3.

对于其他的输入参数，访存延迟和峰值算术运算吞吐率，我们使用的值与本章之前的相同。这些值并不总是与编程指南中的值相同，但却更合适：它们是通过试验得到的，对特定的GPU，特定的设置（比如，假设没有双发射）调整得到，与之前的工作更一致，比如，Volkov and Demmel [2008], Hong and Kim [2009] and Wong et al. [2010]中给出的延迟。这些参数和得到的估计也在表4.3中列出。

The comparison with experiment is shown in Figure 4.10. Experimental data is shown as dots, the original estimate is shown as a thick grey line, and the better estimate, which includes the additional term alu_lat ∙ alu_thru, is shown as a thin black line. According to the figure, the vendor’s model systematically underestimates the needed occupancy by a factor of about 2. The better estimate is nearly twice as large and is substantially more accurate, demonstrating that arithmetic latency is an important factor. The accuracy is still limited due to the unaccounted gradual saturation effect, which is especially pronounced on the G80 GPU. In the case of the Kepler GPU, the better estimate (74 warps) exceeds 100% occupancy and is not shown.

图4.10给出了与试验进行的对比。试验数据用点进行展示，原始的估计展示为厚的灰色线，更好的估计，即包括了额外的项alu_lat ∙ alu_thru，用细的黑线进行展示。根据图示，供应商的模型系统的低估了需要的吞吐率，大约只有一半。更好的估计大约是2倍大，更加准确，说明算术运算延迟是一个重要的因素。这个准确率仍然是有限的，因为渐进饱和效果并没有纳入考虑，这在G80 GPU上非常明显。在Kepler GPU的情况中，更好的估计（74 warps）超过了100%的占用率，所以没有进行显示。

The programming guide is not fully oblivious of arithmetic latency. For example, it is considered in another example which involves no memory accesses. The difficulty, therefore, is in understanding how to include both arithmetic and memory latency in the same analysis at the same time – or in realizing that it is necessary.

编程指南并非完全忽略算术运算延迟。比如，在另一个例子中，完全没有访存指令，进行了考虑。因此，难处在于，在同一个分析中同时考虑算术运算延迟和访存延迟。

## 4.9 Similar solutions in prior work

## 4.10 Similar models but with arithmetic latency

## 4.11 Considering execution of arithmetic and memory instructions in isolation

## 4.12 Do instruction times add or overlap?

## 4.13 Models with no memory bandwidth limits