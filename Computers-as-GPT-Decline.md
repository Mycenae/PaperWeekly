# The Decline of Computers as a General Purpose Technology

Neil C. Thompson & Svenja Spanuth @ MIT

Technological and economic forces are now pushing computing away from being general purpose and toward specialization. 技术和经济力量正在将计算从通用目标推向专用。

key insights:

- Moore’s Law was driven by technical achievements and a “general purpose technology” (GPT) economic cycle where market growth and investments in technical progress reinforced each other. These created strong economic incentives for users to standardize to fast-improving CPUs, rather than designing their own specialized processors.

摩尔定律是由技术进步和通用目标技术(GPT)经济周期推动的，其中市场的增长和在技术进步上的投资互相强化。这对用户创造了强劲的经济刺激，使用标准化的快速改进的CPUs，而不是设计其自己的专用处理器。

- Today, the GPT cycle is unwinding, resulting in less market growth and slower technical progress. 今天，GPT的周期正在展开，市场增长和技术进步放缓。

- As CPU improvement slows, economic incentives will push users toward specialized processors, which threatens to fragment computing. In such a computing landscape, some users will be in the ‘fast lane,’ benefiting from customized hardware, and others will be left in the ‘slow lane,’ stuck on CPUs whose progress fades.

由于CPU改进放缓，经济刺激会将用户推向专用处理器，这会使计算碎片化。在这样一种计算情境中，一些用户会在快车道上，从定制的硬件中获益，其他的会被留在慢车道上，在技术进步放缓的CPUs赛道上。

## 1. Introduction

Perhaps in no other technology has there been so many decades of large year-over-year improvements as in computing. It is estimated that a third of all productivity increases in the U.S. since 1974 have come from information technology, making it one of the largest contributors to national prosperity.

没有其他的技术在好几十年内都有逐年的很大的改进，除了计算。据估计，自从1974年以后，美国生产力的增长的1/3都是从信息技术而来，使其成为国家兴旺最大的贡献者。

The rise of computers is due to technical successes, but also to the economics forces that financed them. Bresnahan and Trajtenberg coined the term general purpose technology (GPT) for products, like computers, that have broad technical applicability and where product improvement and market growth could fuel each other for many decades. But, they also predicted that GPTs could run into challenges at the end of their life cycle: as progress slows, other technologies can displace the GPT in particular niches and undermine this economically reinforcing cycle. We are observing such a transition today as improvements in central processing units (CPUs) slow, and so applications move to specialized processors, for example, graphics processing units (GPUs), which can do fewer things than traditional universal processors, but perform those functions better. Many high profile applications are already following this trend, including deep learning (a form of machine learning) and Bitcoin mining.

计算机的兴起是因为技术成功，也是因为资助其的经济力量。Bresnahan和Trajtenberg创造了通用目标技术(GPT)的术语，用于像计算机这样的产品，有很广泛的技术应用性，其中产品改进和市场增长可以在几十年中互相刺激。但是，他们也预测，GPTs会在其声明周期的最后，受到挑战：随着进步放缓，其他技术会取代GPT，特别是生态位，逐渐破坏这种经济上互相增强的循环。我们今天观察到这样一种迁移，因为CPU的进步正在放缓，所以应用正移到专用处理器，比如，GPUs，比传统的通用处理器做的事情少，但可以更好的发挥作用。很多高调的应用已经跟随这个趋势，包括深度学习（一种机器学习的形式），和比特币挖矿。

With this background, we can now be more precise about our thesis: “The Decline of Computers as a General Purpose Technology.” We do not mean that computers, taken together, will lose technical abilities and thus ‘forget’ how to do some calculations. We do mean that the economic cycle that has led to the usage of a common computing platform, underpinned by rapidly improving universal processors, is giving way to a fragmentary cycle, where economics push users toward divergent computing platforms driven by special purpose processors.

有了这个背景，我们可以对我们的主题更加精确：计算机作为通用目标技术的衰落。我们的意思不是计算机会损失技术能力，忘掉怎样做一些计算。我们的意思是，经济周期带来了通用计算平台的使用，迅速改进的通用处理器进行了增强，这个正在让位于一个碎片化的周期，其中经济正将用户推向不同的计算平台，这是受特殊目的的处理驱动的。

This fragmentation means that parts of computing will progress at different rates. This will be fine for applications that move in the ‘fast lane,’ where improvements continue to be rapid, but bad for applications that no longer get to benefit from field-leaders pushing computing forward, and are thus consigned to a ‘slow lane’ of computing improvements. This transition may also slow the overall pace of computer improvement, jeopardizing this important source of economic prosperity.

这种碎片化意味着计算的不同部分会以不同的速度来进行。这对于在快车道的应用非常好，其中的改进仍然很快，但对于从中无法获得好处的应用就不太好了，因此成为了计算改进的慢车道。这种迁移会使计算改进的总体步伐变慢，对这种经济增长的重要源头造成伤害。

## 2. Universal and Specialized Computing

**Early days—from specialized to universal**. Early electronics were not universal computers that could perform many different calculations, but dedicated pieces of equipment, such as radios or televisions, designed to do one task, and only one task. This specialized approach has advantages: design complexity is manageable and the processor is efficient, working faster and using less power. But specialized processors are also ‘narrower,’ in that they can be used by fewer applications.

**早期-从专用到通用**。早期电子并不是通用计算机，可以进行很多不同的计算，而是专用设备，比如收音机或电视，设计来做一种工作，而且只做一种工作。这种专用的方法有优势：设计复杂度是有限的，处理器是高效的，工作速度很快，使用的能耗很小。但专用的处理器同时也更窄，只能进行很少几种应用。

Early electronic computers, even those designed to be ‘universal,’ were in practice tailored for specific algorithms and were difficult to adapt for others. For example, although the 1946 ENIAC was a theoretically universal computer, it was primarily used to compute artillery range tables. If even a slightly different calculation was needed, the computer would have to be manually re-wired to implement a new hardware design. The key to resolving this problem was a new computer architecture that could store instructions. This architecture made the computer more flexible, making it possible to execute many different algorithms on universal hardware, rather than on specialized hardware. This ‘von Neumann architecture’ has been so successful that it continues to be the basis of virtually all universal processors today.

早期电子计算机，即使是那些设计用于通用的，也是为特定算法定制的，很难改变用于其他的。比如，虽然1946 ENIAC是一个理论上的通用计算机，但其主要用于计算火炮射程表。即使要用于计算稍微不同的计算，这个计算机就必须手动重新连线，实现一种新的硬件设计。解决这个问题的关键，是一种新的计算机架构，可以存储指令。这种架构使计算机更加灵活，使其可以在通用硬件上执行很多不同的算法，而不是在专用硬件上执行。这种冯诺依曼架构非常成功，是今天所有通用处理器的基础。

**The ascent of universal processors**. Many technologies, when they are introduced into the market, experience a virtuous reinforcing cycle that helps them develop (Figure 1a). Early adopters buy the product, which finances investment to make the product better. As the product improves, more consumers buy it, which finances the next round of progress, and so on. For many products, this cycle winds down in the short-to-medium term as product improvement becomes too difficult or market growth stagnates.

**通用处理器的崛起**。很多技术在进入到市场时，会经历一个互相强化的周期，帮助它们发展（图1a）。早期的采用者购买产品，这些资金使得产品更好。随着产品改进，更多的消费者进行购买，会资助下一轮的改进，等等。对很多产品，这个周期在从短期到中期的时间内进行循环，直到产品改进变得太难，或市场增长停滞。

GPTs are defined by the ability to continue benefiting from this virtuous economic cycle as they grow—as universal processors have for decades. The market has grown from a few high-value applications in the military, space, and so on, to more than two billion PCs in use worldwide. This market growth has fueled ever-greater investments to improve processors. For example, Intel has spent 183 billion dollars on R&D and new fabrication facilities over the last decade. This has paid enormous dividends: by one estimate processor performance has improved about 400,000x since 1971.

GPTs的定义，是其能够持续的从这个经济周期中受益，大家可以看到，通用处理器的这个周期进行了几十年。市场从几个高价值应用，如军用，航空等，增长到世界范围内使用的超过20亿PCs。这个市场增长保证了持续的更大的投资来改进处理器。比如，Intel在过去十年中，在研发和新制造设备上花费了183 billion dollars。据估计，自从1971年，处理器性能改进了大约40万倍。

**The alternative: Specialized processors**. A universal processor must be able to do many different calculations well. This leads to design compromises that make many calculations fast, but none optimal. The performance penalty from this compromise is high for applications well suited to specialization, that is those where:

**替代品：专用处理器**。通用处理器要能够进行很多不同的计算。这会带来设计上的折中，使很多计算都很快，但是每个都不是最优的。这种折中的性能惩罚，对于非常适用于专用的应用是很高的，这些应用是：

- substantial numbers of calculations can be parallelized 很多计算可以并行化

- the computations to be done are stable and arrive at regular intervals (‘regularity’) 要进行的计算是稳定的，有稳定的周期（规则性）

- relatively few memory accesses are needed for a given amount of computation (‘locality’) 对一定数量的计算，需要的内存访问数量相对较少（局部性）

- calculations can be done with fewer significant digits of precision. 用更低的精度也可以进行计算

In each of these cases, specialized processors (for example, Applicationspecific Integrated Circuits (ASICs)) or specialized parts of heterogeneous chips (for example, I.P. blocks) can perform better because custom hardware can be tailored to the calculation.

这些情况的每一个，专用处理器或异质芯片的专用部分都可以表现更好，因为对这些计算可以定制硬件。

The extent to which specialization leads to changes in processor design can be seen in the comparison of a typical CPU—the dominant universal processor—and a typical GPU—the mostcommon type of specialized processor (see the accompanying table).

专用化带来的处理器设计的改变的程度，可以在表中看出，这是一个典型的CPU和一个典型的GPU的对比。

The GPU runs slower, at about a third of the CPU’s frequency, but in each clock cycle it can perform ~100x more calculations in parallel than the CPU. This makes it much quicker than a CPU for tasks with lots of parallelism, but slower for those with little parallelism.

GPU运行的更慢，是CPU频率的大约1/3，但在每个时钟周期中，比CPU可以进行的并行计算多大约100x。这使其在有很多并行的任务中，比CPU快很多，但在基本没有并行性的任务中慢了很多。

GPUs often have 5x–10x more memory bandwidth (determining how much data can be moved at once), but with much longer lags in accessing that data (at least 6x as many clock cycles from the closest memory). This makes GPUs better at predictable calculations (where the data needed from memory can be anticipated and brought to the processor at the right time) and worse at unpredictable ones.

GPUs的内存带宽通常要高5x-10x（决定了一次可以移动多少数据），但访问数据时，延迟更长（至少6倍的时钟周期）。这使GPU在可预测的计算中效果更好（在内存中需要的数据可以在正确的时间带到处理器中），在不可预测的计算中效果差一些。

For applications that are well matched to specialized hardware (and where programming models, for example CUDA, are available to harness that hardware), the gains in performance can be substantial. For example, in 2017, NVIDIA, the leading manufacturer of GPUs, estimated that Deep Learning (AlexNet with Caffe) got a speed-up of 35x+ from being run on a GPU instead of a CPU. Today, this speed-up is even greater.

与专用硬件匹配的很好的应用，性能的提升可能会非常大。比如，在2017年，NVIDIA估计，深度学习(AlexNet with Caffe)在GPU上运行，比在CPU上运行加速了35x+。今天，这个加速更大。

Another important benefit of specialized processors is that they use less power to do the same calculation. This is particularly valuable for applications limited by battery life (cell phones, Internet-of-things devices), and those that do computation at enormous scales (cloud computing/datacenters, supercomputing).

专用处理器的另一种重要的好处是，进行相同的计算它们使用的能量更少。这对于受限于电池寿命的应用来说，是非常宝贵的，比如手机，IoT设备，对于那些很大尺度上的计算，也是很重要的（云计算/数据中心，超算）。

As of 2019, 9 out of the top 10 most power efficient supercomputers were using NVIDIA GPUs. 在2019年，功耗效率最高的超算中，10个中的9个都使用了NVIDIA GPUs。

Specialized processors also have important drawbacks: they can only run a limited range of programs, are hard to program, and often require a universal processor running an operating system to control (one or more of) them. Designing and creating specialized hardware can also be expensive. For universal processors, their fixed costs (also called non-recurring engineering costs (NRE)) are distributed over a large number of chips. In contrast, specialized processors often have much smaller markets, and thus higher per-chip fixed costs. To make this more concrete, the overall cost to manufacture a chip with specialized processors using leading-edge technology is about 80 million dollars (as of 2018). Using an older generation of technology can bring this cost down to about 30 million dollars.

专用处理器的缺陷也很明显：它们只能运行有限的程序，而且编程很困难，通常需要一个通用处理器上运行一个操作系统来控制它们。设计创建专用硬件也很昂贵。对于通用处理器，它们的固定成本（也称为NRE）是分布在大量芯片中的。对比起来，专用处理器的市场通常更小，因此每个芯片的固定成本会更高。更具体一些，使用先进技术制造一个专用处理器芯片的总体成本，在2018年大约是80 million dollars。使用更老的工艺，会使成本降低到30 million dollars。

Despite the advantages that specialized processors have, their disadvantages were important enough that there was little adoption (except for GPUs) in the past decades. The adoption that did happen was in areas where the performance improvement was inordinately valuable, including military applications, gaming and cryptocurrency mining. But this is starting to change.

尽管专用处理器有一些优势，但它们的劣势也非常重要，所以在过去几十年采用的很少，除了GPUs。确实采用的，都是在一些价值非常大的性能改进中，包括军用应用，游戏和加密货币挖矿。但这开始改变了。

**The state of specialized processors today**. All the major computing platforms, PCs, mobile, Internet-of-things (IoT), and cloud/supercomputing, are becoming more specialized. Of these, PCs remain the most universal. In contrast, energy efficiency is more important in mobile and IoT because of battery life, and thus, much of the circuitry on a smartphone chip, and sensors, such as RFID-tags, use specialized processors.

**今天专用处理器的状态**。所有主要的计算平台，PCs，移动平台，IoT，和云计算、超算，都在更加专用。其中，PCs是最通用的。对比起来，功耗效率在移动设备和IoT中更重要，因为有电池寿命问题，因此，在智能手机芯片，RFID这样的传感器上的电路，都使用的是专用处理器。

Cloud/supercomputing has also become more specialized. For example, 2018 was the first time that new additions to the biggest 500 supercomputers derived more performance from specialized processors than from universal processors.

云/超算也变得更加专用。比如，2018年新加入的最强的500个超算中，专用处理器的性能第一次超过了通用处理器。

Industry experts at the International Technology Roadmap for Semiconductors (ITRS), the group which coordinated the technology improvements needed to keep Moore’s Law going, implicitly endorsed this shift toward specialization in their final report. They acknowledged the traditional one-solution-fits-all approach of shrinking transistors should no longer determine design requirements and instead these should be tailored to specific applications.

ITRS中的工业专家在其最终报告中隐式的同意了向专用的转换。他们承认，传统的一个解决方案都适用的方法，即缩小晶体管，不能再确定设计需求了，而是要为特定的应用定制。

The next section explores the effect that the movement of all of the major computing platforms toward specialized processors will have on the economics of producing universal processors.

下一节会探索，所有主要计算平台向专用处理器的迁移，对生产通用处理器的经济的影响。

## 3. The Fragmentation of a General Purpose Technology

The virtuous cycle that underpins GPTs comes from a mutually reinforcing set of technical and economic forces. Unfortunately, this mutual reinforcement also applies in the reverse direction: if improvements slow in one part of the cycle, so will improvements in other parts of the cycle. We call this counterpoint a ‘fragmenting cycle’ because it has the potential to fragment computing into a set of loosely-related siloes that advance at different rates.

加强GPTs的周期是来自技术和经济力量的相互增强。不幸的是，这种相互强化反方向也是成立的：如果周期中的一部分改进变慢了，周期中的其他部分的改进也会变慢。我们称这个为碎片化的周期，因为可能会将计算碎片化为松散相关的竖井，会以不同的速率推进。

As Figure 1(b) shows, the fragmenting cycle has three parts: 图1(b)展示了碎片化的周期有三个部分：

- Technology advances slow 技术进步变慢

- Fewer new users adopt 更少有新用户采用

- Financing innovation is more difficult 金融创新更加困难

The intuition behind this cycle is straightforward: if technology advances slow, then fewer new users adopt. But, without the market growth provided by those users, the rising costs needed to improve the technology can become prohibitive, slowing advances. And thus each part of this synergistic reaction further reinforces the fragmentation.

这个周期后的直觉是很直观的：如果技术进步变慢，采用的新用户会更少。但是，没有了这些用户带来的市场增长，改进技术的越来越高的成本会变得不可行，减缓进步。因此，这个协同性的相互作用的每一部分，都会进一步强化这种碎片化。

Here, we describe the state of each of these three parts of the cycle for computing and show that fragmentation has already begun. 这里，我们描述了计算周期中的三个部分的每个的状态，这表明碎片化已经开始了。

**Technology advancements slow**. To measure the rate of improvement of processors we consider two key metrics: performance and performance-per-dollar. Historically, both of these metrics improved rapidly, largely because miniaturizing transistors led to greater density of transistors per chip (Moore’s Law) and to faster transistor switching speeds (via Dennard Scaling). Unfortunately, Dennard Scaling ended in 2004/2005 because of technical challenges and Moore’s Law is coming to an end as manufacturers hit the physical limits of what existing materials and designs can do, and these limits take ever more effort to overcome. The loss of the benefits of miniaturization can be seen vividly in the slowdown of improvements to performance and performance-per-dollar.

**技术进步变慢**。为度量处理器的改进速度，我们考虑两个关键度量：性能，和每dollar的性能。历史上，这两个度量都增长迅速，主要因为晶体管的缩小带来了芯片上的晶体管密度更大，晶体管切换状态的速度更快。不幸的是，Dennard定律在2004/2005年停止了，因为技术挑战太大，而且Moore定律也因为遇到了现存的物质的物理极限也停止了，这些限制需要很大的努力来克服。没有了缩小的好处，可以从性能和每dollar性能的减缓明显看出来。

Figure 2(a), based Hennessy and Patterson’s characterization of progress in SPECInt, as well as Figure 2(b) based on the U.S. Bureau of Labor Statistics’ producer-price index, show how dramatic the slowdown in performance improvement in universal computers has been. To put these rates into perspective, if performance per dollar improves at 48% per year, then in 10 years it improves 50x. In contrast, if it only improves at 8% per year, then in 10 years it is only 2x better.

图2(a)是基于Hennessy和Patterson在SPECInt的特征的过程，图2(b)是基于US劳动统计局的物价指数得到，这表明通用计算机的性能改进的减缓是很明显的。如果每年每dollar性能改进48%，在10年中，会提升50x。对比起来，如果每年只改进8%，那么10年只会提升2x。

**Fewer new users adopt**. As the pace of improvement in universal processors slows, fewer programs with new functionality will be created, and thus customers will have less incentive to replace their computing devices. Intel CEO Krzanich confirmed this in 2016, saying that the replacement rate of PCs had risen from every four years to every 5–6 years. Sometimes, customers even skip multiple generations of processor improvement before it is worth updating. This is also true on other platforms, for example U.S. smartphones were upgraded on average every 23 months in 2014, but by 2018 this had lengthened to 31 months.

**新用户采用的更少**。随着通用处理器的改进步伐变慢，有新功能的程序会更少，用户维修其计算设备的动力会更小。Intel CEO在2016年确认了这个，说PCs的替换率从每4年提升到了每5-6年。有时，用户甚至跳过多代处理器改进，然后再进行升级。这在其他平台上也是对的，比如，US智能手机在2014年是每23个月更新，但在2018年，加长到了每31个月。

The movement of users from universal to specialized processors is central to our argument about the fragmentation of computing, and hence we discuss it in detail. Consider a user that could use either a universal processor or a specialized one, but who wants the one that will provide the best performance at the lowest cost. Figures 3(a) and 3(b) present the intuition for our analysis. Each panel shows the performance over time of universal and specialized processors, but with different rates at which the universal processor improves. In all cases, we assume that the time, T, is chosen so the higher price of a specialized processor is exactly balanced out by the costs of a series of (improving) universal processors. This means that both curves are cost equivalent, and thus superior performance also implies superior performance-perdollar. This is also why we depict the specialized processor as having constant performance over this period. (At the point where the specialized processor would be upgraded, it too would get the benefit of whatever process improvement had benefited the universal processor and the user would again repeat this same decision process.)

用户从通用处理器迁移到专用处理器，是计算碎片化观点的核心，因此我们会详细讨论。考虑一个用户，可以使用一个通用处理器，也可以使用一个专用处理器，但会需要以最低的代价得到最好的性能。图3(a)和3(b)给出了我们的分析的图示。每个面板展示了通用处理器和专用处理器随着时间的性能，但是通用处理器性能改进的速度不一样。在所有情况中，我们假设时间T的选择，专用处理器较高的价格，与一系列改进的通用处理器的价格相平衡的。这意味着，两条曲线是价格等价的，因此更好的性能也意味着，更好的每dollar性能。这也是为什么，我们认为专用处理器在这段时间内的性能是常数。

A specialized processor is more attractive if it provides a larger initial gain in performance. But, it also becomes more attractive if universal processor improvements go from a rapid rate, as in panel (a), to a slower one, as in panel (b). We model this formally by considering which of two time paths provides more benefit. That is, a specialized processor is more attractive if

专用处理器如果能提供性能的很大的初始收益，就会很好。但是，如果通用处理器的改进从高速度变到低速度，如面板(a)到面板(b)，那么专用处理器也会非常有吸引力。我们正式对其进行建模，考虑这两条时间路径哪条提供更多的优势。即，专用处理器在下式成立时更有吸引力

$$int_0^T P_s dt >= \int_0^T P_{u,t_0} e^{rt} dt$$

Where universal and specialized processors deliver performance Pu, and Ps, over time T, while the universal processor improves at r. We present our full derivation of this model in the online appendix (https://doi.org/10.1145/3430936). That derivation allows us to numerically estimate the volume needed for the advantages of specialization to outweigh the higher costs (shown in Figure 3(c) for a slowdown from 48% to 8% in the per-year improvement rate of CPUs).

其中通用处理器和专用处理器在时间T中提供的性能为Pu和Ps，通用处理器的性能改进速度为r。我们在附录中进行了完整推导。这个推导使我们可以数值上估计需要的体量，让专用处理器的优势，超过更高的代价。

Not surprisingly, specialized processors are more attractive when they provide larger speedups or when their costs can be amortized over larger volumes. These cutoffs for when specialization becomes attractive change, however, based on the pace of improvement of the universal processors. Importantly, this effect does not arise because we are assuming different rates of progress between specialized and universal processors overall—all processors are assumed to be able to use whatever is the cutting-edge fabrication technology of the moment. Instead, it arises because the higher perunit NRE of specialized processors must be amortized and how well this compares to upgrading universal processors over that period.

并不令人惊讶的是，专用处理器如果提供了更大的加速，或其价格以更大的体量分担，就会变得更有吸引力。专用什么时候变得有吸引力，是基于通用处理器改进的速度的。重要的是，这个效果并没有出现，因为我们假设专用和通用处理器的速度是不同的，所有的处理器都可以使用最新的制造工艺。其出现是因为，专用处理器的每单元NRE必须被分摊，这与升级通用处理器相比会怎样。

A numerical example makes clear the importance of this change. At the peak of Moore’s Law, when improvements were 48% per year, even if specialized processors were 100x faster than universal ones, that is, Ps/Pu = 100 (a huge difference), then ~83,000 would need to be built for the investment to pay off. At the other extreme, if the performance benefit were only 2x, ~1,000,000 would need to be built to make specialization attractive. These results make it clear why, during the heyday of Moore’s Law, it was so difficult for specialized processors to break into the market.

这种变化的重要性，可以从一个数值例子中看出来。在Moore定律的峰值处，改进是每年48%，即使专用处理器比通用处理器快了100x，即Ps/Pu = 100，那么就需要83000的量使投资来平衡。在另一个极端，如果性能收益只有2x，那么就需要1000000来使专用处理器有吸引力。这就很清楚了，在Moore定律的全盛期，专用处理器很难打入市场。

However, if we repeat our processor choice calculations using an improvement rate of 8%, the rate from 2008–2013, these results change markedly: for applications with 100x speed-up, the number of processors needed falls from 83,000 to 15,000, and for those with 2x speed-up it drops from 1,000,000 to 81,000. Thus, after universal processor progress slows, many more applications became viable for specialization.

但是，如果我们重复处理器选择的计算，但使用的改进速率为8%，也就是2008-2013年的速率，那么结果变化会非常大：对于有100x加速的应用，需要的处理器数量从83000下降到了15000，对于2x的加速，就从1000000下降到了81000。因此，在通用处理器改进减慢后，很多应用都可以采用专用处理器来进行。

**Financing innovation is harder**. In 2017, the Semiconductor Industry Association estimated that the cost to build and equip a fabrication facility (‘fab’) for the next-generation of chips was roughly 7 billion dollars. By “next-generation,” we mean the next miniaturization of chip components (or process ‘node’).

**金融创新更难**。在2017年，半导体工业协会估计，建造并装备一个下一代芯片的fab厂的价格为大约7 billion dollars。下一代的意思是，芯片组成部分的下一次缩小。

The costs invested in chip manufacturing facilities must be justified by the revenues that they produce. Perhaps as much as 30% of the industry’s 343 billion dollars annual revenue (2016) comes from cutting-edge chips. So revenues are substantial, but costs are growing. In the past 25 years, the investment to build leading-edge fab (as shown in Figure 4a) rose 11% per year (!), driven overwhelmingly by lithography costs. Including process development costs into this estimate further accelerates cost increases to 13% per year (as measured for 2001 to 2014 by Santhanam et al.). This is well known by chipmakers who quip about Moore’s “second law”: the cost of a chip fab doubles every four years.

芯片制造设备上投资的价值，必须要被产生的收入平衡。可能2016年工业年产值的343 billion的最多30%是从最尖端的芯片来的。所以收入是可观的，但是价值在不断增长。在过去25年中，建造尖端fab的投资每年提升11%（如图4a所示），主要是由光刻的价格推高的。将工艺开发纳入到这里，增速会提高到13%每年。芯片制造者都知道这个Moore第二定律：芯片制造的价格每4年翻倍。

Historically, the implications of such a rapid increase in fixed cost on unit costs was only partially offset by strong overall semiconductor market growth (CAGR of 5% from 1996–2016), which allowed semiconductor manufacturers to amortize fixed costs across greater volumes. The remainder of the large gap between fixed costs rising 13% annually and the market growing 5% annually, would be expected to lead to less-competitive players leaving the market and remaining players amortizing their fixed costs over a larger number of chips.

历史上，这样快速的固定价值增长，部分被强劲的半导体市场增速补偿，使半导体制造商将固定代价分摊到更大的体量中。剩余的很大的空白，固定价格每年增加13%，市场增速5%每年，会使没那么有竞争力的玩家离开市场，剩余的玩家分摊这些固定价值。

As Figure 4(b) shows, there has indeed been enormous consolidation in the industry, with fewer and fewer companies producing leading-edge chips. From 2002/2003 to 2014/2015/2016, the number of semiconductor manufacturers with a leading-edge fab has fallen from 25 to just 4: Intel, Taiwan Semiconductor Manufacturing Company (TSMC), Samsung and GlobalFoundries. And GlobalFoundries recently announced that they would not pursue development of the next node.

如图4(b)所示，工业上有很大的合并，越来越少的工厂生产尖端的芯片。从2002/2003到2014/2015/2016，有尖端工艺的半导体制造商的数量从25下降到4：Intel，TSMC，Samsung和GF。GF最近宣布，他们不会进行下一个节点的研发。

We find it very plausible this consolidation is caused by the worsening economics of rapidly rising fixed costs and only moderate market size growth. The extent to which market consolidation improves these economics can be seen through some back-of-the-envelope calculations. If the market were evenly partitioned amongst different companies, it would imply a growth in average market share from 4% = 100%/25 in 2002/2003 to 25% = 100%/4 in 2014/2015/2016. Expressed as a compound annual growth rate, this would be 14%. This means that producers could offset the worsening economics of fab construction through market growth and taking the market share of those exiting (13%<5%+14%).

我们发现这种合并很可能是由于固定价格的迅速攀升，和市场规模增长缓慢引起的。市场合并改进这些经济到什么程度，可以通过一些粗略计算得到。如果市场由不同公司平均分配，那么平均市场份额的增速从2002/2003年的4% = 100%/25，到在2014/2015/2016年的25% = 100%/4。复合年均增速会是14%。这意味着，厂商会用市场增长来弥补经济的恶化，拿走退出者的市场份额。

In practice, the market was not evenly divided, Intel had dominant share. As a result, Intel would not have been able to offset fixed cost growth this way. And indeed, over the past decade, the ratio of Intel’s fixed costs to its variable costs has risen from 60% to over 100%. This is particularly striking because in recent years Intel has slowed the pace of their release of new node sizes, which would be expected to decrease the pace at which they would need to make fixed costs investments.

实践中，市场并不是平均分配的，Intel的市场份额最大。结果是，Intel不能通过这种方法弥补固定价格。确实，在过去十年中，Intel的固定价格与其可变价格比，从60%增长到了100%。这是很惊人的，因为最近几年，Intel已经放缓了新节点的步伐。

The ability for market consolidation to offset fixed cost increases can only proceed for so long. If we project forward current trends, then by 2026 to 2032 (depending on market growth rates) leading-edge semiconductor manufacturing will only be able to support a single monopolist manufacturer, and yearly fixed costs to build a single new facility for each node size will be equal to yearly industry revenues (see endnote for details). We make this point not to argue that in late 2020s this will be the reality, but precisely to argue that current trends cannot continue and that within only about 10 years(!) manufacturers will be forced to dramatically slow down the release of new technology nodes and find other ways to control costs, both of which will further slow progress on universal processors.

市场合并来弥补固定价格增长的能力，只能维持一段时间。如果我们延续当前的趋势，那么到2026年到2032年，尖端半导体制造只能支撑一个垄断制造商，建造一个新的下一代节点的工厂的固定价格会等于其年产出。我们认为在20世纪20年代末会成为现实，但精确的说，目前的趋势无法持续，在10年以内，新技术节点的研发会急剧放缓，找到其他方式来控制代价，这两者都会减缓通用处理器的进度。

**The fragmentation cycle**. With each of the three parts of the fragmentation cycle already reinforcing each other, we expect to see more and more users facing meager improvements to universal processors and thus becoming interested in switching to specialized ones. For those with sufficient demand and computations well-suited to specialization (for example, deep learning), this will mean orders of magnitude improvement. For others, specialization will not be an option and they will remain on universal processors improving ever-more slowly.

**碎片化周期**。碎片化周期的三个部分都在相互强化，我们期待看到更多用户面临通用处理器的很小改进，因此会对切换到专用处理器感兴趣。对于那些有适用于专用计算的充足需求和计算的（比如，深度学习），这意味着几个数量级的改进。对于其他的，专用处理器不会是一个选项，会停留在改进逐渐放缓的通用处理器上。

## 4. Implications

**Who will specialize**. As shown in Figure 3(c), specialized processors will be adopted by those that get a large speedup from switching, and where enough processors would be demanded to justify fixed costs. Based on these criteria, it is perhaps not surprising that big tech companies have been amongst the first to invest in specialized processors, for example, Google, Microsoft, Baidu, and Alibaba. Unlike the specialization with GPUs, which still benefited a broad range of applications, or those in cryptographic circuits, which are valuable to most users, we expect narrower specialization going forward because only small numbers of processors will be needed to make the economics attractive.

**谁会专用化**。如图3(c)所示，专用处理器会被那么加速很大的应用采用，会需要足够数量的处理器来平衡固定价格。基于这些准则，大公司是第一个投资专用处理器的，比如，Google，微软，百度和阿里。GPU的专用会使相当多的应用受益，与之不同的是，我们期待更窄的专用会发展，因为只有小批量处理器需要，才会使经济更有吸引力。

We also expect significant usage from those who were not the original designer of the specialized processor, but who re-design their algorithm to take advantage of new hardware, as deep learning users did with GPUs.

我们还期待，很多并不是专用处理器的设计者，而是算法的重设计者来利用并从中受益，比如深度学习使用者从GPUs中受益。

**Who gets left behind**. Applications that do not move to specialized processors will likely fail to do so because they: 不会迁移到专用处理器的大概率会失败，因为：

- Get little performance benefit, 性能提升很少

- Are not a sufficiently large market to justify the upfront fixed costs, or 并没有足够大的市场来弥补固定价格

- Cannot coordinate their demand. 不能协调其需求

Earlier, we described four characteristics that make calculations amenable to speed-up using specialized processors. Absent these characteristics, there are only minimal performance gains, if any, to be had from specialization. An important example of this is databases. As one expert we interviewed told us: over the past decades, it has been clear that a specialized processor for databases could be very useful, but the calculations needed for databases are poorly-suited to being on a specialized processor.

前面，我们描述了加速使用专用处理器的四个特征。没有这些特征，性能提升会很小，如果有的话，也是从专用化来的。一个重要的例子是数据库。一位专家告诉我们：在过去十年中，数据库的专用处理器会非常有用，但数据库所需要的计算，不能成为一个专用处理器。

The second group that will not get specialized processors are those where there is insufficient demand to justify the upfront fixed costs. As we derived with our model, a market of many thousands of processors are needed to justify specialization. This could impact those doing intensive computing on a small scale (for example, research scientists doing rare calculations) or those whose calculations change rapidly over time and thus whose demand disappears quickly.

第二个群体不会得到专用处理器的，是需求量不足以弥补固定价格的。我们从模型中推导出，需要数千个处理器的市场，来弥补专用化。这会影响到那些小体量密集计算的（比如，研究科学家进行稀有的计算），或那么随着时间变化快速的计算，所以其需求会很快消失。

A third group that is likely to get left behind are those where no individual user represents sufficient demand, and where coordination is difficult. For example, even if thousands of small users would collectively have enough demand, getting them to collectively contribute to producing a specialized processor would be prohibitively difficult. Cloud computing companies can play an important role in mitigating this effect by financing the creation of specialized processors and then renting these out.

第三个被拉下的群体，是那些没有单独的用户代表充足的需求的，那么协调就很困难。比如，即使数千个小用户一起有足够的的需求，使其一起来生产专用处理器会非常困难。云计算公司在这其中会发挥很大的作用，可以创建专用处理器，并将其租出去。

**Will technological improvement bail us out**? To return us to a convergent cycle, where users switch back to universal processors, would require rapid improvement in performance and/or performance-per-dollar. But technological trends point in the opposite direction. For example on performance, it is expected that the final benefits from miniaturization will come at a price premium, and are only likely to be paid for by important commercial applications. There is even a question whether all of the remaining, technically-feasible, miniaturization will be done. Gartner predicts that more will be done, with 5nm node sizes being produced at scale by 2026, and TSMC recently announced plans for a 19.5B dollars 3nm plant for 2022. But many of the interviewees that we contacted for this study doubt were skeptical about whether it would be worthwhile miniaturizing for much longer.

**技术进步会使我们拜托困境吗**？为让我们回到收敛的循环，其中用户切换回通用处理器，会需要性能和每美元性能的快速改进。但技术趋势指向了相反的方向。比如性能，缩小的最终好处，会有很高的代价，只会由重要的商业应用支付。所有剩余的技术上可行的缩小是否会进行，这都是一个问题。Gartner预测，会做更多，5nm会在2026年量产，TSMC最近宣布2022年投资19.5B到3nm的工厂中。但我们采访的很多人都怀疑，是否值得缩小到这种程度。

Might another technological improvement restore the pace of universal processor improvements? Certainly, there is a lot of discussion of such technologies: quantum computing, carbon nanotubes, optical computing. Unfortunately, experts expect that it will be at least another decade before industry could engineer a quantum computer that is broader and thus could potentially substitute for classical universal computers. Other technologies that might hold broader promise will likely still need significantly more funding to develop and come to market.

会有另一个技术进步恢复通用处理器改进的步伐吗？这种技术是有很多讨论的，当然：量子计算，碳纳米管，光子计算。不幸的是，专家估计，至少需要十年，工业能够制造出一个量子计算机，能够替代经典通用计算机。其他的很有希望的技术仍然需要更多的投资，才能开发并到市场中。

## 5. Conclusion

Traditionally, the economics of computing were driven by the general purpose technology model where universal processors grew ever-better and market growth fuels rising investments to refine and improve them. For decades, this virtuous GPT cycle made computing one of the most important drivers of economic growth.

传统上，计算经济是受到通用目标技术模型驱动的，其中通用处理器增长非常好，市场增长为技术进步投资，并得到改进。在几十年中，GPT周期使计算成为经济增长最重要的驱动力量。

This article provides evidence that this GPT cycle is being replaced by a fragmenting cycle where these forces work to slow computing and divide users. We show each of the three parts of the fragmentation cycle are already underway: there has been a dramatic and ever-growing slowdown in the improvement rate of universal processors; the economic trade-off between buying universal and specialized processors has shifted dramatically toward specialized processors; and the rising fixed costs of building ever-better processors can no longer be covered by market growth rates.

本文给出了证据，GPT周期正被碎片化周期替代，其中这些力量减缓了计算，分化了用户。我们展示了，碎片化周期的这三个部分正在发生：通用处理器的改进速度放缓；购买通用还是专用处理器的折中，已经向专用处理器偏移；更好的处理器的增加的固定成本，不能被市场增长速度覆盖。

Collectively, these findings make it clear that the economics of processors has changed dramatically, pushing computing into specialized domains that are largely distinct and will provide fewer benefits to each other. Moreover, because this cycle is self-reinforcing, it will perpetuate itself, further fragmenting general purpose computing. As a result, more applications will split off and the rate of improvement of universal processors will further slow.

这些发现说明，处理器的经济已经极大的变化了，使计算进入专用领域。而且，因为这个周期是自我强化的，会永久化，进一步碎片化通用目标计算。结果是，更多应用会剥离，通用处理器的改进速率会进一步降低。

Our article thus highlights a crucial shift in the direction that economics is pushing computing, and poses a challenge to those who want to resist the fragmentation of computing.

本文因此强调了，计算的关键转换的方向，对那些想抵制碎片化计算趋势的人，提出了挑战。