# A New Golden Age for Computer Architecture

John L. Hennessy & David A. Patterson

We began our Turing Lecture June 4, 2018 with a review of computer architecture since the 1960s. In addition to that review, here, we highlight current challenges and identify future opportunities, projecting another golden age for the field of computer architecture in the next decade, much like the 1980s when we did the research that led to our award, delivering gains in cost, energy, and security, as well as performance.

我们以对1960s以来的计算机架构的回顾，来开始我们的2018.6.4的Turing讲座。除了这个回顾，这里，我们还要强调目前的挑战，找到未来的机会，在未来十年中投射一个计算机架构领域的另一个黄金年代，与1980s颇为相像，那时候我们的研究在价格，能耗和安全，以及性能方面带来了回报。

Software talks to hardware through a vocabulary called an instruction set architecture (ISA). By the early 1960s, IBM had four incompatible lines of computers, each with its own ISA, software stack, I/O system, and market niche—targeting small business, large business, scientific, and real time, respectively. IBM engineers, including ACM A.M. Turing Award laureate Fred Brooks, Jr., thought they could create a single ISA that would efficiently unify all four of these ISA bases. They needed a technical solution for how computers as inexpensive as those with 8-bit data paths and as fast as those with 64-bit data paths could share a single ISA. The data paths are the “brawn” of the processor in that they perform the arithmetic but are relatively easy to “widen” or “narrow.” The greatest challenge for computer designers then and now is the “brains” of the processor—the control hardware. Inspired by software programming, computing pioneer and Turing laureate Maurice Wilkes proposed how to simplify control. Control was specified as a two-dimensional array he called a “control store.” Each column of the array corresponded to one control line, each row was a microinstruction, and writing microinstructions was called microprogramming. A control store contains an ISA interpreter written using micro-instructions, so execution of a conventional instruction takes several microinstructions. The control store was implemented through memory, which was much less costly than logic gates.

软件与硬件通过一个字典进行对话，称为指令集架构(ISA)。在1960s早期，IBM有4个不兼容的计算机线，每个都其自己的ISA，软件栈，I/O系统，和市场定位，分别是小型企业，大型企业，科学计算和实时计算。很多著名的IBM工程师，都认为他们可以创造一个ISA，将这四种ISA进行统一起来。他们需要一个技术解决方案，让廉价如8-bit数据通路的，和昂贵如64-bit数据通路的计算机，都共用一个ISA。数据通路是处理器中的体力工作，因为进行的是代数运算，但相对容易来进行加宽或缩窄。计算机设计者最大的挑战是处理器的大脑 - 控制硬件。受软件编程启发，计算先驱和Turing获奖者Maurice Wilkes提出，怎样简化控制部分。控制指定为二维阵列，他称之为控制仓库。阵列的每一列对应着一条控制线，每一行是一条微指令，写微指令称为微编程。一个控制仓库包含一个ISA解释器，使用微指令写成，所以一条常规指令的执行，是几条微指令。控制仓库是用存储实现的，这比逻辑门要便宜的多。

The table here lists four models of the new System/360 ISA IBM announced April 7, 1964. The data paths vary by a factor of 8, memory capacity by a factor of 16, clock rate by nearly 4, performance by 50, and cost by nearly 6. The most expensive computers had the widest control stores because more complicated data paths used more control lines. The least-costly computers had narrower control stores due to simpler hardware but needed more micro-instructions since they took more clock cycles to execute a System/360 instruction.

这里的表格列出了1964.4.7 IBM宣布的新的System/360 ISA的四个模型。数据通路的不同是以8为因子的，存储能力是以16为因子的，时钟速率以接近4为因子，性能以接近50，价格以接近6。最昂贵的计算机有着最宽的控制仓库，因为更复杂的数据通路使用了更多的控制线。最便宜的计算机其控制仓库更窄，因为硬件更简单，但是需要更多的微指令，因为需要更多的时钟周期来执行一条System/360指令。

Facilitated by microprogramming, IBM bet the future of the company that the new ISA would revolutionize the computing industry and won the bet. IBM dominated its markets, and IBM mainframe descendants of the computer family announced 55 years ago still bring in 10 billion dollar in revenue per year.

受微编程促进，IBM认为在公司的未来，新的ISA会使计算工业发生革命，赢得这场赌注。IBM主宰市场，55年前宣布的IBM主机的后代，仍然每年会带来10 billion美元的收益。

As seen repeatedly, although the marketplace is an imperfect judge of technological issues, given the close ties between architecture and commercial computers, it eventually determines the success of architecture innovations that often require significant engineering investment.

我们重复看到的是，虽然市场对科技问题并不是完美的判断，但鉴于架构和商用电脑之间紧密的联系，这最终决定了架构创新的胜利，通常需要显著的工程投资。

## 1. Integrated circuits, CISC, 432, 8086, IBM PC.

When computers began using integrated circuits, Moore’s Law meant control stores could become much larger. Larger memories in turn allowed much more complicated ISAs. Consider that the control store of the VAX-11/780 from Digital Equipment Corp. in 1977 was 5,120 words × 96 bits, while its predecessor used only 256 words × 56 bits.

当计算机开始使用集成电路，摩尔定律意味着控制仓库应当变得大的多。然后，更大的内存会允许复杂的多的ISAs。考虑1977年DEC的VAX-11/780的控制仓库是5,120 words × 96 bits，而其前序只是256 words × 56 bits。

Some manufacturers chose to make microprogramming available by letting select customers add custom features they called “writable control store” (WCS). The most famous WCS computer was the Alto, Turing laureates Chuck Thacker and Butler Lampson, together with their colleagues, created for the Xerox Palo Alto Research Center in 1973. It was indeed the first personal computer, sporting the first bit-mapped display and first Ethernet local-area network. The device controllers for the novel display and network were microprograms stored in a 4,096-word × 32-bit WCS.

一些制造商选择让微编程可用，让选定的客户增加定制的特征，他们称之为可写的控制仓库(writable control store, WCS)。最著名的WCS计算机是Alto，这是图灵获奖者Chuck Thacker和Butler Lampson及其同事在1973年为Xerox Palo Alto研究中心创建的。这确实是第一台个人计算机，首先支持bit映射的显示，首先支持Ethernet局域网。新型显示和网络的设备控制器，是微程序，存储在4,096-word × 32-bit WCS中。

Microprocessors were still in the 8-bit era in the 1970s (such as the Intel 8080) and programmed primarily in assembly language. Rival designers would add novel instructions to outdo one another, showing their advantages through assembly language examples.

微处理器在1970s仍然处在8-bit时代（比如Intel 8080），主要采用汇编语言进行编程。竞争的设计者会增加新的指令，以互相超越，通过汇编语言例子，来展示其优势。

Gordon Moore believed Intel’s next ISA would last the lifetime of Intel, so he hired many clever computer science Ph.D.’s and sent them to a new facility in Portland to invent the next great ISA. The 8800, as Intel originally named it, was an ambitious computer architecture project for any era, certainly the most aggressive of the 1980s. It had 32-bit capability-based addressing, object-oriented architecture, variable-bit-length instructions, and its own operating system written in the then-new programming language Ada.

Gordon Moore相信，Intel的下一个ISA会持续Intel的终生，所以他雇佣了很多聪明的计算机科学博士，将他们送到了Portland一处新的设施内，发明下一个伟大的ISA。Intel将其命名为8800，在任何时代，都是一个有野心的计算机架构工程，在1980s当然是非常激进的。它有32-bit寻址的能力，面向对象的架构，变长指令，它自己的操作系统，用当时新的编程语言Ada写成。

This ambitious project was alas several years late, forcing Intel to start an emergency replacement effort in Santa Clara to deliver a 16-bit microprocessor in 1979. Intel gave the new team 52 weeks to develop the new “8086” ISA and design and build the chip. Given the tight schedule, designing the ISA took only 10 person-weeks over three regular calendar weeks, essentially by extending the 8-bit registers and instruction set of the 8080 to 16 bits. The team completed the 8086 on schedule but to little fanfare when announced.

这个有野心的工程推后了几年，迫使Intel 1979年在Santa Clara开始了一个紧急替代品工程，给出一个16-bit的微处理器。Intel给新团队52周时间，研发新的8086 ISA，设计并构建这个芯片。由于日程非常紧张，设计ISA只耗费了10人周，只有3个常规周，基本就是拓展了8080的8-bit寄存器和指令集到16 bits。团队按时完成了8086，但宣布的时候很少有炫耀。

To Intel’s great fortune, IBM was developing a personal computer to compete with the Apple II and needed a 16-bit microprocessor. IBM was interested in the Motorola 68000, which had an ISA similar to the IBM 360, but it was behind IBM’s aggressive schedule. IBM switched instead to an 8-bit bus version of the 8086. When IBM announced the PC on August 12, 1981, the hope was to sell 250,000 PCs by 1986. The company instead sold 100 million worldwide, bestowing a very bright future on the emergency replacement Intel ISA.

Intel运气好的是，IBM正在研发一种个人电脑，以与Apple II竞争，需要一种16-bit微处理器。IBM对Motorola 68000很感兴趣，其ISA与IBM 360类似，但落后于IBM激进的安排。IBM切换到8086的8-bit版本的总线上。当IBM在1981年8月12日宣布这个PC，希望到1986年能够卖出250000台PC。但公司在世界范围内卖出了1亿台，这给Intel ISA的紧急替代品一个非常光明的未来。

Intel’s original 8800 project was renamed iAPX-432 and finally announced in 1981, but it required several chips and had severe performance problems. It was discontinued in 1986, the year after Intel extended the 16-bit 8086 ISA in the 80386 by expanding its registers from 16 bits to 32 bits. Moore’s prediction was thus correct that the next ISA would last as long as Intel did, but the marketplace chose the emergency replacement 8086 rather than the anointed 432. As the architects of the Motorola 68000 and iAPX-432 both learned, the marketplace is rarely patient.

Intel的原创8800项目重命名为iAPX-432，最终在1981年宣布，但需要几块芯片，而且有严重的性能问题。项目在1986年停止，前一年Intel将16-bit的8086 ISA拓展到80386，将16位的寄存器拓展到了32位。Moore的预测因此是对的，下一个ISA会与Intel寿命一样，但市场选择了紧密替代品8086，而不是选定的432。Motorola 68000和iAPX-432的架构师都学习到了，市场很少是有耐心的。

## 2. From complex to reduced instruction set computers

The early 1980s saw several investigations into complex instruction set computers (CISC) enabled by the big microprograms in the larger control stores. With Unix demonstrating that even operating systems could use high-level languages, the critical question became: “What instructions would compilers generate?” instead of “What assembly language would programmers use?” Significantly raising the hardware/software interface created an opportunity for architecture innovation.

1980s早期，更大的控制仓库存储了大型的微程序，对CISC进行了几次探索。当Unix展示了，即使是操作系统也可以使用高级语言，关键的问题变成了：“编译器会生成什么样的指令？”而不是“程序员会使用什么样的编程语言？”这极大的提升了硬件/软件接口的地位，为架构创新提供了一次机会。

Turing laureate John Cocke and his colleagues developed simpler ISAs and compilers for minicomputers. As an experiment, they retargeted their research compilers to use only the simple register-register operations and load-store data transfers of the IBM 360 ISA, avoiding the more complicated instructions. They found that programs ran up to three times faster using the simple subset. Emer and Clark found 20% of the VAX instructions needed 60% of the microcode and represented only 0.2% of the execution time. One author (Patterson) spent a sabbatical at DEC to help reduce bugs in VAX microcode. If microprocessor manufacturers were going to follow the CISC ISA designs of the larger computers, he thought they would need a way to repair the microcode bugs. He wrote such a paper, but the journal Computer rejected it. Reviewers opined that it was a terrible idea to build microprocessors with ISAs so complicated that they needed to be repaired in the field. That rejection called into question the value of CISC ISAs for microprocessors. Ironically, modern CISC microprocessors do indeed include microcode repair mechanisms, but the main result of his paper rejection was to inspire him to work on less-complex ISAs for microprocessors—reduced instruction set computers (RISC).

Turing获奖者John Cocke和他的同事为微型机提出了更简单的ISA和编译器。作为一个试验，他们将其研究的编译器的目标重新定义为，改变IBM 360 ISA，只使用简单的寄存器-寄存器运算，和load-store数据传输，避免更复杂的指令。他们发现，只使用简单的子集，程序运行的速度快了3倍。Emer和Clark发现，20%的VAX指令需要60%的微码，只有0.2%的执行时间。一位作者在DEC休假期间，帮助减少VAX微码的bugs。如果微处理器制造商要按照更大的计算机的CISC ISA设计，他认为他们需要一种方法来修复微码bugs。他写了这样一篇文章，但期刊Computer拒绝发表。审稿人认为，要用这么复杂的ISA来制造微处理器，是一个很糟糕的主意，他们需要在实地进行修复。这个拒绝发表质疑了CISC ISAs对微处理器的价值。讽刺的是，现代CISC微处理器确实包括了微码修复机制，但论文拒绝发表的主要结果是，启发了他在没那么复杂的ISAs上努力，即RISC。

These observations and the shift to high-level languages led to the opportunity to switch from CISC to RISC. First, the RISC instructions were simplified so there was no need for a microcoded interpreter. The RISC instructions were typically as simple as microinstructions and could be executed directly by the hardware. Second, the fast memory, formerly used for the microcode interpreter of a CISC ISA, was repurposed to be a cache of RISC instructions. (A cache is a small, fast memory that buffers recently executed instructions, as such instructions are likely to be reused soon.) Third, register allocators based on Gregory Chaitin’s graph-coloring scheme made it much easier for compilers to efficiently use registers, which benefited these register-register ISAs. Finally, Moore’s Law meant there were enough transistors in the 1980s to include a full 32-bit datapath, along with instruction and data caches, in a single chip.

这些观察，和向高级语言的转变，带来了从CISC向RISC转变的机会。首先，RISC指令是简化的，所以没有必要有一个微编码的解释器。RISC指令一般与微指令一样简单，可以由硬件直接执行。第二，CISC ISA之前使用的微码解释器的快速存储，现在成为了RISC指令的Cache。（缓存是一种小型的快速存储，缓存了最近执行的指令，因为这样的指令很可能就被重新使用。）第三，基于Gregory Chaitin的图上色方案的寄存器分配器，使编译器更高效的使用寄存器变得更容易，使这些寄存器-寄存器ISAs受益。最后，Moore定律意味着1980s有足够多的晶体管，在一个芯片上就包含一个完整的32-bit数据通路，和指令和数据缓存。

For example, Figure 1 shows the RISC-I and MIPS microprocessors developed at the University of California, Berkeley, and Stanford University in 1982 and 1983, respectively, that demonstrated the benefits of RISC. These chips were eventually presented at the leading circuit conference, the IEEE International Solid-State Circuits Conference, in 1984. It was a remarkable moment when a few graduate students at Berkeley and Stanford could build microprocessors that were arguably superior to what industry could build.

比如，图1展示了RISC-I和MIPS微处理器，分别是1982和1983年开发的，证明了RISC的好处。这些芯片最后在顶级电路会议IEEE ISSCC 1984上展示。这是一个非凡的时刻，几个Berkeley和Stanford的学生，就可以创造出比工业界更优秀的微处理器。

These academic chips inspired many companies to build RISC microprocessors, which were the fastest for the next 15 years. The explanation is due to the following formula for processor performance:

这些学术芯片启发了很多公司来构建RISC微处理器，在之后的15年内都是最快的。这是因为处理器的性能是由下式决定的：

Time/Program = Instructions / Program × (Clock cycles) / Instruction × Time / (Clock cycle)

DEC engineers later showed that the more complicated CISC ISA executed about 75% of the number instructions per program as RISC (the first term), but in a similar technology CISC executed about five to six more clock cycles per instruction (the second term), making RISC microprocessors approximately 4× faster.

后来DEC的工程师展示了，在每个程序中，更复杂的CISC ISA执行的指令数量是RISC的75%（第一项），但相似技术的CISC执行的每指令时钟周期数，是RISC的5到6倍（第二项），使RISC微处理器快了大概4倍。

Such formulas were not part of computer architecture books in the 1980s, leading us to write Computer Architecture: A Quantitative Approach in 1989. The subtitle suggested the theme of the book: Use measurements and benchmarks to evaluate trade-offs quantitatively instead of relying more on the architect’s intuition and experience, as in the past. The quantitative approach we used was also inspired by what Turing laureate Donald Knuth’s book had done for algorithms.

这样的公式在1980s还不是计算机架构书籍的一部分，这使我们在1989年写了Computer Architecture: A Quantitative Approach一书。副标题表明本书的主题：使用度量和基准测试来定量的评估折中，而不是依赖于架构师的直觉和经验，这是过去做事的方法。我们使用的量化方法也受到了图灵获奖者Donald Knuth的书的启发。

## 3. VLIW, EPIC, Itanium

The next ISA innovation was supposed to succeed both RISC and CISC. Very long instruction word (VLIW) and its cousin, the explicitly parallel instruction computer (EPIC), the name Intel and Hewlett Packard gave to the approach, used wide instructions with multiple independent operations bundled together in each instruction. VLIW and EPIC advocates at the time believed if a single instruction could specify, say, six independent operations—two data transfers, two integer operations, and two floating point operations—and compiler technology could efficiently assign operations into the six instruction slots, the hardware could be made simpler. Like the RISC approach, VLIW and EPIC shifted work from the hardware to the compiler.

下一个ISA的创新被认为是要超越RISC和CISC的。VLIW及其表亲，EPIC，这是Intel和HP命名的，使用了宽指令，在每个指令中捆绑了多个独立的操作。那时候VLIW和EPIC相信，如果一个指令可以包含6个独立的操作，2个数据传输，2个整型运算，2个浮点型运算，编译器技术会高效的将运算指定到6个指令槽中，硬件可以做的更简单。与RISC方法类似，VLIW和EPIC将工作从硬件转移到了编译器。

Working together, Intel and Hewlett Packard designed a 64-bit processor based on EPIC ideas to replace the 32-bit x86. High expectations were set for the first EPIC processor, called Itanium by Intel and Hewlett Packard, but the reality did not match its developers’ early claims. Although the EPIC approach worked well for highly structured floating-point programs, it struggled to achieve high performance for integer programs that had less predictable cache misses or less-predictable branches. As Donald Knuth later noted: “The Itanium approach ... was supposed to be so terrific—until it turned out that the wished-for compilers were basically impossible to write.” Pundits noted delays and underperformance of Itanium and rechristened it “Itanic” after the ill-fated Titantic passenger ship. The marketplace again eventually ran out of patience, leading to a 64-bit version of the x86 as the successor to the 32-bit x86, and not Itanium.

Intel和HP一起，基于EPIC的思想设计了一个64-bit处理器，替换了32-bit的x86。对第一代EPIC处理器给予了很高期望，Intel和HP将其称为Itanium，但现实并不负荷开发者早期的声称。虽然EPIC方法对高度结构化的浮点型程序效果很好，但对于整数型程序来说，其可预测的缓存misses更少，可预测的分支也很少，其效果就不太好了。如同Donald Knuth后来说的，Itanium方法开始认为会非常好，但是最后发现，对于编译器来说，不可能写的出来。市场再一次失去了耐心，导致出现了64-bit版的x86，是32-bit x86的后续，而不是Itanium。

The good news is VLIW still matches narrower applications with small programs and simpler branches and omit caches, including digital-signal processing. 好消息是，VLIW仍然匹配的上更窄的应用，程序要小，分支要简单，忽略了缓存，包括数字信号处理。

## 4. RISC vs. CISC in the PC and Post-PC Eras

AMD and Intel used 500-person design teams and superior semiconductor technology to close the performance gap between x86 and RISC. Again inspired by the performance advantages of pipelining simple vs. complex instructions, the instruction decoder translated the complex x86 instructions into internal RISC-like microinstructions on the fly. AMD and Intel then pipelined the execution of the RISC microinstructions. Any ideas RISC designers were using for performance—separate instruction and data caches, second-level caches on chip, deep pipelines, and fetching and executing several instructions simultaneously—could then be incorporated into the x86. AMD and Intel shipped roughly 350 million x86 microprocessors annually at the peak of the PC era in 2011. The high volumes and low margins of the PC industry also meant lower prices than RISC computers.

AMD和Intel使用500人的设计团队，和优秀的半导体技术，来弥补x86和RISC之间的性能空白。再次受到了将简单vs复杂指令流水线化的性能优势的启发，指令译码器将复杂的x86指令翻译成了内部的类似RISC的运行中的微指令。AMD和Intel然后将RISC微指令的执行进行流水线化。RISC设计者用来改进性能的设计，如将指令和数据缓存分开，片上二级缓存，深流水线，多条指令同时取和执行，这时都可以应用到x86中。AMD和Intel在PC时代的顶峰2011年时，每年卖出大约3.5亿颗x86微处理器。PC工业的大体量和低利润率也意味着比RISC计算机价格要低。

Given the hundreds of millions of PCs sold worldwide each year, PC software became a giant market. Whereas software providers for the Unix marketplace would offer different software versions for the different commercial RISC ISAs—Alpha, HP-PA, MIPS, Power, and SPARC—the PC market enjoyed a single ISA, so software developers shipped “shrink wrap” software that was binary compatible with only the x86 ISA. A much larger software base, similar performance, and lower prices led the x86 to dominate both desktop computers and small-server markets by 2000.

每年在世界范围内卖出数亿台PC，那么PC软件也成为了一个巨大的市场。尽管Unix市场的软件提供者可以对不同的商用RISC ISAs提供不同的软件版本，Alpha，HP-PA，MIPS，Power和SPARC，但是PC市场仍然享受单个ISA，所以软件开发者只开发与x86 ISAs兼容的软件。到2000年，巨大的软件基础，类似的性能，和更低的价格，使x86主宰了桌面计算机和小型服务器市场。

Apple helped launch the post-PC era with the iPhone in 2007. Instead of buying microprocessors, smartphone companies built their own systems on a chip (SoC) using designs from other companies, including RISC processors from ARM. Mobile-device designers valued die area and energy efficiency as much as performance, disadvantaging CISC ISAs. Moreover, arrival of the Internet of Things vastly increased both the number of processors and the required trade-offs in die size, power, cost, and performance. This trend increased the importance of design time and cost, further disadvantaging CISC processors. In today’s post-PC era, x86 shipments have fallen almost 10% per year since the peak in 2011, while chips with RISC processors have skyrocketed to 20 billion. Today, 99% of 32-bit and 64-bit processors are RISC.

Apple用iPhone在2007年开启了后PC时代。智能手机厂家不购买微处理器，而是构建其自己的SoC，使用其他厂商的设计，包括ARM的RISC处理器。移动设备设计者将die面积和功耗与性能一样看中，这使CISC ISAs就没有了优势。而且，IoT的到来，极大的增加了处理器的数量，和在die大小，功耗，价格和性能之间的折中。这个趋势增加了设计时间和价格的重要性，进一步使CISC处理器丧失了优势。在今天的后PC时代，x86的出货量自从2011年的巅峰时刻每年降低了接近10%，而RISC处理器芯片激增到200亿。今天，99%的32-bit和64-bit处理器都是RISC的。

Concluding this historical review, we can say the marketplace settled the RISC-CISC debate; CISC won the later stages of the PC era, but RISC is winning the post-PC era. There have been no new CISC ISAs in decades. To our surprise, the consensus on the best ISA principles for general-purpose processors today is still RISC, 35 years after their introduction.

对历史回顾给出结论，我们可以说，市场给RISC-CISC的争论下了结论；CISC赢了PC时代的后面阶段，但RISC赢了后PC时代。几十年以内都没有新的CISC ISAs了。令我们惊讶的是，通用目的处理器的最好ISA仍然是RISC，自从其提出已经过了35年。

## 5. Current Challenges for Processor Architecture

While the previous section focused on the design of the instruction set architecture (ISA), most computer architects do not design new ISAs but implement existing ISAs in the prevailing implementation technology. Since the late 1970s, the technology of choice has been metal oxide semiconductor (MOS)-based integrated circuits, first n-type metal–oxide semiconductor (nMOS) and then complementary metal–oxide semiconductor (CMOS). The stunning rate of improvement in MOS technology—captured in Gordon Moore’s predictions—has been the driving factor enabling architects to design more aggressive methods for achieving performance for a given ISA. Moore’s original prediction in 1965 called for a doubling in transistor density yearly; in 1975, he revised it, projecting a doubling every two years. It eventually became called Moore’s Law. Because transistor density grows quadratically while speed grows linearly, architects used more transistors to improve performance.

之前的小节关注的是ISA的设计，多数计算机架构师不设计新的ISAs，而是在流行的实现技术中实现已有的ISAs。自从1970s后期，可选择的技术是基于MOS的IC，首先是n型MOS，然后是CMOS。MOS技术的改进速度令人震惊，这就是Moore定律的预测，这一直是架构师设计更激进的方法来对给定ISA获得性能的驱动因素。Moore在1965年最初的预测是每年晶体管密度加倍；在1975年进行了修正，每2年翻倍。因为晶体管密度以平方增长，而速度线性增长，架构师使用更多的晶体管来改进性能。

## 6. End of Moore’s Law and Dennard Scaling

Although Moore’s Law held for many decades (see Figure 2), it began to slow sometime around 2000 and by 2018 showed a roughly 15-fold gap between Moore’s prediction and current capability, an observation Moore made in 2003 that was inevitable. The current expectation is that the gap will continue to grow as CMOS technology approaches fundamental limits.

虽然摩尔定律在几十年内都是成立的（见图2），但在2000年左右开始变慢，到2018年，在Moore的预测和当前的能力之间，有了大约15倍的差距。当前的预期是，这个差距会持续增大，因为CMOS技术接近了基本的极限。

Accompanying Moore’s Law was a projection made by Robert Dennard called “Dennard scaling,” stating that as transistor density increased, power consumption per transistor would drop, so the power per mm^2 of silicon would be near constant. Since the computational capability of a mm^2 of silicon was increasing with each new generation of technology, computers would become more energy efficient. Dennard scaling began to slow significantly in 2007 and faded to almost nothing by 2012 (see Figure 3).

随着Moore定律，还有一个Dennard定律，说的是，随着晶体管密度增加，每个晶体管的功耗会下降，所以每mm^2的功耗接近常数。由于新技术的出现，每mm^2硅的计算能力一直在增加，计算机的功耗效率越来越高。Dennard定律在2007年显著变慢，到了2012年逐渐消失。

Between 1986 and about 2002, the exploitation of instruction level parallelism (ILP) was the primary architectural method for gaining performance and, along with improvements in speed of transistors, led to an annual performance increase of approximately 50%. The end of Dennard scaling meant architects had to find more efficient ways to exploit parallelism.

在1986年和2002年之间，ILP的利用是获得性能的主要架构方法，和晶体管速度的改进一起，使性能提升每年大约都是50%。Dennard定律的消失，意味着架构师必须找到更高效的方式来探索并行性。

To understand why increasing ILP caused greater inefficiency, consider a modern processor core like those from ARM, Intel, and AMD. Assume it has a 15-stage pipeline and can issue four instructions every clock cycle. It thus has up to 60 instructions in the pipeline at any moment in time, including approximately 15 branches, as they represent approximately 25% of executed instructions. To keep the pipeline full, branches are predicted and code is speculatively placed into the pipeline for execution. The use of speculation is both the source of ILP performance and of inefficiency. When branch prediction is perfect, speculation improves performance yet involves little added energy cost—it can even save energy—but when it “mispredicts” branches, the processor must throw away the incorrectly speculated instructions, and their computational work and energy are wasted. The internal state of the processor must also be restored to the state that existed before the mispredicted branch, expending additional time and energy.

为理解为什么增加ILP导致了更大的低效，考虑一个现代处理器核，比如ARM，Intel和AMD的。假设有15级流水线，每时钟周期可以发射4条指令。因此，在流水线的任意时刻都有60条指令，包含大约15个分支，因为它们大约占了执行指令的25%。为保持流水线满，分支要进行预测，推测改进了性能，但增加的能耗很小，甚至可以节约能耗，但当其预测分支错误时，处理器必须扔掉那些错误预测的指令，这些计算工作和能量就都浪费了。处理器的内部状态也必须恢复到错误预测分支存在之前的状态，消耗额外的时间和能量。

To see how challenging such a design is, consider the difficulty of correctly predicting the outcome of 15 branches. If a processor architect wants to limit wasted work to only 10% of the time, the processor must predict each branch correctly 99.3% of the time. Few general purpose programs have branches that can be predicted so accurately.

要看到这样一个设计多有挑战，考虑正确预测15个分支的输出的难度。如果一个处理器架构师要将浪费的工作限制在10%的时间，处理器必须在99.3%的时间内对每个分支都预测正确。通用目标程序的分支很少能够被预测的这么准确。

To appreciate how this wasted work adds up, consider the data in Figure 4, showing the fraction of instructions that are effectively executed but turn out to be wasted because the processor speculated incorrectly. On average, 19% of the instructions are wasted for these benchmarks on an Intel Core i7. The amount of wasted energy is greater, however, since the processor must use additional energy to restore the state when it speculates incorrectly. Measurements like these led many to conclude architects needed a different approach to achieve performance improvements. The multicore era was thus born.

要知道这些浪费的工作有多少，看一下图4的数据，给出的是已经执行的但是发现是浪费的指令的比例，因为处理器预测的错误。平均来说，对这些基准测试，在Intel Core i7上，19%的指令都被浪费掉了。浪费的能量的数量甚至更大，因为处理器预测错误时，需要用额外的能量来恢复状态。很多这样的度量结果，使很多人得出结论，架构师需要不同的方法来获得性能改进。多核时代就是这样产生的。

Multicore shifted responsibility for identifying parallelism and deciding how to exploit it to the programmer and to the language system. Multicore does not resolve the challenge of energy-efficient computation that was exacerbated by the end of Dennard scaling. Each active core burns power whether or not it contributes effectively to the computation. A primary hurdle is an old observation, called Amdahl’s Law, stating that the speedup from a parallel computer is limited by the portion of a computation that is sequential. To appreciate the importance of this observation, consider Figure 5, showing how much faster an application runs with up to 64 cores compared to a single core, assuming different portions of serial execution, where only one processor is active. For example, when only 1% of the time is serial, the speedup for a 64-processor configuration is about 35. Unfortunately, the power needed is proportional to 64 processors, so approximately 45% of the energy is wasted.

多核将识别并行性和决定怎样利用其的责任转移到了程序员和语言系统那里。多核并没有解决能耗效率计算的挑战，在Dennard定律消失的时候得到加剧。每个活跃的核都消耗能量，不管其对计算是否有有效贡献。一个主要的障碍是一个旧的观察结论，称之为Amdahl定律，说的是一个并行计算机的加速，受到顺序计算的比例限制。为认识到这个观察的重要性，考虑图5，这展示的是一个应用在64核上的运行，比在单核上运行的速度快了多少，假设序列执行有不同的比例，其中只有一个处理器是活跃的。比如，当只有1%的时间是序列处理时，64核处理器配置的加速是大约35。不幸的是，需要的功耗是与64核成比例的，所以大约45%的功耗被浪费了。

Real programs have more complex structures of course, with portions that allow varying numbers of processors to be used at any given moment in time. Nonetheless, the need to communicate and synchronize periodically means most applications have some portions that can effectively use only a fraction of the processors. Although Amdahl’s Law is more than 50 years old, it remains a difficult hurdle.

当然，真实的程序有更复杂的结构，这些比例可以使不同数量的处理器在任意时刻被使用。尽管如此，周期性通信和同步的需求，意味着多数应用都有一部分可以有效的利用处理器的一部分。虽然Amdahl定律已经超过了50年，这仍然是一个困难的障碍。

With the end of Dennard scaling, increasing the number of cores on a chip meant power is also increasing at nearly the same rate. Unfortunately, the power that goes into a processor must also be removed as heat. Multicore processors are thus limited by the thermal dissipation power (TDP), or average amount of power the package and cooling system can remove. Although some high-end data centers may use more advanced packages and cooling technology, no computer users would want to put a small heat exchanger on their desks or wear a radiator on their backs to cool their cellphones. The limit of TDP led directly to the era of “dark silicon,” whereby processors would slow on the clock rate and turn off idle cores to prevent overheating. Another way to view this approach is that some chips can reallocate their precious power from the idle cores to the active ones.

随着Dennard定律的消失，芯片上核数量的增加，意味着功耗也以同样的速率增加。不幸的是，进入处理器的功率也需要以热的形式散发出去。多核处理器因此受到热耗散功率(TDP)限制，或者说，封装和制冷系统可以耗散的平均功率。虽然一些高端数据中心会使用更高级的封装和散热技术，但是没有计算机使用者希望将一个小型热交换器放到其桌面上，或在其手机背后安装一个辐射器。TDP的限制，直接带来了暗硅时代的到来，借此处理器会降低时钟频率，关掉闲置的核，以防止过热。看待这种方法另一种方式是，一些芯片会将可贵的功耗从限制的核重新配置到活跃的核上。

An era without Dennard scaling, along with reduced Moore’s Law and Amdahl’s Law in full effect means inefficiency limits improvement in performance to only a few percent per year (see Figure 6). Achieving higher rates of performance improvement—as was seen in the 1980s and 1990s—will require new architectural approaches that use the integrated-circuit capability much more efficiently. We will return to what approaches might work after discussing another major shortcoming of modern computers—their support, or lack thereof, for computer security.

没有Dennard定律的时代，与缩减的Moore定律，和完整的Amdahl定律，意味着低效限制了性能的改进，每年只会有几个百分点（见图6）。获得更快的性能改进，就像1980s和1990s，会需要新的架构方法，更高效的使用IC的能力。我们在讨论现代计算机的另一个主要缺点后，再回来讨论什么方法会奏效，这个缺陷就是计算机安全方面的问题。

## 7. Overlooked Security

In the 1970s, processor architects focused significant attention on enhancing computer security with concepts ranging from protection rings to capabilities. It was well understood by these architects that most bugs would be in software, but they believed architectural support could help. These features were largely unused by operating systems that were deliberately focused on supposedly benign environments (such as personal computers), and the features involved significant overhead then, so were eliminated. In the software community, many thought formal verification and techniques like microkernels would provide effective mechanisms for building highly secure software. Unfortunately, the scale of our collective software systems and the drive for performance meant such techniques could not keep up with processor performance. The result is large software systems continue to have many security flaws, with the effect amplified due to the vast and increasing amount of personal information online and the use of cloud-based computing, which shares physical hardware among potential adversaries.

在1970s年代，处理器架构师有很多注意力在增强计算机的安全性上，产生了protection rings到capabilities之类的概念。这些架构师非常理解，多数bugs都是在软件中的，但他们相信，架构上的支持会起到帮助作用。这些特征操作系统基本没有使用，关注的都是设想中的恶意环境（比如个人电脑），这些特征涉及到显著的代价，因此被去除了。在软件团体中，很多认为形式验证和微核之类的技术，会为构建高度安全的软件提供有效的机制。不幸的是，我们集体软件系统的规模，和对性能的驱动，意味着这些技术不能跟上处理器性能。结果是，大型软件系统一直有很多安全缺陷，由于个人信息越来越多的在线，基于云计算的使用，和可能的对手共享了同样的物理硬件，这种效果被放大了。

Although computer architects and others were perhaps slow to realize the growing importance of security, they began to include hardware support for virtual machines and encryption. Unfortunately, speculation introduced an unknown but significant security flaw into many processors. In particular, the Meltdown and Spectre security flaws led to new vulnerabilities that exploit vulnerabilities in the microarchitecture, allowing leakage of protected information at a high rate. Both Meltdown and Spectre use so-called side-channel attacks whereby information is leaked by observing the time taken for a task and converting information invisible at the ISA level into a timing visible attribute. In 2018, researchers showed how to exploit one of the Spectre variants to leak information over a network without the attacker loading code onto the target processor. Although this attack, called NetSpectre, leaks information slowly, the fact that it allows any machine on the same local-area network (or within the same cluster in a cloud) to be attacked creates many new vulnerabilities. Two more vulnerabilities in the virtual-machine architecture were subsequently reported.37,38 One of them, called Foreshadow, allows penetration of the Intel SGX security mechanisms designed to protect the highest risk data (such as encryption keys). New vulnerabilities are being discovered monthly.

虽然计算机架构师和其他人在意识到安全的重要性上很慢，但他们开始对虚拟机器和加密，包含了硬件支持。不幸的是，推测向很多处理器引入了一种未知但明显的安全漏洞。特别是，Meltdown和Spectre安全漏洞会带来新的脆弱性，利用了微架构中的脆弱性，会以很高的速度泄露受保护的信息。Meltdown和Spectre都使用了所谓的side-channel攻击，信息的泄露，是通过观察一个任务消耗的时间，将在ISA层次不可见的信息，转化成时序可见的属性。在2018年，研究者展示了，怎样利用Spectre的一个变体，在攻击者不在目标处理器上载入代码的情况下，在网络中泄露信息。虽然这种称为NetSpectre的攻击泄露信息非常缓慢，但这允许在同样局域网中的任何机器（或在云中的同一集群）都会受到攻击，这个事实创建的新的脆弱性。

Side-channel attacks are not new, but in most earlier cases, a software flaw allowed the attack to succeed. In the Meltdown, Spectre, and other attacks, it is a flaw in the hardware implementation that exposes protected information. There is a fundamental difficulty in how processor architects define what is a correct implementation of an ISA because the standard definition says nothing about the performance effects of executing an instruction sequence, only about the ISA-visible architectural state of the execution. Architects need to rethink their definition of a correct implementation of an ISA to prevent such security flaws. At the same time, they should be rethinking the attention they pay computer security and how architects can work with software designers to implement more-secure systems. Architects (and everyone else) depend too much on more information systems to willingly allow security to be treated as anything less than a firstclass design concern.

Side-channel攻击并不新，但多数更早的情况中，软件的漏洞使攻击得以成功。在Meltdown，Spectre和其他攻击中，在硬件实现中的漏洞暴露了受保护的信息。处理器架构师定义什么是ISA的正确实现，有很基本的困难，因为标准定义没有任何内容是关于执行一个指令序列的性能效果的，只是关于执行的ISA可见的架构状态。架构师需要重新思考一个ISA的正确实现，以防止这样的安全漏洞。同时，他们需要重新思考对计算机安全的关注，以及架构师怎样与软件设计者一起工作，来实现更安全的系统。架构师（和其他任何人）对信息系统的依赖性太多了，让安全被当做一个第一流设计的考虑。

## 8. Future Opportunities in Computer Architecture

Inherent inefficiencies in general-purpose processors, whether from ILP techniques or multicore, combined with the end of Dennard scaling and Moore’s Law, make it highly unlikely, in our view, that processor architects and designers can sustain significant rates of performance improvements in general-purpose processors. Given the importance of improving performance to enable new software capabilities, we must ask: What other approaches might be promising?

通用目的处理器内在就缺乏效率，无论是ILP或多核，随着Dennard定律和Moore定律的消失，在我们看来，处理器架构师和设计者基本不可能维持通用目标处理的高速性能改进。改进性能以使软件能力更高非常重要，我们必须要问：什么样的其他方法会更有希望呢？

There are two clear opportunities, as well as a third created by combining the two. First, existing techniques for building software make extensive use of high-level languages with dynamic typing and storage management. Unfortunately, such languages are typically interpreted and execute very inefficiently. Leiserson et al. used a small example—performing matrix multiply—to illustrate this inefficiency. As in Figure 7, simply rewriting the code in C from Python—a typical high-level, dynamically typed language—increases performance 47-fold. Using parallel loops running on many cores yields a factor of approximately 7. Optimizing the memory layout to exploit caches yields a factor of 20, and a final factor of 9 comes from using the hardware extensions for doing single instruction multiple data (SIMD) parallelism operations that are able to perform 16 32-bit operations per instruction. All told, the final, highly optimized version runs more than 62,000× faster on a multicore Intel processor compared to the original Python version. This is of course a small example, one might expect programmers to use an optimized library for. Although it exaggerates the usual performance gap, there are likely many programs for which factors of 100 to 1,000 could be achieved.

有两个很清楚的机会，还有第三个，是将前两者结合在一起。第一，现有构建软件的技术，大量使用了高级语言，有动态类型和存储管理。不幸的是，这样的语言一般是解释并执行的，效率很低。Leiserson等使用了一个小型例子，即矩阵乘法，来描述这种低效性。如图7所示，用C重写Python的代码，性能就提升了47倍。使用多核上的并行loop，会继续提升大约7倍。优化内存布局来利用缓存，得到大约20倍的加速，使用SIMD的硬件扩展会再提升9倍性能，这可以每条指令执行16条32位运算。与原始Python版本相比，在多核Intel处理器上，所有方法优化过的版本，运行速度快了62000倍。这当然是一个小例子，程序员还是可以使用一个优化的库的。虽然这夸大了通常的性能空白，但很多程序还是可能获得100到1000倍的加速的。

An interesting research direction concerns whether some of the performance gap can be closed with new compiler technology, possibly assisted by architectural enhancements. Although the challenges in efficiently translating and implementing high-level scripting languages like Python are difficult, the potential gain is enormous. Achieving even 25% of the potential gain could result in Python programs running tens to hundreds of times faster. This simple example illustrates how great the gap is between modern languages emphasizing programmer productivity and traditional approaches emphasizing performance.

一个有趣的研究方向是，这些性能空白是否能够用新的编译器技术来弥补，可能还受到架构增强的支持。高效的翻译并执行高级脚本语言如Python，这个挑战是很难的，但潜在的收益也是巨大的。即使只获得25%的潜在收益，可会使Python程序运行速度提升数十上百倍。这个简单的例子说明了，现代语言强调程序员的产出，和强调性能的传统方法，它们之间有多大的空白。

## 9. Domain-specific architectures

A more hardware-centric approach is to design architectures tailored to a specific problem domain and offer significant performance (and efficiency) gains for that domain, hence, the name “domain-specific architectures” (DSAs), a class of processors tailored for a specific domain—programmable and often Turing-complete but tailored to a specific class of applications. In this sense, they differ from application-specific integrated circuits (ASICs) that are often used for a single function with code that rarely changes. DSAs are often called accelerators, since they accelerate some of an application when compared to executing the entire application on a general-purpose CPU. Moreover, DSAs can achieve better performance because they are more closely tailored to the needs of the application; examples of DSAs include graphics processing units (GPUs), neural network processors used for deep learning, and processors for software-defined networks (SDNs). DSAs can achieve higher performance and greater energy efficiency for four main reasons:

一个更加以硬件为中心的方法是，为特定问题领域设计定制的架构，为该领域提供显著的性能提升，因此，领域专用架构(DSA)的名称，为特定领域定制的一类处理器，可编程的，通常是图灵完备的，但是为特定应用类别定制的。在这个意义中上，他们与ASIC不同，ASIC经常是用于一个基本不变的函数代码的。DSAs通常称为加速器，因为与将整个应用在一个通用目标CPU相比，它们加速了一个应用的一部分。而且，DSAs可以获得更好的性能，因为它们是为应用的需求紧密定制的；DSAs的例子包括GPUs，用于深度学习的神经网络处理器，以及用于软件定义网络(SDN)的处理器。DSAs可以获得更高的性能，和更高的能效，这主要有四个原因：

First and most important, DSAs exploit a more efficient form of parallelism for the specific domain. For example, single-instruction multiple data parallelism (SIMD), is more efficient than multiple instruction multiple data (MIMD) because it needs to fetch only one instruction stream and processing units operate in lockstep. Although SIMD is less flexible than MIMD, it is a good match for many DSAs. DSAs may also use VLIW approaches to ILP rather than speculative out-of-order mechanisms. As mentioned earlier, VLIW processors are a poor match for general-purpose code but for limited domains can be much more efficient, since the control mechanisms are simpler. In particular, most high-end general-purpose processors are out-of-order superscalars that require complex control logic for both instruction initiation and instruction completion. In contrast, VLIWs perform the necessary analysis and scheduling at compile-time, which can work well for an explicitly parallel program.

第一，也是最重要的，DSAs对特定领域利用了更高效的并行形式。比如，SIMD比MIMD更高效，因为它只需要只需要取一个指令流，处理单元同步运算。虽然SIMD没有MIMD那么灵活，但是可以匹配很多DSAs。DSAs也可以使用VLIW方法来ILP，而不用预测性的乱序机制。就像之前提到的，VLIW处理器与通用目标代码不匹配，但对于一些领域会非常高效，因为控制机制会更简单。特别是，多数高端通用目标处理器是乱序超标量的，需要复杂的控制逻辑进行指令初始化和指令完成。对比起来，VLIWs在编译时进行必须的分析和调度，对显式并行的程序效果很好。

Second, DSAs can make more effective use of the memory hierarchy. Memory accesses have become much more costly than arithmetic computations, as noted by Horowitz. For example, accessing a block in a 32-kilobyte cache involves an energy cost approximately 200× higher than a 32-bit integer add. This enormous differential makes optimizing memory accesses critical to achieving high-energy efficiency. General-purpose processors run code in which memory accesses typically exhibit spatial and temporal locality but are otherwise not very predictable at compile time. CPUs thus use multilevel caches to increase bandwidth and hide the latency in relatively slow, off-chip DRAMs. These multilevel caches often consume approximately half the energy of the processor but avoid almost all accesses to the off-chip DRAMs that require approximately 10× the energy of a last-level cache access.

第二，DSAs可以更高效的利用内存层次结构。内存访问已经比代数计算更加昂贵，如Horowitz指出。比如，在一个32kB的缓存中访问一个block，大约是32-bit整数加法的能耗的200x。这种巨大的差异使得，要获得高能效比，就必须优化内存访问。通用目标处理器运行的代码，通常展现出空间和时间的局部性，否则在编译时就不会那么有预测性。CPUs因此使用多级缓存来增加带宽，以隐藏在片外的相对缓慢的DRAMs的延迟。这些多级缓存通常消耗处理器能耗的一半，但避免了所有对片外DRAMs的访问，这需要大约对上一级缓存访问能量的10x。

Caches have two notable disadvantages: 缓存有两个著名的缺陷：

*When datasets are very large*. Caches simply do not work well when datasets are very large and also have low temporal or spatial locality; and 当数据集很大时，缓存通常效果不会很好，因为时间或空间的局部性不会很好。

*When caches work well*. When caches work well, the locality is very high, meaning, by definition, most of the cache is idle most of the time. 当缓存效果不错时，局部性很高，这意味着，多数时间里多数缓存是闲置的。

In applications where the memory access patterns are well defined and discoverable at compile time, which is true of typical DSLs, programmers and compilers can optimize the use of the memory better than can dynamically allocated caches. DSAs thus usually use a hierarchy of memories with movement controlled explicitly by the software, similar to how vector processors operate. For suitable applications, user-controlled memories can use much less energy than caches.

在一些应用中，内存访问的模式是明确的，在编译的时候就可以发现，这对典型的DSLs是成立的，程序员和编译器可以优化内存的使用，这比动态的分配缓存效果更好。DSAs因此通常使用的内存层次结构，其移动显式的由软件来控制，与向量处理器运算的方式类似。对于合适的应用，用户控制的内存使用的能量比缓存要低的多。

Third, DSAs can use less precision when it is adequate. General-purpose CPUs usually support 32- and 64-bit integer and floating-point (FP) data. For many applications in machine learning and graphics, this is more accuracy than is needed. For example, in deep neural networks (DNNs), inference regularly uses 4-, 8-, or 16-bit integers, improving both data and computational throughput. Likewise, for DNN training applications, FP is useful, but 32 bits is enough and 16 bits often works.

第三，DSAs可以利用低精度。通用目标CPSs通常支持32位和64位整数和浮点数据。对于机器学习和图形学的很多应用，这种准确率是没有必要的。比如，在DNN中，使用4位，8位或16位整型进行推理就可以了，这可以改进数据和计算的吞吐量。类似的，对于DNN训了的应用，浮点是有用的，但32位就足够了，16位也是可以的。

Finally, DSAs benefit from targeting programs written in domain-specific languages (DSLs) that expose more parallelism, improve the structure and representation of memory access, and make it easier to map the application efficiently to a domain-specific processor.

最后，用DSL写的程序会展现出更高的并行性，这会是DSA受益，改进了内存访问的结构和表示，使其更容易将应用高效的映射到领域专用处理器。

## 10. Domain-Specific Languages

DSAs require targeting of high-level operations to the architecture, but trying to extract such structure and information from a general-purpose language like Python, Java, C, or Fortran is simply too difficult. Domain specific languages (DSLs) enable this process and make it possible to program DSAs efficiently. For example, DSLs can make vector, dense matrix, and sparse matrix operations explicit, enabling the DSL compiler to map the operations to the processor efficiently. Examples of DSLs include Matlab, a language for operating on matrices, TensorFlow, a dataflow language used for programming DNNs, P4, a language for programming SDNs, and Halide, a language for image processing specifying high-level transformations.

DSAs需要让高级运算面向架构，但从通用目标语言，如Python，Java，C或Fortran中提取这样的结构信息，就是太难了。领域专用语言(DSL)使这个过程成为可能，可以更高效的对DSA进行编程。比如，DSLs可以使向量，密集矩阵，和稀疏矩阵运算显式化，使DSL编译器将运算更高效的映射到处理器中。DSLs的例子包括Matlab，TensorFlow，P4和Halide，分别用于处理矩阵，DNNs数据流，SDNs和指定高级变换的图像处理。

The challenge when using DSLs is how to retain enough architecture independence that software written in a DSL can be ported to different architectures while also achieving high efficiency in mapping the software to the underlying DSA. For example, the XLA system translates Tensorflow to heterogeneous processors that use Nvidia GPUs or Tensor Processor Units (TPUs). Balancing portability among DSAs along with efficiency is an interesting research challenge for language designers, compiler creators, and DSA architects.

使用DSL的挑战是，怎样获得足够的架构独立性，以DSL写的软件，怎样移植到不同的架构中，同时将软件映射到潜在的DSA时，保持高效性。比如，XLA系统将TensorFlow翻译到异质处理器中，使用NVIDIA GPUSs或TPUs。在DSA中保持可移植性和效率的均衡，对于语言设计者，编译器创建者，和DSA架构师来说，都是一个有趣的研究挑战。

**Example DSA: TPU v1**. As an example DSA, consider the Google TPU v1, which was designed to accelerate neural net inference. The TPU has been in production since 2015 and powers applications ranging from search queries to language translation to image recognition to AlphaGo and AlphaZero, the DeepMind programs for playing Go and Chess. The goal was to improve the performance and energy efficiency of deep neural net inference by a factor of 10.

作为一个DSA的例子，考虑Google TPUv1，设计用于加速神经网络推理。TPU在2015年就开始生产，为搜索查询，语言翻译，到图像识别，到AlphaGo，和AlphaZero这样的应用赋能。其目标是，改进DNN推理的性能和能效比10倍。

As shown in Figure 8, the TPU organization is radically different from a general-purpose processor. The main computational unit is a matrix unit, a systolic array structure that provides 256 × 256 multiply-accumulates every clock cycle. The combination of 8-bit precision, highly efficient systolic structure, SIMD control, and dedication of significant chip area to this function means the number of multiply-accumulates per clock cycle is approximately 100× what a general-purpose single-core CPU can sustain. Rather than caches, the TPU uses a local memory of 24 megabytes, approximately double a 2015 general-purpose CPU with the same power dissipation. Finally, both the activation memory and the weight memory (including a FIFO structure that holds weights) are linked through user-controlled high-bandwidth memory channels. Using a weighted arithmetic mean based on six common inference problems in Google data centers, the TPU is 29× faster than a general-purpose CPU. Since the TPU requires less than half the power, it has an energy efficiency for this workload that is more than 80× better than a general-purpose CPU.

如图8所示，TPU的组织与通用目标处理器非常不一样。主要计算单元是一个矩阵单元，脉动阵列结构，每个时钟周期可以进行256x256个乘法累加计算。8-bit精度，高效的脉动阵列结构，SIMD控制，大量芯片面积都是实现这个函数，这些组合意味着每个时钟周期的乘法累加计算数量，大约是通用目标单核CPU的100x。TPU没有使用缓存，而是使用了一个24MB的局部存储，是2015年通用目标CPU的2倍，功率耗散相同。最后，激活内存和权重内存（包含了一个保存权重的FIFO结构）是通过用户控制的高带宽的内存通道连接的。在谷歌数据中心中的6个常见的推理问题中使用一个加权的基于代数平均值，TPU比通用目标CPU快29x。由于TPU需要的能量少于一半，在这种工作上的功耗效率比通用目标CPU高80x。

## 11. Summary

We have considered two different approaches to improve program performance by improving efficiency in the use of hardware technology: First, by improving the performance of modern high-level languages that are typically interpreted; and second, by building domain-specific architectures that greatly improve performance and efficiency compared to general-purpose CPUs. DSLs are another example of how to improve the hardware/software interface that enables architecture innovations like DSAs. Achieving significant gains through such approaches will require a vertically integrated design team that understands applications, domain-specific languages and related compiler technology, computer architecture and organization, and the underlying implementation technology. The need to vertically integrate and make design decisions across levels of abstraction was characteristic of much of the early work in computing before the industry became horizontally structured. In this new era, vertical integration has become more important, and teams that can examine and make complex trade-offs and optimizations will be advantaged.

我们已经考虑了两种不同的方法，来改进程序性能，通过改进使用硬件技术中的效率：第一，改进现代高级语言的性能，这些语言通常是解释执行的；第二，构建DSAs，与通用目标CPUs相比，可以极大的改进性能和效率。DSLs是怎样改进硬件、软件接口的另一个例子，使DSAs这样的架构创新成为可能。通过这种方法来获得显著的收益，会需要一个垂直整合的设计团队，理解应用，DSL和相关的编译器技术，计算机架构和组织，和潜在的实现技术。垂直整合的需求，在很多抽象级之间进行决策，是计算中的很多早期工作的特征，然后工业界才会变得水平结构化起来。在这个新时代，垂直整合已经变得更加重要，能够检查并进行复杂的折中和优化的团队，会更加有优势。

This opportunity has already led to a surge of architecture innovation, attracting many competing architectural philosophies: 这个机会已经带来了架构创新的激增，吸引了很多正在竞争的架构哲学：

GPUs. Nvidia GPUs use many cores, each with large register files, many hardware threads, and caches;

GPUs使用很多核，每个都有大型的寄存器组，很多硬件线程，和缓存；

TPUs. Google TPUs rely on large two-dimensional systolic multipliers and software-controlled on-chip memories;

TPUs依赖于大型二维脉动阵列乘法器和软件控制的片上存储；

FPGAs. Microsoft deploys field programmable gate arrays (FPGAs) in its data centers it tailors to neural network applications; and

微软在其数据中心中部署了FPGAs，是对神经网络应用定制的；

CPUs. Intel offers CPUs with many cores enhanced by large multi-level caches and one-dimensional SIMD instructions, the kind of FPGAs used by Microsoft, and a new neural network processor that is closer to a TPU than to a CPU.

Intel提供了多核CPUs，有大型多级缓存，一维SIMD指令，微软使用的那种FPGAs，和一个神经网络处理器，接近于TPU。

In addition to these large players, dozens of startups are pursuing their own proposals. To meet growing demand, architects are interconnecting hundreds to thousands of such chips to form neural-network supercomputers.

除了这些大型玩家，几十个初创企业正在追求其自己的目标。为满足增长的需求，架构师正在将成百上千个这样的芯片互相连接起来，形成神经网络超级计算机。

This avalanche of DNN architectures makes for interesting times in computer architecture. It is difficult to predict in 2019 which (or even if any) of these many directions will win, but the marketplace will surely settle the competition just as it settled the architectural debates of the past.

DNN架构的大量涌入，使得这个时候对计算机架构非常有趣。在2019年，很难预测这些很多方向哪个会赢，但市场肯定会使这场竞争确定下来，就像在过去确定下那些架构争论一样。

## 12. Open Architectures

Inspired by the success of open source software, the second opportunity in computer architecture is open ISAs. To create a “Linux for processors” the field needs industry-standard open ISAs so the community can create open source cores, in addition to individual companies owning proprietary ones. If many organizations design processors using the same ISA, the greater competition may drive even quicker innovation. The goal is to provide processors for chips that cost from a few cents to 100 dollar.

受到开源软件的启发，计算机架构中的第二个机会是开放ISAs。为创建一个处理器界的Linux，这个领域需要工业级的开放ISAs，这样团体可以创建开源核，作为对单个公司拥有专利ISAs的补充。如果很多组织都用同样的ISA来设计处理，竞争更激烈，会驱动更快的创新。目标是，为价格为几个cents到100美元的芯片提供处理器。

The first example is RISC-V (called “RISC Five”), the fifth RISC architecture developed at the University of California, Berkeley. RISC-V’s has a community that maintains the architecture under the stewardship of the RISC-V Foundation (http://riscv.org/). Being open allows the ISA evolution to occur in public, with hardware and software experts collaborating before decisions are finalized. An added benefit of an open foundation is the ISA is unlikely to expand primarily for marketing reasons, sometimes the only explanation for extensions of proprietary instruction sets.

第一个例子是RISC-V，Berkeley大学开发的第5代RISC架构。RISC-V在RISC-V基金会的组织下维护这个架构。这个ISA是开放的，使ISA的演化是公开的，软件和硬件专家在决策之前进行合作。开放基金会的另一个好处是，基本上不会因为市场原因来扩张，对于有专利的指令集来说，这是其扩展的唯一解释。

RISC-V is a modular instruction set. A small base of instructions run the full open source software stack, followed by optional standard extensions designers can include or omit depending on their needs. This base includes 32-bit address and 64-bit address versions. RISC-V can grow only through optional extensions; the software stack still runs fine even if architects do not embrace new extensions. Proprietary architectures generally require upward binary compatibility, meaning when a processor company adds new feature, all future processors must also include it. Not so for RISC-V, whereby all enhancements are optional and can be deleted if not needed by an application. Here are the standard extensions so far, using initials that stand for their full names:

RISC-V是一个模块化的指令集。小型基础指令集就可以运行完整的开源软件栈，设计者还可以增加可选的标准扩展，也可以忽略掉，这要视其需求而定。基础指令包括32位寻址和64-位寻址的版本。RISC-V只会通过可选的扩展来进行增长；即使架构师不选择新的扩展，软件栈也会运行的很好。有专利的架构一般需要向上的二进制兼容性，意味着处理器公司加入新的特征时，未来的所有处理器也必须包括之。RISC-V则不是这样，其中所有的强化都是可选的，如果应用不需要，则可以删掉。这里是目前的标准扩展，使用缩写来表示其完整名称：

M. Integer multiply/divide;

A. Atomic memory operations;

F/D. Single/double-precision floating-point; and

C. Compressed instructions.

A third distinguishing feature of RISC-V is the simplicity of the ISA. While not readily quantifiable, here are two comparisons to the ARMv8 architecture, as developed by the ARM company contemporaneously:

RISC-V的第三个优异特征是，ISA的简洁性。虽然还没有量化，但这里是与ARMv8架构的两个比较，这是ARM公司同时期开发的：

Fewer instructions. RISC-V has many fewer instructions. There are 50 in the base that are surprisingly similar in number and nature to the original RISC-I. The remaining standard extensions—M, A, F, and D—add 53 instructions, plus C added another 34, totaling 137. ARMv8 has more than 500; and

指令更少。RISC-V的指令更少。基础有50个，与原始的RISC-I中的数量和本质都非常相似。剩下的是标准扩展，M，A，F和D，增加了53条指令，C增加了34条，总计137。ARMv8的指令则超过了500条。

Fewer instruction formats. RISC-V has many fewer instruction formats, six, while ARMv8 has at least 14.

指令格式更少。RISC-V的指令格式只有6种，而ARMv8则有至少14种。

Simplicity reduces the effort to both design processors and verify hardware correctness. As the RISC-V targets range from data-center chips to IoT devices, design verification can be a significant part of the cost of development.

简洁性使设计处理器和验证硬件正确性都更简单。由于RISC-V的目标是从数据中心芯片到IoT设备，设计验证是开发代价的一大部分。

Fourth, RISC-V is a clean-slate design, starting 25 years later, letting its architects learn from mistakes of its predecessors. Unlike first-generation RISC architectures, it avoids microarchitecture or technology-dependent features (such as delayed branches and delayed loads) or innovations (such as register windows) that were superseded by advances in compiler technology.

第四，RISC-V是一个从头开始的设计，是从25年后开始的，使其架构师学习到了很多其前辈的错误。与第一代RISC架构不同的是，其避免了依赖于微架构或技术的特征（比如延迟分支和延迟载入）或创新（比如寄存器窗口），这些都有编译器技术的进展取代了。

Finally, RISC-V supports DSAs by reserving a vast opcode space for custom accelerators. 最后，RISC-V支持DSAs，为定制加速器保留了大量的opcode空间。

Beyond RISC-V, Nvidia also announced (in 2017) a free and open architecture it calls Nvidia Deep Learning Accelerator (NVDLA), a scalable, configurable DSA for machine-learning inference. Configuration options include data type (int8, int16, or fp16 ) and the size of the two-dimensional multiply matrix. Die size scales from 0.5 mm^2 to 3 mm^2 and power from 20 milliWatts to 300 milliWatts. The ISA, software stack, and implementation are all open.

在RISC-V之外，NVIDIA还宣布了一种自由开放的架构，其称之为NVDLA，一种可扩展可配置的DSA，用于机器学习推理。可配置的选项包括数据类型(int8, int16, fp16)，和二维乘法矩阵的大小。Die大小从0.5 mm^2到3 mm^2，功率从20 milliWatts到300 milliWatts。ISA，软件栈和实现都是开放的。

Open simple architectures are synergistic with security. First, security experts do not believe in security through obscurity, so open implementations are attractive, and open implementations require an open architecture. Equally important is increasing the number of people and organizations who can innovate around secure architectures. Proprietary architectures limit participation to employees, but open architectures allow all the best minds in academia and industry to help with security. Finally, the simplicity of RISC-V makes its implementations easier to check. Moreover, the open architectures, implementations, and software stacks, plus the plasticity of FPGAs, mean architects can deploy and evaluate novel solutions online and iterate them weekly instead of annually. While FPGAs are 10× slower than custom chips, such performance is still fast enough to support online users and thus subject security innovations to real attackers. We expect open architectures to become the exemplar for hardware/software co-design by architects and security experts.

开放的简单架构与安全是协同的。第一，安全专家不相信不开放的安全，所以开放的实现很有吸引力，开放的实现需要开放的架构。同样重要的是，人员和组织数量的增加，可以围绕安全架构进行创新。有专利的架构限制参与，但开放的架构允许最优最优秀的头脑参与，学术界和工业界的一起，来帮助安全问题。最后，RISC-V的简洁性，使其实现更容易核实。而且，开放的架构，实现，和软件栈，加上FPGA的可塑性，意味着架构师可以在线部署并评估新的解决方案，每周进行迭代，而不是每年迭代。FPGAs比定制的芯片要慢10x，但这样的性能仍然足够快，可以支持在线用户，因此相关的安全创新要留给真正的攻击者。我们期望开放的架构会通过架构师和安全专家成为硬件/软件协同设计的样本。

## 12. Agile Hardware Development

The Manifesto for Agile Software Development (2001) by Beck et al. revolutionized software development, overcoming the frequent failure of the traditional elaborate planning and documentation in waterfall development. Small programming teams quickly developed working-but-incomplete prototypes and got customer feedback before starting the next iteration. The scrum version of agile development assembles teams of five to 10 programmers doing sprints of two to four weeks per iteration.

敏捷软件开发宣言革命了软件开发，克服了瀑布式开发中传统复杂的计划和文档导致的频繁失败。小型程序团队迅速开发了能工作但不完整的原型，得到客户反馈，然后开始下一轮的迭代。敏捷开发的scrum版，召集5-10人的程序员队伍，每次迭代进行2-4周的冲刺开发。

Once again inspired by a software success, the third opportunity is agile hardware development. The good news for architects is that modern electronic computer aided design (ECAD) tools raise the level of abstraction, enabling agile development, and this higher level of abstraction increases reuse across designs.

由软件成功的启发，第三个机会是敏捷硬件开发。对架构师来说，好消息是现代ECAD工具提升了抽象层次，使敏捷开发成为可能，这种更高层次的抽象，在多个设计之间增加了重用。

It seems implausible to claim sprints of four weeks to apply to hardware, given the months between when a design is “taped out” and a chip is returned. Figure 9 outlines how an agile development method can work by changing the prototype at the appropriate level. The innermost level is a software simulator, the easiest and quickest place to make changes if a simulator could satisfy an iteration. The next level is FPGAs that can run hundreds of times faster than a detailed software simulator. FPGAs can run operating systems and full benchmarks like those from the Standard Performance Evaluation Corporation (SPEC), allowing much more precise evaluation of prototypes. Amazon Web Services offers FPGAs in the cloud, so architects can use FPGAs without needing to first buy hardware and set up a lab. To have documented numbers for die area and power, the next outer level uses the ECAD tools to generate a chip’s layout. Even after the tools are run, some manual steps are required to refine the results before a new processor is ready to be manufactured. Processor designers call this next level a “tape in.” These first four levels all support four-week sprints.

4周冲刺开发对硬件似乎行不通，因为一个设计流片到返回一个芯片就需要几个月。图9展示了，通过在适当的层次改变原型，敏捷开发方法怎样可以工作。最里面的层次是一个软件模拟器，如果一个模拟器可以满足一次迭代，那么最容易最快的进行改变的地方就是这里。下一个层次是FPGAs，比一个细节的软件模拟器的运行速度会快上百倍。FPGAs可以运行操作系统和完整的基准测试，比如SPEC中的，可以对原型进行更精确的评估。AWS在云中提供FPGAs，所以架构师不需要首先购买硬件，就可以使用FPGAs，设置一个实验室。为对die面积和功耗的数值计入文档，下一层使用ECAD工具来生成芯片的布局。即使工具运行了之后，仍然需要一些手工步骤来提炼结果，然后才能准备好生产新的处理器。处理器设计者称下一个层次为tape in。这前四个层次都支持4周的冲刺开发。

For research purposes, we could stop at tape in, as area, energy, and performance estimates are highly accurate. However, it would be like running a long race and stopping 100 yards before the finish line because the runner can accurately predict the final time. Despite all the hard work in race preparation, the runner would miss the thrill and satisfaction of actually crossing the finish line. One advantage hardware engineers have over software engineers is they build physical things. Getting chips back to measure, run real programs, and show to their friends and family is a great joy of hardware design.

为研究目的，我们可以在tape in层次停一下，因为面积，功耗和性能估计都是高度精确的。但是，就像在长跑中，在终点线之前100米要停止下来，因为跑者可以精确的预测最终时间。尽管在跑步准备中有很多艰苦的工作，跑者会错过实际跨越终点线的满足感。硬件工程师比软件工程师的一个优势是，他们构建实物。把芯片拿回来进行测量，运行实际的程序，给朋友和家人进行展示，是硬件设计中很享受的事。

Many researchers assume they must stop short because fabricating chips is unaffordable. When designs are small, they are surprisingly inexpensive. Architects can order 100 1-mm^2 chips for only 14,000 dollar. In 28 nm, 1 mm^2 holds millions of transistors, enough area for both a RISC-V processor and an NVLDA accelerator. The outermost level is expensive if the designer aims to build a large chip, but an architect can demonstrate many novel ideas with small chips.

很多研究者认为，他们需要停止short，因为制造芯片是很昂贵的。当设计很小时，它们其实很便宜。架构师可以订购100个1 mm^2的芯片，只需要14000美元。在28nm工艺下，1mm^2上有上百万个晶体管，对一个RISC-V处理器和NVLDA加速器来说足够了。如果设计者想要构建一个大型芯片，最外层是昂贵的，但架构师可以用小型芯片展示很多新的思想。

## 13. Conclusion

To benefit from the lessons of history, architects must appreciate that software innovations can also inspire architects, that raising the abstraction level of the hardware/software interface yields opportunities for innovation, and that the marketplace ultimately settles computer architecture debates. The iAPX-432 and Itanium illustrate how architecture investment can exceed returns, while the S/360, 8086, and ARM deliver high annual returns lasting decades with no end in sight.

为从历史的教训中学习，架构师必须感谢，软件创建也可以启发架构师，提升硬件/软件接口的抽象层次，可以产生创新的机会，市场最终会终结计算机架构的争论。iAPX-432和Itanium描述了，架构投资会怎样超越回报，而S/360，8086和ARM会得到几十年的高额回报，现在仍然看不到尽头。

The end of Dennard scaling and Moore’s Law and the deceleration of performance gains for standard microprocessors are not problems that must be solved but facts that, recognized, offer breathtaking opportunities. High-level, domain-specific languages and architectures, freeing architects from the chains of proprietary instruction sets, along with demand from the public for improved security, will usher in a new golden age for computer architects. Aided by open source ecosystems, agilely developed chips will convincingly demonstrate advances and thereby accelerate commercial adoption. The ISA philosophy of the general-purpose processors in these chips will likely be RISC, which has stood the test of time. Expect the same rapid improvement as in the last golden age, but this time in terms of cost, energy, and security, as well as in performance.

Dennard定律和Moore定律的终结，和标准微处理器性能提升的减速，不是必须要解决的问题，而是现实，如果承认的话，会有惊人的机会。高级领域专用语言和架构，使架构师从专利指令集中解放出来，和公众对改进的安全性的需求一起，会引领计算机架构的一个黄金时代。有了开源生态的帮助，敏捷开发的芯片肯定会有很大进展，加速商业的应用。这些芯片中的通用目标处理器的ISA哲学很可能是RISC，这已经经历了时间的考验。期待有像上一个黄金年代一样的迅速改进，但这次是价格，功耗和安全，以及性能。

The next decade will see a Cambrian explosion of novel computer architectures, meaning exciting times for computer architects in academia and in industry.

下一个十年会看到新的计算机架构的寒武纪大爆发，意味着计算机架构在学术界和工业界的激动人心的时刻。