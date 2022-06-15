# Protocol Verification as a Hardware Design Aid

David L. Dill, Andreas J. Drexler, Alan J. Hu, C. Han Yang @ Stanford University

## 0. Abstract

The role of automatic formal protocol verification in hardware design is considered. Principles are identified that maximize the benefits of protocol verification while minimizing the labor and computation required. A new protocol description language and verifier (both called Murphi) are described, along with experiences in applying them to two industrial protocols that were developed as part of hardware designs.

本文考虑了，硬件设计中自动形式协议验证的角色。原则是，在最小化需要的劳动和计算量的情况下，最大化协议验证的收益。本文描述了一种新的协议描述语言和验证器，称为Murphi，将其应用到了两种工业协议中，是硬件设计的一部分。

## 1. Introduction

Most complex digital designs must be regarded as concurrent systems: individual modules run in parallel and must coordinate by explicit synchronization and communication. Complexity will continue to increase, portending a shift in total design effort from, for instance, faster arithmetic circuits, to mechanisms for coordination. Those mechanisms usually involve protocols: rules that, if followed by each party in a coordinated action, assure a desired outcome.

多数复杂的数字设计都可以认为是并发系统：单个模块并行运行，必须通过显式同步和通信协调。复杂度会持续增加，预示着总计的设计努力，比如，会从更快的算术电路，转换到协调的机制。这些机制通常涉及到协议：这些规则由被协调的动作的每一方遵守，会确保得到期望的输出。

Unfortunately, protocol design is a subtle art. Even when a designer exercises the utmost care, there is a strong possibility that he or she will fail to anticipate some possible interaction among rules, resulting in errors and deadlocks. Even worse, the nondeterminism resulting from differing internal states or delays means that resulting errors are often not reliably repeatable, making testing and debugging extremely difficult.

不幸的是，协议设计是一个微妙的艺术。即使设计者非常谨慎小心，仍然很有可能在这些规则下，不能得到期望的可能的互动，反而得到错误和死锁。更坏的情况是，不同的内部状态或延迟，得到的不确定性的结果，意味着得到的错误，通常并不能可靠的重复，使测试和调试非常困难。

Consequently, the protocol design problem seems an obvious target for computer assistance. However, while protocol simulation is an effective way to catch "obvious" errors, many protocol problems arise when many unusual conditions arise at once; catching these problems reliably would require unrealistic amounts of simulation.

结果是，协议设计问题似乎是计算机辅助的一个明显目标。但是，虽然协议仿真是捕获明显错误的一个有效的方式，在很多不寻常的条件一旦出现，很多协议问题就会出现；可靠的捕获这些问题，需要的仿真数量是不现实的。

Because of these long-standing problems, formal verification of protocols has stimulated a great deal of interest. In particular, automatic methods ("perturbation analysis") that explicitly enumerate all the system states reached under a particular protocol have been used for many years [16, 1, 8]. Generally, these methods have been applied to communication and network protocols.

因为这些长期存在的问题，协议的形式验证激发了广泛的兴趣。特别是，显式的枚举在特定协议下达到的所有系统状态，这种自动方法（扰动分析）已经使用了很多年了。一般来说，这些方法已经应用到通信和网络协议中。

We believe that protocol verification is now a digital CAD problem; protocol verifiers should be in every digital designer's toolbox. There have already been some initial steps in this area: AT&T's COSPAN protocol verifier has been used for hardware designs [12], and McMillan's SMV program was recently applied to a cache coherence protocol for a shared-memory multiprocessor [10].

我们相信，协议验证现在是一个数字CAD问题；协议验证器应当在每个数字设计者的工具箱里。在这个领域中，已经有一些初始的工作：AT&T的COSPAN协议验证器已经用于硬件设计，McMillan的SMV程序最近应用到一个共享存储多处理器中的缓存一致性协议中。

We report here on our experiences using a protocol description language and verifier (both of our own design) called Murphi on some industrial digital design problems. We discuss factors that increase the chance for successful use of a protocol verifier, overview the particular description language and verifier we used, and describe the results of our efforts.

在一些工业数字设计上，我们使用了一个协议描述语言和协议验证器（都是我们自己设计的），称为Murphi，这里我们给出一些经验。我们讨论了，提高成功的使用一个协议验证器的可能性的因素，概述了我们使用的描述语言和验证器，描述了试验结果。

### 1.1 Protocol Verification

Protocols are needed in hardware to achieve coordinated action in the presence of complicating factors such as communication channels that impose long and possibly varying delays or deliver messages unreliably or out-of-order, bounded buffers or other scarce resources that may cause deadlocks, and components that have unpredictable delays and other nondeterministic behavior. Of special interest at this time are protocols for communication over networks or inter-processor switches and protocols for maintaining cache coherence in shared-memory multiprocessors.

硬件中需要协议来协调动作，因为可能有复杂的因素，比如通信通道会有很长但是可能不同的延迟，或传递信息不可靠，或乱序，有限的buffers，或其他稀有资源可能会导致死锁，组成部分有不可预测的延迟和其他不确定性的行为。此时我们特别感兴趣的，是网络或处理器switches之间的通信协议，和在共享存储多处理器之间缓存一致性的维护协议。

Formal verification of a protocol proceeds by describing the protocol in some language and then comparing the behavior of this description with a specification of the desired behavior. A verifier generates states from the description, comparing them with the specification as it goes. If the veryfier detects an inconsistency, this fact is reported, along with an example sequence of states that illustrates how the problem can occur, to aid in diagnosis. The description of the protocol can be in many forms: a program-like notation, a collection of finite-state machines, or a Petri net, for example. The simplest specifications are for fixed properties, such as absence of deadlock, or invariants, which are properties that should be true of individual states. More sophisticated verification systems can handle specifications in the form of temporal logic formulas or automata.

一个协议的形式验证，首先要用某种语言描述这个协议，然后比较这个描述的行为，与期望行为的规范。一个验证器从描述中生成状态，在不断进行的时候，将其与规范进行比较。如果验证器检测到了一个不一致的地方，就汇报这个fact，和一个状态序列的例子，描述这个问题是怎样发生的，以帮助诊断。协议的描述可以是多种形式的：像程序的符号，有限状态机的集合，或Petri网。最简单的规范是固定性质的，比如不存在死锁，或变体，这是个体状态应当为真的性质。更复杂的验证系统可以以时域逻辑公式或自动机的形式来处理规范。

The usual assumption about the role of formal verification is to provide a guarantee of design correctness. Although this is a worthy goal, it is very difficult to achieve. We take a more "economic" view: the main potential of formal verification is to reduce the cost and time of product development.

形式化验证的角色的通常假设，是确保设计的正确性。虽然这是一个值得的目标，但非常难以达到。我们采取了一个竞价经济的视角：形式化验证的主要潜力是，降低产品开发的代价和时间。

One of the most important ways to make verification of large systems possible is down-scaling - pretending that they are small systems. Most of the bugs in a protocol to coordinate thousands of processes can be demonstrated using two or three processes. In this case, down-scaling would be formally verifying the protocol with two or three processes. In some sense, this is the opposite end of the spectrum from simulation: instead of testing a small fraction of the possibilities for a large model of the system, we check all of the possibilities for a small instance of the system. Neither method is guaranteed to catch all of the problems, but down-scaling will almost certainly catch some problems that simulation will not (and vice-versa - we are not advocating the elimination of simulation).

对大型系统进行验证的一种最重要的方式是缩小，假装它们是小型系统。一个协调上千个过程的协议中的多数bugs，都可以使用2个或3个过程进行展示。在这种情况中，缩小就是用2个或3个过程来对协议进行形式化验证。在某种程度上，这是仿真的对立面：我们检查系统的一个小型实例的所有可能性，而不是测试系统的一个大型模型的一小部分所有可能性。没有哪种方法可以确保捕获到所有的问题，但缩小几乎肯定可以捕获到一些仿真不会捕获到的问题（反之亦然，我们并不是用户消除掉仿真）。

## 2. Description language

The Murphi description was designed to be the simplest possible usable language that supports nondeterministic, scalable descriptions. Murphi meets these particular goals (especially simplicity) better than existing hardware and protocol description languages. [2, 3, 11, 8, 9, 14, 13].

Murphi描述设计为最简单的可用语言，支持不确定性，可扩展的描述。Murphi满足这些特定的目标（尤其是简单），比现有的硬件和协议描述语言更好。

Murphi describes a system using a set of iterated guarded commands, like Chandy and Misra's Unity language (which inspired it) [4].

Murphi使用迭代的保护的命令集合来描述系统，与Chandy和Misra的Unity语言类似（也是受到这些语言的启发）。

### 2.1. Murphi Language

A Murphi description consists of constant and type declarations, variable declarations, procedure declarations, rule definitions, a description of the start state, and a collection of invariants. An invariant is a Boolean expression that references the variables.

一个Murphi描述包含常量和类型声明，变量声明，过程声明，规则定义，初始状态描述，和不变量的集合。一个不变量是一个布尔表达式，参考了变量。

Rules are written 规则书写如下

```
Rule
    Boolean-expression
=>
    stmSeq
```

Each rule is a guarded command [6], consisting of a condition and an action. The condition is a Boolean expression consisting of constants, declared variables, and operators. The action is a sequence of statements.

每条规则是一个守护命令，由一个条件和一个行为组成。条件是一个布尔表达式，由常数，声明的变量和算子组成。行为是一系列语句。

A state is a function that gives values to all of the variables. An execution of the system is a finite or infinite sequence of states s0, s1, ..., where s0 is determined by the description of the start state that is part of the description. If s_i is any state in the sequence, s_i+1 can be obtained by applying some rule whose condition is true in si and whose action transforms si to si+1. In general, si can satisfy several conditions, so there is more than one execution (nondeterminism). A simulator for Murphi might choose the rule randomly; a verifier must somehow cover all the possibilities. In either case, the invariants are applied whenever a state is explored; if any invariant is violated, an error is reported.

一个状态是对所有变量赋值的函数。系统的一次执行，是一个有限或无限的状态序列，s0, s1, ..., 其中s0是初始状态描述确定的，是描述的一部分。如果s_i是序列中的任意状态，s_i+1的获得，是应用一些规则得到的，其条件在s_i上是真值，其行为将si转换到si+1。一般来说，si可以满足几个条件，所以会有不止一个执行（不确定性）。一个Murphi的仿真器会随机选择规则；一个验证器必须在某种程序上覆盖所有可能性。在任一情况下，不论在什么时候探索一个状态时，都要应用不变量；如果任意不变量违反了，就给出一个错误。

Murphi is well-suited for an asynchronous, interleaving model of concurrency, where atomic steps of individual processes are assumed to happen in sequence, and one process can perform any number of steps between the steps of the other. When two steps are truly concurrent, there will be executions that allow them to happen in either order in the interleaving model. In Murphi, concurrent composition is very easy: to model two processes in parallel, just form a new description using the union of their rules.

Murphi非常适应异步的，交叉的并发模型，其中单个过程的原子步骤假设是序列发生的，一个过程可以进行任意次数的步骤，然后再进行其他的步骤。当两个步骤真正是并发的，会有允许它们在交替模型中以任一顺序发生的执行。在Murphi中，并发composition是很容易的：为并行的建模两个过程，只需要使用其规则的并集，形成一个新的描述。

Given the importance of down-scaling in verification, we have put some effort into making it possible to change the scale of a Murphi description by changing a single parameter. A Murphi description of a protocol that coordinates n processes can be written with a declared constant (e.g. NumProcesses). Then a subrange Processes : 0..NumProcesses-1 can be declared, and the states of the processes stored in an array indexed by Processes. Finally, a collection of nearly identical rules can be defined using the ruleset construct:

由于验证中缩小非常重要，我们努力使其在Murphi中成为可能，改变一个Murphi描述的规模，只需要改变一个参数。一个协议的Murphi描述，协调了n个过程，可以用一个声明的常数来写，如NumProcesses。然后就可以生成过程的子范围：0...NumProcesses-1，过程的状态存储在一个阵列中，索引为过程。最终，几乎一致的规则的集合，可以使用ruleset构建来定义：

```
Ruleset formal : range Do
    ruleSet
Endruleset;
```

A ruleset can be defined that allows the rules for a process to be instantiated for every process number in the type Processes.

一个ruleset的定义，可以使一个过程的规则对每个过程都进行实例化，其过程数是类型Processes的。

A description written in this style can be scaled by changing only the constant declarations.

以这种方式书写的描述，可以通过变化常数声明来进行缩放。

Quantifiers in expressions also promote scalability: 表达式中的量化器也可以促进缩放性：

```
Forall a: addressType Do v[a] = w[a] Endforall
```

is a Boolean expression which is true if all v[a] equal w[a] over some given address range.

这是一个布尔表达式，如果对给定的地址范围v[a]都等于w[a]，那么就为真。

Statements have sequential semantics, i.e. assignments take place in the environment that has been modified by all previous assignments. The usual conditional statements if-then-elsif-else and switch (case) are part of Murphi. There is a restricted for statement that must have compile-time constant loop bounds. Murphi procedures are essentially "macros" with parameter type-checking. These constructs will probably be generalized in the future, but they were suffcient for examples described here.

语句有顺序的语义，即，赋值发生的环境，已经被之前所有的赋值改变了。通常的条件表达式if-then-elsif-else和switch (case)是Murphi的一部分。有一个有约束的for语句，必须有编译时的常数循环界限。Murphi过程实际上是宏，带有参数的类型检验。这些构建很可能在将来进行泛化，但对这里描述的例子是足够的。

### 2.2 Specifications

Murphi has several features for detecting design errors. First, it can detect dead locks, which are defined to be states with no successors other than themselves. Second, an Error statement can appear in the body of a rule (almost always imbedded in a conditional). This feature is especially useful when some branches of an If or Switch statements are not intended to be reachable. The Error statement prints a user-supplied error message and causes an error trace to be printed. There is also an Assert statement, which is an abbreviation for a conditional error statement. Finally, the user can define invariants in a separate part of the Murphi description. An invariant is a Boolean expression that is desired to be true in every state. When an invariant is violated, an error message and error trace a generated.

Murphi有几种特征，可以检测设计的错误。首先，可以检测死锁，即除了它们自己没有其他后继者的状态。第二，在一条规则的主题中可以出现一条Error语句（几乎永远嵌入到一个条件语句中）。当一些If或Switch语句的分支本来就不能达到的时候，这种特征就非常有用。Error语句打印出一条用户提供的错误信息，导致打印出一条错误迹。同时还有一条Assert语句，是条件错误语句的简写。最后，用户可以在Murphi描述的单独部分定义invariants。一个invariant是一个布尔表达式，期望在每个状态中都是真值。当违反了一个invariant，就生成一条错误信息和错误trace。

These specification facilities are limited, because they do not allow one to directly express properties of sequential behavior. Another important limitation is the lack of general facilities for dealing with liveness or fairness properties. For example, we cannot detect livelocks, a deadlock in part of the system is masked by activity in another part of the system. However, we have been able to verify important properties of real examples using only deadlock, error, and invariant checking. The specification facilities of the system will be expanded in the future.

这些规范是有限的，因为不允许直接表达顺序行为的性质。另一个重要的局限是，缺少通用设施处理性质的liveness或fairness。比如，我们不能检测到活锁，系统中部分的一个死锁由系统的另外一个部分中的行为掩膜掉了。但是，我们曾经能够只使用死锁，错误和invariant检查来验证真实例子的重要性质。系统的规范设施会在将来进行拓展。

### 2.3 Murphi Compiler and Verifier

The Murphi compiler takes a Murphi source description and generates a C++ program, which is compiled together with code for a verifier which check for invariant violations, error statements, assertion violations, and deadlock.

Murphi编译器以一个Murphi源描述为输入，生成一个C++程序，与一个验证器的源码一起进行编译，验证器可以检查invariant违反，错误语句，assertion违反和死锁。

The verifier attempts to enumerate all the states of the system, checking for error conditions as it proceeds. Because space is at a premium in verification, states are represented compactly by encoding all scalar types in the minimum possible number of bits, then concatenating the fields without regard to byte and word alignment constraints. This slows down access to fields somewhat, but is justified by the massive space savings that result. A hash table with double hashing that stores reached states is used to decide efficiently if a newly-reached state is old (has been reached already) or new (has not been reached already). New states are stored in a queue of active states (states that still need to be explored). Depending on the organization of this queue, the verifier does a breadth-first search or a depth-first search. Every state in the hash table has a pointer to a predecessor state that can be used to generate an error trace if a problem is detected. Breadth-first search is used by default, because it causes the error-traces to be as short as possible.

验证器试图枚举系统的所有状态，在这个过程中检查错误条件。因为在验证过程中空间是优先的，状态是紧凑的进行表达的，将所有标量类型，以最小数量的bits进行编码，然后不考虑byte和word的对其限制，将各个fields拼接起来。这在一定程序上减缓了对fields的访问，但毕竟节约了大量了空间。两次哈希的哈比表，存储了达到的状态，用于高效的决定，一个新达到的状态是否是旧的（曾经达到过），或是新的（之前未曾达到过）。新的状态存储在活跃状态队列中（仍然需要探索的状态）。依赖于这个队列的组织，验证器进行一个宽度优先搜索或深度优先搜索。哈希表中的每个状态，都有一个指向之前状态的指针，如果检测到一个问题，可以用于生成一个错误trace。默认使用宽度优先搜索，因为这会让错误traces尽可能的短。

## 3. Experience on larger examples

We have used Murphi on two hardware designs that are "real" in the sense that they were intended to become commercial products: a directory-based cache coherence protocol for a multiprocessor, and a synchronous link-level communication protocol. In both cases, we began verifying early in the design phase, basing our Murphi descriptions on an informal design specification. In both cases, we found significant errors and omissions and spent a great deal of time modifying and enhancing the designs to meet our correctness conditions.

我们在两个真实的硬件设计中使用了Murphi，这两种设计是要成为商业产品的：一种是微处理器的基于目录的缓存一致性协议，一种是同步的连接级的通信协议。在两种情况中，我们在设计阶段就进行早期验证，基于我们的对非正式设计指标的Murphi描述。在两种情况中，我们发现了明显的错误和疏忽，花费了相当的时间修正和增强设计，以满足我们的正确性条件。

Verification goes through several stages: deciding how to model the problem (especially, what details to omit); writing the description; using verification to find description errors; and only then discovering genuine design errors.

验证有几个阶段：决定怎样建模问题（尤其是，要忽略哪些细节）；书写描述；使用验证来找到描述错误；然后发现真正的设计错误。

When the first serious design error is discovered, the system design needs to be modified. But, even if the modification avoids introducing more bugs, more bugs in the original design are uncovered. The verification process then enters a tight loop redesign-reverify loop much like the more traditional edit-compile-debug that programmers experience.

当发现了第一个严重的设计错误时，系统设计需要被修正。但是，即使这个修正避免了引入更多的bugs，还是发现了原始设计的更多bugs。验证过程然后进入了一个重新设计-重新验证的循环，与传统的程序员经历的编辑-编译-调试非常相像。

### 3.1 Cache coherence protocol

Directory-based cache coherence is a way of implementing a shared-memory abstraction on top of a message-passing network, by recording in a central directory which processors have cached readable or writable copies of a memory location. Maintaining cache coherence can be somewhat complicated. For example, if a processor p wants a writable copy of a location which is cached read-only by processors {qi}, a request for a writable copy is first sent from p to the directory. The directory then sends a writable copy to p (which can then proceed) and an invalidation message to every qi. Each qi invalidates its copy and sends and acknowledge back to the directory, which is waiting for all the invalidations to arrive before processing any more transactions on that location.

基于目录的缓存一致性，是在一个信息通过网络之上，实现一个共享内存抽象的方法，在一个集中的目录中记录了，哪个处理器缓存了一个内存位置的可读的或可写的副本。维护缓存一致性在一定程度上是复杂的。比如，如果一个处理器p需要一个位置的可写副本，而这个位置被处理器{qi}缓存为只读，那么就首先从p送到目录中一个可写副本的请求。目录然后发送一个可写的副本到p（马上就可以进行），然后给每个qi发送一个无效信息。每个qi将其副本置为无效，向目录发送并确认反馈，目录在等待所有的置无效到来，然后在这个位置上才能进行下一步的事务。

Although this single transaction sounds simple enough, the problem becomes more complicated when one considers scenarios in which several different transactions on the same location have been initiated at the same time, especially when messages are not guaranteed to arrive in the same order they were sent. A protocol verifier methodically explores all of these possibilities.

虽然单个事务足够简单，但考虑一下下面的场景，几个不同的事务在相同的位置上在同时被发起，这个问题就会变得非常复杂，尤其是，信息并不能保证以相同的顺序到达其被发送的地方。一个协议验证器有条理的探索所有可能性。

Since Murphi has no built-in support for message communication, the network was modeled as an array with a counter of the number of messages it contained. Out-of-order message reception was modeled using a rule set that had the position of the message in the array as a parameter. The has the effect of nondeterministically choosing a message to process, regardless of the order of message transmission.

由于Murphi对信息通信没有内建的支持，网络建模为一个阵列，有一个计数器是其包含的信息数量。乱序信息接收的建模，使用的是一个ruleset，在阵列中有信息的位置作为参数。这个的效果就是，不确定性的选择一条信息来处理，不管信息传输的顺序。

The description has separate scaling parameters (constant declarations) for: number of main memories and directories, number of caches and processors, number of addresses, number of legal memory values, size of directory entry (number of cached entries that can be kept track of), and capacity of the message network.

这个描述有分离的缩放参数（常量声明），如，主存和目录的数量，缓存和处理器的数量，地址的数量，合法的内存值的数量，目录entry的大小（可以追踪的缓存entry的数量），信息网络的容量。

The specification of the protocol is not complete. Instead, we have specified a set of properties that seem to be obvious necessary conditions for correct operation. Our specification made use of in-line error statements, invariants, and deadlock checking. The in-line error statements were used for several purposes, including reporting on common description errors, such as over owing the network array or the directory. However, the most important error statements were those that we inserted methodically on every unused branch of an if or case, to flag presumed impossible occurrences. These error statements were especially useful for detecting "unspecified receptions" of messages.

协议的规范是不完整的，我们指定了性质的集合，似乎是正确运算明显必须的条件。我们的规范使用了内联错误语句，invariants，和死锁检查。内联错误语句的使用有几个目的，包括汇报常见的描述错误，比如网络阵列或目录的错误。但是，多数重要的错误语句，是那些我们插入到每个未使用的if或case分支中的，以标记预先假设的不可能的情况。这些错误语句在检测“未指定接收”到的信息上尤其有用。

Other properties were specified using three invariants. The first checked for conditions that were empirically likely to be violated by description errors. For example, if the directory state for a particular memory address is "INV" (indicating that there are no cached copies), the directory list of cached copies should be empty.

其他性质是使用3个invariants来指定的。第一个检查的条件是，从经验上来看很可能被描述错误违反的。比如，如果特定内存地址的目录状态是INV（表明没有缓存的副本），缓存的拷贝的目录列表应当是空的。

The other two invariants check for cache consistency properties. One of these basically asserts that there are never two modifiable cached copies of the same address, although the condition is made much more complicated by various exceptions for transient states. For example, in this particular protocol there may legally be two modifiable copies if one is already being written back to main memory. Most of these conditions were determined experimentally by starting with simple invariant, running the verifier, and inspecting the results to see whether the violation is because the invariant is too strong or because of a genuine error.

另外两个invariants检查的是缓存一致性属性。一个基本上就是断言，不会有同样地址的两个可修改的缓存的副本，但这个条件会变得更加复杂，因为有各种短暂状态的异常。比如，在这个特定的协议中，有可能合理的有两个可修改的副本，如果一个已经被写回到主存中。这些条件的多数是由试验确定的，由简单的invariant开始，运行验证器，检查结果以查看，违反规则是由于invariant太强了，还是因为这是一个真正的错误。

The final invariant asserts that if a cache entry is read-only, its value is the same as the corresponding value in main memory. This, too, is tempered by various exceptions for transient conditions. For example, the protocol allows a modifiable copy to be converted to a read-only copy by writing back the modified value to main memory and changing the cache entry state. While the writeback message is in transit, the value of the (now) read-only cache entry may be different from the (not-yet-written-back) memory value.

最后一个invariant断言，如果一个缓存entry是只读的，其值与主存中的对应值是一样的。这同样有各种暂时条件导致的异常。比如，协议允许一个可修改的副本转化到一个可读副本，只要将修改的值写回到主存，然后改变缓存entry的状态。当写回信息在传输中时，（现在）可读的缓存entry的值，可能与（尚未写回）的主存中的值不同。

Surprisingly, almost all of the errors found were found with a description consisting of one main memory/directory, two processors/caches, and one memory location with one possible value (zero bits of data). Verifying at this scale required examining on the order of two thousand states. Scaling up to three processors, two values, and two main memories revealed only trivial errors, such as the use of the constant 0 for a value instead of the proper variable. In this case, hundreds of thousands of states were examined. The state explosion problem was only an issue in verifying scaled-up versions of the system, where, in fact, no additional problems were discovered.

令人惊讶的是，几乎发现的所有错误，其描述都是由一个主存/目录，两个处理器/缓存，一个存储位置有一个可能的值（0 bits数据）组成的。在这个规模进行验证，需要在两千个状态的级别进行检查。放大到3个处理器，2个值，和两个主存，只会找到无关紧要的错误，比如给一个值使用常数0，而不是合适的值。在这个情况中，要检查几十万个状态。状态爆炸问题，只是验证放大版系统的一个问题，实际上，并没有发现其他的问题。

Many of the in-line error statements were triggered, every invariant was violated, and several deadlocks were detected. There were many errors in the modeling, particularly in the handling of the network (e.g. failing to remove a message after it had been handled). Another more significant common error was a message arriving at a processor in an unexpected state, detected by an error statement in the default case of a switch statement. In many cases, this represented a legitimate possibility that could be handled by augmenting the design. In other cases, deeper changes were required. Many other errors were manifested in illegal global states, such as two processors having writable copies of a location. Only one memory value was required because most problems that would lead to inconsistencies showed up earlier as illegal states.

触发了很多内联的错误语句，违反了每个invariant，检测到了几个死锁。在建模中有很多错误，尤其是在处理网络时（如，在处理了一条信息之后，无法移除信息）。另一个更显著的常见错误是，一条信息以一种不被预期的状态到达处理器，由一个switch语句的默认case检测到一个错误语句。在很多情况中，这表示了一种合法的可能性，可以通过扩增设计来处理。在其他情况中，需要更深的变化。很多其他错误在非法的全局状态中展现出，比如两个处理器都有一个位置的可写的副本。只需要一个内存值，因为多数后来导致不一致的问题，之前都会展现为非法状态。

### 3.2 Link-level protocol

We also applied Murphi to the problem of verifying a link-level communication protocol. The protocol is basically a complicated version of the well-known alternating bit protocol, in that it uses one-bit sequence numbers to catch lost and duplicated messages. One of the complications is that the protocol has the capability of transmitting a group of several packets as a single unit.

我们还将Murphi应用到了验证连接级通信协议的问题上。这个协议基本上是一个复杂版的著名的比特交换协议，因为它使用1 bit序列数来捕获丢失和重复的信息。一个复杂的地方在于，协议有能力将几个包作为一个单元进行传输。

Verification with Murphi caught several fundamental errors in the initial design. Many of these stemmed from a group of packets being disrupted by the retransmission of another single packet or group of packets. Redesigning the protocol to be correct and also meet given performance goals was quite difficult, and required over a month of effort (with countless iterations of the verification- redesign cycle).

用Murphi进行验证，在初始设计时捕获了几个基本的错误。很多这些是由于，一组包被另外一个包或几个包的重新传输所扰乱。重新设计协议使其正确，也使其满足给定的性能目标，是非常困难的，而且需要超过一个月的工作（还有无数的验证-重新设计循环的迭代）。

Three major properties were specified. The first two were that that messages were not lost or duplicated. These were specified entirely by in-line error checks by exploiting data independence [15]: the control of the protocol does not depend on the data being sent.

指定了三种主要的性质。前两个是，信息不能丢失或重复。这两个是完全由内联错误检查指定的，利用了数据的独立性：协议的控制不依赖于发送的数据。

The description checks for lost and duplicated packets by sending exactly one packet with a "1" value; all other packets have value 0. The time at which to send the "1" packet packet is chosen nondeterministically. If there is a possibility of a packet being lost or duplicated, there is a possibility that that packet will be the "1" packet. So it is sufficient to verify that the "1" packet is not lost or duplicated. This trick works because the verifier considers all of the possible rule executions.

描述检查的是丢失的和重复的包，严格发送一个值为1的包；所有其他的包的值都是0。发送"1"的包的时间，是不确定性的选择的。如果有一种可能性，包丢失了或重复了，那么就有可能，这个包是"1"的包。只要验证“1”包没有丢失或重复，就足够了。这个会好用，因为验证器考虑了所有可能的规则执行的可能性。

The third property was that a group of packets arrived together. This property was checked by choosing to send no more than one group of packets, all of where the data value in every packet of the group was 1. All other packets carried the value 0. The description contains in-line error statements that look for the acceptance of at most one group of packets with the all values set to 1. A group has been disrupted if and only if some of its packets are 1 and some are 0.

第三个属性是，一组包同时到来。这个属性是通过选择发送不多于1组包进行检查的，在每个包中的数据值所有都是1。所有其他包都是值0。描述包含内联错误语句，查找接受最多一组所有值都设为1的包。如果一些包是1一些是0，同时只有在这种情况时，一个组才是被干扰的。

We believe that this specification is essentially complete. However, the same approach as we suggested for the cache coherence protocol of comparing the implementation protocol with another, more abstract protocol could be applied. In this case, the more abstract protocol would model communication over a reliable channel.

我们相信，这个规范是完整的。但是，与另一种实现协议进行比较时，我们建议了相同的缓存一致性协议，可以应用更抽象的协议。在这种情况下，更抽象的协议会对一个更可靠的通道进行建模通信。

Verifying the link-level protocol generally required dealing with larger state spaces than the cache-coherence protocol; however, all detected design errors were found examining fewer than 1 million states, which used 11 megabytes of memory.

验证连接级的协议一般需要处理更大的状态空间，这是与缓存一致性协议进行比较的；但是，所有检测到的设计错误是在检查少于一百万状态下找到的，使用了11MB内存。

## 4 Conclusions

In summary, automatic formal protocol verification can be a valuable design aid if 总结起来，自动形式化协议验证可以是一个宝贵的设计辅助工具

- it is used by a designer in the earliest design phases; 在最早期设计阶段，设计者进行使用；

- it is regarded as a debugging tool, not a guarantee of total correctness; 被当做一个调试工具，而并不是保证完全的正确性；

- the system is modeled at a high level of abstraction; and 系统在较高的抽象层次进行建模；

- the system description is down-scaled. 系统的描述被缩小。

The adoption of these principles maximizes the utility of verification given the current state of the art: they gain maximum economic advantage by catching the most expensive design errors as early as possible, and reduce the sizes of the state spaces that need to be explored, making verification computationally feasible.

采用这些原则，最大化了验证的利用：通过尽可能早的捕获最昂贵的设计错误，降低了需要探索的状态空间的大小，使得验证在计算上是可行的，可以得到最大化的经济优势。

Our findings on the industrial examples we have tried are: 我们在工业例子上的发现是：

- There are many bugs to be found in the early design phase. Verification finds them quickly. 在早期设计阶段有更多的bugs要发现。验证可以很迅速的发现这些bugs。

- The state explosion problem was not severe (because of adherence to the principles above). 状态爆炸问题并不是很严重（因为坚持以上的原则）。

- We were able to catch many errors using relatively weak specification methods, such as invariants and deadlock checking. 我们可以捕获很多错误，使用相对较弱的规范的方法，比如invariants和死锁检查。

Formal verification is also feasible without these assumptions, for example in comparing low-level sequential circuits [5, 7]. Techniques will advance in the future to increase the payoff for a broader range of problems. However, for the near-term future, we believe that the highest payoff can be obtained with these principles.

协议验证在没有这些假设的时候也是可行的，比如在比较底层顺序电路的时候。在未来，技术会进步，以在更广阔范围的问题中增加收益。但是，在较近期的未来，我们相信用这些原则就可以得到最高的收益。