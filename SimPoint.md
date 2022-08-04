# Automatically Characterizing Large Scale Program Behavior

Timothy Sherwood, Erez Perelman, Greg Hamerly, Brad Calder @ University of California, San Diego

## 0. Abstract

Understanding program behavior is at the foundation of computer architecture and program optimization. Many programs have wildly different behavior on even the very largest of scales (over the complete execution of the program). This realization has ramifications for many architectural and compiler techniques, from thread scheduling, to feedback directed optimizations, to the way programs are simulated. However, in order to take advantage of time-varying behavior, we must first develop the analytical tools necessary to automatically and efficiently analyze program behavior over large sections of execution.

理解程序行为是计算机架构和程序优化的基础。很多程序即使是在最大的尺度上（在程序的完整执行上），也有非常不同的行为。这种实现对很多架构和编译器技术有复杂的影响，从线程调度，到反馈引导的优化，到程序仿真的方式。但是，为利用时变的行为，我们必须首先开发必须的分析工具，以自动高效的在大规模的执行中分析程序行为。

Our goal is to develop automatic techniques that are capable of finding and exploiting the Large Scale Behavior of programs (behavior seen over billions of instructions). The first step towards this goal is the development of a hardware independent metric that can concisely summarize the behavior of an arbitrary section of execution in a program. To this end we examine the use of Basic Block Vectors. We quantify the effectiveness of Basic Block Vectors in capturing program behavior across several different architectural metrics, explore the large scale behavior of several programs, and develop a set of algorithms based on clustering capable of analyzing this behavior. We then demonstrate an application of this technology to automatically determine where to simulate for a program to help guide computer architecture research.

我们的目标是，开发自动的技术，可以找到和利用程序的大规模行为（在数十亿条指令中发现的行为）。为实现这个目标，第一步是开发出一个与硬件无关的度量，可以精确的总结一个程序段的执行中的行为。为此，我们试验了BBV的使用。我们量化了BBV在几种不同的架构度量中捕获程序行为的有效性，探索了几个程序的大规模行为，开发了一系列基于聚类的算法，可以分析这种行为。然后我们展示了这种技术的一种应用，自动的确定去仿真一个程序中的哪个部分，以帮助引导计算机架构的研究。

## 1. Introduction

Programs can have wildly different behavior over their run time, and these behaviors can be seen even on the largest of scales. Understanding these large scale program behaviors can unlock many new optimizations. These range from new thread scheduling algorithms that make use of information on when a thread’s behavior changes, to feedback directed optimizations targeted at not only the aggregate performance of the code but individual phases of execution, to creating simulations that accurately model full program behavior. To enable these optimizations, we must first develop the analytical tools necessary to automatically and efficiently analyze program behavior over large sections of execution.

程序运行时会有非常不同的行为，甚至在最大的尺度上，都可以看到这些行为。理解这些大规模程序行为，可以解锁很多新的优化。这包括新的线程调度算法，利用了一个线程何时行为变化的信息，到反馈引导的优化，其目标不仅是聚积代码的性能，还包括执行的单独阶段，以创建可以精确的对完整的程序行为进行建模的仿真。为达到这些优化，我们必须首先开发必须的分析工具，以自动和高效的在大量执行中分析程序行为。

In order to perform such an analysis we need to develop a hardware independent metric that can concisely summarize the behavior of an arbitrary section of execution in a program. In [19], we presented the use of Basic Block Vectors (BBV), which uses the structure of the program that is exercised during execution to determine where to simulate. A BBV represents the code blocks executed during a given interval of execution. Our goal was to find a single continuous window of executed instructions that match the whole program’s execution, so that this smaller window of execution can be used for simulation instead of executing the program to completion. Using the BBVs provided us with a hardware independent way of finding this small representative window.

为进行这样的分析，我们需要开发一种与硬件无关的度量，可以精确的总结程序中任意一段执行的行为。在[19]中，我们提出使用BBV，在程序执行中使用程序的结构，来确定对哪里进行仿真。一个BBV表示的是，在程序执行的给定间隔内，所执行的代码片段。我们的目标是，找到执行指令的单个的连续窗口，与整个程序的执行进行匹配，这样可以对这个更小的执行窗口进行仿真，而不用执行完整的程序。使用BBV给我们提供了一种硬件独立的找到这种小型代表性窗口的方法。

In this paper we examine the use of BBVs for analyzing large scale program behavior. We use BBVs to explore the large scale behavior of several programs and discover the ways in which common patterns, and code, repeat themselves over the course of execution. We quantify the effectiveness of basic block vectors in capturing this program behavior across several different architectural metrics (such as IPC, branch, and cache miss rates).

本文中，我们试验了使用BBVs来分析大规模程序行为。我们使用BBVs来探索了几个程序的大规模行为，发现了常见的模式和代码在执行的过程中重复其本身的方式。我们量化了BBVs在几种不同的架构度量中（如IPC，分支和缓存丢失率）捕获这种程序行为中的有效性。

In addition to this, there is a need for a way of classifying these repeating patterns so that this information can be used for optimization. We show that this problem of classifying sections of execution is related to the problem of clustering from machine learning, and we develop an algorithm to quickly and effectively find these sections based on clustering. Our techniques automatically break the full execution of the program up into several sets, where the elements of each set are very similar. Once this classification is completed, analysis and optimization can be performed on a per-set basis.

除了这些，需要一种对这些重复的模式进行分类的方法，这样这些信息可以用于优化。我们展示了，分类运行的片段的问题，与机器学习中的聚类问题相关，我们提出了一种算法，可以基于聚类迅速的有效的找到这些片段。我们的技术自动的将程序的完整执行分割成几个集合，每个集合的元素都非常相似。一旦分类完成，分析和优化都可以在per-set的基础上进行。

We demonstrate an application of this cluster-based behavior analysis to simulation methodology for computer architecture research. By making use of clustering information we are able to accurately capture the behavior of a whole program by taking simulation results from representatives of each cluster and weighing them appropriately. This results in finding a set of simulation points that when combined accurately represents the target application and input, which in turn allows the behavior of even very complicated programs such as gcc to be captured with a small amount of simulation time. We provide simulation points (points in the program to start execution at) for Alpha binaries of all of the SPEC 2000 programs. In addition, we validate these simulation points with the IPC, branch, and cache miss rates found for complete execution of the SPEC 2000 programs.

我们展示了这种基于聚类的行为分析，在计算机体系结构的仿真方法的应用。通过使用聚类信息，从每个聚类的代表中获取仿真结果，对其进行合理的加权，我们可以准确的捕获整个程序的行为。这会得到一系列仿真点，当准确的结合在一起时，可以表示目标应用和输入，然后又使得，即使是非常复杂的程序，如gcc，其行为也可以在较少的仿真时间内得到。我们给出了所有SPEC 2000程序的Alpha binaries的仿真点（程序中开始执行的点）。此外，我们完整执行了SPEC 2000程序，核实了这些仿真点的IPC，分支和缓存丢失率。

The rest of the paper is laid out as follows. First, a summary of the methodology used in this research is described in Section 2. Section 3 presents a brief review of basic block vectors and an in depth look into the proposed techniques and algorithms for identifying large scale program behaviors, and an analysis of their use on several programs. Section 4 describes how clustering can be used to analyze program behavior, and describes the clustering methods used in detail. Section 5 examines the use of the techniques presented in Sections 3 and 4 on an example problem: finding where to simulate in a program to achieve results representative of full program behavior. Related work is discussed in Section 6, and the techniques presented are summarized in Section 7.

本文剩余部分组织如下。首先，第2部分给出了本文使用的方法总结。第3部分给出了BBV的简要概览，详细介绍了提出的技术和算法，识别大规模程序行为，分析了在几个程序中的应用。第4部分描述了聚类怎样用于分析程序行为，详细的描述了使用的聚类方法。第5部分将第3和第4部分提出的技术在一个例子问题上进行了试验：找到对程序的哪里进行仿真，以得到可以代表完整程序行为的结果。相关的工作在第6部分给出，第7部分总结了该技术。

## 2. Methodology

In this paper we used both ATOM [21] and SimpleScalar 3.0c [3] to perform our analysis and gather our results for the Alpha AXP ISA. ATOM is used to quickly gather profiling information about the code executed for a program. SimpleScalar is used to validate the phase behavior we found when clustering our basic block profiles showing that this corresponds to the phase behavior in the programs performance and architecture metrics. The baseline microarchitecture model we simulated is detailed in Table 1. We simulate an aggressive 8-way dynamically scheduled microprocessor with a two level cache design. Simulation is execution-driven, including execution down any speculative path until the detection of a fault, TLB miss, or branch mis-prediction.

本文中，我们使用了ATOM和SimpleScalar 3.0c进行我们的分析，对Alpha AXP ISA得到我们的结果。ATOM用于迅速收集一个程序执行的代码的profiling信息。SimpleScalar用于核实我们在聚类我们的basic block profiles时候发现的阶段行为，展示了这对应着程序性能和架构度量中的阶段行为。我们仿真的基准架构模型如表1所示。仿真是执行驱动的，包括执行执行任意预测性路径，直到检测到一个错误，一个TLB丢失，或分支误预测。

We analyze and simulated all of the SPEC 2000 benchmarks compiled for the Alpha ISA. The binaries we used in this study and how they were compiled can be found at: http://www.simplescalar.com/.

我们分析和仿真了在Alpha ISA上编译的所有SPEC 2000基准测试。我们在本文中所使用的binaries，以及是怎样编译的，可以在下面的网站上找到。

## 3. Using Basic Block Vectors

A basic block is a section of code that is executed from start to finish with one entry and one exit. We use the frequencies with which basic blocks are executed as the metric to compare different sections of the application’s execution to one another. The intuition behind this is that the behavior of the program at a given time is directly related to the code it is executing during that interval, and basic block distributions provide us with this information.

一个basic block是一段代码，从头到尾执行，有一个入口，一个退出处。我们使用basic blocks执行的频率作为度量，来比较应用执行的片段之间的差异。这之后的道理是，程序在给定时间上的行为，是与在这段时间内执行的代码直接相关的，basic block的分布，给我们提供了这样的信息。

A program, when run for any interval of time, will execute each basic block a certain number of times. Knowing this information provides us with a fingerprint for that interval of execution, and tells us where in the code the application is spending its time. The basic idea is that knowing the basic block distribution for two different intervals gives us two separate fingerprints which we can then compare to find out how similar the intervals are to one another. If the fingerprints are similar, then the two intervals spend about the same amount of time in the same code, and the performance of those two intervals should be similar.

一个程序在其执行的任意时间段内，都会执行每个basic block一定数量次数。知道这种信息会给我们提供这段执行时间的特征，告诉我们应用在这段代码中将时间花费在哪里。其基本思想是，知道两个不同的片段的basic block分布，会给我们两个不同的特征，我们可以比较，以发现这些片段与其他片段有多相似。如果特征是相似的，那么两个片段在相同的代码中会花费相似的时间，这两个片段的性能就是类似的。

### 3.1 Basic Block Vector

A Basic Block Vector (BBV) is a single dimensional array, where there is a single element in the array for each static basic block in the program. For the results in this paper, the basic block vectors are collected in intervals of 100 million instructions throughout the execution of a program. At the end of each interval, the number of times each basic block is entered during the interval is recorded and a new count for each basic block begins for the next interval of 100 million instructions. Therefore, each element in the array is the count of how many times the corresponding basic block has been entered during an interval of execution, multiplied by the number of instructions in that basic block. By multiplying in the number of instructions in each basic block we insure that we weigh instructions the same regardless of whether they reside in a large or small basic block. We say that a Basic Block Vector which was gathered by counting basic block executions over an interval of N × 100 million instructions, is a Basic Block Vector of duration N.

一个BBV是一个一维阵列，对程序中的每个静态basic block，阵列中都有一个元素。对本文中的结果，BBV的收集是在程序执行的过程中，每1亿条指令的执行中收集的。在每个间隔的最后，在这个间隔中每个basic block进入的次数被记录下来，对下面1亿条指令的间隔中，每个basic block会有新的计数。因此，阵列中的每个元素是，在这段执行中，对应的basic block进入了多少次，乘以在这个basic block中有多少条指令。乘以了每个basic block所包含的指令数量后，我们确保对指令的加权是一样的，不论其是在一个大的或小的basic block中。我们说，在N×1亿条指令的间隔中收集的basic block执行，是持续为N的BBV。

Because we are not interested in the actual count of basic block executions for a given interval, but rather the proportions between time spent in basic blocks, a BBV is normalized by having each element divided by the sum of all the elements in the vector.

因为我们对在给定时间段内basic block执行的实际数量并不感兴趣，而是对在basic blocks中花费的时间比例感兴趣，所以对BBV进行了归一化，每个元素都除以向量中所有元素之和。

### 3.2 Basic Block Vector Difference

In order to find patterns in the program we must first have some way of comparing two Basic Block Vectors. The operation we desire takes as input two Basic Block Vectors, and outputs a single number which tells us how close they are to each other. There are several ways of comparing two vectors to one another, such as taking the dot product or finding the Euclidean or Manhattan distance. In this paper we use both the Euclidean and Manhattan distances for comparing vectors.

为找到程序中的模式，我们需要首先有比较两个BBV的方法。这种比较应当以两个BBV为输入，输出一个数，告诉我们它们有多接近。有几种方法比较两个向量，比如取其点积，或计算其欧式距离，或曼哈顿距离。本文中，我们使用欧式距离和曼哈顿距离来比较向量。

The Euclidean distance can be found by treating each vector as a single point in D-dimensional space. The distance between two points is simply the square root of the sum of squares just as in c^2 = a^2+b^2. The formula for computing the Euclidean distance of two vectors a and b in D-dimensional space is given by:

将每个向量视为D维空间中的一个点，我们就可以得到欧式距离。两个点之间的距离，就是各元素差值的平方和的平方根。计算两个D维向量a、b的欧式距离的公式为：

$$EuclideanDist(a,b) = sqrt_{\sum_{i=1}^D (a_i - b_i)^2}$$

The Manhattan distance on the other hand is the distance between two points if the only paths you can take are parallel to the axes. In two dimensions this is analogous to the distance traveled if you were to go by car through city blocks. This has the advantage that it weighs more heavily differences in each dimension (being closer in the x-dimension does not get you any closer in the y-dimension). The Manhattan distance is computed by summing the absolute value of the element-wise subtraction of two vectors. For vectors a and b in D-dimensional space, the distance can be computed as:

曼哈顿距离是两个点之间的沿着轴走过的距离。在二维的情况中，与在城市中开车经过的方块距离类似。这种距离的优势是，在维度差值大的权重就大（在x维上距离近，并不会在y维度上近）。曼哈顿距离的计算，是两个向量的元素之差的绝对值之和。对D维空间中的向量a和b，距离可以计算为：

$$ManhattanDist(a,b) = \sum_{i=1}^D |a_i - b_i|$$

Because we have normalized all of the vectors, the Manhattan distance will always be a single number between 0 and 2 (because we normalize each BBV to sum to 1). This number can then be used to compare how closely related two intervals of execution are to one another. For the rest of this section we will be discussing distances in terms of Manhattan distance, because we found that it more accurately represented differences in our high-dimensional data. We present the Euclidean distance as it pertains to the clustering algorithms presented in Section 4, since it provides a more accurate representation for data with lower dimensions.

因为我们归一化了所有的向量，曼哈顿距离是为0到2之间的数（因为每个BBV归一化到了1）。这个数可以用于比较，两个执行间隔之间关联密接的程度。本节剩余部分，我们会用曼哈顿距离来讨论，因为我们发现，这可以精确的表示高维数据的差异。我们给出欧式距离，因为适用于第4部分的聚类算法，因为这会给出较低维数据更精确的表示。

### 3.3 Basic Block Similarity Matrix

Now that we have a method of comparing intervals of program execution to one another, we can now concentrate on finding phase-based behavior. A phase of program behavior can be defined in several ways. Past definitions are built around the idea of a phase being a contiguous interval of execution during which a measured program metric is relatively stable. We extend this notion of a phase to include all similar sections of execution regardless of temporal adjacency.

现在我们有一种比较程序执行间隔的方法，我们现在可以集中在寻找基于阶段的行为上。程序行为的一个阶段，可以以几种方式定义。过去的定义是执行的一个相邻间隔，在这个间隔内，测量的程序度量是相对稳定的。我们将这个阶段的定义拓展为包括所有类似的执行的片段，而不论其在时间上是否相邻。

A key observation from this paper is that the phase behavior seen in any program metric is directly a function of the code being executed. Because of this we can use the comparison between the Basic Block Vectors as an approximate bound on how closely related any other metrics will be between those two intervals.

本文中一个关键的观察是，在任何程序度量中看到的阶段行为，都是执行的代码的函数。为此，我们可以使用BBV之间的比较，作为这两个片段之间其他度量接近程度的近似界限。

To find how intervals of execution relate to one another we create a Basic Block Similarity Matrix. The similarity matrix is an upper triangular N × N matrix, where N is the number of intervals in the program’s execution. An entry at (x, y) in the matrix represents the Manhattan distance between the basic block vector at interval x and the basic block vector at interval y.

为找到执行间隔怎样相互关联，我们创建了Basic Block相似性矩阵。相似性矩阵是一个上三角N × N矩阵，其中N是程序执行中的间隔数量。矩阵中位置(x,y)的元素，表示在间隔x和y处的BBV的曼哈顿距离。

Figures 1(left and right) and 4(left) shows the similarity matrices for gzip, bzip, and gcc using the Manhattan distance. The diagonal of the matrix represents the program’s execution over time from start to completion. The darker the points, the more similar the intervals are (the Manhattan distance is closer to 0), and the lighter they are the more different they are (the Manhattan distance is closer to 2).

图1（左右）和图4（左）展示了gzip，bzip和gcc使用曼哈顿距离的相似性矩阵。矩阵的对角表示在时间上程序执行从开始到结束。点越暗，片段之间就越相似（曼哈顿距离接近于0），点越亮，片段之间就越不同（曼哈顿距离接近于2）。

The top left corner of each graph is the start of program execution and is the origin of the graph, (0, 0), and the bottom right of the graph is the point (N − 1, N − 1) where N is the number of intervals that the full program execution was divided up into. The way to interpret the graph is to start considering points along the diagonal axis drawn. Each point is perfectly similar to itself, so the points directly on the axis all are drawn dark. Starting from a given point on the diagonal axis of the graph, you can begin to compare how that point relates to it’s neighbors forward and backward in execution by tracing horizontally or vertically. If you wish to compare a given interval x with the interval at x + n, you simply start at the point (x, x) on the graph and trace horizontally to the right until you reach (x, x + n).

每幅图的左上角，是程序执行的开始，是图的原点(0,0)，图的右下角是点(N-1, N-1)，其中N是程序执行分割成的间隔数量。解释这幅图的方法，要从对角线的点开始考虑。每个点与其本身完全相似，所以在对角线上的点是全黑的。从图中对角线上的点开始，垂直或水平开始追踪，可以比较这个点与其前后的点的相似关系。如果你希望比较一个给定的间隔x和x+n处的间隔，你只需要从点(x,x)处开始，水平向右找到点(x,x+n)。

To examine the phase behavior of programs, let us first examine gzip because it has behavior on such a large scale that it is easy to see. If we examine an interval taken from 70 billion instructions into execution, in Figure 1 (left), this is directly in the middle of a large phase shown by the triangle block of dark color that surrounds this point. This means that this interval is very similar to it’s neighbors both forward and backward in time. We can also see that the execution at 50 billion and 90 billion instructions is also very similar to the program behavior at 70 billion. We also note, while it may be hard to see in a printed version that the phase interval at 70 billion instructions is similar to the phases at interval 10 and 30 billion, but they are not as similar as to those around 50 and 90 billion. Compare this with the IPC and data cache miss rates for gzip shown in Figure 2. Overall, Figure 1(left) shows that the phase behavior seen in the similarity matrix lines up quite closely with the behavior of the program, with 5 large phases (the first 2 being different from the last 3) each divided by a small phase, where all of the small phases are very similar to each other.

为检查程序的阶段行为，我们首先检查gzip，因为其行为在大尺度上容易看到。如果我们检查从70 billions指令开始执行的间隔，在图1左中，这是在一个大型阶段的中间，有一个很大的暗色三角形。这意味着，这个片段与其前后时间上的片段都很类似。我们还可以看到，在50 billion和90 billion指令处的执行与在70 billions处的指令执行都是很类似的。我们还要指出，在打印版上可能比较难以看到，在70 billion处的指令执行与10 billion和30 billion处的虽然很类似，但是与其与50 billion和90 billion处的类似程度是不一样的。将这个与图2中gzip的IPC和数据缓存丢失率相比较。总体上，图1左展示了相似性矩阵中的阶段行为，与程序的行为非常接近，有5个大型的阶段（前2个与后3个非常不同），每两个之间都有一个小型阶段分隔，而所有的小型阶段之间则非常类似。

The similarity matrix for bzip (shown on the right of Figure 1) is very interesting. Bzip has complicated behavior, with two large parts to it’s execution, compression and decompression. This can readily be seen in the figure as the large dark triangular and square patches. The interesting thing about bzip is that even within each of these sections of execution there is complex behavior. This, as will be shown later, makes the behavior of bzip impossible to capture using a small contiguous section of execution.

bzip的相似性矩阵（图1右）则非常有趣。Bzip有复杂的行为，其执行有2个大的部分，压缩和解压。这可以在图中的大型暗三角和方形块中看出。bzip有趣的事是，即使在执行的每个部分的内部，都有复杂的行为。后面会看到，这使得bzip的行为用小型相邻的片段无法捕获到。

A more complex case for finding phase behavior is gcc, which is shown on the left of Figure 4. This similarity matrix shows the results for gcc using the Manhattan distance. The similarity matrix on the right will be explained in more detail in Section 4.2.1. This figure shows that gcc does have some regular behavior. It shows that, even here, there is common code shared between sections of execution, such as the intervals around 13 billion and 36 billion. In fact the strong dark diagonal line cutting through the matrix indicates that there is good amount of repetition between offset segments of execution. By analyzing the graph we can see that interval x is very similar to interval (x + 23.6B) for all x.

找到阶段行为的一个更复杂的案例是gcc，如图4左所示。相似性矩阵展示了使用曼哈顿距离的gcc的结果。右边的相似性矩阵在4.2.1节会更详细的解释。这个图展示了，gcc确实有一些规则的行为。这个图说明了，即使是这里，在执行的片段中有一些共享的通用代码，比如在13 billion和36 billion的附近。实际上，将矩阵切割开来的暗对角线说明，在执行的片段中有很多重复的地方。通过分析这个图，我们可以看到，间隔x与间隔x+23.6B处都是很类似的。

Figures 2 and 5 show the time varying behavior of gzip and gcc. The average IPC and data cache miss rate is shown for each 100 million interval of execution over the complete execution of the program. The time varying results graphically show the same phase behavior seen by looking at only the code executed. For example, the two phases for gcc at 13 billion and 36 billion, shown to be very similar in Figure 4, are shown to have the same IPC and data cache miss rate in Figure 5.

图2和图5展示了gzip和gcc的时变行为。在程序的完整执行过程中，每1亿条指令间隔的平均IPC和数据缓存丢失率进行了展示。图示的时变结果，通过观察执行的代码，展示了相同的阶段行为。比如，在13 billion和36 billion处gcc的两个阶段，在图4中可以看到很类似，在图5中也可以看到有相同的IPC和数据缓存丢失率。

## 4. Clustering

The basic block vectors provide a compact and representative summary of the program’s behavior for intervals of execution. By examining the similarity between them, it is clear that there exists a high level pattern to each program’s execution. In order to make use of this behavior we need to start by delineating a method of finding and representing the information. Because there are so many intervals of execution that are similar to one another, one efficient representation is to group the intervals together that have similar behavior. This problem is analogous to a clustering problem. Later, in Section 5, we demonstrate how we use the clusters we discover to find multiple simulation points for irregular programs or inputs like gcc. By simulating only a single representative from each cluster, we can accurately represent the whole program’s execution.

BBV对程序执行的片段的行为给出了紧凑和有代表性的总结。通过检查其之间的相似性，可以很明显看出来，对每个程序的执行，有高层的模式。为利用这些行为，我们需要一种找到和表示这些信息的方法。因为有太多的执行间隔互相之间很相似，一种有效的表示是，将有相似行为的间隔分到一组里。这个问题与聚类问题类似。后面在第5部分，我们展示了我们怎样使用发现的clusters来对不规则程序或输入，如gcc，来找到多个仿真点。我们对每个cluster只仿真一个有代表，可以精确的表示整个程序的执行。

### 4.1 Clustering Overview

The goal of clustering is to divide a set of points into groups such that points within each group are similar to one another (by some metric, often distance), and points in different groups are different from one another. This problem arises in other fields such as computer vision [10], document classification [22], and genomics [1], and as such it is an area of much active research. There are many clustering algorithms and many approaches to clustering. Classically, the two primary clustering approaches are Partitioning and Hierarchical:

聚类的目标是，将点集分成组，这样每个组内的点是互相相似的（利用某个度量，通常是距离），而不同组的点则是不同的。这个问题出现在其他领域中，如计算机视觉，文档分类，和基因组学，因此这是一个非常活跃的研究领域。有很多聚类算法，很多方法可以聚类。两种基本的聚类方法是分割和层次化：

Partitioning algorithms choose an initial solution and then use iterative updates to find a better solution. Popular algorithms such as k-means [14] and Gaussian Expectation-Maximization [2, pages 59–73] are in this family. These algorithms tend to have run time that is linear in the size of the dataset.

分割算法选择一个初始解，然后使用迭代的更新来找到一个更好的解。流行的算法如k-means和高斯期望最大化，就是这个类别中的。这些算法这些算法的运行时间一般与数据集大小是线性关系。

Hierarchical algorithms [9] either combine together similar points (called agglomerative clustering, and conceptually similar to Huffman encoding), or recursively divides the dataset into more groups (called divisive clustering). These algorithms tend to have run time that is quadratic in the size of the dataset.

层次化算法要么将类似的点结合到一起（称为凝结聚类，概念上与Huffman编码类似），或递归的将数据集分成更多的组（称为分裂式聚类）。这些算法的运行时间通常与数据集的大小的平方成正比。

### 4.2 Phase Finding Algorithm

For our algorithm, we use random linear projection followed by k-means. We choose to use the k-means clustering algorithm, since it is a very fast and simple algorithm that yields good results. To choose the value of k, we use the Bayesian Information Criterion (BIC) score [11, 17]. The following steps summarize our algorithm, and then several of the steps are explained in more detail:

在我们的算法中，我们使用随机线性投影，然后进行k-means聚类。我们选择使用k-means聚类算法，因为算法速度快，简单，效果也很好。为选择k的值，我们使用贝叶斯信息准则(BIC)分数。下面的步骤总结了算法，一些步骤会更详细的进行解释：

1. Profile the basic blocks executed in each program to generate the basic block vectors for every 100 million instructions of execution.

对每个程序中执行的basic block获取其特征，以对每1亿条指令执行生成BBV。

2. Reduce the dimension of the BBV data to 15 dimensions using random linear projection.

使用随机线性投影将BBV数据降维到15。

3. Try the k-means clustering algorithm on the low-dimensional data for k values 1 to 10. Each run of k-means produces a clustering, which is a partition of the data into k different clusters.

在低维数据上使用1到10的k值运行k-means聚类算法。k-means的每次运行都产生一种聚类，即将数据分成k个不同的聚类。

4. For each clustering (k = 1 ... 10), score the fit of the clustering using the BIC. Choose the clustering with the smallest k, such that it’s score is at least 90% as good as the best score.

对每种聚类(k=1,...,10)，对聚类结果使用BIC进行评分。选择最小k的聚类，使其分数至少是最佳分数的90%。

#### 4.2.1 Random Projection

For this clustering problem, we have to address the problem of dimensionality. All clustering algorithms suffer from the so-called “curse of dimensionality”, which refers to the fact that it becomes extremely hard to cluster data as the number of dimensions increases. For the basic block vectors, the number of dimensions is the number of executed basic blocks in the program, which ranges from 2,756 to 102,038 for our experimental data, and could grow into the millions for very large programs. Another practical problem is that the running time of our clustering algorithm depends on the dimension of the data, making it slow if the dimension grows too large.

对这个聚类问题，我们要处理维度的问题。所有聚类算法都有所谓的维度诅咒的问题，这是指在数据维度提升的情况下，数据聚类的难度会非常高。对于BBV，维度数量是程序中运行的basic blocks的数量，对我们的试验数据来说，是2756到102038，对于非常大型的程序可能增长到上百万。另一个实际的问题是，我们的聚类算法的运行时间与数据维度有关，如果维度过高，运行速度会很慢。

Two ways of reducing the dimension of data are dimension selection and dimension reduction. Dimension selection simply removes all but a small number of the dimensions of the data, based on a measure of goodness of each dimension for describing the data. However, this throws away a lot of data in the dimensions which are ignored. Dimension reduction reduces the number of dimensions by creating a new lower-dimensional space and then projecting each data point into the new space (where the new space’s dimensions are not directly related to the old space’s dimensions). This is analogous to taking a picture of 3 dimensional data at a random angle and projecting it onto a screen of 2 dimensions.

有两种数据降维的方式，分别是维度选择和维度缩减。维度选择就是去掉大部分数据的维度，选择一小部分，基于每个维度描述数据的好坏程度。但是，这丢掉了很多维度的数据。维度缩减通过创造一个新的低维空间，将每个数据点投影到新空间（新空间的维度与老空间的维度不直接相关），降低了维度。这与对三维数据在随机角度拍照，将其投影到2维屏幕上类似。

For this work we choose to use random linear projection [5] to create a new low-dimensional space into which we project the data. This is a simple and fast technique that is very effective at reducing the number of dimensions while retaining the properties of the data. There are two steps to reducing a dataset X (which is a matrix of basic block vectors and is of size N_intervals × D_numbb, where D_numbb is the number of basic blocks in the program) down to D_new dimensions using random linear projection:

这个工作我们选择使用随机线性投影[5]来创建一个新的低维空间，然后投影数据。这是一个简单快速的技术，可以很有效的降低维数，同时保持数据的性质。将数据集X（这是BBV的一个矩阵，大小为N_intervals × D_numbb，其中D_numbb是程序中basic blocks的数量）使用随机线性投影降低到D_new维，有两个步骤：

- Create a D_numbb×D_new projection matrix M by choosing a random value for each matrix entry between -1 and 1.

创建一个D_numbb×D_new的投影矩阵M，矩阵中每个元素为-1到1之间的随机值。

- Multiply X times M to obtain the new lower-dimensional dataset X' which will be of size N_intervals × D_new.

用X乘以M，以得到新的低维数据集X'，大小为N_intervals × D_new。

For clustering programs, we found that using D_new = 15 dimensions is sufficient to still differentiate the different phases of execution. Figure 7 shows why we chose to project the data down to 15 dimensions. The graph shows the number of dimensions on the x-axis. The y-axis represents the k value found to be best on average, when the programs were projected down to the number of dimensions indicated by the x-axis. The best k is determined by the k with the highest BIC score, which is discussed in Section 4.2.3. The y-axis is shown as a percent of the maximum k seen for each program so that the curve can be examined independent of the actual number of clusters found for each program. The results show that for 15 dimensions the number of clusters found begins to stabilize and only climbs slightly. Similar results were also found using a different method of finding k in [6].

对聚类程序，我们发现使用D_new=15足以区分程序执行的不同阶段。图7展示了，为什么我们选择将数据投影到15维上。图中x轴是维度数量。y轴表示当程序投影到x轴的维数时，平均找到的最佳的k值。最佳k值是与最高BIC分数的k比较得到的，这在4.2.3节讨论。y轴表示对每个程序找到的最大k数值的比例，这样曲线可以与每个程序的实际clusters数量无关。结果表明，15是开始稳定的数值，更大的维度只会略有上升。使用不同的寻找k的方法，可以得到类似的结果。

The advantages of using linear projections are twofold. First, creating new vectors with a low dimension of 15 is extremely fast and can even be done at simulation time. Secondly, using only 15 dimensions speeds up the k-means algorithm significantly, and reduces the memory requirements by several orders of magnitude over using the original basic block vectors.

使用线性投影的好处有2个。第一，创建新15维低维的向量是很快的，可以在仿真的时间完成。第二，使用15维显著加速了k-means算法，与使用原始BBV相比，内存需求降低了好几个数量级幅度。

Figure 4 shows the similarity matrix for gcc on the left using original BBVs, whereas the similarity matrix on the right shows the same matrix but on the data that has been projected down to 15 dimensions. For the reduced dimension data we use the Euclidean distance to measure differences, rather than the Manhattan distance used on the full data. After the projection, some information will be blurred, but overall the phases of execution that are very similar with full dimensions can still be seen to have a strong similarity with only 15 dimensions.

图4左展示了对gcc使用原始BBVs的相似性矩阵，而右边的相似性矩阵展示了投影到15维后的结果。对于降维后的数据，我们使用欧式距离来度量差异，而不是在完整数据时使用的曼哈顿距离。在投影后，一些信息被模糊了，但总体上，执行的阶段与完整维度时的情况是类似的，15维可以看到非常强的相似性。

#### 4.2.2 K-means

The k-means algorithm is an iterative optimization algorithm, which executes as two phases, which are repeated to convergence. The algorithm begins with a random assignment of k different centers, and begins its iterative process. The iterations are required because of the recursive nature of the algorithm; the cluster centers define the cluster membership for each data point, but the data point memberships define the cluster centers. Each point in the data belongs to, and can be considered a member of, a single cluster.

k-means算法是一种迭代优化算法，包含两个阶段，重复进行以得到收敛的结果。算法开始时随机指定k个不同的中心，然后开始其迭代过程。迭代是必须的，因为算法的递归本质；聚类中心定义了每个数据点的聚类归属，数据点的归属定义了聚类中心。数据中的每个点属于一个聚类，可以考虑是一个聚类的的一个成员。

We initialize the k cluster centers by choosing k random points from the data to be clustered. After initialization, the k-means algorithm proceeds in two phases which are repeated until convergence:

我们初始化k个聚类中心，从数据中选择k个随机点以聚类。在初始化之后，k-means算法以两个阶段进行，重复直到收敛：

- For each data point being clustered, compare its distance to each of the k cluster centers and assign it to (make it a member of) the cluster to which it is the closest.

对每个要聚类的数据点，比较其与k个聚类中心的距离，将其指定到距离最近的聚类中。

- For each cluster center, change its position to the centroid of all of the points in its cluster (from the memberships just computed). The centroid is computed as the average of all the data points in the cluster.

对每个聚类中心，将其位置调整为其聚类中所有点的重心。重心计算为聚类为所有数据点的平均。

This process is iterated until membership (and hence cluster centers) cease to change between iterations. At this point the algorithm terminates, and the output is a set of final cluster centers and a mapping of each point to the cluster that it belongs to. Since we have projected the data down to 15 dimensions, we can quickly generate the clusters for k-means with k from 1 to 10. In doing this, there are efficient algorithms for comparing the clusters that are formed for these different values of k, and choosing one that is good but still uses a small value for k is the next problem.

这个过程要重复进行，直到迭代过程不会改变聚类中心和点的聚类归属。这样算法就终止了，输出是最终的聚类中心，还有每个点到其所属的聚类的映射。因为我们将数据投影到15维，我们可以迅速生成聚类，k-means的k值为1到10。对不同的k值进行聚类，然后进行比较，有高效的算法，选择一个较小的但效果较好的k，是下一个问题。

#### 4.2.3 Bayesian Information Criterion

To compare and evaluate the different clusters formed for different k, we use the Bayesian Information Criterion (BIC) as a measure of the “goodness of fit” of a clustering to a dataset. More formally, the BIC is an approximation to the probability of the clustering given the data that has been clustered. Thus, the larger the BIC score, the higher the probability that the clustering being scored is a “good fit” to the data being clustered. We use the BIC formulation given in [17] for clustering with k-means, however other formulations of the BIC could also be used.

为比较和评估不同的k值形成的不同聚类，我们使用贝叶斯信息准则(BIC)作为一个数据集的聚类好的程度的度量。更正式的，BIC是在给定了被聚类的数据后，聚类概率的近似。因此，BIC值越大，评分的聚类越可能是一个好的聚类。我们使用[17]中的BIC对k-means聚类进行评分，但是也可以使用其他的BIC形式。

More formally, the BIC score is a penalized likelihood. There are two terms in the BIC: the likelihood and the penalty. The likelihood is a measure of how well the clustering models the data. To get the likelihood, each cluster is considered to be produced by a spherical Gaussian distribution, and the likelihood of the data in a cluster is the product of the probabilities of each point in the cluster given by the Gaussian. The likelihood for the whole dataset is just the product of the likelihoods for all clusters. However, the likelihood tends to increase without bound as more clusters are added. Therefore the second term is a penalty that offsets the likelihood growth based on the number of clusters. The BIC is formulated as

更正式的，BIC分数是一个惩罚的似然。BIC中有两项：似然和惩罚。似然是聚类对数据建模程度的度量。为得到似然，每个聚类都认为是由一个球形高斯分布生成，一个聚类中数据的似然是聚类中每个点的概率的乘积。整个数据集的似然，就是所有聚类的似然的乘积。但是，随着加入更多的聚类，似然会倾向于没有界限的增加。因此，第二项是一个惩罚项，基于聚类的数量对似然的增加加入偏置。BIC如下式所示

$$BIC(D,k) = l(D|k) - \frac {p_j}{2} log(R)$$

where l(D|k) is the likelihood, R is the number of points in the data, and pj is the number of parameters to estimate, which is (k − 1) + dk + 1 for (k − 1) cluster probabilities, k cluster center estimates which each require d dimensions, and 1 variance estimate. To compute l(D|k) we use

其中l(D|k)是似然，R是数据中点的数量，pj是要估计的参数的数量，对于k-1个聚类概率，是(k − 1) + dk + 1，k个聚类中心估计，每个需要d维，和1个方差估计。为计算l(D|k)，我们使用

$$l(D|k) = \sum_{i=1}^k (-R_i log(2π) - R_i d log(σ^2) - (R_i - 1) + 2R_i log(R_i/R) )/2 $$

where R_i is the number of points in the ith cluster, and σ^2 is the average variance of the Euclidean distance from each point to its cluster center.

其中R_i是第i个聚类中点的数量，σ^2是每个点到其聚类中心的欧式距离的平均方差。

For a given program and inputs, the BIC score is calculated for each k-means clustering, for k from 1 to N. We then choose the clustering that achieves a BIC score that is at least 90% of the spread between the largest and smallest BIC score that the algorithm has seen. Figure 8 shows the benefit of choosing a BIC with a high value and its relationship with the variance in IPC seen for that cluster. The y-axis shows the percent of IPC variance seen for a given clustering, and the corresponding BIC score the clustering received. Each point on the graph represents the average or max IPC variance for all points in the range of ±5% of the BIC score shown. The results show that picking clusterings that represent greater than 80% of the BIC score resulted in an IPC variance of less than 20% on average. The IPC variance was computed as the weighted sum of the IPC variance for each cluster, where the weight for a cluster is the number of points in that cluster. The IPC variance for each cluster is simply the variance of the IPC for all the points in that cluster.

对给定的程序的输入，BIC分数对每个k-means聚类都进行计算，k从1到N。我们然后选择的，是BIC分数至少是算法看到的最大和最小BIC之间差异的90%。图8展示了选择高BIC的好处，以及与IPC变化之间的关系。y轴展示的是对给定聚类的IPC变化，和对应聚类的BIC分数。图中的每个点表示，在±5%的BIC分数变化范围内，平均或最大IPC变化。结果表明，选择的聚类大于80%的BIC分数，会得到IPC变化小于20%。IPC变化的计算，是每个聚类的IPC变化的加权和，其中一个聚类的加权，是聚类中点的数量。对每个聚类的IPC变化，只是这个聚类中所有点的IPC的变化。

### 4.3  Clusters and Phase Behavior

Figures 3 and 6 show the 6 clusters formed for gzip and the 4 clusters formed for gcc. The X-axis corresponds to the execution of the program in billions of instructions, and each interval (each of 100 million instructions) is tagged to be in one of the N clusters (labeled on the Y-axis). These figures, just as for Figures 1 and 4, show the execution of the programs to completion.

图3和图6展示了，形成gzip的6个聚类，和形成gcc的4个聚类。X轴对应着程序在十亿次运行的单位的执行，每个间隔，每个间隔（每1亿条指令）会标记为在N个聚类中的一个（在Y轴标记）。这些图，就像图1和4，展示程序运行到结束。

For gzip, the full run of the execution is partitioned into a set of 6 clusters. Looking to Figure 1(left) for comparison, we see that the cluster behavior captured by our tool lines up quite closely with the behavior of the program. The majority of the points are contained by clusters 1,2,3 and 6. Clusters 1 and 2 represent the large sections of execution which are similar to one another. Clusters 3 and 6 capture the smaller phases which lie in between these large phases, while cluster 5 contains a small subset of the larger phases, and cluster 4 represents the initialization phase.

对于gzip，完整运行分割成6个聚类。与图1左进行比较，我们看到我们的工具捕获的聚类行为，与程序的行为是非常吻合的。多数点都包含在聚类1，2，3，6中。聚类1和2表示执行的大型片段，互相之间很相似。聚类3和6捕获了更小的阶段，在大型阶段之间，而聚类5更大型阶段的一个很小的子集，聚类4表示初始化阶段。

In the cluster graph for gcc, shown in Figure 6, the run is now partitioned into 4 different clusters. Looking to Figure 4 for comparison, we see that even the more complicated behavior of gcc is captured correctly by our tool. Clusters 2 and 4 correspond to the dark boxes shown parallel to the diagonal axis. It should also be noted that the projection does introduce some degree of error into the clustering. For example, the first group of points in cluster 2 are not really that similar to the other points in the cluster. Comparing the two similarity matrices in Figure 4, shows the introduction of a dark band at (0,30) on the graph which was not in the original (un-projected) data. Despite these small errors, the clustering is still very good, and the impact of any such errors will be minimized in the next section.

在gcc的聚类图中，如图6所示，运行分割成4个不同的聚类。与图4进行比较，我们看到，更复杂的gcc的行为，我们的工具也捕获的很正确。聚类2和4对应着暗色矩形，与对角线轴相平行。还应当指出，投影确实对聚类引入了一些错误。比如，聚类2中的第一组点，与该聚类中的其他点并没有那么相似。与图4的两个相似性矩阵相比，在图中的(0,30)处引入了一个暗色带，在未投影的数据中是没有的。尽管有一些小的错误，聚类仍然是非常好的，任何这样的错误的影响，在下一节都会得到最小化。

## 5. Finding Simulation Points

Modern computer architecture research relies heavily on cycle accurate simulation to help evaluate new architectural features. While the performance of processors continues to grow exponentially, the amount of complexity within a processor continues to grow at an even a faster rate. With each generation of processor more transistors are added, and more things are done in parallel on chip in a given cycle while at the same time cycle times continue to decrease. This growing gap between speed and complexity means that the time to simulate a constant amount of processor time is growing. It is already to the point that executing programs fully to completion in a detailed simulator is no longer feasible for architectural studies. Since detailed simulation takes a great deal of processing power, only a small subset of a whole program can be simulated.

现代计算机架构研究严重依赖于周期精确的仿真来帮助评估新的架构特征。处理器的性能一直在呈指数级增长，处理器内部的复杂度以更快的速度在增长。每一代处理器都加入了更多的晶体管，在芯片中在给定的周期内更多的处理在并行进行，同时周期时间在持续降低。速度与复杂度之间的空白逐渐增加，这意味着仿真固定数量的处理器时间所需的时间在增长。已经达到了这样一个点，在一个详细的仿真器中执行程序到结束，对于架构研究已经不再可行。因为详细的仿真要消耗大量处理能力，只能对完整程序的一小部分进行仿真。

SimpleScalar [3], one of the faster cycle-level simulators, can simulate around 400 million instructions per hour. Unfortunately many of the new SPEC 2000 programs execute for 300 billion instructions or more. At 400 million instructions per hour this will take approximately 1 month of CPU time.

SimpleScalar是最快的周期级的仿真器之一，每个小时可以仿真大约4亿条指令。不幸的是，很多新的SPEC 2000程序包含3000亿条指令或更多。以4亿条指令每小时的速度，这大约需要1个月的CPU时间。

Because it is only feasible to execute a small portion of the program, it is very important that the section simulated is an accurate representation of the program’s behavior as a whole. The basic block vector and cluster analysis presented in Sections 3 and 4 will allow us to make sure that this is the case.

因为只能执行一小部分程序，那么非常重要的是，仿真的是程序对其行为要具有精确代表性。第3和第4部分介绍的BBV和聚类分析，会确保是这种情况。

### 5.1 Single Simulation Points

In [19], we used basic block vectors to automatically find a single simulation point to potentially represent the complete execution of a program. A Simulation Point is a starting simulation place (in number of instructions executed from the start of execution) in a program’s execution derived from our analysis. That algorithm creates a target basic block vector, which is a BBV that represents the complete execution of the program. The Manhattan distance between each interval BBV and the target BBV is computed. The BBV with the lowest Manhattan distance represents the single simulation point that executes the code closest to the complete execution of the program. This approach is used to calculate the long single simulation points (LongSP) described below.

在[19]中，我们使用BBV来自动找到一个仿真点，以代表程序的完整执行。一个仿真点，是从我们的分析推导出来的，程序执行中一个开始仿真的位置（从开始执行处以指令数为计）。算法创建了一个目标BBV，这是能代表程序完整执行的BBV。计算每个间隔BBV和目标BBV的曼哈顿距离。具有最小曼哈顿距离的BBV，表示仿真点，执行这段代码，与程序的完整执行是最接近的。这种方法用于计算长仿真点(LongSP)，下面进行描述。

In comparison, the single simulation point results in this paper are calculated by choosing the BBV that has the smallest Euclidean distance from the centroid of the whole dataset in the 15-dimensional space, a method which we find superior to the original method. The 15-dimensional centroid is formed by taking the average of each dimension over all intervals in the cluster.

比较起来，本文中的仿真点结果，是选择与整个数据集在15维空间中的重心欧式距离最小的BBV得到的，这种方法与原方法相比效果更好。15维重心是通过聚类中所有片段的每一维的平均得到的。

Figure 9 shows the IPC estimated by executing only a single interval, all 100 million instructions long but chosen by different methods, for all SPEC 2000 programs. This is shown in comparison to the IPC found by executing the program to completion. The results are from SimpleScalar using the architecture model described in Section 2, and all fast forwarding is done so that all of the architecture structures are completely warmed up when starting simulation (no cold-start effect).

图9展示了对所有的SPEC 2000程序，通过执行单个片段，所有1亿条指令那么长，但是通过不同方法选择的，估计得到的IPC。这与执行了整个程序到结束得到的IPC进行比较。结果是用SimpleScalar，使用第2节描述的架构模型得到的，所有快进都做好了，这样所有架构结构是完全预热过的，然后开始仿真（没有冷启动的效果）。

The first bar, labeled none, is the IPC found when executing only the first 100 million instructions from the start of execution (without any fast forwarding). The second bar, FF-Billion shows the results after blindly fast forwarding 1 billion instructions before starting simulation. The third bar, SimPoint shows the IPC using our single simulation point analysis described above, and the last bar shows the IPC of simulating the program to completion (labeled Full). Because these are actual IPC values, values which are closer to the Full bar are better.

第一个方块，标记为None，是从开始执行处执行前1亿条指令得到的IPC（没有任何快进）。第二个方块，FF-Billion展示的盲目快进1亿条指令，然后开始仿真的结果。第三个方块，SimPoint展示的是使用我们的单仿真点分析得到的IPC，最后一个方块展示的是仿真整个程序到结束得到的IPC（标记为Full）。因为这些是实际的IPC值，所以与Full方块接近的值是更好的。

The results in Figure 9 shows that the single simulation points are very close to the actual full execution of the program, especially when compared against the ad-hoc techniques. Starting simulation at the start of the program results in an average error of 210%, whereas blindly fast forwarding results in an average 80% IPC error. Using our single simulation point analysis we reduce the average IPC error to 18%. These results show that it is possible to reasonably capture the behavior of the most programs using a very small slice of execution.

图9中的结果表明，单仿真点的结果，与程序实际完整执行的结果很接近，尤其是与ad-hoc技术相比。在程序开始处开始仿真的结果平均误差为210%，而盲目快进的结果IPC误差平均为80%。使用我们的单仿真点分析，我们将平均IPC误差降低到了18%。这些结果表明，用程序很小的一个片段，来捕获多数程序的行为，是可行的。

Table 2 shows the actual simulation points chosen along with the program counter (PC) and procedure name corresponding to the start of the interval. If an input is not attached to the program name, then the default ref input was used. Columns 2 through 4 are in terms of the number of intervals (each 100 million instruction long). The first column is the number of instructions executed by the program, on the specific input, when run to completion. The second column shows the end of initialization phase calculated as described in [19]. The third column shows the single simulation point automatically chosen as described above. This simulation point is used to fast forward to the point of desired execution. Some simulators, debuggers, or tracing environments (e.g., gdb) provide the ability to fast forward based upon a program PC, and the number of times that PC was executed. We therefore, provide the instruction PC for the start of the simulation point, the procedure that PC occurred in, and the number of times that PC has to be executed in order to arrive at the desired simulation point.

表2展示了选择的实际仿真点，和PC值，和过程名称，对应片段的开始。如果输入与程序名称没有连接，那么就使用默认的ref输入。第2列到第4列是片段的数字（每个是1亿条指令长）。第1列是程序执行的指令数量，在特定输入时，运行到结束。第2列展示的是初始化阶段结束，按[19]中描述的方法计算。第3列展示的是按上述方法自动选择的单仿真点。这个仿真点用于快进到此，然后进行期望的执行。一些仿真器，debuggers，或追踪环境（如，gdb）有基于程序PC，和PC执行的次数进行快进的能力。因此，我们给出仿真点的开始的指令PC，这条PC发生所在的过程，和PC要执行的次数，以达到理想的仿真点。

These results show that a single simulation point can be accurate for many programs, but there is still a significant amount of error for programs like bzip, gzip and gcc. This occurs because there are many different phases of execution in these programs, and a single simulation point will not accurately represent all of the different phases. To address this, we used our clustering analysis to find multiple simulation points to accurately capture these programs behavior, which we describe next.

这些结果表明，单仿真点对很多程序是很精确的，但对于bzip，gzip和gcc这样的程序，仍然有明显的误差。这是因为，在这些程序中有很多不同的执行阶段，单仿真点不能准确的代表所有不同的阶段。为处理这个问题，我们使用聚类分析找到多个仿真点，以准确的捕获这些程序行为，下面我们进行描述。

### 5.2 Multiple Simulation Points

To support multiple simulation points, the simulator can be run from start to stop, only performing detailed simulation on the selected intervals. Or the simulation can be broken down into N simulations, where N is the number of clusters found via analysis, and each simulation is run separately. This has the further benefit of breaking the simulation down into parallel components that can be distributed across many processors. This is the methodology we use in our simulator. For both cases results from the separate simulation points need to be weighed and combined to arrive at overall performance for the program [4]. Care must be taken to combine statistics correctly (simply averaging will give incorrect results for statistics such as rates).

为支持多个仿真点，仿真器的运行从开始到停止，只在选择的片段上进行详细的仿真。或者仿真要分裂成N个仿真，其中N是通过分析发现的聚类数量，每个仿真单独运行。这还有一个好处，可以将仿真分成并行进行的部分，在多个处理器上分布式运行。这是我们在我们的仿真器中使用的方法。在这两种情况中，不同仿真点得到的结果需要进行加权，然后结合到一起，得到程序的总体性能。要小心的将统计数据正确的结合到一起（简单的平均，会得到统计结果的不正确结果，比如rates）。

Knowing the clustering alone is not sufficient to enable multiple point simulation because the cluster centers do not correspond to actual intervals of execution. Instead, we must first pick a representative for each cluster that will be used to approximate the behavior of the the full cluster. In order to pick this representative, we choose for each cluster the actual interval that is closest to the center (centroid) of the cluster. In addition to this, we weigh any use of this representative by the size of the cluster it is representing. If a cluster has only one point, it’s representative will only have a small impact on the overall outcome of the program.

只知道聚类，不足以开启多点仿真，因为聚类中心并不对应着实际的执行片段。我们首先选择每个聚类的一个代表，用于近似完整聚类的行为。为选择这个代表，对每个聚类，我们选择的是与聚类中心（重心）距离最近的实际片段。此外，我们对代表的加权，是用其代表的聚类的大小。如果一个聚类只有一个点，那么其代表对程序整体输出的影响将会非常小。

Table 2 shows the multiple simulation points found for all of the SPEC 2000 benchmarks. For these results we limited the number of clusters to be at most six for all but the most complex programs. This was done, in order to limit the number of simulation points, which also limits the amount of warmup time needed to perform the overall simulation. The cluster formation algorithm in Section 4 takes as an input parameter the max number of clusters to be allowed. Each simulation point contains two numbers. The first number is the location of the simulation point in 100s of millions of instructions. The second number in parentheses is the weight for that simulation point, which is used to create an overall combined metric. Each simulation point corresponds to 100 million instructions.

表2展示了对所有SPEC 2000基准测试找到的多仿真点。对这些结果，我们限制聚类数量最多是6个，但最复杂的程序除外。这是为了限制仿真点的数量，也限制了需要的预热时间来进行整体的仿真。第4部分的聚类形成算法，其一个输入参数，是允许的最大聚类数量。每个仿真点包含两个数。第一个数，是仿真点的位置。第二个数在括号中，是这个仿真点的权重，用于创建总体的结合的度量。每个仿真点对应着1亿条指令。

Figure 10 shows the IPC results for multiple simulation points. The first bar shows our single simulation points simulating for 100 million instructions. The second bar LongSP chooses a single simulation point, but the length of simulation is identical to the length used for multiple simulation points (which may go up to 1 billion instructions). This is to provide a fair comparison between the single simulation points and multiple. The Multiple bar shows results using the multiple simulation points, and the final bar is IPC for full simulation. As in Figure 9, the closer the bar is to Full, the better.

图10展示了多仿真点的IPC结果。第一个方块展示了我们的单仿真点结果，仿真了1亿条指令。第二个方块LongSP选择了一个仿真点，但仿真的长度与多仿真点一样（最多可能到10亿条指令）。这是为了对单仿真点和多仿真点进行公平比较。多仿真点的方块展示了相应的结果，最后的方块是完整仿真的IPC。如图9所示，与完整方块最接近的方法效果最好。

The results show that the average IPC error rate is reduced to 3% using multiple simulation points, which is down from 17% using the long single simulation point. This is significantly lower than the average 80% error seen for blindly fast forwarding. The benefits can be most clearly seen in the programs bzip, gcc, ammp, and galgel. The reason that the long contiguous simulation points do not do much better is that they are constrained to only sample at one place in the program. For many programs this is sufficient, but for those with interesting long term behavior, such as bzip, it is impossible to approximate the full behavior.

结果表明，使用多仿真点，平均IPC误差率降低到了3%，使用LongSP则为17%。这比盲目快进的平均80%误差降低的非常明显。在程序bzip，gcc，ammp和galgel中可以很清楚的看到其好处。连续长仿真点没有得到更好的效果的原因，是它们局限在程序的一个位置进行采样。对很多程序，这是足够的，但对于那些有趣的长期行为的，比如bzip，则不可能近似完整的行为。

Figure 11 is the average over all of the floating point programs (top graph) and integer programs (bottom graph). Errors for IPC, branch miss rate, instruction and data cache miss rates, and the unified L2 cache miss rate for the architecture presented in Section 2 are shown. The errors are with respect to these metrics for the full length of simulation using SimpleScalar. Results are shown for starting simulation at the start of the program None, blindly fast forwarding a billion instructions FF-Billion, single simulation points of duration 1 (SimPoint) and k (LongSP), and multiple simulation points (Multiple).

图11是对所有的浮点程序（上图）和整型程序（下图）的平均结果。第2部分中给出的IPC，分支错误率，指令缓存和数据缓存丢失率，和统一L2缓存丢失率的误差进行了展示。这些误差是相对于用SimpleScalar进行完整仿真结果来说的。从程序开始进行仿真的结果是None，盲目快进1亿条指令的是FF-Billion，单仿真点一个片段的为SimPoint，k个片段的为LongSP，多仿真点为Multiple。

The first thing to note is that using the just a single small simulation point performs quite well on average across all of the metrics when compared to blindly fast-forwarding. Even though a single SimPoint does well, it is clearly beaten by using the clustering based scheme presented in this paper across all of the metrics examined. One thing that stands out on the graphs is that the error rate of the instruction cache and L2 cache appear to be high (especially for the integer programs) despite the fact that our technique is doing quite well in terms of overall performance. This is due to the fact that we present here an arithmetic mean of the errors, and there are several programs that have high error rates due to the very small number of cache misses. If there are 10 misses in the whole program, and we estimate there to be 100, that will result in a error of 10X. We point to the overall IPC as the most important metric for evaluation as it implicitly weighs each of the metrics by it’s relative importance.

第一个要说明的是，只使用单个小的仿真点，与盲目快进相比，其所有度量值平均看起来已经很好了。即使单个SimPoint效果很好，使用本文中给出的聚类方法，在所有度量上都可以得到更好的结果。图中一个很明显的点是，指令缓存和L2缓存的误差似乎很高（尤其是对于整型程序），而我们的技术在整体性能上表现非常好。这是因为，我们给出的是误差的代数平均，有几个程序有很高的误差率，因为非常少的缓存丢失。如果在整个程序中有10次misses，我们估计有100个，那么就会得到10倍的误差率。我们认为整体的IPC是要评估的最重要的度量，因为其隐式的对每个度量用其相对重要性进行了加权。

## 6. Related Work

*Time Varying Behavior of Programs*: In [18], we provided a first attempt at showing the periodic patterns for all of the SPEC 95 programs, and how these vary over time for cache behavior, branch prediction, value prediction, address prediction, IPC and RUU occupancy.

程序的时变行为：在[18]中，我们我们首次尝试展示所有SPEC 95程序的周期性模式，以及对于缓存的行为，分支预测，值的预测，地址的预测，IPC和RUU occupancy来说，这些是怎样随着时间变化的。

*Training Inputs and Finding Smaller Representative Inputs*: One approach for reducing the simulation time is to use the training or test inputs from the SPEC benchmark suite. For many of the benchmarks, these inputs are either (1) still too long to fully simulate, or (2) too short and place too much emphasis on the startup and shutdown parts of the program’s execution, or (3) inaccurately estimate behavior such as cache accesses do to decreased working set size.

训练输入和找到更小的代表性输入：一种减少仿真时间的方法是，从SPEC基准测试包中使用训练或测试输入。对很多基准测试，这些输入要么是(1)太长了，不能完全仿真，或(2)太短了，过去强调程序执行的开始阶段和停止阶段，或(3)不准确的估计行为，比如缓存访问，以降低working set大小。

KleinOsowski et. al [12], have developed a technique where they manually reduce the input sets of programs. The input sets were developed using a range of approaches from truncating of the input files to modification of source code to reduce the number of times frequent loops were traversed. For these input sets they develop, they make sure that they have similar results in terms of IPC, cache, and instruction mix.

KleinOsowski等[12]提出了一种技术，手工减少程序的输入集。输入集是用一系列方法开发的，从截断输入文件，到修改源码来减少频繁的循环的次数。对这些他们开发的输入集，他们确保在IPC，缓存和指令混合上都有类似的结果。

*Fast Forwarding and Check-pointing*: Historically researchers have simulated from the start of the application, but this usually does not represent the majority of the program’s behavior because it is still in the initialization phase. Recently researchers have started to fast-forward to a given point in execution, and then start their simulation from there, ideally skipping over the initialization code to an area of code representative of the whole. During fast-forward the simulator simply needs to act as a functional simulator, and may take full advantage of optimizations like direct execution. After the fast-forward point has been reached, the simulator switches to full cycle level simulation.

快进和检查点：历史上研究者都是从应用的开始进行仿真，但是这通常不代表程序的主要行为，因为这仍然在初始化阶段。最近，研究者开始快进到给定的执行点，然后从那里开始其仿真，理想的跳过了初始化代码，到达对整体具有代表性的代码区域。在快进的过程中，仿真器只是需要作为一个功能仿真器，可以完全利用优化功能，如直接执行。在到达快进点后，仿真器切换到了完整周期级的仿真。

After fast-forwarding, the architecture state to be simulated is still cold, and a warmup time is needed in order to start collecting representative results. Efficiently warming up execution only requires references immediately proceeding the start of simulation. Haskins and Skadron [7] examined probabilistically determining the minimum set of fast-forward transactions that must be executed for warm up to accurately produce state as it would have appeared had the entire fast-forward interval been used for warm up [7]. They recently examined using reuse analysis to determine how far before full simulation warmup needs to occur [8].

在快进后，要仿真的架构状态仍然是冷的，需要预热时间以开始收集有代表性的结果。高效的预热执行只需要立刻参考仿真开始的过程。Haskins等[7]检查了概率上确定最小集合快进事务，必须要执行以预热，以准确的产生状态，就像它已经出现有完整的快进片段用于预热。他们最近检查了使用重用分析，以确定在完整仿真之前，预测需要多久。

An alternative to fast forwarding is to use check-pointing to start the simulation of a program at a specific point. With check-pointing, code is executed to a given point in the program and the state is saved, or checkpointed, so that other simulation runs can start there. In this way the initialization section can be run just one time, and there is no need to fast forward past it each time. The architectural state (e.g., caches, register file, branch prediction, etc) can either be stored in the trace (if they are not going to change across simulation runs) or can be warmed up in a manner similar to described above.

快进的一个替代品，是使用检查点来在特定点开始程序仿真。有了检查点，代码在程序给定点进行执行，状态得到保存，或保存到检查点，这样其他仿真可以在这里开始。这样，初始化部分可以只运行一次，没必要每次都快进经过这里。架构状态（如，缓存，寄存器组，分支预测，等）可以要么存储在trace中（如果他们在仿真运行时不会变化），或可以以之前描述的方式进行预热。

*Automatically Finding Where to Simulate*: Our work is based upon the basic block distribution analysis in [19] as described in prior sections. Recent work on finding simulation points for data cache simulations is presented by Lafage and Seznec [13]. They proposed a technique to gather statistics over the complete execution of the program and use them to choose a representative slice of the program. They evaluate two metrics, one which captures memory spatial locality and one which captures memory temporal locality. They further propose to create specialized metrics such as instruction mix, control transfer, instruction characterization, and distribution of data dependency distances to further quantify the behavior of the both the program’s full execution and the execution of samples.

自动找到仿真的位置：我们的工作是基于[19]中的basic block分布分析，在之前的章节中描述过。最近的Lafage等[13]给出了在数据缓存仿真中寻找仿真点的工作。他们提出了一种技术，收集程序完整执行的统计数据，用这些数据来选择程序的一个有代表性的切片。他们评估两个度量，一个捕获的存储空间局部性，一个捕获的存储时间局部性。他们进一步提出来创建专用的度量，比如指令混合，控制迁移，指令特征化，数据依赖性距离的分布，以进一步量化程序完整执行和样本执行的行为。

*Statistical Sampling*: Several different techniques have been proposed for sampling to estimate the behavior of the program as a whole. These techniques take a number of contiguous execution samples, referred to as clusters in [4], across the whole execution of the program. These clusters are spread out throughout the execution of the program in an attempt to provide a representative section of the application being simulated. Conte et. al [4] formed multiple simulation points by randomly picking intervals of execution, and then examining how these fit to the overall execution of the program for several architecture metrics (IPC and branch and data cache statistics). Our work is complementary to this, where we provide a fast and metric independent approach for picking multiple simulation points based just on basic block vector similarity. When an architect gets a new binary to examine they can use our approach to quickly find the simulation points, and then validate these with detailed simulation in parallel with using the binary.

统计采样：已经提出了几种不同的技术来采样，来估计程序整体的行为。这些技术以几个相邻的执行样本为输入，在[4]中称之为clusters，在程序的完整执行过程中都进行。这些clusters分布在程序的执行过程中，以给出要仿真的应用有代表性的部分。Conte等通过随机选择执行的间隔，然后检查这些在几种架构度量上（IPC，分支和数据缓存统计数据）怎样拟合程序的整体执行。我们的工作是这个的互补，我们提供了一个快速的、与度量无关的方法来选择多个仿真点，只是基于BBV相似性。当一个架构师得到一个新的binaries要检查，可以使用我们的方法以迅速找到仿真点，然后用详细的仿真核实这个的正确性，同时使用binary并行进行仿真。

*Statistical Simulation*: Another technique to improve simulation time is to use statistical simulation [16]. Using statistical simulation, the application is run once and a synthetic trace is generated that attempts to capture the whole program behavior. The trace captures such characteristics as basic block size, typical register dependencies and cache misses. This trace is then run for sometimes as little as 50-100,000 cycles on a much faster simulator. Nussbaum and Smith [15] also examined generating synthetic traces and using these for simulation and was proposed for fast design space exploration. We believe the techniques presented in this paper are complementary to the techniques of Oskin et al. and Nussbaum and Smith in that more accurate profiles can be determined using our techniques, and instead of attempting to characterize the program as a whole it can be characterized on a per-phase basis.

统计仿真：另一种改进仿真时间的技术是，使用统计仿真。使用统计仿真，应用运行一次，生成一个合成的trace，试图捕获整体程序的行为。trace捕获的特征包括basic block大小，典型的寄存器依赖关系，和缓存丢失情况。这个trace然后在一个更快的仿真器上运行50-100000个周期。Nussbaum等也检查生成合成trace，使用这些用于仿真，然后用于快速设计空间探索。我们相信，本文中给出的技术与Oskin等和Nassbaum和Smith等的技术是互补的，因为使用我们的技术可以得到更精确的profiles，不用将程序作为一个整体得到其特征，而是在一个逐阶段的基础上进行。

## 7. Summary

At the heart of computer architecture and program optimization is the need for understanding program behavior. As we have shown, many programs have wildly different behavior on even the very largest of scales (over the full lifetime of the program). While these changes in behavior are drastic, they are not without order, even in very complex applications such as gcc. In order to help future compiler and architecture researchers in exploiting this large scale behavior, we have developed a set of analytical tools that are capable of automatically and efficiently analyzing program behavior over large sections of execution.

在计算机架构和程序优化的核心，是需要理解程序的行为。我们已经展示了，即使在非常大的尺度上，很多程序有非常不同的行为（在程序的完全声明周期内）。这些在行为上的变化是剧烈的，但它们并不是没有规则的，即使是非常复杂的应用，如gcc。为帮助未来的编译器和架构师研究者探索这种大规模的行为，我们开发一系列分析工具，可以在很多执行片段上自动的、高效的分析程序行为。

The development of the analysis is founded on a hardware independent metric, Basic Block Vectors, that can concisely summarize the behavior of an arbitrary section of execution in a program. We showed that by using Basic Block Vectors one can capture the behavior of programs as defined by several architectural metrics (such as IPC, and branch and cache miss rates).

这些分析工具的开发，是基于一种硬件无关的度量，BBV，可以简明的总结程序中任意片段执行的行为。我们展示了，使用BBV，可以捕获程序的行为，由几种架构度量所定义（如，IPC，分支预测率，缓存丢失率）。

Using this framework, we examine the large scale behavior of several complex programs like gzip, bzip, and gcc, and find interesting patterns in their execution over time. The behavior that we find shows that code and program behavior repeat over time. For example, in the input we examined in detail for gcc we see that program behavior repeats itself every 23.6 billion instructions. Developing techniques that automatically capture behavior on this scale is useful for architectural, system level, and runtime optimizations. We present an algorithm based on the identification of clusters of basic block vectors that can find these repeating program behaviors and group them into sets for further analysis. For two of the programs gzip and gcc we show how the clustering algorithm results line up nicely with the similarity matrix and correlate with the time varying IPC and data cache miss rates.

使用这种框架，我们检查几种复杂程序的大规模行为，如gzip，bzip和gcc，以找到在其执行中随着时间的进行的有趣模式。我们发现的行为表明，代码和程序的行为随着时间重复。比如，在输入中，我们检查了gcc的细节，我们看到程序行为每23.6 billion指令就会自我重复。开发一些技术可以自动捕获这种尺度上的行为，对于架构，系统级和运行时优化是有用的。我们提出一种算法，基于识别BBV的聚类，可以找到这些重复的程序行为，将其分组到集合中，以进行进一步分析。为其中两个程序gzip和gcc，我们展示了聚类算法的结果与相似性矩阵如何类似，与时变的IPC和数据缓存丢失率是如何相关的。

It is increasingly common for computer architects and compiler designers to use a small section of a benchmark to represent the whole program during the design and evaluation of a system. This leads to the problem of finding sections of the program’s execution that will accurately represent the behavior of the full program. We show how our clustering analysis can be used to automatically find multiple simulation points to reduce simulation time and to accurately model full program behavior. We call this clustering tool to find single and multiple simulation points SimPoint. SimPoint along with additional simulation point data can be found at: http://www.cs.ucsd.edu/~calder/simpoint/. For the SPEC 2000 programs, we found that starting simulation at the start of the program results in an average error of 210% when compared to the full simulation of the program, whereas blindly fast forwarding resulted in an average 80% IPC error. Using a single simulation point found, using our basic block vector analysis, resulted in an average 17% IPC error. When using the clustering algorithm to create multiple simulation points we saw an average IPC error of 3%.

对于计算机架构师和编译器设计者来说，在系统的设计和评估阶段，使用一个基准测试的一小部分来代表整个程序，这越来越常见。这带来了一个问题，即找到这个程序执行的片段，要能准确的代表完整程序的行为。我们展示了，我们的聚类分析怎样用于自动的找到多个仿真点，以减少仿真时间，和准确的对完整程序行为建模。我们称这种聚类工具为SimPoint，可以找到单个和多个仿真点。SimPoint和额外的仿真点数据可以在下面链接中找到。对SPEC 2000程序，我们发现在程序的开始进行仿真，与程序的完整仿真相比，平均有210%的误差，而盲目快进得到的平均IPC误差有80%。使用找到的单个仿真点，使用我们的BBV分析，得到平均17%的IPC误差。当使用聚类算法得到多个仿真点，我们可以将IPC误差降低到平均3%。

Automatically identifying the phase behavior using clustering is beneficial for architecture, compiler, and operating system optimizations. To this end, we have used the notion of basic block vectors and a random projection to create an efficient technique for identifying phases on-the-fly [20], which can be efficiently implemented in hardware or software. Besides identifying phases, this approach can predict not only when a phase change is about to occur, but to which phase it is about to transition. We believe that using phase information can lead to new compiler optimizations with code tailored to different phases of execution, multi-threaded architecture scheduling, power management, and other resource distribution problems controlled by software, hardware or the operating system.

使用聚类自动识别阶段行为，对架构，编译器和操作系统优化，都是有好处的。为此，我们使用BBV和随机投影的概念，来创建一种高效的技术识别on-the-fly技术，可以在硬件和软件中高效的实现。除了识别阶段，这种方法可以预测阶段变化就要发生了，还可以预测要迁移到哪个阶段了。我们相信，使用阶段信息会得到新的编译器优化，代码对不同阶段的执行的定制，多线程架构调度，功耗管理，和其他资源分布问题，由软件，硬件或操作系统控制。