# SimPoint 3.0: Faster and More Flexible Program Phase Analysis

Greg Hamerly, Erez Perelman et. al. @ Baylor University & University of California, San Diego

## 0. Abstract

This paper describes the new features available in the SimPoint 3.0 release. The release provides two techniques for drastically reducing the run-time of SimPoint: faster searching to find the best clustering, and efficiently clustering large numbers of intervals. SimPoint 3.0 also provides an option to output only the simulation points that represent the majority of execution, which can reduce simulation time without much increase in error. Finally, this release provides support for correctly clustering variable length intervals, taking into consideration the weight of each interval during clustering. This paper describes SimPoint 3.0’s new features, how to use them, and points out some common pitfalls.

本文描述了SimPoint 3.0中可用的新特征。这个版本有两种技术，极大的降低了SimPoint的运行时间：更快的搜索，以找到最佳的聚类；和对大量区间的高效聚类。SimPoint 3.0还给出了一个选项，可以只输出表示主要执行的仿真点，可以降低仿真时间，而不会大幅度增加误差。最后，这个版本可以对变长的区间进行正确的聚类，在聚类的时候，考虑了每个区间的权重。本文描述了SimPoint 3.0的新特征，怎样使用，并指出了一些常见的陷阱。

## 1. Introduction

Modern computer architecture research requires understanding the cycle level behavior of a processor during the execution of an application. To gain this understanding, researchers typically employ detailed simulators that model each and every cycle. Unfortunately, this level of detail comes at the cost of speed, and simulating the full execution of an industry standard benchmark can take weeks or months to complete, even on the fastest of simulators. To make matters worse, architecture researchers often simulate each benchmark over a variety of architecture configurations and designs to find the set of features that provide the best trade-off between performance, complexity, area, and power. For example, the same program binary, with the exact same input, may be run hundreds or thousands of times to examine how the effectiveness of an architecture changes with cache size. Researchers need techniques to reduce the number of machine-months required to estimate the impact of an architectural modification without introducing an unacceptable amount of error or excessive simulator complexity.

现代计算机架构研究需要在应用的执行过程中，理解处理器的周期级行为。为获得这种理解，研究者一般会采用细节仿真器，对每个周期进行建模。不幸的是，这种级别的细节的代价是速度，对一个工业级的标准基准测试进行完整执行仿真，即使是在最快的仿真器中，也需要几个星期，甚至几个月来完成。更加恶化的是，架构研究者通常要对每个基准测试，在多个架构配置和设计上进行仿真，以找到在性能，复杂度，面积和功耗能够最佳折中的特征集合。比如，相同的程序binary，用相同的输入，会被运行成百上千次，以检查一个cache大小架构变化的有效性。研究者需要技术来降低估计架构变化的影响所需的机器-时间，而不能带来很大的误差，或带来过多的仿真器复杂度。

At run-time, programs exhibit repetitive behaviors that change over time. These behavior patterns provide an opportunity to reduce simulation time. By identifying each of the repetitive behaviors and then taking only a single sample of each repeating behavior, we can perform very fast and accurate sampling. All of these representative samples together represent the complete execution of the program. The underlying philosophy of SimPoint [1, 2, 3, 4, 5, 6] is to use a program’s behavior patterns to guide sample selection. SimPoint intelligently chooses a very small set of samples called Simulation Points that, when simulated and weighed appropriately, provide an accurate picture of the complete execution of the program. Simulating only these carefully chosen simulation points can save hours to days of simulation time with very low error rates. The goal is to run SimPoint once for a binary/input combination, and then use these simulation points over and over again (potentially for thousands of simulations) when performing a design space exploration.

在运行时，程序会表现出重复性的行为，随着时间而变化。这些行为模式提供了降低仿真时间的机会。通过识别每种重复性行为，然后对每个重复性行为只取一个样本，我们就可以进行非常快速准确的采样。所有这些代表性的样本一起，就表示了程序的完整执行。SimPoint的潜在哲学是，使用一个程序的行为模式，来引导样本的选择。SimPoint智能的选择很少量的样本，称为仿真点，如果进行合适的仿真和加权，是可以代表程序的完整执行的性能。只对这些仔细选择的仿真点进行仿真，可以节省数小时到数天的仿真时间，而带来的误差却很低。我们的目标是，对一个binary/input组合，运行一次SimPoint，然后在进行设计空间探索时，不断使用这些仿真点（可能进行数千次仿真）。

This paper describes the new SimPoint 3.0 release. In Section 2 we present an overview of the SimPoint approach. Section 4 describes the new SimPoint features, and describes how and when to tune these parameters. It also provides a summary of SimPoint’s results and discusses some suggested configurations. Section 5 discusses the common pitfalls to watch for when using SimPoint, and Section 6 summarizes this paper. Finally, the appendix describes in detail the command line options for SimPoint 3.0.

本文描述了新版SimPoint 3.0。在第2部分，我们给出了SimPoint方法的概览。第4部分描述了新的SimPoint特征，描述了怎样、何时来调节这些参数；还给出了SimPoint结果的总结，讨论了一些建议的配置。第5部分讨论了在使用SimPoint时常见的注意点，第6部分总结了这篇文章。最后，附录详细描述了SimPoint 3.0的命令行选项。

The major new features for the SimPoint 3.0 release include: 新版的主要新特征包括：

- **Efficient searching to find the best clustering**. Instead of trying every value, or every Nth value, of k when running the k-means algorithm, we provide a binary search method for choosing k. This typically reduces the execution time of SimPoint by a factor of 10.

高效的搜索，以找到最佳的聚类。在运行k-means算法时，不运行每个k值，也不每N个值运行一个，我们给出了选择k的一个binary搜索方法。这一般会将SimPoint的运行时间降低到之前的1/10。

- **Faster SimPoint analysis when processing many intervals**. To speed the execution of SimPoint on very large inputs (100s of thousands to millions of intervals), we sub-sample the set of intervals that will be clustered. After clustering, the intervals not selected for clustering are assigned to phases based on their nearest cluster.

当处理很多区间时，SimPoint分析的速度会更快。在输入很大时（数十万到上百万片段），为加速SimPoint的分析，我们对将要进行聚类的区间集合进行亚采样。在聚类后，那些没有选择进行聚类的区间，就基于其最近的聚类指定给那些状态。

- **Support for Variable Length Intervals**. Prior versions of SimPoint assumed fixed length intervals, where each interval represents the same amount of dynamic execution. For example, in the past, each interval represented 1, 10, or 100 million dynamic instructions. SimPoint 3.0 provides support for clustering variable length intervals, where each interval can represent different amounts of dynamic execution. With variable length intervals, the weight of each interval must be considered during clustering.

支持变长区间。之前版本的SimPoint假设是固定长度的区间，每个区间表示相同数量的动态执行。比如，在过去，每个区间表示一百万，一千万或一亿条动态指令。SimPoint 3.0支持对变长区间的聚类，这里每个片段会表示不同量的动态执行。在可变长度区间下，需要在聚类时考虑每个区间的权重。

- **Reduce the number of simulation points by representing only the majority of executed instructions**. We provide an option to output only the simulation points whose clusters account for the majority of execution. This reduces simulation time, without much increase in error.

只表示主要的执行指令，从而降低仿真点的数量。我们提供了一个选项，输出的仿真点，其聚类只包含主要的执行。这降低了仿真时间，而误差没有很大的增加。

## 2. Background

Several other researchers have worked on phase analysis, and we review some of the related work here.

几个其他的研究者在状态分析上进行研究，我们回顾一些相关的工作。

### 2.1 Related Work on Phase Analysis

The recurring use of different areas of memory in a program is first noted by Denning and Schwarz [7] and is formalized as the idea of working sets. While working sets have driven the development of caches for decades, recently many of the more subtle implications of recurring behaviors have been explored by researchers in the computer architecture community.

程序重复使用内存的不同区域，这种现象在[7]中首次被注意到，形成了working sets的概念。Working sets驱动了caches几十年的发展，最近计算机架构研究者研究了这种重复行为很多更微妙的含义。

Balasubramonian et al. [8] proposed using hardware counters to collect miss rates, CPI and branch frequency information for every 100,000 instructions. They use these miss rates and the total number of branches executed for each interval to dynamically evaluate the program’s stability. They used their approach to guide dynamic cache reconfiguration to save energy without sacrificing performance.

[8]提出使用硬件计数器来收集每10万条指令的miss率，CPI和分支频率信息。他们使用这些每个区间内的miss率和执行的总计分支数量，以动态的评估程序的稳定性。他们使用这种方法来引导动态cache重配置，在不损失性能的情况下节约能耗。

Dhodapkar and Smith [9, 10, 11] found a relationship between phases and instruction working sets, and show that phase changes occur when the working set changes. They proposed a method by which the dynamic reconfiguration of multi-configuration units can be controlled in response to phase changes indicated by working set changes. Through a working set analysis of the instruction cache, data cache and branch predictor they derive methods to save energy.

[9，10，11]发现了状态和指令working sets之间的关系，展示了状态会在working set变化的时候同时变化。他们提出了一种方法，多配置单元的动态重配置，可以被控制以响应状态的变化，这是由working set的变化指示的。通过指令cache、数据cache和分支预测器的working set分析，可以推导出节省能量的方法。

Hind et al. [12] provide a framework for defining and reasoning about program phase classifications, focusing on how to best define granularity and similarity to perform phase analysis.

Hind等[12]提出了一个框架，对程序状态的分类进行定义和推理，聚焦在怎样对粒度和相似度进行最好的定义，以进行状态分析。

Isci and Martonosi [13, 14] have shown the ability to dynamically identify the power phase behavior using power vectors. Deusterwald et al. [15] recently used hardware counters and other phase prediction architectures to find phase behavior.

[13,14]展示了用能耗向量来动态识别能耗状态行为的能力。[15]最近使用硬件计数器和其他状态预测架构来找到状态行为。

These related methods offer alternative techniques for representing programs for the purpose of finding phase behaviors. They each also offer methods for using the data to find phases. Our work on SimPoint frames the problem as a clustering problem in the machine learning setting, using data clustering algorithms to find related program behaviors. This problem is a natural application of data clustering, and works well.

这些相关的方法提供了为寻找状态行为表示程序的其他技术。他们都给出了使用数据来寻找状态的方法。我们在SimPoint上的工作，将这个问题表示成一个聚类问题，使用数据聚类算法来找到相关的程序行为。这个问题是数据聚类的一个自然应用，应用效果很好。

### 2.2 Phase Vocabulary

To ground our discussion in a common vocabulary, the following is a list of definitions we use to describe the analysis performed by SimPoint.

我们的讨论要基于共同的词汇，下面是我们用于描述SimPoint的分析的词汇的定义：

- Interval - A section of continuous execution (a slice in time) of a program. All intervals are assumed to be non-overlapping, so to perform our analysis we break a program’s execution into contiguous non-overlapping intervals. The prior versions of SimPoint required all intervals to be the same size, as measured in the number of instructions committed within an interval (e.g., interval sizes of 1, 10, or 100 million instructions were used in [3]). SimPoint 3.0 still supports fixed length intervals, but also provides support for Variable Length Intervals (VLI), which allows the intervals to account for different amounts of executed instructions as described in [16].

区间：程序连续执行的一个片段。所有区间都是非重叠的，所以要进行分析，我们将程序的执行分解成相邻的非重叠区间。之前版本的SimPoint需要所有的区间都是相同的大小，这是以区间内的指令数量来度量的（如，[3]中使用了一百万，一千万和一亿条指令的区间大小）。SimPoint 3.0仍然支持固定长度的区间，但也提供了对变长区间(VLI)的支持，允许区间包含不同数量的指令执行，如[16]所示。

- Phase - A set of intervals within a program’s execution with similar behavior. A phase can consist of intervals that are not temporally contiguous, so a phase can re-appear many times throughout execution.

状态：程序执行中，有类似行为的区间集合。一个状态所包含的区间，并不一定在时域上是连续的，所以一个状态可能在执行过程中重复出现很多次。

- Similarity - Similarity defines how close the behavior of two intervals are to one another as measured across some set of metrics. Well-formed phases should have intervals with similar behavior across various architecture metrics (e.g. IPC, cache misses, branch misprediction).

相似性：相似性定义了两个区间的行为互相接近的程度，这会用几种度量进行衡量。好的状态所包含的区间，应当在不同的架构度量下，都有类似的行为（如，IPC，cache misses，分支误预测率）。

- Frequency Vector - Each interval is represented by a frequency vector, which represents the program’s execution during that interval. The most commonly used frequency vector is the basic block vector [1], which represents how many times each basic block is executed in an interval. Frequency vectors can also be used to track other code structures [5] such as all branch edges, loops, procedures, registers, opcodes, data, or program working set behavior [17] as long as tracking usage of the structure provides a signature of the program’s behavior.

频率向量：每个区间都是由一个频率向量来表示的，这表示程序在这个区间的执行。最常使用的频率向量是bbv，表示了每个basic block在一个区间内执行的次数。频率向量也可以用于追踪其他的代码结构，比如所有的分支边缘，循环，过程，寄存器，opcodes，数据，或程序working set行为，只要跟踪结构的使用会给出程序行为的签名。

- Similarity Metric - Similarity between two intervals is calculated by taking the distance between the corresponding frequency vectors from the two intervals. SimPoint determines similarity by calculating the Euclidean distance between the two vectors.

相似度度量：两个区间之间的相似度，是通过计算两个区间的频率向量的距离得到的。SimPoint通过计算两个向量的欧氏距离，来得到其相似度。

- Phase Classification - Phase classification groups intervals into phases with similar behavior, based on a similarity metric. Phase classifications are specific to a program binary running a particular input (a binary/input pair).

状态分类：状态分类将区间根据类似行为分类成状态，这是基于相似度度量的。状态分类是对一个程序binary运行特定输入进行的(a binary/input pair)。

### 2.3 Similarity Metric - Distance Between Code Signatures

SimPoint represents intervals with frequency vectors. A frequency vector is a one dimensional array, where each element in the array tracks usage of some way to represent the program’s behavior. We focus on code structures, but a frequency vector can consist of any structure (e.g., data working sets, data stride access patterns [5, 17]) that may provide a signature of the program’s behavior. A frequency vector is collected from each interval. At the beginning of each interval we start with a frequency vector containing all zeros, and as the program executes, we update the current frequency vector as structures are used.

SimPoint用频率向量来表示区间。频率向量是一个一维阵列，其中阵列中的每个元素跟踪一些用途，来表示程序的行为。我们聚焦在代码结构中，但是一个频率向量可以包含任何结构（如，数据working sets，数据步长访问模式），只要能够提供程序行为的特征。从每个区间中都会收集得到一个频率向量。在每个区间的开始，我们以全0的频率向量开始，随着程序的执行，我们随着结构的使用更新频率向量。

A common frequency vector we have used is a list of static basic blocks [1] (called a Basic Block Vector (BBV)). If we are tracking basic block usage with frequency vectors, we count the number of times each basic block in the program has been entered in the current interval, and we record that count in the frequency vector, weighted by the number of instructions in the basic block. Each element in the frequency vector is a count of how many times the corresponding basic block has been entered in the corresponding interval of execution, multiplied by the number of instructions in that basic block.

我们曾经使用的一个常见的频率向量，是静态basic block的列表（称为basic block vector，bbv）。如果我们用频率向量来跟踪basic block的使用，我们对每个basic block在当前区间中进入的次数进行计数，将这个计数记录在频率向量中，并以这个basic block中的指令数进行加权。频率向量中的每个元素，是对应的basic block在对应的区间中进入执行的次数，乘以这个basic block中的指令数量。

We use basic block vectors (BBV) for the results in this paper. The intuition behind this is that the behavior of the program at a given time is directly related to the code executed during that interval [1]. We use the basic block vectors as signatures for each interval of execution: each vector tells us what portions of code are executed, and how frequently those portions of code are executed. By comparing the BBVs of two intervals, we can evaluate the similarity of the two intervals. If two intervals have similar BBVs, then the two intervals spend about the same amount of time in roughly the same code, and therefore we expect the behavior of those two intervals to be similar. Prior work showed that loop and procedure vectors can also be used, where each entry represents the number of times a loop or procedure was executed, performs comparably to basic block vectors [5], while using fewer dimensions.

我们在本文中使用BBV。这背后的直觉是，程序在给定时间内的行为，与在那个区间内执行的代码是直接相关的。我们使用bbv作为每个区间的特征：每个向量都代表了执行了哪些代码片段，以及这些代码片段执行的频繁程度。通过比较两个区间的BBVs，我们可以评估两个区间的相似度。如果两个区间有类似的BBVs，那么这两个区间在大致相同的代码上花费了大约相同的时间，因此我们期望这两个区间的行为是类似的。之前的工作表明，也可以使用循环向量和过程向量，这里每个入口表示一个循环或过程执行的次数，与bbv的性能类似，而使用的维度更少。

To compare two frequency vectors, SimPoint 3.0 uses the Euclidean distance, which has been shown to be effective for off-line phase analysis [2, 3]. The Euclidean distance is calculated by viewing each vector as a point in D-dimensional space, and calculating the straight-line distance between the two points.

为比较两个频率向量，SimPoint 3.0使用的是欧式距离，这对于离线状态分析[2,3]是有效的。欧式距离的计算，是将每个向量用作D维空间中的一个点，计算两个点之间的直线距离。

### 2.4 Using k-Means for Phase Classification

Clustering divides a set of points into groups, or clusters, such that points within each cluster are similar to one another (by some metric, usually distance), and points in different clusters are different from one another. The k-means algorithm [18] is an efficient and well-known clustering algorithm which we use to quickly and accurately split program behavior into phases. The k in k-means refers to the number of clusters (phases) the algorithm will search for.

聚类将点集分成组，或聚类，这样每个聚类中的点，都互相类似（按照某种度量，通常是距离），不同聚类中的点互不相同。k-means算法是一种高效的、著名的聚类算法，我们使用这种算法将程序行为迅速准确的分类成状态。k-means中的k，指的是聚类（状态）的数量。

The following steps summarize the phase clustering algorithm at a high level. We refer the interested reader to [2] for a more detailed description of each step.

下面的步骤，在很高的层次上总结了状态聚类算法。有兴趣的读者可以参考[2]得到每个步骤更详细的描述。

1. Profile the program by dividing the program’s execution into contiguous intervals, and record a frequency vector for each interval. Each frequency vector is normalized so that the sum of all the elements equals 1. 将程序执行分成连续的区间，对每个区间记录一个频率向量。每个频率向量都进行归一化，即所有元素的和为1。

2. Reduce the dimensionality of the frequency vector data to a smaller number of dimensions using random linear projection. 使用随机线性投影，将频率向量数据的维度降低到较小的维度。

3. Run the k-means clustering algorithm on the reduced-dimension data for a set of k values. 用数个k值，在降维数据上运行k-means聚类算法。

4. Choose from among these different clusterings a well-formed clustering that also has a small number of clusters. To compare and evaluate the different clusters formed for different values of k, we use the Bayesian Information Criterion (BIC) [19] as a measure of the “goodness of fit” of a clustering to a dataset. We choose the clustering with the smallest k, such that its BIC score is close to the best score that has been seen. Here “close” means it is above some percentage of the range of scores that have been seen. The chosen clustering represents our final grouping of intervals into phases.

从这些不同的聚类中，找到一个好的聚类，同时聚类数量要较小。为比较评估不同k值形成的聚类，我们使用BIC作为一种聚类作为一个数据集的拟合程度的度量。我们选择的聚类其k值最小，而且其BIC分数要与最好的分数要接近。这里接近意思是高于最高分数的一定百分比。选择的聚类代表我们将区间分组成状态的最终分类结果。

5. The final step is to select the simulation points for the chosen clustering. For each cluster (phase), we choose one representative interval that will be simulated in detail to represent the behavior of the whole cluster. By simulating only one representative interval per phase we can extrapolate and capture the behavior of the entire program. To choose a representative, SimPoint picks the interval in each cluster that is closest to the centroid (center) of each cluster. Each simulation point also has an associated weight, which reflects the fraction of executed instructions that cluster represents.

最后的步骤，是对被选的聚类选择仿真点。对每个聚类（状态），我们选择一个有代表性的区间，对其进行详细仿真，就代表了整个聚类的行为。每个状态只仿真一个有代表性的区间，我们可以外插并捕获整个程序的行为。为选择一个代表，SimPoint选择每个聚类中与聚类中心最接近的区间。每个仿真点都有相关的权重，这反应的是这个聚类代表的执行的指令部分。

6. With the weights and the detailed simulation results of each simulation point, we compute a weighted average for the architecture metric of interest (CPI, miss rate, etc.). This weighted average of the simulation points gives an accurate representation of the complete execution of the program/input pair.

有了每个仿真点的详细仿真结果和权重，我们对感兴趣的架构度量计算一个加权平均(CPI, miss rate, etc)。这种仿真点的加权平均，可以准确的代表program/input对的完整执行。

## 3. Methodology

We performed our analysis for the complete set of SPEC2000 programs for multiple inputs using the Alpha binaries on the SimpleScalar website. We collect all of the frequency vector profiles (basic block vectors) using SimpleScalar [20]. To generate our baseline fixed length interval results, all programs were executed from start to completion using SimpleScalar. The baseline microarchitecture model is detailed in Table 1.

我们进行分析的对象是SPEC2000程序的完整集合，有多个输入，使用的是SimpleScalar网站上的Alpha binaries。我们使用SimpleScalar收集了所有的频率向量profile (basic block vectors)。为生成我们的基准固定长度区间结果，使用SimpleScalar从头到尾运行所有程序。基准微架构模型如表1所示。

To examine the accuracy of our approach we provide results in terms of CPI error and k-means variance. CPI error is the percent error in CPI between using simulation points from SimPoint and the baseline CPI of the complete execution of the program.

为检查我们方法的准确率，我们给出了CPI误差和k-means方差的结果。CPI误差是使用SimPoint得到的仿真点，和完整运行程序的基准CPI之间的误差百分比。

The k-means variance is the average squared distance between every vector and its closest center. Lower variances are better. When sub-sampling, we still report the variance based on every vector (not just the sub-sampled ones). The relative k-means variance reported in the experiments is measured on a per-input basis as the ratio of the k-means variance observed for clustering on a sample to the k-means variance observed for clustering on the whole input.

k-means方差是每个向量及其最近的中心点的距离均方。方差越小越好。当下采样时，我们仍然汇报基于每个向量的方差（并不仅仅是下采样的那些）。实验中给出的相对k-means方差，是基于每个输入进行度量的，是作为一个样本的聚类上观察到的k-means方差，到作为整个输入上的聚类的k-means方差的比率。

## 4. SimPoint 3.0 Features

In this section we describe and analyze the SimPoint features that affect the running time of the SimPoint algorithm, and the resulting simulation time and accuracy of the simulation points.

本节中，我们描述和分析了影响SimPoint算法的运行时间的SimPoint特征，得到的仿真时间，和仿真点的准确率。

### 4.1 Choosing an Interval Size

When using SimPoint one of the first decisions to make is the interval size. The interval size along with the number of simulation points chosen by SimPoint will determine the simulation time of a binary/input combination. Larger intervals allow more aggregation of profile information, allowing SimPoint to search for large scale repeating behavior. In comparison, smaller intervals allow for more fine-grained representations and searching for smaller scale repeating behavior.

当使用SimPoint时，要做的第一个决定就是区间的大小。区间大小，以及SimPoint选择的仿真点数量，会决定一个binary/input组合的仿真时间。更大的区间会聚积更多的profile信息，使SimPoint可以搜索大尺度的重复行为。比较起来，更小的区间允许的是更细粒度的表示，可以搜索更小尺度的重复行为。

The interval size affects the number of simulation points; with smaller intervals more simulation points are needed than when using larger intervals to represent the same proportion of a program. We showed that using smaller interval sizes (1 million or 10 million) results in high accuracy with reasonable simulation limits [3]. The disadvantage is that with smaller interval sizes warmup becomes more of an issue, but there are efficient techniques to address warmup as discussed in [21, 22]. In comparison, warmup is not really an issue with larger interval sizes, and this may be preferred for some simulation environments [23]. For all of the results in this paper we use an interval size of 10 million instructions.

区间大小影响的是仿真点的数量；区间更小，就需要更多的仿真点；当使用更大的区间来表示一个程序的相同部分时，就需要更少的仿真点。我们展示了，使用更小的区间大小（一百万，或一千万），会在合理的仿真限制下，得到高准确率。劣势是，更小的区间大小，预热成为了很大的问题，但是有有效的技术来处理预热的问题，如[21,22]所示。比较起来，预热在更大的区间大小下，就不是一个真正的问题，有些仿真环境会倾向于这个选择。对于本文中的所有结果，我们使用的区间大小都是一千万指令。

#### 4.1.1 Support for Variable Length Intervals

Ideally we should align interval boundaries with the code structure of a program. In [24], we examine an algorithm to produce variable length intervals aligned with the procedure call, return and loop transition boundaries found in code. A Variable Length Interval (VLI) is represented by a frequency vector as before, but each interval’s frequency vector can account for different amounts of the program’s execution.

理想情况下，我们应当将区间边界与程序的代码结构对齐。在[24]中，我们试验了一种算法，产生的是变长区间，与代码中的过程调用、返回和循环迁移的边界对齐。一个变长区间(VLI)也是由频率向量表示的，但是每个区间的频率向量会代表程序执行的不同数量部分。

To be able to pick simulation points with these VLIs, we need to change the way we do our SimPoint clustering to include the different weights for these intervals. SimPoint 3.0 supports VLIs, and all of the detailed changes are described in [16]. At a high level the changes focused around the following three parts of the SimPoint algorithm:

为能够用这些VLIs选择仿真点，我们需要改变我们用SimPoint聚类的方式，以对这些区间包含不同的权重。SimPoint 3.0支持VLIs，所有详细的变化请见[16]。在高层次上，这些变化是围绕SimPoint的下面三部分的：

- Computing k-means cluster centers – With variable length intervals, we want the kmeans cluster centers to represent the centroid of the intervals in the cluster, based on the weights of each interval. Thus k-means must include the interval weights when calculating the cluster’s center. This is an important modification to allow k-means to better model those intervals that represent a larger proportion of the program.

计算k-means聚类中心：在变长区间下，我们希望k-means聚类中心表示的是聚类中的区间中心，基于每个区间的权重。因此，k-means在计算聚类中心时，必须包含区间的权重。这是一个重要的修改，可以允许k-means更好的建模这些区间，表示程序的更大的部分。

- Choosing the Best Clustering with the BIC – The BIC criterion is the log-likelihood of the clustering of the data, minus a complexity penalty. The likelihood calculation sums a contribution from each interval, so larger intervals should have greater influence, and we modify the calculation to include the weights of the intervals. This modification does not change the BIC calculated for fixed-length intervals.

用BIC选择最好的聚类：BIC准则是数据聚类的log似然，减去复杂度惩罚。似然计算从每个区间中加上一个贡献，所以更大的区间应当有更大的影响，我们将计算进行修改，加入区间的权重。这个修改不会改变对于固定长度的区间计算的BIC。

- Computing cluster centers for choosing the simulation points – Similar to the above, the centroids should be weighted by how much execution each interval in the cluster accounts for.

计算聚类中心以选择仿真点：与上面的类似，中心的加权应该是每个区间的执行量在整个聚类中占多少。

When using VLIs, the format of the frequency vector files is the same as before. A user can either allow SimPoint to determine the weight of each interval or specify the weights themselves (see the options in the Appendix). If the user allows SimPoint to determine the weights automatically, SimPoint will assign the weights from the frequency vector counts. For example, if one frequency vector summed to 100, and another summed to 200, then the second would have twice as much weight as the first. However, the default behavior of SimPoint is to assume fixed-length vectors, and give all vectors equal weight.

当使用VLIs时，频率向量文件的格式与之前一样。用户可以允许SimPoint来确定每个区间的权重，或自己指定权重（见附录中的选项）。如果用户允许SimPoint来自动确定权重，SimPoint会从频率向量计数中指定权重。比如，如果一个频率向量总和等于100，另一个总和等于200，那么第二个的权重是第一个的2倍。但是，SimPoint默认的行为是假设固定长度的向量，给所有的向量相等的权重。

### 4.2 Methods for Reducing the Run-Time of K-Means

Even though SimPoint only needs to be run once per binary/input combination, we still want a fast clustering algorithm that produces accurate simulation points. To address the run-time of SimPoint, we first look at three options that can greatly affect the running time of a single run of k-means. The three options are the number of intervals to cluster, the size (dimension) of the intervals being clustered, and the number of iterations it takes to perform a clustering.

即使SimPoint只需要对每个binary/input组合运行一次，我们仍然希望有一个快速的聚类算法，可以生成精确的仿真点。为处理SimPoint的运行时间，我们首先三个可以极大影响k-means一次运行的运行时间的选项。这三个选项是，要聚类的区间数量，要聚类的区间的大小（维度），进行一次聚类所需要的迭代次数。

To start we first examine how the number of intervals affects the running time of the SimPoint algorithm. Figure 1 shows the time in seconds for running SimPoint varying the number of intervals (vectors) as we vary the number of clusters (value of k). For this experiment, the interval vectors are randomly generated from uniformly random noise in 15 dimensions.

我们首先检查区间数量如何影响SimPoint算法的运行时间。图1展示了运行SimPoint的时间（以秒为计），变化的是区间的数量（向量），变化的还有聚类的数量（k的值）。对于这个试验，区间向量是从15维的均匀随机噪声中随机生成的。

The results show that as the number of vectors and clusters increases, so does the amount of time required to cluster the data. The first graphs show that for 100,000 vectors and k = 128, it took about 3.5 minutes for SimPoint 3.0 to perform the clustering. It is clear that the number of vectors clustered and the value of k both have a large effect on the run-time of SimPoint. The run-time changes linearly with the number of clusters and the number of vectors. Also, we can see that dividing the time by the multiplication of the number of iterations, clusters, and vectors to provide the time per basic operation continues to give improving performance for larger k.

结果表明，随着向量的数量和聚类增加，聚类数据需要的时间也在增加。第一张图展示了，对于10万个向量和k=128，SimPoint 3.0需要3.5分钟来进行聚类。很明显，要聚类的向量数量，和k值，都对SimPoint的运行时间有很大的影响。运行时间随着聚类数量和向量数量线性变化。同时，我们可以看到，将运行时间除以迭代次数，聚类数量和向量数量的乘积，以得到每个基础运算所需的时间，更大的k会得到改进的性能。

#### 4.2.1 Number of Intervals and Sub-sampling

The k-means algorithm is fast: each iteration has run-time that is linear in the number of clusters, and the dimensionality. However, since k-means is an iterative algorithm, many iterations may be required to reach convergence. We already found in prior work [2], and revisit in Section 4.2.2 that we can reduce the number of dimensions down to 15 and still maintain the SimPoint’s clustering accuracy. Therefore, the main influence on execution time for SimPoint is the number of intervals.

k-means算法是很快的：每次迭代的运行时间与聚类数量和维度是线性的关系。但是，由于k-means是一种迭代算法，会需要很多次迭代才能达到收敛。我们已经在之前的工作发现，在4.2.2中会重温，我们可以维度降低到15，仍然保持SimPoint的聚类准确率。因此，SimPoint在执行时间上的主要影响是区间数量。

To show this effect, Table 2 shows the SimPoint running time for gcc-166 and crafty-ref, which are at the lower and upper ranges for the number of intervals and basic block vectors seen in SPEC 2000 with an interval size of 10 million instructions. The second and third column shows the number of intervals (vectors) and the original number of dimensions for each vector (these are projected down to 15 dimensions when performing the clustering). The last three columns show the time it took to execute SimPoint searching for the best clustering from k=1 to 100, with 5 random initializations (seeds) per k. SP2 is the time it took for SimPoint 2.0. The second to last column shows the time it took to run SimPoint 3.0 when searching over all k in the same manner as SimPoint 2.0, and the last column shows the clustering time when using our new binary search described in Section 4.4.3. The results show that increasing the number of intervals by 4 times increased the running time of SimPoint around 10 times. The results show that we significantly reduced the running time for SimPoint 3.0, and that combined with the new binary search functionality results in 10x to 50x faster choosing of simulation points over SimPoint 2.0. The results also show that the number of intervals clustered has a large impact on the running time of SimPoint, since it can take many iterations to converge, which is the case for crafty.

为展示这种效果，表2展示了SimPoint对于gcc-166和crafty-ref的运行时间，这在使用一千万指令的区间大小的SPEC2000中，对于所有的区间数量和bbv，是在较低和较高的范围中的。第2列和第3列展示了区间的数量（向量的数量），和每个向量的原始维度（在进行聚类时，这些投影到15维）。最后3列展示了SimPoint搜索得到最佳聚类的时间，k=1到100，每个k有5个随机初始化（种子）。SP2是SimPoint 2.0所耗费的时间。倒数第2列是SimPoint 3.0搜索所有k所耗费的时间，方式与SimPoint 2.0一样；最后一列展示了使用4.4.3中的binary搜索进行聚类所耗费的时间。结果表明，区间数量增加4倍，SimPoint的运行时间增加约10倍。结果表明，我们显著降低了SimPoint 3.0的运行时间，与新的binary搜索功能结合到一起，比SimPoint 2.0要快10x-50x。结果还表明，聚类的区间数量，对SimPoint的运行时间有很大影响，因为需要耗费很多次迭代来收敛，对于crafty就是这个情况。

The effect of the number of intervals on the running time of SimPoint becomes critical when using very small interval sizes like 1 million instructions or smaller, where there can be millions of intervals to cluster. To speed the execution of SimPoint on these very large inputs, we sub-sample the set of intervals that will be clustered, and run k-means on only this sample. We sample the vector dataset using weighted sampling for VLIs, and uniform sampling for fixed-length vectors. The number of desired intervals is specified, and then SimPoint chooses that many intervals (without replacement). The probability of each interval being chosen is proportional to the weight of its interval (the number of dynamically executed instructions it represents).

区间数量对SimPoint的运行时间的效果，在使用非常小的区间大小，如一百万条指令或更小的时候，会变得非常严重，因为需要聚类的区间数量会有上百万。为加速SimPoint在这些非常大数量输入上的运行，我们对这些要聚类的区间进行下采样，只在这些下采样的样本上运行k-means。我们对VLIs使用加权采样，对定长向量采用均匀采样，来采样这个向量数据集。期望区间的数量是指定的，然后SimPoint选择这些区间（没有替代）。每个区间被选择的概率，与其区间的权重（其代表的动态执行的指令数量）成正比。

Sampling is common in clustering for datasets which are too large to fit in main memory [25, 26]. After clustering the dataset sample, we have a set of clusters with centroids. We then make a single pass through the unclustered intervals and assign each to the cluster that has the nearest center (centroid) to that interval. This then represents the final clustering from which the simulation points are chosen. We originally examined using sub-sampling for variable length intervals in [16]. When using VLIs we had millions of intervals, and had to sub-sample 10,000 to 100,000 intervals for the clustering to achieve a reasonable running time for SimPoint, while still providing very accurate simulation points.

对数据集的聚类，如果数据集太大，无法载入到主存中，那么采样就是很常见的手段。在聚类数据集样本后，我们就有了聚类集合，带有聚类中心。我们将未聚类的区间跑一遍，将每个区间指定给与区间中心距离最近的聚类。这就代表了最终的聚类结果，仿真点就从这里进行选择。我们在[16]中对变长区间使用下采样进行了试验。当使用VLIs时，我们有上百万个区间，要下采样到10000到100000个区间进行聚类，以得到合理的运行时间，同时仍然可以给出很精确的仿真点。

The experiments shown in Figure 2 show the effects of sub-sampling across all the SPEC 2000 benchmarks using 10 million interval size, 30 clusters, 15 projected dimensions, and sub-sampling sizes that used 1/8, 1/4, 1/2, and all of the vectors in each program. The first two plots show the effects of sub-sampling on the CPI errors and k-means variance, both of which degrade gracefully when smaller samples are used. The average SPEC INT and SPEC FP results are shown.

图2中所示的试验表明，在所有SPEC 2000基准测试中，使用一千万区间大小，30个聚类，投影维度15，下采样大小为1/8, 1/4, 1/2和每个程序中的所有向量，进行下采样的效果。前两个图表明，下采样对CPI误差和k-means方差的效果，在使用更少的样本时，这两者的下降都很小。图中给出了平均的SPEC INT和SPEC FP的结果。

As shown in the second graph of Figure 2, sub-sampling a program can result in k-means finding a slightly less representative clustering, which results in higher k-means variance and higher CPI errors, on average. Even so, when sub-sampling, we found in some cases that it can reduce the k-means variance and/or CPI error (compared to using all the vectors), because sub-sampling can remove unimportant outliers in the dataset that k-means may be trying to fit. It is interesting to note the difference between floating point and integer programs, as shown in the first two plots. It is not surprising that it is easier to achieve lower CPI errors on floating point programs than on integer programs, as the first plot indicates. In addition, the second plot suggests that floating point programs are also easier to cluster, as we can do quite well even with only small samples. The third plot shows the effect of the number of vectors on the running time of SimPoint. This plot shows the time required to cluster the full run of all of the benchmark/input combinations and the three (1/8, 1/4 and 1/2) sub-sampled runs. In addition, we have fit a logarithmic curve with least-squares to the points to give a rough idea of the growth of the run-time. The main variance in time, when two different datasets with the same number of vectors are clustered, is due to the number of k-means iterations required for the clustering to converge.

如图2的第二幅图所示，下采样一个程序会导致，k-means会找到略微没那么有代表性的聚类，平均会导致更高的k-means方差和更高的CPI误差。即使这样，当下采样的时候，我们发现在一些情况中，可以降低k-means方差和/或CPI误差（与使用所有向量相比），因为下采样会去掉数据集中不重要的outliers，k-means会试图拟合这些outliers。要指出的有趣之处是，浮点和整形程序之间的差异，在前两幅图中可以看到。并不意外的是，与整型程序相比，在浮点程序上更容易获得更低的CPI误差，如第一幅图所示。此外，第二幅图说明，浮点程序更容易进行聚类，因为我们只用很少的样本就可以做的很好。第3幅图展示了向量的数量对SimPoint运行时间的影响。这幅图说明运行完整的基准测试/输入的组合所需的时间，和3种下采样(1/8, 1/4 and 1/2)的运行的时间。此外，我们将这些点拟合了一个logarithmic曲线，给出了运行时间增长的大致概念。当聚类同样数量向量的两个不同的数据集，时间上的不同主要是由于需要的k-means迭代次数以得到收敛的聚类。

#### 4.2.2 Number of Dimensions and Random Projection

Along with the number of vectors, the other most important aspect in the running time of k-means is the number of dimensions used. In [2] we chose to use random linear projection to reduce the dimension of the clustered data for SimPoint, which dramatically reduces computational requirements while retaining the essential similarity information. SimPoint allows the user to define the number of dimensions to project down to. We have found that SimPoint’s default of 15 dimensions is adequate for SPEC 2000 applications as shown in [2]. In that earlier work we looked at how much information or structure of frequency vector data is preserved when projecting it down to varying dimensions. We did this by observing how many clusters were present in the low-dimensional version. We noted that at 15 dimensions, we were able to find most of the structure present in the data, but going to even lower dimensions removed too much structure.

除了向量数量，运行k-means时间的其他重要因素是，使用的维度数量。在[2]中，我们使用随机线性投影，来给SimPoint降低聚类数据的维度，这极大的降低了计算需求，同时保持了基础的相似性信息。SimPoint允许用户定义投影到的维数。我们发现，SimPoint默认的15维对于[2]中的SPEC 2000应用是足够的。在这之前的工作中，我们将频率向量数据投影到不同的维度中，查看信息或结构保留的情况。我们通过观察在低维版本中，还存在多少聚类，还进行这个试验。我们注意到，在15维的情况下，我们可以找到数据中存在的多数结构，但更低的维度，会去掉太多的结构。

To examine random projection, Figure 3 shows the effect of changing the number of projected dimensions on both the CPI error (left) and the run-time of SimPoint (right). For this experiment, we varied the number of projected dimensions from 1 to 100. As the number of dimensions increases, the time to cluster the vectors increases linearly, which is expected. Note that the run-time also increases for very low dimensions, because the points are more “crowded” and as a result k-means requires more iterations to converge.

为检查随机投影的效果，图3展示了改变投影维度数值对CPI误差和SimPoint运行时间的效果。对于这个试验，我们变化了投影维度从1到100。随着维度增加，聚类向量所需要的时间线性增长，这是符合期望的。注意，对非常低的维度，运行时间也在增长，因为这些点太拥挤了，所以k-means需要更多的迭代来收敛。

It is expected that by using too few dimensions, not enough information is retained to accurately cluster the data. This is reflected by the fact that the CPI errors increase rapidly for very low dimensions. However, we can see that at 15 dimensions, the SimPoint default, the CPI error is quite low, and using a higher number of dimensions does not improve it significantly and requires more computation. Using too many dimensions is also a problem in light of the well-known “curse of dimensionality” [27], which implies that as the number of dimensions increase, the number of vectors that would be required to densely populate that space grows exponentially. This means that higher dimensionality makes it more likely that a clustering algorithm will converge to a poor solution. Therefore, it is wise to choose a dimension that is low enough to allow a tight clustering, but not so low that important information is lost.

使用过少的维度，保留的信息就不足了，不能准确的对数据进行聚类。在非常低的维度下，CPI误差上升的非常快，这也反应了这个事实。但是，我们可以看到，在15维时，也就是SimPoint默认的维度，CPI误差是很低的，使用更高的维度，误差并没有明显改进，而且需要更多的计算量。使用更多的维度也会导致“维度诅咒”的问题，这意思是说，随着维度的增加，需要在这个空间中密集populate的向量的数量增加太多。这意味着更高的维度很可能会导致聚类算法收敛到一个较差的解。因此，选择一个较低的维度，得到一个紧致的聚类，但不要太低，这会导致丢失重要的信息。

#### 4.2.3 Number of Iterations Needed

The final aspect we examine for affecting the running time of the k-means algorithm is the number of iterations it takes for a run to converge.

我们检查的最后的影响k-means算法的运行时间的方面，是收敛需要的迭代次数。

The k-means algorithm iterates either until it hits a user-specified maximum number of iterations, or until it reaches a point where no further improvement is possible, whichever is less. k-means is guaranteed to converge, and this is determined when the centroids no longer change. In SimPoint, the default limit is 100 iterations, but this can easily be changed (and can be turned off). More than 100 iterations may be required, especially if the number of intervals is very large compared to the number of clusters. The interaction between the number of intervals and the number of iterations required is the reason for the large SimPoint running time for crafty-ref in Table 2.

k-means算法是迭代进行的，直到达到了用户指定的最大迭代次数，或到了不能改进的程度再停止。k-means算法一定会收敛，在中心不再变化之后就收敛了。在SimPoint中，默认的限制是100次迭代，但这个改变起来很容易（而且可以关掉）。可能需要多于100次迭代，尤其是在区间数量比聚类数量大很多时。区间数量和需要的迭代次数之间的关系，是在表2中对于crafty-ref SimPoint需要很多运行时间的原因。

For our results, we observed that only 1.1% of all runs on all SPEC 2000 benchmarks reach the limit of 100 iterations. This experiment was with 10-million instruction intervals, k=30, 15 dimensions, and with 10 random (seeds) initializations (runs) of k-means. Figure 4 shows the number of iterations required for all runs in this experiment. Out of all of the SPEC program and input combinations run, only crafty-ref, gzip-program, perlbmk-splitmail had runs that had not converged by 100 iterations. The longest running clusterings for these programs reached convergence in 160, 126, and 101 iterations, respectively.

对于我们的结果，我们观察到，在所有SPEC 2000的基准测试中，只有1.1%达到了100次迭代的上限。这个试验的配置是，一千万指令区间，k=30，15维，k-means有10个随机初始化（种子）。图4展示了在试验中需要的所有运行的次数。在所有的SPEC程序和输入组合中，只有crafty-ref，gzip-program，perlbmk-splitmail的运行在100次的时候还没有收敛。这些程序分别在第160，126，101次迭代的时候，达到了收敛的次数。

### 4.3 MaxK and Controlling the Number of Simulation Points

The number of simulation points that SimPoint chooses has a direct effect on the simulation time that will be required for those points. The maximum number of clusters, MaxK, along with the interval size as discussed in Section 4.1, represents the maximum amount of simulation time that will be needed. When fixed length intervals are used, MaxK ∗ interval size puts a limit on the instructions simulated.

SimPoint选择的仿真点数量，对这些点需要的仿真时间有直接的影响。聚类的最大数量MaxK，和4.1节讨论的区间大小一起，表示需要的最大的仿真时间。当使用定长区间时，MaxK ∗ 区间大小对仿真的指令加上了一个限制。

SimPoint enables users to trade off simulation time with accuracy. Researchers in architecture tend to want to keep simulation time to below a fixed number of instructions (e.g., 300 million) for a run. If this is desirable, we find that an interval size of 10M with MaxK=30 provides very good accuracy (as we show in this paper) with reasonable simulation time (below 300 million and around 220 million instructions on average). If even more accuracy is desired, then decreasing the interval size to 1 million and setting MaxK=300 or MaxK equal to the square root of the total number of intervals: $\sqrt_n$ performs well. Empirically we discovered that as the granularity becomes finer, the number of phases discovered increases at a sub-linear rate. The upper bound defined by this heuristic works well for the SPEC 2000 benchmarks.

SimPoint使用户可以在仿真时间和准确率之间折中。架构研究者希望在一次运行中使仿真时间低于固定数量的指令（如，3亿条指令）。如果这是可取的，我们发现区间大小一千万，MaxK=30，会给出很好的准确率，仿真时间也比较合理（在3亿条以下，平均在2.2亿条指令）。如果希望更高的准确率，那么就降低区间大小到一百万，设MaxK=300，或MaxK等于区间总计数量的平方根：$\sqrt_n$ 效果较好。经验上来说，我们发现，随着粒度变得更精细，发现的状态数量会呈亚线性速率增长。这种直觉定义的上限，对SPEC 2000基准测试的效果是良好的。

Finally, if the only thing that matters to a user is accuracy, then if SimPoint chooses a number of clusters that is close to the maximum allowed, then it is possible that the maximum is too small to capture all of the unique behaviors. If this is the case and more simulation time is acceptable, it is better to double the maximum k and re-run the SimPoint analysis.

最后，如果用户关心的唯一的就是准确率，那么如果SimPoint选择的聚类数量接近于最大值，那么有可能最大值也很小，不能捕获所有唯一的行为。如果是这种情况，而且可以接受更多的仿真时间，那么可以将最大值k加倍，重新运行SimPoint分析。

#### 4.3.1 Choosing Simulation Points to Represent the Top Percent of Execution

One advantage to using SimPoint analysis is that each simulation point has an associated weight, which tells how much of the original program’s execution is represented by the cluster that simulation point represents. The simulation points can then be ranked in order of importance. If simulation time is too costly, a user may not want to simulate simulation points that have very small weights. SimPoint 3.0 allows the user to specify this explicitly with the -coveragePct p option. When this option is specified, the value of p sets a threshold for how much of the execution should be represented by the simulation points that are reported in an extra set of files for the simulation points and weights. The default is p = 1.0: that the entire execution should be represented.

使用SimPoint分析的一个优势是，每个仿真点都有相关联的权重，这包含了这个仿真点表示的这个聚类在程序原始执行中的占比。仿真点可以以重要性排序。如果仿真时间很昂贵，用户就可能不想去仿真那些权重很小的点。SimPoint 3.0允许用户用-coveragePct p选项来显式指定。当指定了这个选项，p值设置了一个阈值，即仿真点应当表示多少执行，才应当在额外的文件中表示其仿真点和权重。默认的是p=1.0：即应当表示完整的执行。

For example, if p = 0.98 and the user has specified -saveSimpoints and -saveWeights, then SimPoint will report simulation points and associated weights for all the non-empty clusters in two files, and also for the largest clusters which make up at least 98% of the program’s weight. Using this reduced-coverage set of simulation points can potentially save a lot of simulation time if there are many simulation points with very small weights without severely affecting the accuracy of the analysis.

比如，如果p=0.98，用户也指定了选项-saveSimpoints和-saveWeights，那么SimPoint会在两个文件中对所有非空聚类中给出仿真点和相关的权重，还有最大的聚类，组成了至少程序权重的98%。用这个降低覆盖的仿真点集合，可能会节约很多仿真时间，因为可能会有很多仿真点的权重很小，不对其进行仿真，不会严重影响分析的准确率。

Figure 5 shows the effect of varying the percentage of coverage that SimPoint reports. These experiments use binary search with MaxK=30, 15 dimensions, and 5 random seeds. The left graph shows the CPI error and the right shows the number of simulation points chosen when only representing the top 95%, 98%, 99% and 100% of execution. The three bars show the maximum value, the second highest value (max-1), and the average. The results show that when the coverage is reduced from 100%, the average number of simulation points decreases, which reduces the simulation time required, but this is at the expense of the CPI error, which goes up on average. For example, comparing 100% coverage to 95%, the average number of simulation points is reduced from about 22 to about 16, which is a reduction of about 36% in required simulation time for fixed-length vectors. At the same time, the average CPI error increases from 1.5% to 2.8%. Depending on the user’s goal, a practitioner can use these types of results to decide on the appropriate trade off between simulation time and accuracy. Out of all of the SPEC binary/input pairs there was one combination (represented by the maximum) that had a bad error rate for 95% and 98%. This was ammp-ref, and the reason was that a simulation point was removed that had a small weight (1-2% of the executed instructions) but its behavior was different enough to affect the estimated CPI.

图5展示了SimPoint的覆盖率不同的情况下的效果。这些试验使用binary搜索，MaxK=30，15维，5个随机种子。左图展示了CPI误差，右图展示了在只表示最高的95%，98%，99%和100%的执行时所选择的仿真点的情况。三个bars分别展示了最高值，第二高的值(max-1)，和平均值。结果表明，当覆盖率从100%的情况进行下降，仿真点的平均数量也下降了，需要的仿真时间也减少了，但代价是CPI误差在增长。比如，比较100%的覆盖和95%的覆盖率，仿真点平均数量从22减少到16，相对于固定长度向量的仿真时间，减少了36%。同时，平均CPI误差从1.5%增长到了2.8%。依赖于用户目标，实践者可以使用这种类型的结果，来决定仿真时间和准确率的合适折中。在所有的SPEC binary/input对中，有一个组合（表示为最大），其95%和98%的误差率比较差。这就是ammp-ref，原因是有一个权重很小的仿真点被去掉了(1-2%的执行指令)，但其行为非常不同，足以影响估计的CPI。

Note, when using simulation points for an architecture design space exploration, the CPI error compared to the baseline is not as important as making sure that this error is consistent between the different architectures being examined. What is important is that a consistent relative error is seen across the design space exploration, and SimPoint has this consistent bias as shown in [3]. Ignoring a few simulation points using -coveragePct p will create a consistent bias across the different architecture runs when compared to complete simulation. This is because a small fraction of behavior will be ignored during the design space exploration, but the same simulation points representing the top percent of execution will be represented. This can be acceptable technique for reducing simulation time, especially when performing large design space exploration trade-offs.

注意，当对架构设计空间探索使用仿真点时，与基准相比较的CPI误差，并没有误差在不同的架构之间是一致的相比更重要。重要的是，在设计空间探索时，可以观察到一致的相对误差，[3]中展示了，SimPoint有这种一致的bias。使用-coveragePct p忽略几个仿真点，与完整仿真进行比较时，在不同的架构运行时，会有一致的bias。这是因为，在设计空间探索时，会忽略行为的一小部分，但表示执行最高比例的相同仿真点会被表示。这在降低仿真时间上是可以接受的技术，尤其是进行大型设计空间探索的trade-offs。

### 4.4 Searching for the Smallest k with Good Clustering

As described above, we suggest setting MaxK as appropriate for the maximum amount of simulation time a user will tolerate for a given run. We then use three techniques to search over the possible values of k, which we describe now. The goal is to try to pick a k that reduces simulation time, but also provides an accurate picture of the program’s execution.

如同上面描述的，我们建议合理的设置MaxK，让一次运行的仿真时间达到用户所能忍受的最大值。然后我们使用三种技术来搜索可能的k值，我们现在会进行描述。目标是选择一个减少仿真时间的k值，同时还能给出程序执行的准确表示。

#### 4.4.1 Setting the BIC Percentage

As we examine several clusterings and values of k, we need to have a method for choosing the best clustering. The Bayesian Information Criterion (BIC) [19] gives a score of the goodness of the clustering of a set of data. These BIC scores can then be used to compare different clusterings of the same data. The BIC score is a penalized likelihood of the clustering of the vectors, and can be considered the approximation of a probability. However, the BIC score often increases as the number of clusters increase. Thus choosing the clustering with the highest BIC score can lead to often selecting the clustering with the most clusters. Therefore, we look at the range of BIC scores, and select the score which attains some high percentage of this range. The SimPoint default BIC threshold is 0.9. When the BIC rises and then levels off, this method chooses a clustering with the fewest clusters that is near the maximum value. Choosing a lower BIC percent would prefer fewer clusters, but at the risk of less accurate simulation.

我们研究了k的几种取值及聚类，我们需要一种方法来选择最佳聚类。BIC给出了数据聚类质量好坏程度的分数。这些BIC分数可以用于比较相同数据的不同聚类。BIC分数是向量聚类的惩罚似然，可以认为是概率的近似。但是，BIC分数通常随着聚类数量的增加而增加。因此，选择BIC分数最高的聚类，通常会导致选出数量最多的聚类。因此，我们查看BIC分数的范围，选择在这个范围内获得一定很高百分比的分数。SimPoint默认的BIC阈值为0.9，当BIC升高然后保持稳定，该方法选择的分数会获得这个范围内较高百分比的值，同时聚类数最少。选择较低的BIC百分比会得到更少的聚类数，但风险是仿真的精确率会很低。

Figure 6 shows the effect of changing the BIC threshold on both the CPI error (left) and the number of simulation points chosen (right). These experiments are for using binary search with MaxK=30, 15 dimensions, and 5 random seeds. BIC thresholds of 70%, 80%, 90% and 100% are examined. As the BIC threshold decreases, the average number of simulation points decreases, and similarly the average CPI error increases. At the 70% BIC threshold, perlbmk-splitmail has the maximum CPI error in the SPEC suite. This is due to a clustering that was picked at that threshold which has only 9 clusters. This anomaly is an artifact of the looser threshold, and better BIC scores point to better clusterings and better error rates, which is why we recommend the BIC threshold to be set at 90%.

图6展示了改变BIC阈值对CPI误差（左）和仿真点选择数量（右）的效果。这些试验中使用的是binary search，MaxK=30，15维，5个随机种子。BIC阈值试验了70%，80%，90%和100%四个值。随着BIC阈值降低，仿真点的平均数量也在降低，类似的，平均CPI误差在增加。在70%的BIC阈值上，perlbmk-splitmail在SPEC包中有最大的CPI误差。这是因为在这个阈值上选择的聚类只有9个聚类。这种异常是因为阈值太低导致的，更好的BIC分数会得到更好的聚类，和更好的误差率，所以我们推荐的BIC阈值为90%。

#### 4.4.2 Varying the Number of Random Seeds, and k-means initialization

The k-means clustering algorithm is essentially a hill-climbing algorithm, which starts from a randomized initialization, which requires a random seed. Because of this, running k-means multiple times can produce very different results depending on the initializations. Sometimes this means k-means can converge to a locally-good solution that is poor compared to the best clustering on the same data for that number of clusters. Therefore the conventional suggests that it is good to run k-means several times using a different randomized starting point each time, and take the best clustering observed, based on the k-means variance or the BIC. SimPoint has the functionality to do this, using different random seeds to initialize k-means each time. Based on our experience, we have found that using 5 random seeds works well.

k-means聚类算法实际上是一个爬坡算法，从随机初始化开始，这需要一个随机种子。因此，多次运行k-means会由于不同的初始化而产生非常不同的结果。有时候，这意味着k-means会收敛到一个局部很好的解，与在相同的数据上，对这个数量的聚类来说，相对于最佳聚类结果，这可能是一个不太好的结果。因此，传统上一般推荐运行k-means多次，每次使用不同的随机初始点，基于k-means方差或BIC，取观察到的最佳聚类。SimPoint具有这种功能，每次使用不同的随机种子来初始化k-means。基于我们的经验，我们发现使用5个随机种子效果良好。

Figure 7 shows the effect on CPI error of using two different k-means initialization methods (furthest-first and sampling) along with different numbers of initial k-means seeds. These experiments are for using binary search with MaxK=30, 15 dimensions, and a BIC threshold of .9. When multiple seeds are used, SimPoint runs k-means multiple times with different starting conditions and takes the best result.

图7展示了使用两种不同的k-means初始化方法，与不同数量的初始k-means种子，对CPI误差的影响。这些试验使用的是binary search，MaxK=30，15维，BIC阈值为.9。当使用了多个种子时，SimPoint用不同的初始条件多次运行k-means，取最好的结果。

Based on these results we see that sampling outperforms furthest-first k-means initialization. This can be attributed to the data we are clustering, which has a large number of anomaly points. The furthest-first method is likely to pick those anomaly points as initial centers since they are the furthest points apart. The sampling method randomly picks points, which on average does better than the furthest-first method. It is also important to try multiple seed initializations in order to avoid a locally minimal solution. The results in Figure 7 shows that 5 seed initializations is sufficient for finding a good clustering, but using 10 seeds did reduce the maximum error seen from 8% down to 5.5%.

基于这些结果，我们看到采样的效果超过了furthest-first k-means初始化。这可以归因为聚类的数据，有很多异常点。Furthest-first方法很可能选择这些异常点作为初始中心，因为它们是距离最远的点。采样方法随机选取点，平均来说，表现比furthest-first要好。尝试多个种子初始化方法，以避免局部最小解，这也是很重要的。图7中的结果说明，5个种子的初始化，足以找到一个好的聚类，但使用10个种子也确实会将最大误差从8%降低到5.5%。

#### 4.4.3 Binary Search for Picking k

SimPoint 3.0 makes it much faster to find the best clustering and simulation points for a program trace over earlier versions. Since the BIC score generally increases as k increases, SimPoint 3.0 uses this to perform a binary search for the best k. For example, if the maximum k desired is 100, with earlier versions of SimPoint one might search in increments of 5: k = 5, 10, 15,..., 90, 100, requiring 20 clusterings. With the binary search method, we can ignore large parts of the set of possible k values and examine only about 7 clusterings.

SimPoint 3.0与之前的版本相比，可以更快的对一个程序trace找到最佳的聚类和仿真点。因为BIC分数一般随着k增加而增加，SimPoint 3.0用此进行binary search以找到最佳的k值。比如，如果最大k期望是100，对于SimPoint早期的版本，会以递增的形式进行搜索：k = 5, 10, 15,..., 90, 100, 需要20种聚类。在binary search方法下，我们可以忽略大部分可能的k值，只检查大约7种聚类。

The binary search method first clusters 3 times: at k = 1, k = max k, and k = (max k + 1)/2. It then proceeds to divide the search space and cluster again based on the BIC scores observed for each clustering. The binary search may stop early if the window of k values is relatively small compared to the maximum k value. Thus the binary search method requires the user only to specify the maximum k value, and performs at most log(max k) clusterings.

Binary search方法首先聚类3次：k=1， k=max k，和k=(max k+1)/2，然后持续进行，以分裂搜索空间，基于对每个聚类观察到的BIC分数进行聚类。Binary search在k值的窗口与最大k值相比很小的时候，可能会提前停止。因此，binary search方法需要用户指定最大k值，并最多进行log(max k)次聚类。

Figure 8 shows the comparison between the new binary search method for choosing the best clustering, and searching all k values (as was done in SimPoint 2.0). The top graph shows the CPI error for each program, and the bottom graph shows the number of simulation points (clusters) chosen. These experiments are for using binary search with MaxK=30, 15 dimensions, 5 random seeds, and a BIC threshold of .9. SimPoint All performs slightly better than the binary search method, since it searches exhaustively through all k values for MaxK=30. Using the binary search, it is possible that it will not choose as small of clustering as the exhaustive search. This is shown in the bottom graph of Figure 8, where the exhaustive search picked 19 simulation points on average, and binary search chose 22 simulation points on average. In terms of CPI error rates, the average is about the same across the SPEC programs between exhaustive and binary search.

图8展示了新的binary search和搜索所有k值（在SimPoint 2.0里是这样做的），在选择最佳聚类时的比较。上图展示了每个程序的CPI误差，下图展示了选择的仿真点（聚类）的数量。这些试验使用的是binary search，MaxK=30，15维，5个随机种子，BIC阈值为0.9。SimPoint All的效果比binary search略微好一些，因为它对k值进行的是穷举式搜索，MaxK=30。使用binary search，其选择的聚类与穷举式搜索相比，不会那么小。这在图8的下图中进行了展示，其中穷举式搜索平均选择了19个仿真点，而binary search平均选择了22个仿真点。以CPI误差论，穷举式搜索和binary search在SPEC程序上的平均表现类似。

## 5. Common Pitfalls

There are a few important potential pitfalls worth addressing to ensure accurate use of SimPoint’s simulation points.

有几个潜在的困难值得要说一下，以确保准确的使用SimPoint的仿真点。

**Setting MaxK Appropriately** – MaxK must be set based on the interval size used and the maximum number of instructions you are willing to simulate as described in Section 4.3.

The maximum number of clusters and the interval size represent the maximum amount of simulation time needed for the simulation points selected by SimPoint. Finding good simulation points with SimPoint requires recognizing the tradeoff between accuracy and simulation time. If a user wants to place a low limit on the number of clusters to limit simulation time, SimPoint can still provide accurate results, but some intervals with differing behaviors may be grouped together as a result. In such cases it may be advantageous to increase M axK and with that use the option -coveragePct with a value less than 1 (e.g. .98). This can allow different behaviors to be grouped into more clusters, but the final set of simulation points can be smaller since only the most dominant behaviors will be chosen for simulation points.

**Off by One Interval Errors** – SimPoint 3.0 starts counting intervals and cluster IDs at 0. These are the counts and IDs written to a file by -saveSimpoints, where SimPoint indicates which intervals have been selected as simulation points and their respective cluster IDs. A common mistake may be to assume that SimPoint 3.0, like previous versions of SimPoint, counts intervals starting from 1, instead of 0. Just remember that the first interval of execution and the first cluster in SimPoint 3.0 are both numbered 0.

**Reproducible Tracking of Intervals and Using Simulation Points** – It is very important to have a reproducible simulation environment for (a) creating interval vectors, and (b) using the simulation points during simulation. If the instruction counts are not stable between runs, then selection of intervals can be skewed, resulting in additional error.

SimPoint provides the interval number for each simulation point. Interval numbers are zero-based, and are relative to the start of execution, not to the previous simulation point. So for fixed-length intervals, to get the instruction count at the start of a simulation point, just multiply the interval number by the interval size, but watch out for Interval Drift described later. For example, interval number 15 with an interval size of 10 million instructions means that the simulation point starts when 150 million (15*10M) correct path instructions have been fetched. Detailed simulation of this simulation point would occur from instruction 150 million until just before 160 million.

One way to get more reproducible results is to use the first instruction program counter (Start PC) that occurs at the start of each interval of execution, instead of relying on instruction count. The same program counter can reappear many times, so it is also necessary to keep track of how many times a program counter value must appear to indicate the start of an interval of execution. For example, if a simulation point is triggered when PC 0x12000340 is executed the 1000th time. Then detailed simulation starts after that PC is seen 1000 times, and simulation occurs for the length of the interval. For this to work, the user needs to profile PCs in parallel with the frequency vector profile, and record the first PC seen for each interval along with the number of times that PC has executed up to that point in the execution. SimPoint provides the interval chosen for a simulation point, and this data can easily be mapped to this PC profile to determine the Start PC and the Nth occurrence of it where simulation should start.

It is highly recommended that you use the simulation point Start PCs for performing simulations. There are two reasons for this. The first reason deals with making sure you calculate the instructions during fast-forwarding exactly the same as when the simulation points were gathered. The second reason is that there can be slight variations in execution count between different runs of the same binary/input due to subtle changes in the simulation environment. Both of these are discussed in more detail later in this section.

Note, if you use the Start PC and its invocation count you need to make sure that the binary and any shared libraries used are loaded into the same address locations across all of your simulation runs for this to work. In general, this is important for any simulation study, in order to ensure that there are consistent address streams (instruction and global data) seen across the different runs of a program/input pair.

**Interval “Drift”** – When creating intervals, a problem may occur that the counts inside an interval might be just slightly larger than the interval size. Over time these counts can add up, so that if you were to try to find a particular fixed length interval in a simulation environment different from where the intervals were generated, you might be off by a few intervals.

For example, this can occur when forming fixed length intervals of X instructions. After X instructions execute the interval should be created, but this boundary may occur in the middle of a basic block, where there are an additional Y instructions in the basic block over the interval size. A frequency vector profiler that has the problem of interval drift may naively include these additional Y instructions in the interval that was just completed, especially if it was just counting basic blocks. Even though Y may be extremely small, it will accumulate over many thousands of intervals and cause a slow “drift” in the interval endpoints in terms of instruction count.

This is mainly a problem if you use executed instructions to determine the starting location for a simulation point. If you have drift in your intervals, to calculate the starting instruction count, you cannot just multiply the simulation point by the fixed length interval size as described above, since the interval lengths are not exactly the same. This can result in simulating the wrong set of instructions for the simulation point. When using the instruction count for the start of the simulation point, you need to keep track of the total instruction count for each interval if you have interval drift. You can then calculate the instruction count starting location for a simulation point by summing up the exact instruction counts for all of the intervals up to the interval chosen as the simulation point.

A better solution is to just make sure there is no drift at all in your intervals, but ending them precisely at the interval size boundary. In our above example, instead of including Y extra instructions in the interval that just ended, those extra Y instructions should be counted toward their basic block in the next interval. This results in splitting the basic block counts, for the basic blocks that occur on an interval boundary.

**Accurate Instruction Counts (No-ops)** – It is important to count instructions exactly the same for the frequency vector profiles as for the detailed simulation, otherwise they will diverge. Note that the simulation points on the SimPoint website include only correct path instructions and the instruction counts include no-ops. Therefore, to reach these simulation points in a simulator, every committed (correct path) instruction (including no-ops) must be counted.

**System Call Effects** – Some users have reported system call effects when running the same simulation points under slightly different OS configurations on a cluster of machines. This can result in slightly more or fewer instructions being executed to get to the same point in the program’s execution, and if the number of instructions executed is used to find the simulation point, this may lead to variations in the results. To avoid this, we suggest using the Start PC and Execution Count for each simulation point as described above. Another way to avoid variations in startup is to use checkpointing [22], and to use the SimpleScalar EIO files to make sure the system calls are the same between all simulated runs of a program/input combinations.

**Calculating Weighted IPC** – For IPC (instructions/cycle) we cannot just apply the weights directly as is done for CPI. Instead, we must convert all the simulated samples to CPI, compute the weighted average of CPI, and then convert the result back to IPC.

**Calculating Weighted Miss Rates** – To compute an overall miss rate (e.g. cache miss rate), first we must calculate both the weighted average of the number of cache accesses, and the weighted average of the number of cache misses. Dividing the second number by the first gives the estimated cache miss rate. In general, care must be taken when dealing with any ratio because both the numerator and the denominator must be averaged separately and then divided.

**Number of intervals** – There should be a sufficient number of intervals for the clustering algorithm to choose from. A good rule of thumb is to make sure to use at least 1,000 intervals in order for the clustering algorithm to be able to find a good partition of the intervals. If there are too few intervals, one can decrease the interval size to obtain more intervals for clustering.

**Using SimPoint 2.0 with VLIs** – As described in Section 4.1.1, SimPoint 2.0 assumes fixed-length intervals, and should not be used if the vectors to be clustered are variable length. The problem with using VLIs with SimPoint 2.0 is that the data will be clustered with a uniform weight distribution across all intervals, which is not correct for representing the execution properly. This means that the centroids may not be representative of the program’s execution in a cluster. This can result in large error rates, since a vector that is not representative of the majority of the cluster could be chosen as the simulation point.

**Wanting Variable Length, but not asking for it** – If you want variable length weighting for each interval then you need to use the -fixedLength off option. You may need to also use -loadVectorWeights if your vector weights cannot be automatically calculated from the vector’s frequency count values.

## 6. Summary

Modern computer architecture research depends on understanding the cycle level behavior of a processor running an application, and gaining this understanding can be done efficiently by judiciously applying detailed cycle level simulation to only a few simulation points. The level of detail provided by cycle level simulation comes at the cost of simulation speed, but by targeting only one or a few carefully chosen samples for each of the small number of behaviors found in real programs, this cost can be reduced to a reasonable level.

现在计算机架构研究需要理解处理器运行一个应用在周期级的行为，获得这种理解的高效方式，可以是将细节的周期级仿真只应用到几个仿真点上。周期级仿真提供的细节层次，其代价是仿真速度，但只对一个或几个仔细选择的样本进行仿真，这些样本是在真实程序中对几种行为找到的，这个代价是可以降低到合理的层次上的。

The main idea behind SimPoint is the realization that programs typically only exhibit a few unique behaviors which are interleaved with one another through time. By finding these behaviors and then determining the relative importance of each one, we can maintain both a high level picture of the program’s execution and at the same time quantify the cycle level interaction between the application and the architecture. The key to being able to find these phases in a efficient and robust manner is the development of a metric that can capture the underlying shifts in a program’s execution that result in the changes in observed behavior. SimPoint uses frequency vectors to calculate code similarity to cluster a program’s execution into phases.

SimPoint背后的主要思想，是程序一般只表现出几种独特的行为，这些行为随着时间逐个交叠在一起。找到这些行为，确定每种行为的相对重要性，我们可以得到程序执行的高层样子，同时量化应用和架构在周期级的互动。高效并稳健的找到这些状态的关键，是找到了一种度量，可以捕获程序执行的偏移会导致观察行为的变化。SimPoint使用频率向量来计算代码相似度，来将程序的执行聚类成状态。

SimPoint 3.0 automates the process of picking simulation points using an off-line phase classification algorithm, which significantly reduces the amount of simulation time required. These goals are met by simulating only a handful of intelligently picked behaviors of the full program. When these simulation points are carefully chosen, they provide an accurate picture of the complete execution of a program, which gives a highly accurate estimation of performance. This release provides new features for reducing the run-time of SimPoint and simulation points required, and provides support for variable length intervals. The SimPoint software can be downloaded at: http://www.cse.ucsd.edu/users/calder/simpoint/

SimPoint 3.0将利用离线状态分类算法来选择仿真点的过程进行了自动化，这显著降低了需要的仿真时间。只仿真一部分智能的选择得到的行为，就可以达到这个目标。当仿真点是仔细选择的，它们就会给出程序完整执行的准确表示，会给出性能的准确估计。这个版本会给出新特征，可以降低SimPoint的运行时间和需要的仿真点，并支持变长区间。