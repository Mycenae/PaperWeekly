# Machine Learning for Electronic Design Automation: A Survey

GUYUE HUANG et. al. @ Tsinghua University

With the down-scaling of CMOS technology, the design complexity of very large-scale integrated (VLSI) is increasing. Although the application of machine learning (ML) techniques in electronic design automation (EDA) can trace its history back to the 90s, the recent breakthrough of ML and the increasing complexity of EDA tasks have aroused more interests in incorporating ML to solve EDA tasks. In this paper, we present a comprehensive review of existing ML for EDA studies, organized following the EDA hierarchy.

随着CMOS尺寸缩小技术的发展，VLSI设计复杂度逐渐增加。虽然ML技术在EDA中的应用可以追溯历史到90s年代，ML最近的突破和EDA增加的复杂度，引起了很多采用ML来解决EDA任务的兴趣。本文中，我们回顾了ML在EDA中的研究，按照EDA层次结构来进行组织。

## 1. Introduction

As one of the most important fields in applied computer/electronic engineering, Electronic Design Automation (EDA) has a long history and is still under heavy development incorporating cutting-edge algorithms and technologies. In recent years, with the development of semiconductor technology, the scale of integrated circuit (IC) has grown exponentially, challenging the scalability and reliability of the circuit design flow. Therefore, EDA algorithms and software are required to be more effective and efficient to deal with extremely large search space with low latency.

作为应用计算机/电子工程的一个最重要领域，EDA有很长的历史，而且仍然在使用尖端算法和技术进行很多开发。在最近几年中，随着半导体技术的发展，IC的规模呈指数增长，挑战了电路设计流程的可扩展性和可靠性。因此，EDA算法和软件需要更加有效，效率更高，来用很短的时间处理极大的搜索空间。

Machine learning (ML) is taking an important role in our lives these days, which has been widely used in many scenarios. ML methods, including traditional and deep learning algorithms, achieve amazing performance in solving classification, detection, and design space exploration problems. Additionally, ML methods show great potential to generate high-quality solutions for many NP-complete (NPC) problems, which are common in the EDA field, while traditional methods lead to huge time and resource consumption to solve these problems. Traditional methods usually solve every problem from the beginning, with a lack of knowledge accumulation. Instead, ML algorithms focus on extracting high-level features or patterns that can be reused in other related or similar situations, avoiding repeated complicated analysis. Therefore, applying machine learning methods is a promising direction to accelerate the solving of EDA problems.

机器学习今天在我们的生活中的角色越来越重要，在很多场景中被广泛使用。ML方法，包括传统算法和深度学习算法，在处理分类，检测和设计空间探索问题中展现了非常好的性能。另外，ML方法在很多NPC问题中很可能生成高质量的解，这些问题在EDA中很常见，而传统方法需要耗费大量的时间和资源来求解这些问题。传统方法通常从头解决每个问题，缺少知识积累。而ML方法聚焦在提取高层次特征或模式，在相关或类似的情况中可以复用，避免了重复的复杂分析。因此，应用机器学习方法是加速EDA问题解决的一个有希望的方向。

In recent years, ML for EDA is becoming one of the trending topics, and a lot of studies that use ML to improve EDA methods have been proposed, which cover almost all the stages in the chip design flow, including design space reduction and exploration, logic synthesis, placement, routing, testing, verification, manufacturing, etc. These ML-based methods have demonstrated impressive improvement compared with traditional methods.

最近几年里，ML for EDA变得越来越流行，提出了很多使用ML改进EDA的研究，覆盖了芯片设计流程的几乎所有阶段，包括设计空间缩减和探索，逻辑综合，布局布线，测试，验证，制造等。与传统方法相比，这些基于ML的方法展现出了令人印象深刻的进步。

We observe that most work collected in this survey can be grouped into four types: decision making in traditional methods, performance prediction, black-box optimization, and automated design, ordered by decreasing manual efforts and expert experiences in the design procedure, or an increasing degree of automation. The opportunity of ML in EDA starts from decision making in traditional methods, where an ML model is trained to select among available tool chains, algorithms, or hyper-parameters, to replace empirical choice or brute-force search. ML is also used for performance prediction, where a model is trained from a database of previously implemented designs to predict the quality of new designs, helping engineers to evaluate new designs without the time-consuming synthesis procedure. Even more automated, EDA tools utilized the workflow of black-box optimization, where the entire procedure of design space exploration (DSE) is guided by a predictive ML model and a sampling strategy supported by ML theories. Recent advances in Deep Learning (DL), especially Reinforcement Learning (RL) techniques have stimulated several studies that fully automate some complex design tasks with extremely large design space, where predictors and policies are learned, performed, and adjusted in an online form, showing a promising future of Artificial Intelligence (AI)-assisted automated design.

我们观察到，本综述中收集的多数工作可以归类为四类：传统方法中的决策算法，性能预测，黑盒优化，和自动设计，这个排序是按照手工工作和专家经验在设计过程中的降序排列的，或自动化程度的增加。ML在EDA中的机会开始于传统方法中的决策算法，训练一个模型在可用的工具链、算法，或超参数中选择，以替换经验选择，或暴力搜索。ML也用于性能预测，从之前实现的设计的数据库中，训练一个模型，来预测新设计的质量，帮助工程师来评估新的设计，不需要非常耗时的综合过程。自动化程度更高的是，EDA工具利用黑盒优化的工作流，其中设计空间探索的整个过程由预测性ML模型，和采样策略来引导，这个策略由ML理论支撑。最近在DL方面的进展，尤其是RL，仿真了几个研究，使几个复杂的设计任务完全自动化了，有着非常大的设计空间，其中学习并使用了预测器和策略，并以在线的形式进行了调整，展现了AI辅助的自动设计的光明未来。

This survey gives a comprehensive review of some recent important studies applying ML to solve some EDA important problems. The review of these studies is organized according to their corresponding stages in the EDA flow. Although the study on ML for EDA can trace back to the last century, most of the works included in this survey are in recent five years. The rest of this survey is organized as follows. In Section 2, we introduce the background of both EDA and ML. From Section 3 to Section 5, we introduce the studies that focus on different stages of the EDA flow, i.e., high-level synthesis, logic synthesis & physical design (placement and routing), and mask synthesis, respectively. In Section 6, analog design methods with ML are reviewed. ML-powered testing and verification methods are discussed in Section 7. Then, in Section 8, other highly-related studies are discussed, including ML for SAT solver and the acceleration of EDA with deep learning engine. The discussion of various studies from the ML perspective is given in Section 9, which is complementary to the main organization of this paper. Finally, Section 10 concludes the existing ML methods for EDA and highlights future trends in this field.

本文对一些最近的复杂研究，使用ML来解决一些EDA重要问题，给出了一个综合回顾。这些研究的回顾根据其在EDA流中的阶段进行组织。虽然ML在EDA中的应用可以追溯到上个世纪，本综述包含的多数工作都是最近5年以内的。本综述组织如下：第2部分，我们介绍了EDA和ML的背景。从第3到第5部分，我们介绍的工作聚焦在EDA的不同阶段，即分别是，高层次综合，逻辑综合&物理设计（布局布线），掩膜综合。在第6部分，回顾了用ML的模拟设计方法。第7部分讨论了ML赋能的测试和验证方法。在第8部分，讨论了其他高度相关的研究，包括用于SAT求解器的ML，和用深度学习引擎加速的EDA。第9部分是从ML方面的各种研究的讨论，与本文的主要组织形式是互补的。最后，第10部分总结了现有的用于EDA的ML方法，强调了这个领域中的未来趋势。

## 2. Background

### 2.1 Electronic Design Automation

Electronic design automation is one of the most important fields in electronic engineering. In the past few decades, it has been witnessed that the flow of chip design became more and more standardized and complicated. A modern chip design flow is shown in Figure 1.

EDA是电子工程最重要的领域之一。在过去几十年，已经观察到，芯片设计的流程已经变得越来越标准化和复杂。图1展示了一个现代芯片设计的流程。

High-level synthesis (HLS) provides automatic conversion from C/C++/SystemC-based specifications to hardware description languages (HDL). HLS makes hardware design much more convenient by allowing the designer to use high-level descriptions for a hardware system. However, when facing a large-scale system, HLS often takes a long time to finish the synthesis. Consequently, efficient design space exploration (DSE) strategy is crucial in HLS [74, 95, 107, 112, 180].

高层次综合(HLS)可以从基于C/C++/SystemC的产品规格自动转换成HDL。HLS使硬件设计更加方便了，使硬件设计者可以对一个硬件系统使用高层次描述。但是，当面对一个大规模系统，HLS通常耗时很长才能完成综合。结果是，高效的DSE策略在HLS中非常关键。

Logic synthesis converts the behavioral level description to the gate level description, which is one of the most important problems in EDA. Logic synthesis implements the specific logic functions by generating a combination of gates selected in a given cell library, and optimizes the design for different optimization goals. Logic synthesis is a complicated process that usually cannot be solved optimally, and hence the heuristic algorithms are widely used in this stage, which include lots of ML methods [48, 56, 115, 167].

逻辑综合将行为级的描述，转换成门级的描述，这是EDA中最重要的一个问题。逻辑综合通过生成在给定单元库中的门的组合，来实现特定逻辑函数，并对不同的优化目标，来优化设计。逻辑综合是一个复杂的过程，通常不能得到最优的解决方法，因此启发式算法在这个阶段得到了广泛的应用，这包括很多ML方法。

Based on the netlist obtained from synthesis, floorplanning and placement aim to assign the netlist components to specific locations on the chip layout. Better placement assignment implies the potential of better chip area utilization, timing performance, and routability. Routing is one of the essential steps in very large-scale integrated (VLSI) physical design flow based on the placement assignment. Routing assigns the wires to connect the components on the chip. At the same time, routing needs to satisfy the requirements of timing performance and total wire-length without violating the design rules. The placement and routing are strongly coupled. Thus it is crucial to consider the routing performance even in the placement stage, and many ML-based routing-aware methods are proposed to improve the performance of physical design [6, 27, 89, 106, 150, 154].

基于从综合中得到的网表，平面规划和布局的目标是，指定网表的组成部分到芯片布局中特定的位置。更好的布局意味着对芯片面积可能利用的更好，时序性能和布线性更好。布线是VLSI物理设计流中一个基础的步骤，是基于布局指定结果的。布线指定一些线，在芯片中连接这些组成部分。同时，布线需要满足时序性能和总计线长的要求，同时不能违反设计规则。布局和布线是强耦合的。因此，在布局阶段就考虑布线的性能，这非常关键，因此提出了很多基于ML的意识到布线的方法来改进物理设计的性能。

Fabrication is a complicated process containing multiple steps, which has a high cost in terms of time and resources. Mask synthesis is one of the main steps in the fabrication process, where lithography simulation is leveraged to reduce the probability of fabrication failure. Mask optimization and lithography simulation are still challenging problems. Recently, various ML-based methods are applied in the lithography simulation and mask synthesis [20, 43, 159, 163, 165].

制造是一个复杂的过程，包含多个步骤，时间和资源的消耗都很大。掩膜综合是制造过程中的一个主要步骤，其中利用光刻仿真来降低制造失败的概率。掩膜优化和光刻仿真仍然是有挑战的问题。最近，各种基于ML的方法应用到了光刻仿真和掩膜综合中。

To ensure the correctness of a design, we need to perform design verification before manufacturing. In general, verification is conducted after each stage of the EDA flow, and the test set design is one of the major problems. Traditional random or automated test set generation methods are far away from optimal, therefore, there exist many studies that apply ML methods to optimize test set generation for verification [24, 33, 38, 47, 49, 57, 69, 135, 145, 146].

为确保一个设计的正确性，我们需要在制造之前进行设计验证。总体上，在EDA流的每个阶段，都进行验证，测试集的设计是一个主要的问题。传统的随机或自动测试集生成方法远不是最优的，因此，有很多研究应用ML方法来优化测试集生成，以用于验证。

After the chip design flow is finished, manufacturing testing needs to be carried out. The chips need to go through various tests to verify their functionality and reliability. The coverage and the efficiency are two main optimization goals of the testing stage. Generally speaking, a large test set (i.e., a large number of test points) leads to higher coverage at the cost of high resource consumption. To address the high cost of the testing process, studies have focused on applying ML techniques for test set optimization [100, 120, 140, 141] and test complexity reduction [5, 34, 142].

在芯片设计流完成后，需要进行制造测试。芯片需要经过各种测试，验证其功能和可靠性。测试阶段的两个主要优化目标，是覆盖和效率。一般来说，大的测试集（即，大量测试点），会以很高的资源消耗的代价，带来更高的覆盖。为解决测试过程中的高代价问题，有一些研究聚焦在将ML技术应用到测试集优化中，和测试复杂度降低中。

Thanks to decades of efforts from both academia and industry, the chip design flow is well-developed. However, with the huge increase in the scale of integrated circuits, more efficient and effective methods need to be incorporated to reduce the design cost. Recent advancements in machine learning have provided a far-reaching data-driven perspective for problem-solving. In this survey, we review recent learning-based approaches for each stage in the EDA flow and also discuss the ML for EDA studies from the machine learning perspective.

学术界和工业界经过几十年的努力，使芯片设计流程得到了充分发展。但是，随着IC的规模的巨量增长，需要更有效更高效的方法来降低设计代价。在机器学习中的最新进展，提供了一个影响深远的，数据驱动的解决问题的角度。本综述中，我们回顾了最近的基于学习的方法，来在EDA的每个阶段来解决问题，还从机器学习的角度讨论了用ML解决EDA问题的研究。

### 2.2 Machine Learning

Machine learning is a class of algorithms that automatically extract information from datasets or prior knowledge. Such a data-driven approach is a supplement to analytical models that are widely used in the EDA domain. In general, ML-based solutions can be categorized according to their learning paradigms: supervised learning, unsupervised learning, active learning, and reinforcement learning. The difference between supervised and unsupervised learning is whether or not the input data is labeled. With supervised or unsupervised learning, ML models are trained on static data sets offline and then deployed for online inputs without refinement. With active learning, ML models subjectively choose samples from input space to obtain ground truth and refine themselves during the searching process. With reinforcement learning, ML models interact with the environment by taking actions and getting rewards, with the goal of maximizing the total reward. These paradigms all have been shown to be applied to the EDA problems.

机器学习是一类算法，可以从数据集或先验知识中自动提取信息。这样一种数据驱动的方法是解析模型的补充，在EDA领域，解析模型得到了广泛应用。总体上，基于ML的方法可以根据其学习范式来归类：监督学习，无监督学习，主动学习和强化学习。监督学习和无监督学习的差异在于，输入数据是否带有标注。用监督学习或无监督学习，ML模型在静态数据集上进行离线训练，然后部署到在线输入中，不需要提炼优化。在主动学习中，ML模型从输入空间中主观的选择样本，得到真值，在搜索过程中进行提炼优化。在强化学习中，ML模型与环境互动，采取行动，得到回报，目标是最大化总计回报。这些范式都应用到了EDA问题中。

As for the model construction, conventional machine learning models have been extensively studied for the EDA problems, especially for physical design [66, 178]. Linear regression, random forest (RF) [91] and artificial neural networks (ANN) [55] are classical regression models. Support vector machine (SVM) [12] is a powerful classification algorithm especially suitable for tasks with a small size of training set. Other common classification models include K-Nearest-Neighbor (KNN) algorithm [39] and RF. These models can be combined with ensemble or boosting techniques to build more expressive models. For example, XGBoost [23] is a gradient boosting framework frequently used in the EDA problems.

至于模型构建，传统机器学习模型解决EDA问题已经得到了广泛的研究，尤其是对于物理设计。线性回归，随机森林，和人工神经网络是经典的回归模型。SVM是一种很强的分类算法，尤其适用于训练集较小的任务。其他常见的分类模型包括KNN算法和随机森林。这些模型可以用集成或boosting技术结合到一起，构建更有表达力的模型。比如，XGBoost是一个梯度boosting框架，在EDA问题中得到了频繁的应用。

Thanks to large public datasets, algorithm breakthrough, and improvements in computation platforms, there have been efforts of applying deep learning (DL) for EDA. In particular, popular models in recent EDA studies include convolutional neural network (CNN) [37, 111], recurrent neural networks (RNN) [83, 148], generative adversarial network (GAN) [165], deep reinforcement learning (DRL) [113, 147] and graph neural networks (GNN) [147, 168]. CNN models are composed of convolutional layers and other basic blocks such as non-linear activation functions and down-sample pooling functions. While CNN is suitable for feature extraction on grid structure data like 2-D image, RNN is good at processing sequential data such as text or audio. GNN is proposed for data organized as graphs. GAN trains jointly a generative network and a discriminative network which compete against each other to eventually generate high quality fake samples. DRL is a class of algorithms that incorporated deep learning into the reinforcement learning paradigm, where an agent learns a strategy from the rewards acquired with previous actions to determine the next action. DRL has achieved great success in complicated tasks with large decision space (e.g., Go game [138]).

多亏了大型公共数据集，算法突破，和计算平台的改进，将深度学习用于EDA也有很多工作。特别是，在最近的EDA研究中流行的模型包括，CNN，RNN，GAN，DRL和GNN。CNN模型由卷积层和其他基础模块如非线性激活函数和下采样池化函数组成。CNN适用于网格结构数据中的特征提取，如2D图像，RNN擅长处理序列数据，如文本或语音。GNN是用于组织成图的数据。GAN联合训练一个生成网络和区分网络，互相竞争，最终生成高质量假样本。DRL这类算法将深度学习结合到强化学习的范式，其中一个agent从回报中学习一个策略，回报是之前的行为获得的，确定下一个行为。DRL已经在有大量决策空间的复杂任务中获得了极大的成功（如，Go game）。

## 3. High Level Synthesis

High-level synthesis (HLS) tools provide automatic conversion from C/C++/SystemC-based specification to hardware description languages like Verilog or VHDL. HLS tools developed in industry and academia [1, 2, 15] have greatly improved productivity in customized hardware design. High-quality HLS designs require appropriate pragmas in the high-level source code related to parallelism, scheduling and resource usage, and careful choices of synthesis configurations in post-Register-Transfer-Level (RTL) stage. Tuning these pragmas and configurations is a non-trivial task, and the long synthesis time for each design (hours from the source code to the final bitstream) prohibits exhaustive DSE.

高层次综合(HLS)工具可以将基于C/C++/SystemC的指标自动转换到硬件描述语言，如Verilog或VHDL。工业界和学术界开发的HLS工具在定制的硬件设计中极大的改进了生产力。高质量HLS设计在高层源代码中需要合适的编译指示，与并行性、调度和资源使用，在后RTL阶段仔细的选择综合配置。调节这些编译指示和配置，需要不少工作量，对于每个设计都需要很长的综合时间（从源代码到最后的比特流，要几个小时），这就妨碍使用穷举式DSE。

ML techniques have been applied to improve HLS tools from the following three aspects: fast and accurate result estimation [30, 37, 108, 109, 143, 164, 172], refining conventional DSE algorithms [74, 107, 149], and reforming DSE as an active-learning problem [94, 95, 112, 180]. In addition to achieving good results on individual problems, previous studies have also introduced new generalizable techniques about feature engineering [30, 108, 109, 164, 172], selection and customization of ML models [143], and design space sampling and searching strategies [95, 112, 180].

ML技术已经应用与改进HLS工具，主要从下面这三个方面：快速准确的结果估计，改进传统的DSE算法，将DSE重新表述为一个主动学习问题。除了要在单个问题上获得好的结果，之前的研究还引入了新的可泛化的关于特征工程的技术，选择和定制ML模型，设计空间采样和搜索技术。

This section is organized as follows. Section 3.1 introduces recent studies on employing ML for result estimation, often in a static way. Section 3.2 introduces recent studies on adopting ML in DSE workflow, either to improve conventional methods or in the form of active learning.

本节组织如下。3.1节介绍了最近在采用ML在结果估计上的研究，通常是一种静态的方式。3.2节介绍了采用ML进行DSE工作流的研究，要么改进传统方法，或以主动学习的方式。

### 3.1 Machine Learning for Result Estimation

The reports from HLS tools provide important guidance for tuning the high-level directives. However, acquiring accurate result estimation in an early stage is difficult due to complex optimizations in the physical synthesis, imposing a trade-off between accuracy (waiting for post-synthesis results) and efficiency (evaluating in the HLS stage). ML can be used to improve the accuracy of HLS reports through learning from real design benchmarks. In Section 3.1.1, we introduce previous work on predicting the timing, resource usage, and operation delay of an HLS design. In Section 3.1.2 we describe two types of research about cross-platform performance prediction.

HLS工具的报告为调节高层指令给出了重要的引导。但是，在早期阶段获得精确的结果估计是很困难的，因为物理综合过程的优化非常复杂，在准确率（等待综合后的结果）和效率（在HLS阶段的评估）之间给出了一个折中。ML可以从真实的设计基准测试中学习，从而用于改进HLS报告的准确率。在3.1.1节，我们介绍了一个HLS设计在预测时序，资源使用和操作延迟方面以前的工作。在3.1.2节我们描述了关于跨平台性能预测方面两种类型的研究。

**3.1.1 Estimation of Timing, Resource Usage, and Operation Delay**. The overall workflow of timing and resource usage prediction is concluded in Figure 2. This workflow is first proposed by Dai et al. [30] and augmented by Makrani et al. [108] and Ferianc et al. [37]. The main methodology is to train an ML model that takes HLS reports as input and outputs a more accurate implementation report without conducting the time-consuming post-implementation. The workflow proposed by Dai et al. [30] can be divided into two steps: data processing and training estimation models.

时序，资源使用和操作延迟的估计。时序和资源使用预测方面的总体流程如图2所示。这种工作流程经过了提出和改进的过程。主要的方法是训练一个ML模型，以HLS报告为输入，输出一个更精确的实现报告，而不需要进行耗时的后实现操作。Dai等[30]提出的工作流，可以分成2个步骤：数据处理和训练估计模型。

**Step 1: Data Processing**. To enable ML for HLS estimation, we need a dataset for training and testing. The HLS and implementation reports are usually collected across individual designs by running each design through the complete C-to-bitstream flow, for various clock periods and targeting different FPGA devices. After that, one can extract features from the HLS reports as inputs and features from implementation reports as outputs. Besides, to overcome the effect of colinearity and reduce the dimension of the data, previous studies often apply feature selection techniques to systematically remove unimportant features. The most commonly used features are summarized in Table 1.

步骤1：数据处理。为能用ML对HLS进行估计，我们需要一个数据集进行训练和测试。通常要收集单个设计的HLS报告和实现报告，将每个设计，对不同的时钟周期，面向不同的FPGA设备，运行完整的C-to-bitstream的流。在这个之后，可以从HLS报告中提取特征作为输入，实现报告中的特征作为输出。另外，为克服共线的效果，降低数据的维度，之前的研究通常用特征选择技术，来系统性的去除不重要的特征。最常用的特征，如表1所示。

**Step 2: Training Estimation Models**. After constructing the dataset, regression models are trained to estimate post-implementation resource usages and clock periods. Frequently used metrics to report the estimation error include relative absolute error (RAE) and relative root mean squared error (RMSE). For both metrics, lower is better. RAE is defined in Equation (1), where $\hat 𝑦$ is a vector of values predicted by the model, 𝑦 is a vector of actual ground truth values in the testing set, and $\bar 𝑦$ denotes the mean value of 𝑦.

步骤2：训练估计模型。在构建数据集后，训练了回归模型来估计实现后资源使用和时钟周期。报告估计错误所频繁使用的度量包括，相对绝对误差(RAE)，和相对均方误差(RMSE)。对两种度量都是越小越好。RAE由式(1)定义，其中$\hat 𝑦$是模型预测的值的矢量，𝑦是测试集中实际的真值矢量，$\bar 𝑦$表示𝑦的均值。

$$RAE = \frac {\hat 𝑦 - 𝑦}{y - \bar y}$$(1)

Relative RMSE is given by Equation (2), where 𝑁 is the number of samples, and $\hat 𝑦_𝑖$ and $𝑦_𝑖$ are the predicted and actual values of a sample, respectively.

相对RMSE是由式(2)给出，其中N是样本数量，$\hat 𝑦_𝑖$和$𝑦_𝑖$分别是样本的预测值和实际值。

$$Relative RMSE = \sqrt{\sum_{i=1}^N (\frac{\hat 𝑦_𝑖 - 𝑦_𝑖}{𝑦_𝑖})^2/N} × 100%$$(2)

Makrani et al. [108] model timing as a regression problem, and use the Minerva tool [36] to obtain results in terms of maximum clock frequency, throughput, and throughput-to-area ratio for the RTL code generated by the HLS tool. Then an ensemble model combining linear regression, neural network, SVM, and random forest, is proposed to conduct estimation and achieve an accuracy higher than 95%. There are also studies that predict whether a post-implementation is required or not, instead of predicting the implementation results. As a representative study, Liu and Schäfer [94] train a predictive model to avoid re-synthesizing each new configuration.

Makrani等[108]将时序建模为一个回归问题，使用Minerva工具[36]来对HLS工具生成的RTL代码得到最大时钟频率，通量和通量面积比等结果。然后，一个集成模型将线性回归，神经网络，SVM和随机森林，结合到一起，进行估计，得到的准确率高于95%。也有研究预测实现后是否需要，而不是预测实现结果。作为一个代表性研究，Liu and Schäfer [94]训练了一个预测模型，来避免重新综合每个新配置。

ML techniques have been applied recently to reduce the HLS tool’s prediction error of the operation delay [143]. Existing HLS tools perform delay estimations based on the simple addition of pre-characterized delays of individual operations, and can be inaccurate because of the post-implementation optimizations (e.g., mapping to hardened blocks like DSP adder cluster). A customized Graph Neural Network (GNN) model is built to capture the association between operations from the dataflow graph, and train this model to infer the mapping choices about hardened blocks. Their method can reduce the RMSE of the operation delay prediction of Vivado HLS by 72%.

ML技术最近进行了应用，以降低HLS工具的操作延迟预测误差[143]。现有的HLS工具进行延迟估计，是基于单个操作的预先特征化的延迟的简单相加，可能是不准确的，因为还有后实现优化（如，映射到硬化的模块，如DSP加法簇）。构建了一个定制的GNN模型，来捕获数据流图中的操作之间的关联，训练这个模型来推理硬化模块的映射选项。他们的方法可以降低Vivado HLS的操作延迟预测的RMSE达到72%。

**3.1.2 Cross-Platform Performance Prediction**. Hardware/software co-design enables designers to take advantage of new hybrid platforms such as Zynq. However, dividing an application into two parts makes the platform selection difficult for the developers, since there is a huge variation in the application’s performance of the same workload across various platforms. To avoid fully implementing the design on each platform, Makrani et al. [109] propose an ML-based cross-platform performance estimator, XPPE, and its overall workflow is described in Figure 3. The key functionality of XPPE is using the resource utilization of an application on one specific FPGA to estimate its performance on other FPGAs.

3.1.2 跨平台性能预测。硬件/软件协同设计使设计者可以利用新的混合平台，如Zynq。但是，将一个应用分成两部分，使开发者很难选择平台，由于相同的workload在不同的平台上，其应用的性能会有很大的变化。为避免在每个平台对设计进行完整的实现，Makrani等[109]提出一种基于ML的跨平台性能估计器，XPPE，其总体工作流如图3所示。XPPE的关键功能是，使用一个应用在特定FPGA平台的资源利用，来估计其在其他FPGA上性能。

XPPE uses a Neural Network (NN) model to estimate the speedup of an application for a target FPGA over the ARM processor. The inputs of XPPE are available resources on target FPGA, resource utilization report from HLS Vivado tool (extracted features, similar to the features in Table 1), and application’s characteristics. The output is the speedup estimation on the target FPGA over an ARM A-9 processor. This method is similar to Dai et al. [30] and Makrani et al. [108] in that they all take the features in HLS reports as input and aim to avoid the time-consuming post-implementation. The main difference is that the input and output features in XPPE are from different platforms. The relative RMSE between the predictions and the real measurements is used to evaluate the accuracy of the estimator. The proposed architecture can achieve a relative mean square error of 5.1% and the speedup is more than 0.98×.

XPPE使用一个神经网络模型来估计一个应用在目标FPGA上对在ARM处理器上的加速。XPPE的输入是在目标FPGA上的可用资源，HLS Vivado工具的资源利用报告（提取出的特征，与表1中的特征类似），和应用的特征。输出是在目标FPGA上，对在ARM A-9处理器的加速估计。这个方法与Dai等[30]和Makrani等[108]等的方法类似，都以HLS报告中的特征作为输入，以避免耗时的后实现。主要的差别是，XPPE的输入和输出特征，是从不同的平台来的。在预测和实际测量之间的相对RMSE，用于评估估计器的准确率。提出的架构可以得到5.1%的RMSE，加速超过了0.98x。

Like XPPE, O’Neal et al. [116] also propose an ML-based cross-platform estimator, named HLSPredict. There are two differences. First, HLSPredict only takes workloads (the applications in XPPE) as inputs instead of the combination of HLS reports, application’s characteristics and specification of target FPGA devices. Second, the target platform of HLSPredict must be the same as the platform in the training stage. In general, HLSPredict aims to rapidly estimate performance on a specific FPGA by direct execution of a workload on a commercially available off-the-shelf host CPU, but XPPE aims to accurately predict the speedup of different target platforms. For optimized workloads, HLSPredict achieves a relative absolute percentage error ($𝐴𝑃𝐸 = |(𝑦− \hat 𝑦)/𝑦|$) of 9.08% and a 43.78× runtime speedup compared with FPGA synthesis and direct execution.

与XPPE类似，O'Neal等[116]还提出了一种基于ML的跨平台估计器，名为HLSPredict。有两个差异。第一，HLSPredict只将workload作为输入（XPPE中的应用），而不是HLS报告，应用的特征和目标FPGA设备的指标的集合。第二，HLSPredict的目标平台，必须与训练时的平台一样。总体来说，HLSPredict的目标是迅速估计特定FPGA平台的性能，将workload在一个商用开箱即用的宿主CPU进行直接执行，而XPPE的目标是准确的预测不同目标平台的加速。对优化的workloads，HLSPredict获得的相对绝对误差百分比($𝐴𝑃𝐸 = |(𝑦− \hat 𝑦)/𝑦|$)为9.08%，运行时加速为43.78x（FPGA综合和直接执行的对比）。

### 3.2 Machine Learning for Design Space Exploration in HLS

In the previous subsection, we describe how ML models are used to predict the quality of results. Another application of ML in HLS is to assist DSE. The tunable synthesis options in HLS, provided in the form of pragmas, span a very large design space. Most often, the task of DSE is to find the Pareto Frontier Curve, on which every point is not fully dominated by any other points under all the metrics.

在前一小节中，我们描述了ML模型怎样用于预测结果的质量。ML在HLS中的另一个应用是支持DSE。HLS中的可调节的综合选项是以pragma的形式给出的，撑起了一个非常大的设计空间。DSE任务最经常的目标是，找到Pareto front曲线，其中的每个点上，其他任何点都不会在所有度量上dominate这个点。

Classical search algorithms have been applied in HLS DSE, such as Simulated Annealing (SA) and Genetic Algorithm (GA). But these algorithms are unable to learn from the database of previously explored designs. Many previous studies use an ML predictive model to guide the DSE. The models are trained on the synthesis results of explored design points, and used to predict the quality of new designs (See more discussions on this active learning workflow in Section 9.1). Typical studies are elaborated in Section 3.2.1. There is also a thread of work that involves learning-based methods to improve the inefficient or sensitive part of the classical search algorithms, as elaborated inSection 3.2.2. Some work included in this subsection focuses on system-level DSE rather than HLS design [74], or general active learning theories [180].

经典搜索算法曾经应用到HLS DSE上，比如模拟退火，和遗传算法。但这些算法不能从之前探索过的设计中学习。很多之前的研究使用一个ML预测模型来引导DSE。模型是在探索过的设计点的综合结果上进行训练的，用于预测新设计的质量（见9.1节这个主动学习工作流的更多讨论）。典型的研究在3.2.1节中进行详述。还有一条工作线是，基于学习的方法来改进经典搜索方法的低效性或敏感部分，这部分见3.2.2节。本节中包含的一些工作，关注的是系统级的DSE，而不是HLS 设计，或一般的主动学习理论。

**3.2.1 Active Learning**. The four papers visited in this part utilize the active learning approach to perform DSE for HLS, and use predictive ML models to surrogate actual synthesis when evaluating a design. Liu and Schäfer [94] propose a design space explorer that selects new designs to implement through an active learning approach. Transductive experimental design (TED) [95] focuses on seeking the samples that describe the design space accurately. Pareto active learning (PAL) in [180] is proposed to sample designs which the learner cannot clearly classify. Instead of focusing on how accurately the model describes the design space, adaptive threshold non-pareto elimination (ATNE) [112] estimates the inaccuracy of the learner and achieves better performance than TED and PAL.

主动学习。本小节回顾的四篇文章，利用了主动学习的方法，来进行HLS的DSE，在评估一个设计时，使用预测性的ML模型，来代理实际的综合。[94]提出了一个设计空间探索方法，通过一种主动学习方法，来选择新的设计来实现。转换试验设计(TED)[95]聚焦的是寻找可以精确的描述设计空间的样本。Pareto主动学习(PAL)[180]的提出，是为了采样学习者不能清晰的分类的设计。自适应阈值non-pareto消除(ANTE)[112]估计的是学习者的不精确性，而不是聚焦在模型怎样精确的描述设计空间，获得了比TED和PAL更好的性能。

Liu and Schäfer [94] propose a dedicated explorer to search for Pareto-optimal HLS designs for FPGAs. The explorer iteratively selects potential Pareto-optimal designs to synthesize and verify. The selection is based on a set of important features, which are adjusted during the exploration. The proposed method runs 6.5× faster than an exhaustive search, and runs 3.0× faster than a restricted search method but finds results with higher quality.

[94]提出了一个专用explorer，搜索FPGA用的Pareto最优HLS设计。Explorer迭代选择可能的Pareto最优设计来进行综合和验证。这个选择是基于一系列重要的特征，在探索的过程中进行调整。提出的方法比穷举式搜索要快6.5x，比一种受限的搜索方法快3.0x，但找到的结果有更高的质量。

The basic idea of TED [95] is to select representative as well as the hard-to-predict samples from the design space, instead of the random sample used in previous work. The target is to maximize the accuracy of the predictive model with the fewest training samples. The authors formulate the problem of finding the best sampling strategy as follows: TED assumes that the overall number of knob settings is 𝑛(|K| = 𝑛), from which we want to select a training set $\tilde K$ such that $|\tilde K| = 𝑚$. Minimizing the prediction error $𝐻(𝑘) − \tilde 𝐻(𝑘)$ for all 𝑘 ∈ K is equivalent to the following problem:

TED[95]的基本思想是从设计空间中选择有代表性的，同时也是难以预测的样本，而不是之前的工作中使用的随机样本。目标是用最小的训练样本最大化预测模型的准确率。作者将找到最好的采样策略的问题表述如下：TED假设，总体knob设置的数量为𝑛(|K| = 𝑛)，从中我们希望选择一个训练集$\tilde K$，使得$|\tilde K| = 𝑚$。对所有的𝑘 ∈ K最小化预测误差$𝐻(𝑘) − \tilde 𝐻(𝑘)$，等价于如下问题：

$$max_{\tilde K} T[K\tilde K^T(\tilde K\tilde K^T + 𝜇I)^{-1}\tilde KK^T], s.t. \tilde K ⊂ K, |\tilde K| = m$$

where 𝑇[·] is the matrix trace operator and 𝜇 > 0. The authors interpret their solution as sampling from a set $\tilde K$ that span a linear space, to retain most of the information of K [95].

其中𝑇[·]是矩阵迹运算，𝜇 > 0。作者解释其解决方案，采样得到的集合$\tilde K$要张成一个线性空间，保留最多的K中的信息。

PAL [180] is proposed for general active learning scenarios and is demonstrated by a sorting network synthesis DSE problem in the paper. It uses Gaussian Process (GP) to predict Pareto-optimal points in design space. The models predict the objective functions to identify points that are Pareto-optimal with high probabilities. A point 𝑥 that has not been sampled is predicted as $\hat 𝑓(𝑥) = 𝜇(𝑥)$ and 𝜎(𝑥) is interpreted as the uncertainty of the prediction which can be captured by the hyperrectangle

PAL[180]是为了通用主动学习场景而提出，论文中用一个排序网络综合DSE问题来证明。其利用高斯过程(GP)来预测设计空间中的Pareto-optimal点。模型预测目标函数来识别高概率Pareto-optimal点。没有被采样的点𝑥预测为$\hat 𝑓(𝑥) = 𝜇(𝑥)$，𝜎(𝑥)解释为预测的不确定性，可以通过下面的超矩形来捕获

$$𝑄_{𝜇,𝜎,𝛽} (𝑥) = {𝑦: 𝜇(𝑥) − 𝛽^{1/2}𝜎(𝑥) ⪯ 𝑦 ⪯ 𝜇(𝑥) + 𝛽^{1/2}𝜎(𝑥)}$$

where 𝛽 is a scaling parameter to be chosen. PAL focuses on accurately predicting points near the Pareto frontier, instead of the whole design space. In every iteration, the algorithm classifies samples into three groups: Pareto-optimal, Non-Pareto-optimal, and uncertain ones. The next design point to evaluate is the one with the largest uncertainty, which intuitively has more information to improve the model. The training process is terminated when there are no uncertain points. The points classified as Pareto-optimal are then returned.

其中𝛽是一个要选择的缩放参数。PAL聚焦在准确的预测Pareto frontier附近的点上，而不是整个设计空间。在每次迭代中，算法将样本分成三组：Pareto-optimal, Non-Pareto-optimal, 和不确定的样本。要评估的下一个点，是有最大不确定性的那个点，从直觉上来说，有更多的信息来改进模型。在没有不确定点的时候，训练过程停止。分类为Pareto-optimal的点，然后进行返回。

ATNE [112] utilizes Random Forest (RF) to aid the DSE process. This work uses a Pareto identification threshold that adapts to the estimated inaccuracy of the RF regressor and eliminates the non-Pareto-optimal designs incrementally. Instead of focusing on improving the accuracy of the learner, ATNE focuses on estimating and minimizing the risk of losing “good” designs due to learning inaccuracy.

ATNE利用随机森林来帮助DSE过程。这个工作使用一个Pareto识别阈值，随着RF回归器的估计的不准确率来改变，逐渐的消除non-Pareto-optimal设计。ATNE聚焦在估计和最小化损失好的设计的风险（由于学习到了不准确性），而没有聚焦在提升学习者的准确率。

**3.2.2 Machine Learning for Improving Other Optimization Algorithms**. In this part, we summarize three studies that use ML techniques to improve classical optimization algorithms.

机器学习用于改进其他优化算法。这个部分中，我们总结了三个研究，使用ML技术来改进经典优化算法。

STAGE [74] is proposed for DSE of many-core systems. The motivating observation of STAGE is that the performance of simulated annealing is highly sensitive to the starting point of the search process. The authors build an ML model to learn which parts of the design space should be focused on, eliminating the times of futile exploration [13]. The proposed strategy is divided into two stages. The first stage (local search) performs a normal local search, guided by a cost function based on the designer’s goals. The second stage (meta search) tries to use the search trajectories from previous local search runs to learn to predict the outcome of local search given a certain starting point [74].

STAGE的提出是为了众核系统的DSE。STAGE的推动性观察是，模拟退火的性能，对搜索过程的起始点高度敏感。作者构建了一个ML模型，来学习应当聚焦在设计空间的哪个部分，消除无用探索的次数。提出的策略分成两个阶段。第一阶段（局部搜索）进行正常的局部搜索，由基于设计者目标的代价函数来引导。第二个阶段（元搜索）试图使用之前的局部搜索的搜索轨迹，在给定特定起始点的情况下，来学习预测局部搜索的输出。

Fast Simulated Annealing (FSA) [107] utilizes the decision tree to improve the performance of SA. Decision tree learning is a widely used method for inductive inference. The HLS pragmas are taken as input features. FSA first performs standard SA to generate enough training sets to build the decision tree. Then it generates new design configurations with the decision tree and keeps the dominating designs [107].

快速模拟退火(FPA)[107]利用决策树来改进SA的性能。决策树学习是一种广泛使用的方法，可以进行归纳推理。HLS pragma作为输入特征。FSA首先进行标准的SA来生成足够的训练集，来构建决策树。然后用决策树来生成新的设计配置，保留主导的设计。

In a recent study, Wang and Schäfer [149] propose several ML techniques to help decide the hyper-parameter settings of three meta-heuristic algorithms: SA, GA and Ant Colony Optimizations (ACO). For each algorithm, the authors build an ML model that predicts the resultant design quality (measured by Average Distance to the Reference Set, ADRS) and runtime from hyper-parameter settings. Compared with the default hyper-parameters, their models can improve the ADRS by more than 1.92× within similar runtime. The authors also combine SA, GA and ACO to build a new design space explorer, which further improves the search efficiency.

在最近的研究中，[149]提出几种ML技术来帮助决定三种元启发式算法的超参数配置：SA，GA和ACO。对于每种算法，作者构建了一个ML模型，从超参数设置中预测得到的设计质量（由到参考集的平均距离ADRS来度量）和运行时。与默认的超参数比较，他们的模型可以改进ADRS超过1.92x，运行时间类似。作者还结合SA，GA和ACO，构建了一个新的设计空间探索者，这更进一步改进了搜索效率。

### 3.3 Summary of Machine Learning for HLS

This section reviews recent work on ML techniques in HLS, as listed in Table 2. Using ML-based timing/resource/latency predictors and data-driven searching strategies, the engineering productivity of HLS tools can be further improved and higher-quality designs can be generated by efficiently exploring a large design space.

本节回顾了最近ML技术在HLS中的工作，如表2所示。使用基于ML的时序/资源/延迟预测器，和数据驱动的搜索策略，HLS工具的工程生产力可以进一步改进，通过高效的探索大型设计空间，可以生成更高质量的设计。

We believe the following practice can help promote future research of ML in HLS: 我们相信，下面的实践可以帮助推动ML在HLS中的未来研究：

- Public benchmark for DSE problems. The researches about result estimation are all evaluated on public benchmarks of HLS applications, such as Rosetta [174], MachSuite [127], etc. However, DSE researches are often evaluated on a few applications because the cost of synthesizing a large design space for each application is heavy. Building a benchmark that collects different implementations of each application can help fairly evaluate DSE algorithms.

DSE问题的公开基准测试。结果估计的研究者都在HLS应用的公开基准测试上进行评估，比如Rosetta，MachSuite等。但是，DSE研究者通常在几个应用中进行评估，因为为每个应用综合一个大型设计空间的代价是很重的。构建一个基准测试，收集每个应用的不同实现，可以帮助公平的评估DSE算法。

- Customized ML models. Most of the previous studies use off-the-shelf ML models. Combining universal ML algorithms with domain knowledge can potentially improve the performance of the model. For example, Ustun et al. [143] customize a standard GNN model to handle the specific delay prediction problem, which brings extra benefit in model accuracy.

定制ML模型。多数之前的研究使用开箱即用的ML模型。将通用ML算法与领域知识结合起来，可能会改进模型性能。比如，Ustun等[143]定制了一个标准GNN模型，来处理特定延迟预测问题，对模型准确率带来了额外的好处。

## 4. Logic Synthesis and Physical Design

In the logic synthesis and physical design stage, there are many key sub-problems that can benefit from the power of ML models, including lithography hotspot detection, path classification, congestion prediction, placement guide, fast timing analysis, logic synthesis scheduling, and so on. In this section, we organize the review of studies by their targeting problems.

### 4.1 Logic Synthesis

Logic synthesis is an optimization problem with complicated constraints, which requires accurate solutions. Consequently, using ML algorithms to directly generate logic synthesis solutions is difficult. However, there are some studies using ML algorithms to schedule existing traditional optimization strategies. For logic synthesis, LSOracle [115] relies on DNN to dynamically decide which optimizer should be applied to different parts of the circuit. The framework exploits two optimizers, and-inverter graph (AIG) and majority-inverter graph (MIG), and applies k-way partitioning on circuit directed acyclic graph (DAG).

There are many logic transformations in current synthesis tools such as ABC [14]. To select an appropriate synthesis flow, Yu et al. [167] formulate a multi-class classification problem and design a CNN to map a synthesis flow to quality of results (QoR) levels. The prediction on unlabeled flows are then used to select the optimal synthesis flow. The CNN takes the one-hot encoding of synthesis flows as inputs and outputs the possibilities of the input flow belonging to different QoR metric levels.

Reinforcement learning is also employed for logic synthesis in [48, 56]. A transformation between two DAGs with the same I/O behaviors is modeled as an action. In [48], GCN is utilized as a policy function to obtain the probabilities for every action. [56] employs advantage actor critic agent (A2C) to search the optimal solution.

### 4.2 Placement and Routing Prediction

**4.2.1 Traditional Placers Enhancement**. While previous fast placers can conduct random logic placement efficiently with good performances, researchers find that their placement of data path logic is suboptimal. PADE [150] proposes a placement process with automatic data path extraction and evaluation, in which the placement of data path logic is conducted separately from random logic. PADE is a force-directed global placer, which applies SVM and NN to extract and evaluate the data path patterns with high dimensional data such as netlist symmetrical structures, initial placement hints, and relative area. The extracted data path is mapped to bit stack structure and uses SAPT [151] (a placer placed on SimPL [73]) to optimize separately from random logic.

**4.2.2 Routing Information Prediction**. The basic requirements of routing design rules must be considered in the placement stage. However, it is difficult to predict routing information in the placement stage accurately and fast, and researchers recently employ machine learning to solve this. RouteNet [154] is the first work to employ CNN for design rule checking (DRC) hotspot detection. The input features of a customized fully convolutional network (FCN) include the outputs of rectangular uniform wire density (RUDY), a pre-routing congestion estimator. An 18-layer ResNet is also employed to predict design rule violation (DRV) count. A recent work [89] abstracts the pins and macros density in placement results into image data, and utilizes a pixel-wise loss function to optimize an encoder-decoder model (an extension of U-Net architecture). The network output is a heat-map, which represents the location where detailed routing congestion may occur. PROS [21] takes advantages of fully convolution networks to predict routing congestion from global placement results. The framework is demonstrated efficient on industrial netlists. Pui et al. [124] explore the possibilities using ML methods to predict routing congestion in UltraScale FPGAs. Alawieh et al. [6] transfer the routing congestion problem in large-scale FPGAs to an image-to-image problem, and then uses conditional GAN to solve it. In addition, there are some studies that only predict the number of congestions instead of the location of congestion [27, 106]. Maarouf et al. [106] use models like linear regression, RF and MLP to learn how to use features from earlier stages to produce more accurate congestion prediction, so that the placement strategy can be adjusted. Qi et al. [125] predict the detailed routing congestion using nonparametric regression algorithm, multivariate adaptive regression splines (MARS) with the global information as inputs. Another study [18] takes the netlist, clock period, utilization, aspect ratio and BEOL stack as inputs and utilizes MARS and SVM to predict the routability of a placement. This study also predicts Pareto frontiers of utilization, number of metal layers, and aspect ratio. Study in [99] demonstrates the potential of embedding ML-based routing congestion estimator into global placement stage. Recently, Liang et al. [90] build a routing-free crosstalk prediction model by adopting several ML algorithms such as regression, NN, GraphSAGE and GraphAttention. The proposed framework can identify nets with large crosstalk noise before the routing step, which allows us to modify the placement results to reduce crosstalk in advance.

There is also a need to estimate the final wirelength, timing performance, circuit area, power consumption, clock and other parameters in the early stage. Such prediction task can be modeled as a regression task and commonly-used ML models include SVM, Boosting, RF, MARS, etc. Jeong et al. [63] learn a model with MARS to predict performance from a given set of circuit configurations, with NoC router, a specific functional circuit and a specific business tool. In [60], the researchers introduce linear discriminant analysis (LDA) algorithm to find seven combined features for the best representation, and then a KNN-like approach is adopted to combine the prediction results of ANN, SVM, LASSO, and other machine learning models. In this way, Hyun et al. [60] improve the wirelength prediction given by the virtual placement and routing in the synthesis. Cheng et al. [27] predict the final circuit performance in the macro placement stage, and Li and Franzon [82] predict the circuit performance in the global routing stage, including congestion number, hold slack, area and power.

For sign-off timing analysis, Barboza et al. [8] use random forest to give the sign-off timing slack from hand-crafted features. Another research [67] works on sign-off timing analysis and use linear regression to fit the static timing analysis (STA) model, thus reduce the frequency that the incremental static timing analysis (iSTA) tool need to be called. Han et al. [52] propose SI for Free, a regression method to predict expensive signal integrity (SI) mode sign-off timing results by using cheap non-SI mode sign-off timing analysis. [68] propose golden timer extension (GTX), a framework to reduce mismatches between different sign-off timing analysis tools to obtain neither optimistic nor pessimistic results.

Lu et al. [102] employ GAN and RL for clock tree prediction. Flip flop distribution, clock net distribution, and trial routing results serve as input images. For feature extraction, GAN-CTS adopts transfer learning from a pre-trained ResNet-50 on the ImageNet dataset by adding fully-connected (FC) layers. A conditional GAN is utilized to optimize the clock tree synthesis, of which the generator is supervised by the regression model. An RL-based policy gradient algorithm is leveraged for the clock tree synthesis optimization.

**4.2.3 Placement Decision Making**. As the preliminary step of the placement, floorplanning aims to roughly determine the geometric relationship among circuit modules and to estimate the cost of the design. He et al. [53] explore the possibility of acquiring local search heuristics through a learning mechanism. More specifically, an agent has been trained using a novel deep Q-learning algorithm to perform a walk in the search space by selecting a candidate neighbor solution at each step, while avoiding introducing too much prior human knowledge during the search. Google [113] recently models chip placement as a sequential decision making problem and trains an RL policy to make placement decisions. During each episode, the RL agent lays the macro in order. After arranging macros, it utilizes the force-directed method for standard cell placement. GCN is adopted in this work to embed information related to macro features and the adjacency matrix of the netlist. Besides, FC layers are used to embed metadata. After the embedding of the macros, the graph and the metadata, another FC layer is applied for reward prediction. Such embedding is also fed into a deconvolution CNN model, called PolicyNet, to output the mask representing the current macro placement. The policy is optimized with RL to maximize the reward, which is the weighted average of wirelength and congestion.

### 4.3 Power Deliver Network Synthesis and IR Drop Predictions

Power delivery network (PDN) design is a complex iterative optimization task, which strongly influences the performance, area and cost of a chip. To reduce the design time, recent studies have paid attention to ML-based IR drop estimation, a time-consuming sub-task. Previous work usually adopts simulator-based IR analysis, which is challenged by the increasing complexity of chip design. IR drop can be divided into two categories: static and dynamic. Static IR drop is mainly caused by voltage deviation of the metal wires in the power grid, while dynamic IR drop is led by the switching behaviors and localized fluctuating currents. In IncPIRD [54], the authors employ XGBoost to conduct incremental prediction of static IR drop problem, which is to predict IR value changes caused by the modification of the floorplan. For dynamic IR drop estimation, Xie et al. [155] aim to predict the IR values of different locations and models IR drop estimation problem as a regression task. This work introduces a “maximum CNN” algorithm to solve the problem. Besides, PowerNet is designed to be transferable to new designs, while most previous studies train models for specific designs. A recent work [173] proposes an electromigration-induced IR drop analysis framework based on conditional GAN. The framework regards the time and selected electrical features as input images and outputs the voltage map. Another recent work [28] focuses on PDN synthesis in floorplan and placement stages. This paper designs a library of stitchable templates to represent the power grid in different layers. In the training phase, SA is adopted to choose a template. In the inference phase, MLP and CNN are used to choose the template for floorplan and placement stages, respectively. Cao et al. [16] use hybrid surrogate modeling (HSM) that combines SVM, ANN and MARS to predict the bump inductance that represents the quality of the power delivery network.

### 4.4 Design Challenges for 3D Integration

3D integration is gaining more attention as a promising approach to further improve the integration density. It has been widely applied in memory fabrication by stacking memory over logic.

Different from the 2D design, 3D integration introduces die-to-die variation, which does not exist in 2D modeling. The data or clock path may cross different dies in through-silicon via (TSV)-based 3D IC. Therefore, the conventional variation modeling methods, such as on-chip variation (OCV), advanced OCV (AOCV), parametric OCV (POCV), are not able to accurately capture the path delay [131]. Samal et al. [131] use MARS to model the path delay variation in 3D ICs.

3D integration also brings challenges to the design optimization due to the expanded design space and the overhead of design evaluation. To tackle these challenges, several studies [31, 122, 131] have utilized design space exploration methods based on machine learning to facilitate 3D integration optimization.

The state-of-the-art 3D placement methods [75, 121] perform bin-based tier partitioning on 2D placement and routing design. However, the bin-based partitioning can cause significant quality degradation to the 3D design because of the unawareness of the design hierarchy and technology. Considering the graph-like nature of the VLSI circuits, Lu et al. [103] proposed a GNN-based unsupervised framework (TP-GNN) for tier partitioning. TP-GNN first performs the hierarchy-aware edge contraction to acquire the clique-based graph where nodes within the same hierarchy can be contracted into supernodes. Moreover, the hierarchy and the timing information is included in the initial feature of each node before GNN training. Then the unsupervised GNN learning can be applied to general 3D design. After the GNN training, the weighted k-means clustering is performed on the clique-based graph for the tier assignment based on the learned representation. The proposed TP-GNN framework is validated on experiments of RISC-V based multi-core system and NETCARD from ISPD 2012 benchmark. The experiment results indicate 7.7% better wirelength, 27.4% higher effective frequency and 20.3% performance improvement.

### 4.5 Other Predictions

For other parameters, Chan et al. [17] adopt HSM to predict the embedded memory timing failure during initial floorplan design. Bian et al. [11] work on aging effect prediction for high-dimensional correlated on-chip variations using random forest.

### 4.6 Summary of Machine Learning for Logic Synthesis and Physical Design

We summarize recent studies on ML for logic synthesis and physical design in Table 3. For logic synthesis, researchers focus on predicting and evaluating the optimal synthesis flows. Currently, these studies optimize the synthesis flow based on the primitives of existing tools. In the future, we expect to see more advanced algorithms for logic synthesis be explored, and more metrics can be formulated to evaluate the results of logic synthesis. Besides, applying machine learning to logic synthesis for emerging technologies is also an interesting direction.

In the physical design stage, recent studies mainly aim to improve the efficiency and accuracy by predicting the related information that traditionally needs further simulation. A popular practice is to formulate the EDA task as a computer vision (CV) task. In the future, we expect to see more studies that incorporate advanced techniques (e.g., neural architecture search, automatic feature generation, unsupervised learning) to achieve better routing and placement results.

## 5. Lithography and Mask Synthesis

Lithography is a key step in semiconductor manufacturing, which turns the designed circuit and layout into real objects. Two popular research directions are lithography hotspot detection and mask optimization. To improve yield, lithography hotspot detection is introduced after the physical implementation flow to identify process-sensitive patterns prior to the manufacturing. The complete optical simulation is always time-consuming, so it is necessary to analyze the routed layout by machine learning to reduce lithography hotspots in the early stages. Mask optimization tries to compensate diffraction information loss of design patterns such that the remaining pattern after lithography is as close to the design patterns as possible. Mask optimization plays an important role in VLSI design and fabrication flow, which is a very complicated optimization problem with high verification costs caused by expensive lithography simulation. Unlike the hotspot detection studies in Section 5.1 that take placement & routing stages into consideration, mask optimization focuses only on the lithography process, ensuring that the fabricated chip matches the designed layout. Optical proximity correction (OPC) and sub-resolution assist feature (SRAF) insertion are two main methods to optimize the mask and improve the printability of the target pattern.

### 5.1 Lithography Hotspot Detection

For lithography hotspot detection, Ding et al. [32] uses SVM for hotspot detection and small neural network for routing path prediction on each grid. To achieve better feature representation, Yang et al. [162] introduces feature tensor extraction, which is aware of the spatial relations of layout patterns. This work develops a batch-biased learning algorithm, which provides better trade-offs between accuracy and false alarms. Besides, there are also attempts to check inter-layer failures with deep learning solutions. A representative solution is proposed by Yang et al. [161]. They employ an adaptive squish layout representation for efficient metal-to-via failure check. Different layout-friendly neural network architectures are also investigated these include vanilla VGG [160], shallow CNN [162] and binary ResNet [65].

With the increased chip complexity, traditional deep learning/machine learning-based solutions are facing challenges from both runtime and detection accuracy. Chen et al. [22] recently propose an end-to-end trainable object detection model for large scale hotspot detection. The framework takes the input of a full/large-scale layout design and localizes the area that hotspots might occur (see Figure 6). In [44], an attention-based CNN with inception-based backbone is developed for better feature embeddings.

### 5.2 Machine Learning for Optical Proximity Correction

For OPC, inverse lithography technique (ILT) and model-based OPC are two representative mask optimization methodologies, and each of which has its own advantages and disadvantages. Yang et al. [163] propose a heterogeneous OPC framework that assists mask layout optimization, where a deterministic ML model is built to choose the appropriate one from multiple OPC solutions for a given design, as shown in Figure 7.

With the improvement of semiconductor technology and the scaling down of ICs, traditional OPC methodologies are becoming more and more complicated and time-consuming. Yang et al.[159] propose a new OPC method based on generative adversarial network (GAN). A Generator(G) is used to generate the mask pattern from the target pattern, and a discriminator (D) is used to estimate the quality of the generated mask. GAN-OPC can avoid complicated computation in ILT-based OPC, but it faces the problem that the algorithm is hard to converge. To deal with this problem, ILT-guided pre-training is proposed. In the pre-training stage, the D network is replaced with the ILT convolution model, and only the G network is trained. After pre-training, the ILT model that has huge cost is removed, and the whole GAN is trained. The training flow of GAN-OPC and ILT-guided pre-training is shown in Figure 8. The experimental results show that the GAN-based methodology can accelerate ILT based OPC significantly and generate more accurate mask patterns.

Traditional ILT-based OPC methods are costly and result in highly complex masks where many rectangular variable-shaped-beam (VSB) shots exist. To solve this problem, Jiang et al. [64] propose an ML-based OPC algorithm named neural-ILT, which uses a neural network to replace the costly ILT process. The loss function is specially designed to reduce the mask complexity, which gives punishment to complicated output mask patterns. In addition, for fast litho-simulation, a CUDA-based accelerator is proposed as well, which can save 96% simulation time. The experimental results show that neural-ILT achieves a 70× speedup and 0.43× mask complexity compared with traditional ILT methods.

Recently, Chen et al. [20] propose DAMO, an end-to-end OPC framework to tackle the full-chip scale. The lithography simulator and mask generator share the same deep conditional GAN (DCGAN), which is dedicatedly designed and can provide a competitively high resolution. The proposed DCGAN adopts UNet++ [176] backbone and adds residual blocks at the bottleneck of UNet++. To further apply DAMO on full-chip layouts, a coarse-to-fine window splitting algorithm is proposed. First, it locates the regions of high via density and then runs KMeans++ algorithm on each cluster containing the via pattern to find the best splitting window. Results on ISPD 2019 full-chip layout show that DAMO outperforms state-of-the-art OPC solutions in both academia [43] and an industrial toolkit.

### 5.3 Machine Learning for SRAF Insertion

Several studies have investigated ML-aided SRAF insertion techniques. Xu et al. [158] propose an SRAF insertion framework based on ML techniques. Geng et al. [43] propose a framework with a better feature extraction strategy. Figure 9 shows the feature extraction stage. After their concentric circle area sampling (CCAS) method, high-dimension features 𝑥𝑡 are mapped into a discriminative low-dimension features 𝑦𝑡 through dictionary training by multiplication of an atom matrix 𝐷. The atom matrix is the dictionary consists of representative atoms of the original features. Then, the sparse codes 𝑦𝑡 are used as the input of a machine learning model, more specifically, a logistic regression model that outputs a probability map indicating whether SRAF should be inserted at each grid. Then, the authors formulate and solve the SRAF insertion problem as an integer linear programming based on the probability grid and various SRAF design rules.

### 5.4 Machine Learning for Lithography Simulation

There are also studies that focus on fast simulation of the tedious lithography process. Traditional lithography simulation contains multiple steps, such as optical model building, resist model building, and resist pattern generation. LithoGAN [165] proposes an end-to-end lithography modeling method by using GAN, of which the framework is shown in Figure 10. Specifically, a conditional GAN is trained to map the mask pattern to a resist pattern. However, due to the characteristic of GAN, the generated shape pattern is good, while the position of the pattern is not precise. To tackle this problem, LithoGAN adopts a conditional GAN for shape modeling and a CNN for center prediction. The experimental results show that LithoGAN can predict the resist pattern with high accuracy, and this algorithm can reduce the lithography simulation time for several orders of magnitude. [20] is also equipped with a machine learning-based lithography simulator that can output via contours accurately to assist via-oriented OPC.

### 5.5 Summary

This section reviews ML techniques used in the design for manufacturability stage that include lithography hotspot detection, mask optimization and lithography modeling. Related studies are summarized in Table 4.

## 6. Analog Design

Despite the promotion of digital circuits, the analog counterpart is still irreplaceable in applications like nature signal processing, high speed I/O and drive electronics [126]. Unlike digital circuit design, analog design demands lots of manual work and expert knowledge, which often makes it the bottleneck of the job. For example, the analog/digital converter and Radio Frequency (RF) transceiver only occupy a small fraction of area but cost the majority of design efforts in a typical mixed-signal System-on-Chip (SoC), compared to other digital processors [129].

The reason for the discrepancy can be summarized as follows: 1) Analog circuits have a larger design space in terms of device size and topology than digital circuits. Sophisticated efforts are required to achieve satisfactory results. 2) The specifications of analog design are variable for different applications. It is difficult to construct a uniform framework to evaluate and optimize different analog designs. 3) Analog signals are more susceptible to noise and process-voltage-temperature variations, which cost additional efforts in validation and verification.

### 6.1 The Design Flow of Analog Circuits

Gielen and Rutenbar [45] provide the design flow followed by most analog designers. As shown in Figure 11, it includes both top-down design steps from system level to device-level optimizations and bottom-up layout synthesis and verification. In the top-down flow, designers choose proper topology, which satisfies system specifications in the circuit level. Then device sizes are optimized in the device level. The topology design and device sizing constitute the pre-layout design. After the schematic is well-designed, designers draw the layout of the circuit. Then they extract parasitics from the layout and simulate the circuit with parasitics. This is known as post-layout simulations. If the post-layout simulation fails to satisfy the specifications, designers need to resize the parameters and repeat the process again. This process can go for many iterations before the layout is done [136].

Although analog design automation has improved significantly over the past few decades, automatic tools cannot replace manual work in the design flow [10] yet. Recently, researchers are trying to introduce machine learning techniques to solve analog design problems. Their attempts range from topology selection at the circuit level to device sizing at the device level as well as the analog layout in the physical level.

### 6.2 Machine Learning for Circuit Topology Design Automation

6.2.1 Topology Selection.

6.2.2 Topological Feature Extraction.

6.2.3 Topology Generation.

### 6.3 Machine Learning for Device Sizing Automation

6.3.1 Reinforcement Learning Based Device Sizing.

6.3.2 Artificial Neural Network Based Device Sizing.

6.3.3 Machine Learning Based Prediction Methods.

6.3.4 Comparison and Discussion on Device Sizing.

### 6.4 Machine Learning for Analog Layout

### 6.5 Conclusion of Analog Design

## 7. Verification and Testing

### 7.1 Machine Learning for Test Set Redundancy Reduction

7.1.1 Test Set Redundancy Reduction for Digital Design Verification.

7.1.2 Test Set Redundancy Reduction for Analog/RF Design Testing.

7.1.3 Test Set Redundancy Reduction for Semiconductor Technology Testing.

### 7.2 Machine Learning for Test & Diagnosis Complexity Reduction

7.2.1 Test Complexity Reduction for Digital Design.

7.2.2 Verification Diagnosis Complexity Reduction for Digital Design.

7.2.3 Verification & Test Complexity Reduction for Analog/RF Design.

### 7.3 Summary of ML for Verification and Testing

## 8. Other Related Studies

### 8.1 Power Prediction

### 8.2 Machine Learning for SAT Solver

### 8.3 Acceleration with Deep Learning Engine

### 8.4 Auto-tuning design flow

## 9. Discussion From The Machine Learning Perspective

In this section, we revisit some aforementioned research studies from an ML-application perspective.

本节中，我们从ML应用的角度，重新回顾了一些之前提到的研究。

### 9.1 The Functionality of ML

Section 2.2 introduces the major ML models and algorithms used in EDA problems. Based on the functionality of ML in the EDA workflow, we can group most researches into four categories: decision making in traditional methods, performance prediction, black-box optimization, and automated design.

2.2节介绍了在EDA问题中使用的主要的ML模型和算法。基于ML在EDA工作流中的主要功能，我们将主要研究分成四种类别：传统方法中的决策算法，性能预测，黑盒优化，和自动设计。

**Decision making in traditional methods**. The configurations of EDA tools, including the choice of algorithm or hyper-parameters, have a strong impact on the efficiency of the procedure and quality of the outcome. This class of researches utilizes ML models to replace brute-force or empirical methods when deciding configurations. ML has been used to select among available tool-chains for logic synthesis [115, 167] , mask synthesis [163], and topology selection in analog design [111, 117, 137]. ML has also been exploited to select hyper-parameters for non-ML algorithms such as Simulated Annealing, Genetic Algorithm, etc. (refer to Section 3.2.2).

**传统方法中的决策算法**。EDA工具的配置，包括算法或超参数的选择，对输出的过程和质量的效率有很强的影响。当决定配置时，经典的研究利用ML模型来替代暴力或经验方法。ML已经被用于选择可用的工具链进行逻辑综合，掩膜综合，模拟设计中的拓扑选择。ML也被利用来选择非ML算法的超参数，比如模拟退火，遗传算法，等。

**Performance prediction**. This type of tasks mainly use supervised or unsupervised learning algorithms. Classification, regression and generative models are trained by former cases in real production to estimate QoR rapidly, to assist engineers to drop unqualified designs without time-consuming simulation or synthesis.

**性能预测**。这种类型的任务主要使用监督学习或无监督学习算法。用之前的案例来在实际生产中训练分类，回归和生成式模型，来迅速估计QoR，来帮助工程师丢弃不合格的设计，而不用耗时的仿真或综合。

ML-based performance prediction is a very common type of ML application. Typical applications of this type include congestion prediction in placement & routing and hotspot detection in manufacturability estimation (Table 8). The most commonly-used models are Linear Regression, Random Forests, XGBoost, and prevailing CNNs.

基于ML的性能预测是非常常见类型的ML应用。这种类型的典型应用包括布局布线中的拥堵预测，和可制造性估计中的热点检测（表8）。最常用的模型包括，线性回归，随机森林，XGBoost，和流行的CNNs。

**Black-box optimization**. This type of tasks mainly use active learning. Many tasks in EDA are DSE, i.e., searching for an optimal (single- or multi-objective) design point in a design space. Leveraging ML in these problems usually yields black-box optimization, which means that the search for optimum is guided by a surrogate ML model, not an explicit analytical model or hill-climbing techniques. The ML model learns from previously-explored design points and guides the search direction by making predictions on new design points. Different from the first category, the ML model is trained in an active-learning process rather than on a static dataset, and the inputs are usually a set of configurable parameters rather than results from other design stages.

黑盒优化。这种类型的任务主要使用主动学习。EDA中的很多任务是DSE，即，在设计空间中搜索一个最优的设计点（单目标或多目标的）。在这些问题中利用ML通常会得到黑盒优化问题，这意味着搜索最优是由代理ML模型引导的，并不是一个显式的解析模型，或爬坡技术。ML模型从之前探索过的设计点中学习，通过在新设计点上进行预测，来引导搜索方向。与第一种类别不同，ML模型的训练，是一个主动学习过程，而不是在一个静态数据集上，输入通常是可配置的参数集，而不是从其他设计阶段的结果。

Black-box optimization is widely used for DSE in many EDA problems. Related ML theories and how to combine with the EDA domain knowledge are extensively studied in literature. Typical applications of this type include tuning HLS-level parameters and physical parameters of 3D integration (see Table 8). The key techniques are to find an underlying surrogate model and a search strategy to sample new design points. Options of the surrogate model include GP, along with all the models used in performance prediction [105, 112]. Search strategies are usually heuristics from domain knowledge, including uniformly random exploration [95], exploring the most uncertain designs [180], exploring and eliminating the worst designs [112], etc.

在很多EDA问题中，黑盒优化广泛的用于DSE。相关的ML理论和怎样与EDA领域知识结合，在文献中进行了广泛的研究。这种类型的典型应用包括，调节HLS级的参数，和3D集成的物理参数（见表8）。关键技术是找到潜在的代理模型和搜索策略，来采样新的设计点。代理模型的选择包括GP，和在性能预测中用到的所有模型。搜索策略通常是领域知识中的启发式，包括均匀随机搜索，探索最不确定的设计，探索和消除最坏的设计，等。

**Automated design**. Some studies leverage AI to automate design tasks that rely heavily on human efforts. Typical applications are placement [113] and analog device sizing [134, 147, 148]. At first look it is similar to black-box optimization, but we highlight the differences as:

自动化设计：一些研究利用AI来自动化设计任务，这些任务严重依赖于人的努力。典型的应用是布局，和模拟device sizing。第一眼看上去，与黑盒优化很像，但其差异是：

- The design space can be larger and more complex, for example in placement, the locations of all the cells. 设计空间会更大更复杂，比如在布局中，是所有单元的位置。

- Instead of searching in the decision space, there exists a trainable decision-making policy that outputs the decisions, which is usually learned with RL techniques. 并不是在决策空间中搜索，而是有一种可训练的决策策略，输出决策，通常是通过RL技术来学习的。

More complicated algorithms with large volumes of parameters, such as deep reinforcement learning, are used in these problems. This stream of researches show the potential to fully automate IC design.

带有更大体量参数的更复杂算法，比如深度强化学习，也用在了这些问题中。这条线的研究展示了完全自动化IC设计的潜力。

Table 8 summarizes representative work of each category and typical model settings in terms of algorithm, input and output.

表8总结了每个类别的代表性工作，和典型的模型设置，包括算法，输入和输出。

### 9.2 Data Preparation

The volume and quality of the dataset are essential to model performance. Almost all studies we review make some discussions on leveraging EDA domain knowledge to engineer a large, fair and clean dataset.

数据集的体量和质量，对模型性能是非常关键的。我们回顾的几乎所有研究，都会对利用EDA领域知识来生成一个大型的合理的干净的数据集来进行一些讨论。

**Raw data collection**. Raw features and ground truth / labels are two types of data needed by ML models. Raw feature extraction is often a problem-specific design, but there are some shared heuristics. Some studies treat the layout as images and leverage image processing algorithms [32, 89, 154]. Some choose geometric or graph-based features from the netlist [150]. Some use traditional algorithms to generate features [6, 67, 106, 154]. Quite a lot studies choose features manually [6, 11, 16, 17, 27, 82, 115]. To some extend, manual feature selection lacks a theoretical guarantee or practical guidance for other problems. The labels or ground truth are acquired through time-consuming simulation or synthesis. This also drives researchers to improve data efficiency by carefully architect their models and preprocess input features, or use semi-supervised techniques [25] to expand the dataset.

原始数据收集。原始特征和真值/标签是ML模型需要的两种类型的数据。原始特征提取通常是一个问题特定的设计，但是有一些共享的启发式。一些研究将layout视为图像，利用图像处理算法。一些从网表中选择几何特征或基于图的特征。一些使用传统算法来生成特征。相当多的研究是手工选择特征的。在一定程度上，手工特征选择缺少理论保证或对其他问题的实践引导。标签或真值是通过耗时的仿真或综合获得的。这也推动研究者通过仔细的设计模型架构，或预处理输入特征，去改进数据效率，或使用半监督的技术，来扩展数据集。

**Feature preprocessing**. Standard practices like feature normalization and edge data removal are commonly used in the preprocessing stage. Some studies also use dimension reduction techniques like PCA and LDA to further adjust input features [60].

特征预处理。标准的实践如特征归一化，和边缘数据移除，在预处理阶段是常用的。一些研究还使用降维技术，如PCA和LDA，来进一步调整输入特征。

### 9.3 Domain Transfer

There have been consistent efforts to make ML-based solutions more adaptive to domain shift, so as to save training from scratch for every new task. Some researches propose ML models that take specifications of the new application domain and predict results in new domain based on results acquired in original domain. This idea is used in cross-platform performance estimation of FPGA design instances [109, 116]. It would be more exciting to train AI agents to adapt to new task without preliminary information of the new domain, and recent studies show that Reinforcement Learning (RL) might be a promising approach. RL models pre-trained on one task is able to perform nicely on new tasks after a fine-tune training on the new domain [113, 134, 147], which costs much less time than training from scratch and sometimes lead to even better results.

使基于ML的解决方案对领域的变化更加适应，一直都有这方面的努力，这样就可以不用对每个新任务，都从头进行训练。一些研究者提出ML模型，以新应用领域的指标为输入，基于原始领域获得的结果，预测新领域的结果。这个观点在FPGA设计实例的跨平台性能估计中得到了应用。训练AI agents，不需要新领域的初步知识，就可以适应到新任务，这会更加令人激动，最近的研究表明，强化学习可能是一个有希望的方法。在一个任务上预训练的RL模型可以在新任务上经过在新领域的精调训练，就可以表现的很好，这比从头训练的耗时要少很多，有时甚至会带来更好的结果。

## 10. Conclusion and Future Work

It is promising to apply machine learning techniques in accelerating EDA tasks. In this way, the EDA tools can learn from previous experiences and solve the problem at hand more efficiently. So far machine learning techniques have found their applications in almost all stages of the EDA hierarchy. In this paper, we have provided a comprehensive review of the literature from both the EDA and the ML perspectives.

应用机器学习技术加速EDA任务是很有希望的。在这方面，EDA工具可以从之前的经验中进行学习，更高效的求解手头的问题。迄今为止，机器学习技术在EDA的几乎所有阶段都有其应用。本文中，我们从EDA的角度和ML的角度给出了文献的综合回顾。

Although remarkable progress has been made in the field, we are looking forward to more studies on applying ML for EDA tasks from the following aspects. 虽然在这个领域已经有了很多进展，我们期望在下面的方面有更多的研究将ML应用到EDA任务中。

- **Towards full-fledged ML-powered EDA tools**. In many tasks (e.g., analog/RF testing, physical design), the performance of purely using machine learning models is still difficult to meet the industrial needs. Therefore, smart combination of machine learning and the traditional method is of great importance. Current machine learning aided EDA methods may be still restricted to less flexible design spaces, or aim at solving a simplified problem. New models and algorithms are desired to be developed to make the ML models more useful in real applications.

完全成熟的ML赋能的EDA工具。在很多任务中（如，模拟/RF测试，物理设计），纯使用机器学习模型的性能，仍然很难满足工业应用。因此，机器学习和传统方法的结合是非常重要的。目前的机器学习辅助的EDA方法仍然受限于没那么灵活的设计空间，或目标在于求解一个简化的问题。要开发新的模型和算法，使ML模型在实际应用中更加有用。

- **Application of new ML techniques**. Very recently, some new machine learning models and methodologies (e.g., point cloud and GCN) and machine learning techniques (e.g., domain adaptation and reinforcement learning) begin to find their application in the EDA field. We expect to see a broader application of these techniques in the near future.

新ML技术的应用。最近，一些新的机器学习模型和方法学（如，点云和GCN）和机器学习技术（如，领域自适应和强化学习），开始在EDA领域中找到了其应用。我们期望看到这些技术在未来有更广泛的应用。

- **Trusted Machine Learning**. While ML holds the promise of delivering valuable insights and knowledge into the EDA flow, broad adoption of ML will rely heavily on the ability to trust their predictions/outputs. For instance, our trust in technology is based on our understanding of how it works and our assessment of its safety and reliability. To trust a decision made by an algorithm or a machine learning model, circuit designers or EDA tool users need to know that it is reliable and fair, and that it will cause no harm. We expect to see more research along this line making our automatic tool trusted.

受信的机器学习。ML有希望给出EDA流的宝贵的洞见和知识，但是广泛的采用ML，肯定依赖于信任其预测/输出的能力。比如，我们对科技的信任，是基于我们对其怎样工作的理解，和我们对其安全性和可靠性的评估。为信任算法或机器学习模型做出的决策，电路设计者或EDA工具使用者需要知道，这是可靠的，合理的，并且不会导致任何危害。我们期望看到更多的这条线的研究，使我们的自动工具被信任。