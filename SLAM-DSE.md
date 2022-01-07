# Application-oriented Design Space Exploration for SLAM Algorithms

Sajad Saeedi et. al. @ Imperial College London

## 0. Abstract

In visual SLAM, there are many software and hardware parameters, such as algorithmic thresholds and GPU frequency, that need to be tuned; however, this tuning should also take into account the structure and motion of the camera. In this paper, we determine the complexity of the structure and motion with a few parameters calculated using information theory. Depending on this complexity and the desired performance metrics, suitable parameters are explored and determined. Additionally, based on the proposed structure and motion parameters, several applications are presented, including a novel active SLAM approach which guides the camera in such a way that the SLAM algorithm achieves the desired performance metrics. Real-world and simulated experimental results demonstrate the effectiveness of the proposed design space and its applications.

在视觉SLAM中，有很多软件和硬件参数，比如算法阈值和GPU频率，都需要进行调整；但是，这种调整还应当把相机的结构和运动纳入考虑。本文中，我们用几个参数确定结构和运动的复杂性，这几个参数是用信息论计算得到的。依赖于这个复杂度和期望的性能度量，探索并确定了合适的参数。另外，基于提出的结构，和运动参数，给出了几个应用，包括一种新型主动SLAM方法，其引导相机的方法会使SLAM算法获得期望的性能度量。真实世界的试验结果和仿真实现结果都表明提出的设计空间及其应用的有效性。

## 1. Introduction

Recently within the Simultaneous Localization and Mapping (SLAM) and robot vision community, it has been a controversial issue whether SLAM is solved or not. To answer this question, we need to consider three main factors as defined by Cadena et al. in [1]: robot, environment, and performance. In other words, the answer depends on the robot (its motion, resources, batteries, sensors, ...), the environment (indoor, outdoor, dynamic, ...), and the required performance (the desired accuracy, success rate, latency, ...). For instance, 2D grid-based SLAM in indoor environments with a required reconstruction error of below 0.01 m could be considered solved. Similarly, visual SLAM is also considered almost solved, but in some applications, when the robot has very fast dynamics or the environment is highly dynamic, the performance of the mapping and localization degrades. Therefore, research on SLAM is entering a new era where robust performance and application-oriented SLAM is the focus.

最近在SLAM和机器人视觉团体中，有一个有争议的问题，SLAM是否被解决掉了。为回答这个问题，我们需要考虑Cadena等[1]定义的三个主要的因素：机器人，环境和性能。换句话说，答案取决于机器人（其运动，资源，电池，传感器。。。），环境（室内，室外，动态。。。），和需要的性能（期望的准确率，成功率，延迟。。。）。比如，在室内环境中基于2D网格的SLAM，要求重建误差在0.01m以下，可以认为是已经解决了。类似的，视觉SLAM也被认为是几乎解决了，但在一些应用中，当机器人运动很快，或环境非常动态，建图和定位的性能会下降。因此，对SLAM的研究进入了一个新时代，其中稳健的性能和面向应用的SLAM是其核心。

There are several different discrete paradigms for SLAM algorithms, including sparse [2], semi-dense [3], dense [4], and semantic [5]. At the next level, there are possible major choices between components of these algorithms (e.g. type of feature, type of surface representation, etc), and finally, parameter choices within a particular algorithm. The choice of the algorithm is dependent on the application, the available resources, and the required performance metrics. There have been measures and benchmarks for SLAM systems for several years now, and these have been widely used to compare and tune the performance of different algorithms and systems. The majority of these have concentrated on accuracy; mainly of trajectory, because that is straightforward to independently measure, but sometimes of mapping accuracy too.

SLAM算法有几种不同的范式，包括稀疏的，半密集的，密集的，和语义的。下一级，有这些算法的组成部分的可能的主要选择（如，特征的类型，表面表示的类型，等），最后，特定算法中的参数选择。算法的选择是依赖于应用的，可用的资源，和需要的性能度量的。对于SLAM系统，有度量和基准测试几年了，广泛用于比较和调节不同算法和系统的性能。这些中大部分都聚焦于准确率；主要是轨迹trajectory，因为这可以很直接的独立独立，但有时候也有建图准确率。

However, the performance of a SLAM algorithm on an accuracy benchmark actually tells us little about how useful it would be for a particular application. SLAMBench [6] showed how the usefulness of benchmarks could be broadened in an important dimension by considering efficiency of performance on different computing platforms. A SLAM algorithm which is useful for a high accuracy industrial mapping application is almost certainly not the right choice for a low power embedded platform like a drone. This has started to open up research on Design Space Exploration (DSE) [7], where a high level search is made through the possible operating parameters of a SLAM system, in order to find the combinations which work best in terms of an appropriate compromise between accuracy and efficiency. In general, the results of DSE are represented by a Pareto front of possible operating points, where each point on the front represents an optimum set of parameters given the desired performance metrics. But still, the scene and motion are fixed in SLAMBench; all variations of algorithms are tested on a certain synthetic scene dataset with a certain camera motion.

但是，SLAM算法在一个准确率基准测试上的性能，与其在一个具体应用中有多好用，关系不是很大。SLAMBench展示了基准测试的用处怎样被推广，考虑了在不同的计算平台上的性能效率。一个在工业建图应用中准确率很高的SLAM算法，几乎肯定不会适用于低功耗的嵌入式平台，比如无人机。这就开启了设计空间探索(DSE)的研究，对SLAM系统的可能操作参数进行了很多高层次搜索，找到最好的参数组合，以在准确率和效率之间做出合适的折中。总体上，DSE的结果是由可能的操作点的Pareto front来表示的，其中front上的每个点都表示参数的一个最优集合，可以得到理想的性能度量。但是，在SLAMBench中的场景和运动是固定的；算法的所有变化，都是在特定的合成场景数据集上测试的，有特定的相机运动。

In reality, different applications need to work in different environments; and have varying specifications with regard to motion. If a drone must use visual SLAM to navigate through a forest, it will be flying fast past complex, nearby trees; while a robot vacuum cleaner navigates rather slowly on a ground plane, but must deal with a scene which is often distant and textureless. How can we perform design space exploration for SLAM systems as a whole, taking into account this range of applications with different constraints and requirements? It would seem that the specific qualities of the motion and structure involved in an application would need many parameters to specify — the typical linear and rotational velocities and accelerations of motion; the structure complexity of the scene; the average scene depth; the level of texture, and so on.

实际中，不同的应用需要在不同的环境中工作；对不同的运动有不同的指标。如果一个无人机必须使用视觉SLAM导航通过一片森林，就会很快的飞过复杂的附近的树；而一个机器人真空清洁器则很慢的在一个地平面上导航，但需要处理的场景可能很远，而且没有纹理。我们怎样对SLAM系统进行整体的设计空间探索呢，考虑到众多的应用，有着不同的约束和需求？似乎运动和结构的特定质量，需要很多参数来指定，典型的有线性和旋转的速度、加速度；场景的结构复杂度；平均场景深度；纹理的水平，等。

The hypothesis of this paper is that we can use a small set of parameters as a very useful proxy for a full description of the setting and motion of a SLAM application. We call these Motion and Structure (MS) parameters, and define them based on information theory. Specifying and searching through MS parameters in design space exploration allows us to focus within the wide range of possible operating points represented by an accuracy/efficiency Pareto front. Using the MS parameters, we are able to identify how challenging the environment is with a given camera motion, and thus choose a set of more suitable hardware and software parameters from the Pareto front. One of the applications of the proposed MS parameters, as shown in this paper, is the active SLAM with robotic platforms (Fig. 1). Unlike other information theoretic methods, such those which try to maximize mutual information or information gain [8] for rapid exploration, in our method we propose to limit the information divergence, to ensure that the SLAM system is robust with respect to the structure of the observed scene.

本文的假设是，我们可以用很少参数作为一个SLAM应用的运动和设置的完整描述的代理。我们称之为运动和结构参数(MS)，基于信息论对其进行定义。指定MS参数并在其中搜索，使我们可以聚焦在可能的操作点的很广的范围内的准确率/效率Pareto front上。使用MS参数，我们可以在给定的相机运动中，识别环境是多么有挑战，因此从Pareto front中选择一个更合适的硬件和软件参数集。提出的MS参数的一个应用，是主动SLAM（图1）。其他信息论方法大多试图最大化互信息或信息增益，以进行快速探索，我们的方法与之不同，我们提出限制信息散度，以保证SLAM系统对观察到的场景的结构是稳健的。

### 1.1. Contributions

The contributions of this work are as follows: 本文贡献如下：

• Introducing a comprehensive design space, including motion and structure space, for real-time applications, 提出了一种综合的设计空间，包括运动和结构空间，用于实时应用；

• Parameterising the motion and structure space with information theory, and 用信息论对运动和结构空间进行参数化；

• Proposing several applications based on the MS parameters, including an active SLAM algorithm. 基于MS参数提出了几种应用，包括一种主动SLAM算法。

The rest of the paper is organised as follows: Section II presents background and literature review. In Section III the proposed motion and structure space is introduced. In Section IV the design space exploration is explained. In Section V, several applications of the proposed motion and structure space are presented. In Section VI, experiments are presented, and in Section VII, conclusions and future works are presented.

本文组织如下：第2部分给出背景和文献综述，第3部分介绍了提出的运动和结构空间，第4部分解释了设计空间探索，第5部分，提出的运动和结构空间的几种应用，在第6部分，给出了试验，在第7部分，给出了结论和未来的工作。

## 2. Background and Literature Review

In this section, three topics are presented: performance metrics, design space exploration, and information theory. 本节给出三个主题：性能度量，设计空间探索，和信息理论。

### 2.1. Performance Metrics

SLAM algorithms are compared based on various performance metrics such as accuracy, robustness, processing cost, and etc [9], [10], [11], [12]. Strum et al. improve trajectory metrics, absolute trajectory error (ATE) and relative pose error (RPE), by evaluating the root mean squared error over all time indices of the translational components [11].

SLAM算法的比较是基于各种性能度量的，比如准确率，稳健性，处理代价等等。Strum等改进了轨迹度量，绝对轨迹误差(ATE)，和相对姿态误差(RPE)，将平移组成部分的所有时间索引的均方误差进行计算得到。

Other important metrics are related to the quality of the map, such as reconstruction completeness (RCM), defined as the reconstructed percentage of the ground truth points [13], and reconstruction error (RER), defined as the error between the reconstructed and the ground truth map. As an example, in ElasticFusion [4], where the map is shown by surfels, RER is determined by running iterative closest point (ICP) algorithm on the point cloud models of the world and the map. The error of the point cloud matching is used as RER.

其他重要的度量与地图的质量相关，比如重建完整性(RCM)，定义为真值点的重建百分比，重建误差(RE)，定义为重建地图和真值地图的误差。ElasticFusion[4]是一个例子，其中地图用表面基元进行展示，RER是对世界和地图的点云模型运行ICP算法而得到。点云匹配的误差用作RER。

Execution time (EXT), memory usage (MEM), and energy consumption (ENE) per frame are other important metrics which are usually taken into account in real-world applications and on mobile devices. 每帧的执行时间(EXT)，内存使用(MEM)，和能量消耗(ENE)是其他重要的度量，在真实世界应用和在移动设备上的应用会考虑这些度量。

### 2.2. Design Space Exploration

The design parameters of a SLAM algorithm are categorised as either software parameters, including algorithmic and compiler parameters, or hardware parameters. SLAM算法的设计参数归类为软件参数，或硬件参数，其中软件参数包括算法和编译器参数。

Algorithmic parameters are algorithm dependent. For instance, in KinectFusion [14], the ICP convergence threshold, volume resolution, and pyramid level iterations are such algorithmic parameters. Compiler parameters operate at the compiler level and affect the way that the hardware executes the algorithm. Vectorisation and compiler flags for the precision of mathematical operations are examples of such parameters. Hardware parameters include the number of active CPU cores and the GPU processor frequency. By proper selection and tuning of these parameters, the objective is to achieve the desired performance metrics; however, the augmented hardware and software variables form a large vector that is not easy to tune manually. Additionally, there are multiple choices for the desired parameters which are shown by a Pareto front. Figure 7 demonstrates the Pareto front highlighted in green, where each point on the Pareto front is an optimal answer. For every non-Pareto point, there is a point on the front which is better in at least one metric. A user can choose the desired point from the front depending on the trade-off between metrics.

算法参数是依赖于算法的。比如，在KinectFusion中，ICP收敛阈值，体分辨率，和金字塔层级迭代都是这样的算法参数。编译器参数在编译器层级操作，影响硬件执行算法的方式。矢量化和用于数学运算的精度的编译器标志，是这种参数的例子。硬件参数包括，活跃的CPU核的数量，和GPU处理器的频率。合理的选择和调节这些参数，来获得期望的性能度量；但是，扩增的硬件和软件变量，形成了很大的向量，手工调节并不容易。另外，期望的参数有多种选择，表现为一个Pareto front。图7展示了Pareto front，用绿色进行了高亮，Pareto front上的每个点，都是一个最优解。对于每个非Pareto点，在front上都有一个点，至少在一个度量上要更好一些。用户可以从front上选择期望的点，在各种度量中进行折中权衡。

In the recent paper of SLAMBench [6], the idea of adopting the KinectFusion algorithm to run on four different platforms with default algorithmic parameters was proposed. SLAMBench uses ICL-NUIM dataset [15] to do experiment.

在SLAMBench的最近文章中，提出了采用KinectFusion算法在四个不同的平台上运行的思想，采用默认的算法参数。SLAMBench使用ICL-NUIM数据集来进行试验。

Bodin et al. proposed the idea of design space exploration (DSE) which tries to optimise the hardware and software parameters to achieve some of the desired performance metrics, including ATE, ENE, and EXT [7]. The methodology of their work is based on quantifying these indices by playing the KinectFusion algorithm using the ICL-NUIM dataset on two different platforms and exploring the design space parameters.

Bodin等提出DSE的思想，尝试优化硬件和软件参数，获得一些期望的性能度量，包括ATE，ENE和EXT。他们的工作的方法是基于量化这些索引，在ICL-NUIM数据集中，在两种不同的平台上，运行KinectFusion算法，探索设计空间参数。

Zia et al. apply a similar concept in [16], but at a small scale, to only algorithmic parameters of the KinectFusion and LSD-SLAM [3] algorithms. They have done their experiments using ICL-NUIM and TUM-RGBD [11] datasets.

Zia等在[16]中应用了类似的概念，但其规模较小，只是对KinectFusion和LSD-SLAM算法的算法参数应用。他们用ICL-NUIM和TUM-RGBD数据集进行了试验。

### 2.3. Information Divergence

Information theory and its concepts such as entropy and mutual information has many applications in robotics and perception, including path planing [17], SLAM [18], and exploration [19]. In this paper, information divergence is used to assess the quality of mapping and localization.

信息理论及其概念，比如熵和互信息，在机器人学和感知中有很多应用，包括路径规划，SLAM，和探索。本文中，信息散度用于衡量建图和定位的质量。

In information theory, information divergence, which is a measure of difference between two probability distribution functions, has been used in many different fields such as image processing, speech processing, and machine learning [20].

在信息论中，信息散度是两种概率分布函数的差异的度量，在很多不同的领域中得到了应用，如图像处理，语音处理和机器学习。

As an information divergence measure, the Kullback-Leibler divergence also called KL divergence or relative entropy, is a natural distance measure that uses Shannon’s entropy. For a discrete random variable with dimension d, such as X = (X1, . . . , Xd) ∈ R^d with a probability distribution function of p(x1, . . . , xd), the entropy is defined as:

作为一种信息散度度量，Kullback-Leibler散度也称为KL散度或相对熵，是一种自然距离度量，使用了香农熵。对于d维的离散随机变量，比如X = (X1, . . . , Xd) ∈ R^d，其概率分布函数为p(x1, . . . , xd)，其熵定义为

$$H(X) = \sum_{x_1,...,x_d} -p(x_1,...,x_d) logp(x_1,...,x_d)$$(1)

If the random variables X_i, i = 1, . . . , d are independent, equation (1) becomes: 如果随机变量X_i是独立的，式(1)就变成

$$H(X) = \sum_{i=1,...,d} H(X_i)$$(2)

If X_i s are independent and identically distributed, H(X) is 如果X_i是独立同分布的，H(X)就成为

$$H(X) = dH(X_i)$$(3)

Entropy dH(Xi) is the upper bound for the entropy that can be achieved. In other words, the upper bound for H(X) is when Xis are independent and identically distributed. Similarly, by extending the definition in equation (1), the relative entropy or KL divergence distance for two distributions, p(X) and q(X), is defined as:

熵dH(X_i)是可以获得的熵的上限。换句话说，H(X)的上限是当X_i独立同分布时得到的。类似的，通过拓展式(1)的定义，两个分布p(X)和q(X)的KL散度距离，或相对熵可以定义为：

$$δ(p||q) = \sum_{x_1,...,x_d} p(x_1,..., x_d) log \frac {p(x_1,..., x_d)}{q(x_1,..., x_d)}$$(4)

When p(·) and q(·) are equal, the distance is zero. 当p(·)和q(·)相等时，距离为0。

## 3. Motion and Structure Parameter Space

This section explains the Motion and Structure (MS) design space. If a camera mounted on a quadrotor experiences a sudden change in the view of the scene due to the fast dynamics of the quadrotor, depending on the depth of the scene, the SLAM algorithm may fail or succeed to process the following frames because tracking is difficult when sequential images have very different appearances. Therefore it is important to quantify the limits of the physical motions in different environments. In other words, it is desired to represent this complex dependency of motion and structure with a minimum number of parameters which are also easy to compute. For sparse SLAM, Civera et al. achieved this goal by decomposing the state space into metric parameters and dimensionless parameters [21]. The dimensionless parameters are used to tune the SLAM filter without any assumption about the scene. The structure of the scene also offers important cues as to which parameters to use, and we later address this.

本节解释了运动和结构设计空间。如果一个相机安装在一个旋翼飞行器上，因为飞行器的快速运动，经历了场景视野的一个突然变化，依赖于场景的深度，SLAM算法可能会失败，或成功的处理后续的帧，因为在顺序图像的外观非常不同时，跟踪很困难。因此，在不同环境中量化物理运动的极限，非常重要。换句话说，用最少数量的参数来表示这种复杂的运动和结构依赖关系，就非常理想，也很容易计算。对于稀疏的SLAM，Civera等将状态空间分解成度量参数，和无维度的参数，达到了这个目标。无维度参数用于调节SLAM滤波器，对场景没有任何假设。场景的结构，也为使用哪些参数提供了重要的线索，我们后面处理这个问题。

One way to take into account the behaviour of the motion in a structure is to refer to the sensory data. The information gained, from one frame to another, tells us about the motion of the camera relative to the environment. As it is shown, there is a correlation between the change of the information from one frame to the next, and the desired performance metrics. Extremely high rates of change will result in failure of SLAM, as expected. The MS design space identifies the maximum change permitted for a SLAM algorithm to achieve the desired performance metrics.

一种考虑结构中的运动行为的方法，是参考传感器数据。从一帧到另一帧得到的信息，告诉了我们相机相对于环境的运动。从一帧到下一帧之间的信息变化，与期望的性能度量，有一种相关性。极高的变化速度会导致SLAM失败。MS设计空间会给出SLAM算法允许的最大变化，以得到期望的性能度量。

### 3.1. Divergence of Sensor Information

In the rest of the work, it is assumed that the sensor operates in a realistic environment, i.e. the sensor is not blind, and the structure has minimum texture to be mapped. If images are modelled by probability distributions, by knowing the magnitude of divergence in information from one distribution to another, we are able to determine the motion of the sensor in an environment. In an extreme case, a zero divergence means there is no motion. A large divergence may indicate that either the sensor is moving very fast, or the environment has rapidly varying structure.

在本文的剩余部分中，假设传感器处于理想环境中，即，传感器并不是盲的，结构要映射的纹理非常少。如果图像通过概率分布进行建模，知道一个分布与另一个分布之间的信息散度的幅度，我们就可以确定传感器在环境中的运动。在极端的情况，零散度意味着没有运动。大散度意味着传感器运动速度很快，或环境有快速变化的结构。

**1) Approximate KL Divergence for Intensity Images**: We treat an image as a very high-dimensional discrete random variable. An approximate probabilistic model can then be generated by assuming that pixels are individually independent random variables. The reason that this is an approximate model is that in practice the pixels are correlated through the geometry of the environment; however, modelling the geometry is not a trivial task. In this work, for intensity images, an approximate probability distribution model is generated by making a normalised histogram of the intensities of the pixels. This is similar to the model that Shannon created to model English words [22]. The key is that the normalised histogram is an estimate of the underlying probability of each pixel’s intensity.

灰度图像的近似KL散度：我们将图像当做高维离散随机变量。假设这些像素是独立的随机变量，就可以生成一个近似的概率模型。这是一个近似模型的原因是，在实践中，像素是通过环境的几何关联起来的；但是，对几何进行建模是一个比较复杂的工作。在本文中，对于灰度图像，通过得到像素灰度的归一化直方图，可以生成一个近似的概率分布模型。这与香农创建的对英语单词建模的模型类似。关键部分是，归一化的直方图每个像素灰度的潜在概率的估计。

For two intensity images, It and It−1, the normalised histogram of intensity values is considered as their distribution functions. Typically for intensity images, P = 256 bins are considered, where each bin is associated with an integer u = 0, ..., 255. If the distributions of the images are indicated by It and It−1, the intensity information divergence is:

对于两幅灰度图像，It和It-1，灰度值的归一化直方图被认为是其概率函数。对典型的灰度图像，考虑P=256 bins的情况，其中每个bin都有一个整数相关u = 0, ..., 255。如果图像的分布表示为It和It−1，灰度信息散度为

$$δ_I(t) = KL(I_t||I_{t−1}) = \sum_{u=0}^P I_t(u) log \frac{I_t(u)}{I_{t-1}(u)}$$(5)

where subscript I indicates that the distribution and the divergence distance are derived from the intensity images, and δI is the KL divergence. I(u) is the u-th bin of distribution I.

其中下表I表示分布和散度距离是从灰度图像中推导出来的，δI是KL散度。I(u)是分布I的第u个bin。

**2) Approximate KL Divergence for Depth Images**: To create depth distributions, depth values could also be binned similarly; however, this is not trivial given the unbounded range of these depth values. Instead, for two consecutive depth images, Dt and Dt−1, their probability distribution functions are defined using normal vectors of the depth points. To generate the distributions, the method used in [23] has been adopted. First, from the depth images, for each point, a normal vector is calculated. Then the normal vectors are binned to created a histogram. To bin the normal vectors, a unit sphere is partitioned into equal patches (see Fig. 3). Patches are identified by their central azimuth and inclination angles. To have equally distributed patches, regularly-spaced inclination angles are selected by dividing the inclination range (i e. [0, π]) to N equally-distributed angles (equation (6)). For each inclination, the azimuth angles are selected by dividing the azimuth range (i e. [0, 2π]) to M equal angles (equation (7)). Note that as we get closer to the poles, M decreases. The ⌊·⌋ sign denotes the integer part operation.

深度图像的近似KL散度：为创建深度分布，深度值也可以类似的放到bin中；但是，因为深度值的范围是不受限的，这个问题就有些复杂了。对于两个连续的深度图像，Dt和Dt-1，其概率分布函数是使用深度点的法线矢量来定义的。为生成这个分布，采用了[23]中的方法。第一，从深度图像中，对每个点都计算一个法线矢量。然后这些法线矢量放到bin中，创建一个直方图。为将法线矢量放到bin中，将一个单位球分割成相等的小块（见图3）。这些块由其方位角和倾角来识别。为为得到相等分布的块，将倾角范围[0, π]分割成N个相等分布的角（式6），选择得到规则间隔的倾角。对于每个倾角，方位角的选择是将方位角范围即[0, 2π]分割成M个相等的角度（式7）。注意，当我们接近极地，M减小。⌊·⌋符号表示整数部分运算。

$$θ_i = πi/N, i = 1,...,N$$(6)

$$φ_j (θ_i) = 2πj/M, j = 1,...,M, M = ⌊2N sin θ_i⌋ + 1$$(7)

Once the bins are created, the k-th normal vector n_k, k = 1..L, contributes to bin i, j based on the angle between nk and vij, where vij represents bin i, j in Cartesian coordinates:

一旦创建了这些bins，第k个法线向量n_k, k = 1..L, 对bin i, j有贡献，基于在nk和vij之间的角度，其中vij表示在笛卡尔坐标系中的bin i, j

$$w^k_{ij} = \left\{ \begin{array} 0 & if cos^{-1}(n_k·v_{ij})>α \\ n_j·v_{ij}-cosα/1-cosα & else \end{array} \right.$$(8)

In equation (8), α is the angular range of influence for each bin. Based on these weights, the spherical distribution is:

在式8中，α是每个bin影响的角度范围。基于这些权重，球形分布是：

$$D(i,j) = \sum_{k=1}^L w_{i,j}^k, i=1,...,N, j=1,...,M$$(9)

After calculating the contribution of all normals to the bins, the histogram is normalized to sum to one. For two distributions, Dt and Dt−1, the depth information divergence, δD, is:

在计算了所有法线对bins的分布后，直方图进行归一化。对于两个分布，Dt和Dt−1，深度信息散度δD是：

$$δD(t)=KL(Dt||Dt−1) = \sum_{i,j} D_t(i,j) log \frac{D_t(i,j)}{D_{t-1}(i,j)}$$(10)

### 3.2 Motion and Structure Design Space

There is a direct relationship between the KL divergence distance of intensity and depth images, and the performance metrics. A larger divergence means more outliers during image alignment, which introduces more error. If we want to perform better outlier rejection by relying on more iterations or more accurate algorithms, like RANSAC, the hardware requirements increase. In general, the relationship between the divergence and the metrics is not easily proved analytically, and thus it has been shown experimentally here.

灰度和深度图像的KL散度距离，和性能度量之间，有直接的关系。更大的散度，意味着在图像对齐时更多的离群点，这会带来更大的误差。如果我们进行更好的离群点拒绝，依赖于更多的迭代，或更准确的算法，如RANSAC，硬件的需求就会增加。总体上，散度和度量之间的关系，要进行解析的证明，并不容易，所以这里通过试验进行证明。

To efficiently represent the MS design space with information divergence, for a trajectory of T frames (1..T ), the maximum information divergences for intensity and depth images, M_I and M_D, are introduced:

为用信息散度高效的表示MS设计空间，对于T帧的轨迹(1,...,T)，引入灰度和深度图像的最大信息散度，M_I和M_D：

$$M_I = max(δ_I (t) |_{t=1:T})$$(11)

$$M_D = max(δ_D (t) |_{t=1:T})$$(12)

To demonstrate the relationship between these divergence values and performance metrics, a dataset is tested using ElasticFusion [4]. To generate data streams with larger information divergence, frames were skipped in regular intervals. Then the absolute trajectory error (ATE) is calculated. Fig. 4 demonstrates the absolute trajectory error versus divergence for the ICL-NIUM dataset (stream lr kt1 and lr kt2). This figure shows that higher information divergence corresponds to higher trajectory error.

为表明散度值和性能度量之间的关系，使用ElasticFusion测试了一个数据集。为生成更大信息散度的数据流，以有规律的间隔跳过了帧，然后计算ATE。图4展示了在ICL-NIUM数据集上ATE与散度之间的关系。这张图表明，更高的信息散度对应着更高的轨迹误差。

These maximum divergence values parameterise the motion and structure simultaneously. In other words, for a desired ATE, the motion in a given structure should be such that the frame to frame divergence should not exceed these parameters.

这些最大散度值同时对运动和结构进行了参数化。换句话说，对于一个期望的ATE，在给定结构中的运动，应当是帧与帧之间的散度不应当超过这些参数。

## 4. Design Space Exploration

In this section, the design space exploration with the four design spaces, shown in Fig. 2, is explained. The design space exploration is performed with ElasticFusion [4] on an Intel machine. For simplicity, the experiments are performed only on algorithmic and MS parameters. To evaluate the design parameters, ATE and EXT performance metrics are calculated.

本节中，解释了图2所示的4个设计空间的设计空间探索。设计空间探索是用ElasticFusion在Intel机器上进行的。为简化，试验只在算法和MS参数上进行。为评估设计参数，计算了ATE和EXT性能度量。

### 4.1. Design Parameters

The ElasticFusion algorithm is parameterised by the following parameters. For a detailed description, please refer to original paper by Whelan et al. [4].

ElasticFusion算法由下列参数参数化。详细描述请参考Whelan等的原始文章。

- Depth cutoff: Cutoff distance for depth processing. Range: [0 − 10] m, default: 3 m. 截止深度：深度处理的截止距离。范围：[0 - 10]m，默认：3m。

- ICP/RGB tracking weight: This weight determines the ratio of ICP to RGB tracking in visual odometry. Range: [0 − 1], default: 0.1. 追踪权重：这个权重决定视觉里程计中ICP到RGB追踪中的比率。范围：[0 - 1]，默认：0.1。

- Confidence threshold: Surfel confidence threshold. Range: [0 − 12], default: 10. 置信阈值：表面基元置信阈值。范围：[0 - 12]，默认：10。

For the motion and structure parameters, maximum intensity information divergence, and maximum depth information divergence, are used. These two parameters were introduced in Equations (11) and (12). To determine these parameters, a dataset sequence was played with frames being dropped at different rates, and the maximum information divergence was calculated across that sequence. Dropping frames actually occurs in real-world applications, i.e. when there is limited buffer or processing resource, the unprocessed frames are simply discarded. While the parameters for the algorithmic, hardware, and compiler domains were generated in advance, the parameters for the MS space is produced on the fly.

对于运动和结构参数，使用了最大灰度信息散度和最大深度信息散度。这两个参数在式(11)(12)中引入。为确定这些参数，一个数据集序列以不同的速率丢弃帧，最大信息散度在这个序列中计算。丢弃帧在真实世界应用中真的会发生，即，当buffer或处理资源有限时，未处理的帧就简单的被丢弃。算法、硬件和编译器域中的参数是提前生成的，MS空间的参数是在计算中产生的。

### 4.2. Procedure

We wish to determine the Pareto front for those parameters which are defined above. To generate one point on the Pareto plane, we first generate a random sample of the algorithmic space parameters. We then specify a frame drop rate, and by running the algorithm with these parameters on the corresponding image sequence, the EXT and ATE metrics are calculated, together with the corresponding MS parameters (maximum information divergence). This process continues, each time adding a Pareto point, until we have the Pareto front determined. The Pareto front is later used to specify design space parameters based on the trade-off between different performance metrics.

我们希望确定上面定义的这些参数的Pareto front。为在Pareto平面上生成一个点，我们首先生成算法空间参数的随机样本。我们然后指定一个帧丢弃率，用这些参数在对应的图像序列中运行算法，计算出对应的EXT和ATE度量，和对应的MS参数一起（最大信息散度）。这个过程继续下去，每次加上一个Pareto点，直到我们确定了Pareto front。这个Pareto front后面用于指定设计空间参数，基于不同性能度量之间的折中。

## 5. Applications of Design Space Exploration

In this section, four different scenarios are presented which show how the proposed MS parameters and design space exploration are used in real-world applications to meet the objectives of a mission or limitation of the resources. These scenarios are active frame management, run-time adaptation, dataset difficulty level assessment, and active SLAM. Of these, the active SLAM algorithm is explained in detail and some experimental results are presented in the next section.

本节中，给出了四种不同的场景，展示了提出的MS参数和设计空间探索怎样用在真实世界应用中，满足一个任务的目标，或资源的限制。这些场景是active frame management, run-time adaptation, dataset difficulty level assessment, 和active SLAM。其中，active SLAM算法进行了详细解释，一些试验结果在下一节中给出。

### 5.1. Active Frame Management

In real-world applications, optimising resources such as battery is very important. One of the applications of the design space exploration is the ability to decide when to process a frame. If two consecutive frames are statistically very similar, by processing them, we are able to gain more confidence in the map and the pose of the camera; however, this is at the cost of spending other important resources such as battery. In this situation, it is desirable to simply drop the second frame to save the battery. Obviously when there are unlimited resources, it is desirable to process all frames. To manage frames actively, for each frame its information divergence with respect to the previous frame is calculated. If the divergence is less than a threshold, the frame is not passed to the SLAM pipeline. The threshold can be dynamic and could be a function of the available resources such as battery or the processing resources.

在真实世界应用中，优化资源比如电池，是非常重要的。设计空间探索的一个应用，是决定什么时候处理一帧的能力。如果两个连续的帧在统计上非常像，处理它们，我们可以在地图和相机的姿态上得到更高的置信度；但是，其代价是使用了其他重要的资源，比如电池。在这种情况下，简单的丢弃第二帧，节省电池，是更好的选择。很明显，当有无限的资源时，处理所有帧，是理想的情况。为主动的管理帧，对于每一帧，计算其对前一帧信息散度。如果散度小于一个阈值，这一帧就不传送给SLAM流程。阈值可以是动态的，可以是现有可用资源的函数，比如电量，或处理资源。

### 5.2. Run-time Adaptation

Assume that after the design space exploration, a set of parameters have been identified from the Pareto front; these guarantee acceptable performance metrics according to a defined maximum information divergence. If for any reason, the information divergence is higher than the expected values, there is risk of having poor performance. However, this can be counteracted by choosing another set of parameters, which may require higher allocation and consumption of the available resources, but can deal with the higher divergence. In other words, using the proposed method, it is possible to have multiple sets of parameters, and in extreme situations, our method can easily switch from one set of parameters to another to guarantee the required performance metrics.

假设在设计空间探索以后，从Pareto front中识别到一些参数集合；根据定义的最大信息散度，这确保了可接受的性能度量。如果因为任何原因，信息散度高于期望的值，那么就有危险性能会比较差。但是，可以选择另一个集合的参数来应对，会需要更高的可用资源的配置和消耗，但可以处理更高散度的情况。换句话说，使用提出的方法，可能会有多个集合的参数，在极限情况下，我们的方法可以很容易的从一个集合参数切换到另一个，以确保需要的性能度量。

### 5.3. Dataset Difficulty Level Assessment

When proposing a new SLAM algorithm, a de facto is to compare the results with other algorithms by testing them on known datasets. So far there is no measure to assess the difficulty level of the datasets, and thus, the comparison with datasets may not be able to reveal all strengths or weaknesses of a new SLAM algorithm. As a standard metric, the proposed information divergence, without considering software and hardware parameters, can easily be used to assess the difficulty of different datasets. This can be achieved by assigning statistics of the information divergence, such as mean and variance, to the sequence of the data in each dataset. Additionally, the divergence metrics can be evaluated per unit of time. Table I shows some of these statistics for ICL-NUIM datasets (only intensity divergence for simplicity). According to [4], in ICL-NUIM, datasets lr kt2 and lr kt3 are more difficult than lr kt0 and lr kt1 based on the reported performance metrics. These difficult trajectories have a higher difficulty score.

当提出一种新的SLAM算法时，将其与其他算法进行比较，就要在已知的数据集上对其进行测试。目前为止，没有哪种度量来估计数据集的难度水平，因此，数据集的比较可能不能展示出新SLAM算法的优势或弱点。作为一种标准度量，提出的信息散度没有考虑软件参数和硬件参数，可以很容易用于评估不同数据集的难度。可以将信息散度的统计值，比如均值和方差，在每个数据集的数据序列中指定应用，来获得。另外，散度度量可以在每个时间单位上评估。表1展示了ICL-NUIM数据集上的一些统计值（简化起见，只展示了灰度散度）。根据[4]，在ICL-NUIM中，基于报告出的性能度量，数据集lr kt2和lr kt3比lr kt0和lr kt1要难的多。这种困难的轨迹有更高的难度分数。

### 5.4. Active SLAM with Information Divergence

Active SLAM, also known as active vision, view path planning (VPP), or next best-view (NBV), is the problem of determining the optimal camera motion (in some sense) to perform mapping [24]. Active SLAM is closely related to the exploration problem [25], where the objective is to map an unknown environment completely.

主动SLAM，也称为主动视觉，观察路径计划(VPP)，或下一个最好的视角(NBV)，是决定最佳相机运动以进行建图的问题。主动SLAM与探索问题紧密相关，其目标是对未知环境完全建图。

There are several works that perform active SLAM with sensors such as lasers for 2D/3D mapping [26], [27], but Davison and Murray were the first who integrated motion with stereo visual SLAM [28] where their objective was to minimize the trajectory error. Most active SLAM algorithms are either based on maximizing mutual information [8], [29], or maximizing information gain [30], [31] for increased coverage, decreased pose uncertainty, or dense mapping purposes. But our objective is to maintain robustness and achieve the desired performance metrics by controlling the incoming information flow. In other words, we guide the camera such that the information divergence is not more than the permitted maximum information divergence defined in equations (11) and (12).

有几个工作进行主动SLAM，用的传感器比如激光器，用于2D/3D建图，但Davison和Murray是第一个将运动与立体视觉SLAM结合到一起的，其目标是最小化轨迹误差。多数主动SLAM算法是基于最大化互信息的，或对增加的覆盖，降低的姿态不确定性，或密集建图目的，最大化获得信息的。但我们的目标是，通过控制来到的信息流，保持稳健性，获得期望的性能度量。换句话说，我们引导相机，这样信息散度不高于式(11)和(12)允许的最大信息散度。

**1) Active SLAM based on Information Divergence**: Fig. 5 shows the block diagram of the system. The SLAM block implements the ElasticFusion algorithm [4]. The resulting pose and map are used in the motion planning block. MI and MD are the MS parameters that are used for motion planning. The most recent images, It−1 and Dt−1, together with the predicted next images, generated from the current map, are used to determine the next best waypoint for the controller block. The controller guides the robot using inverse kinematics. The details of this controller are beyond the scope of the paper.

基于信息散度的主动SLAM：图5展示了系统的模块图。SLAM模块实现的是ElasticFusion算法。得到的姿态和地图用在运动规划模块中。MI和MD是用于运动规划的MS参数。最近的图像，It-1和Dt-1，和从当前的地图生成的预测的下一幅图像一起，用于确定控制器模块下一个最佳的waypoint。控制器引导机器人使用inverse kinematics。这个控制器的细节超出了本文的范围。

Algorithm 1 explains the proposed motion planning in detail. Inputs to the algorithm are the previous intensity and depth images, (It−1, Dt−1), the previous pose and map estimates, pt−1, mt−1, and the maximum allowed intensity and depth divergence parameters, (MI, MD). Based on these inputs, the algorithm determines the best rotation and translation, T, to maintain the information divergence below the threshold.

算法1详细解释了提出的运动规划。算法的输入是前一个灰度和深度图像(It−1, Dt−1)，前一个姿态和地图估计，pt−1, mt−1, 和最大允许的灰度和深度散度参数，(MI, MD)。基于这些输入，算法确定最佳的旋转和平移T，以保持信息散度低于阈值。

In line 1, ∆I and ∆D, which contain divergence values for candidate poses, are initialised. In line 2, the space around the current pose is decomposed to reachable rotation and translation motions. The decomposed space includes seven translations along the axes of the current local frame: no translation, up, down, left, right, forward, and backward. For each translation, there are seven rotations in the local frame including no rotation, roll right, roll left, pitch forward, pitch backward, yaw anti-clockwise, and yaw clockwise. T contains the set of rotations and translations for the decomposed space. With this simple decomposition, there are 49 elements in T. In line 4, the candidate global poses of the camera, given the previous pose and the next potential pose transformations, are calculated. In line 5, for each of the candidate poses, depth and intensity images are predicted by projecting the current map, mt−1, on the camera plane. ($\hat I^i_t, \hat D^i_t$) are predicted intensity and depth images for the i-th candidate pose $\hat p^i_t$. In lines 6 and 7, for each of the predicted images, the divergence with respect to the last intensity and depth images are calculated. ∆_I(i) and ∆_D(i) contain the corresponding divergences for the i-th candidate pose. Given the predicted images and their divergences, in line 9, a pair of predicted intensity and depth images are chosen which has the optimum divergence distance to the divergence parameters. In this line, two thresholds is introduced, defined as a percentage of the maximum allowed intensity and depth information divergence, denoted by ρ_I and ρ_D. Note that these two thresholds control the exploratory behavior of the algorithm. If these parameters are zero, the algorithm wants to keep the camera almost stationary, and if they are set to 1, the algorithm wants to move the camera to the locations where the image will provide maximum allowed information, defined by MI and MD. Also, λ in this line is a weight parameter, used to adjust the significance of depth over intensity in the optimisation. Since the criterion has a finite number of elements, i.e. only 49 different candidate poses, the optimization is performed exhaustively. Finally, in line 10, the rotation and translation commands associated with the chosen intensity and depth images, are selected and passed to the controller.

在第1行，∆I和∆D被初始化，包含候选姿态的散度值。在第2行，在目前姿态附近的空间被分解为可到达的旋转和平移运动。分解的空间包括7个平移，沿着当前局部框架的轴：没有平移，上，下，左，右，前，后。对每个平移，在局部框架中有7个旋转，包括没有旋转，右roll，左roll，前pitch，后pitch，逆时针yaw，顺时针yaw。T包含分解空间的旋转和平移集合。用这个简单的分解，在T中有49个元素。在第4行，在给定之前的姿态和下一个可能的姿态变换后，计算相机的候选全局姿态。在第5行，对每个候选姿态，将目前的地图mt-1投影到相机平面上，预测深度图像和灰度图像。($\hat I^i_t, \hat D^i_t$)是在第i个候选姿态$\hat p^i_t$下，预测的灰度和深度图像。在第6和第7行，对于每个预测的图像，计算对最后一幅灰度和深度图像的散度。∆_I(i)和∆_D(i)包含第i个候选姿态对应的散度。在第9行，在给定预测的图像和其散度后，选择一对预测的灰度和深度图像，与散度参数的散度距离为最优。在这一行中，引入了两个阈值，定义为最大允许的灰度和深度信息散度百分比，表示为ρ_I和ρ_D。注意，这两个阈值控制着算法的探索行为。如果这两个参数为0，算法希望使相机保持静止。如果这两个参数为1，算法希望将相机移动到的位置，图像会给出最大允许的信息，由MI和MD定义。还有，这一行中的λ是一个权重参数，用于调整在这个优化中深度相对于灰度的显著性。由于规则的元素数量有限，即，只有49个不同的候选姿态，优化是穷举式进行的。最后，在第10行，旋转和平移命令与选定的灰度和深度图像，送入到控制器中。

The proposed motion planning is a local algorithm and does not provide a global destination for the camera. To provide global planning, in line 9, by adding more constraints, the optimisation for the next motion can be combined with any globally planned trajectory. This allows us to guide the camera with global optimality, whilst providing acceptable information divergence.

提出的运动规划是一个局部算法，并不给相机提供全局的目的地。为给出全局规划，在第9行，通过加入更多的约束，对下一个运动的优化可以与任意全局规划的轨迹结合到一起。这使我们可以用全局最优性引导相机，而给出可接受的信息散度。

## 6. Experiments

In this section, we evaluate how our method can optimise parameters to achieve certain desired metrics. Then we provide in-depth exploration of the application to active SLAM, and present both simulated and real-world experiments with a camera mounted on a robotic arm.

本节中，我们评估我们的方法怎样优化参数，获得特定的期望度量。我们给出主动SLAM应用的深度探索，给出仿真的试验和相机安装在机械臂上的真实世界的试验。

### 6.1. Design Space Exploration

This experiment demonstrates the usefulness of DSE in providing better performance metrics using information divergence. Fig. 6 shows maximum ATE vs. EXT per frame for various divergence values in the ICL-NUIM dataset (stream lr tr0). In the legend, the highlighted marks have been sorted from the highest divergence (×) to the lowest (◦). In Fig. 6, as divergence increases, ATE and EXT increase.

这个试验证明了，在使用信息散度提供更好的性能度量上，DSE是有用的。图6展示了在ICL-NUIM数据集中(stream lr tr0)，在各种不同的散度值时，最大ATE vs. EXT每帧。在图标中，高亮的标记进行了排序，从最高的散度(×)到最低的(◦)。在图6中，随着散度增加，ATE和EXT增加。

Next, for one of the divergence values, DSE is implemented as explained in Section IV-B to find the suitable algorithmic parameters. For the point marked with ♦, maximum ATE is 2 cm, and EXT is approximately 0.038 s per frame. In Fig 7, this point has been shown by a black diamond as default parametric configuration. All other points show the results of DSE. The Pareto front has been shown by a green curve. Using DSE, the ATE for this divergence can be reduced down to 1 cm and EXT can be reduced to less than 0.02 s.

下一步，对于其中的一个散度值，实现了前述的DSE，以找到合适的算法参数。标记了♦的点，最大ATE是2cm，EXT是大约每帧0.038s。在图7中，这个点被标记了黑色菱形，作为默认的参数配置。所有其他点展示了DSE的结果。Pareto front表示为一条绿色曲线。使用DSE，这个散度的ATE可以降低到1cm，EXT可以降低到小于0.02s。

### 6.2. Active SLAM in Simulation

This experiment demonstrates the concept of performing active SLAM, in which the motion of the camera is controlled to adjust the information flow to the SLAM pipeline. In the simulation, a pair of intensity and depth images are rendered from a known world model (ICL-NUIM living room) given the current pose of the camera. These images are processed by SLAM, and also by the the motion planner to decide what the next pose of the camera should be. Once the next pose is known, the camera is guided to the desired pose, and the process of rendering images, SLAM, and motion planning continues recursively. To render images from the 3D model, Persistence Of Vision Raytracer, POVRay, is used. POVRay renders much more realistic images compared to similar tools such as Gazebo. In the simulation, two different motion planning algorithms are tested: random walk and the proposed active SLAM. In the random walk, for each frame, one transformation is chosen from the 49 different transformations available (combination of 7 translations and 7 rotations as explained in Section V-D), while in the active SLAM, a transformation that optimises the information divergence is chosen (Algorithm 1). Fig. 8 shows a demonstration of 49 different intensity image predictions and their divergence scores (depth images are not shown for the sake of brevity). The experiment was repeated twice (Table II). In the free motion experiment, rotation and translation were changing as explained. In the fixed translation experiment, the camera was translating along a straight line, and the rotation was optimised (or randomly selected). Table II compares the performance metrics for these experiments. The results show that the active SLAM generated better results in terms of performance metrics.

试验证明了进行主动SLAM的概念，其中相机的运动是为了调整到SLAM流程的信息流。在仿真中，从已知的世界模型(ICL-NUIM起居室)中，在给定相机的当前姿态下，渲染出一对灰度和深度图像。这些图像由SLAM处理，也有运动规划处理，来决定相机的下一个姿态应当是什么。一旦下一个姿态已知，相机引导到期望的姿态，渲染图像，SLAM和运动规划的过程迭代着持续进行。从3D模型中渲染图像，使用了POVRay。与类似的工具相比，如Gazebo，POVRay渲染的是真实的多的图像。在仿真中，两个不同的运动规划算法进行了测试：random walk和提出的主动SLAM。在random walk中，对每一帧，从49个不同的变换中选择一个变换（7个平移和7个旋转的组合），而在主动SLAM中，选择了一种优化信息散度的变换 （算法1）。图8展示了49种不同的灰度图像预测，及其散度值。这个试验重复了两次（表II）。在固定平移试验中，相机沿着一条直线进行平移，旋转被优化（或随机选择）。表II比较了这些试验的性能度量。结果表明，以性能度量来说，主动SLAM生成了更好的结果。

### 6.3. Active SLAM with Robotic Arm

This experiment demonstrates the active SLAM algorithm with a robotic arm. Fig 1 shows the Kinova Mico Arm used for active SLAM. An ASUS RGB-D camera was mounted on the arm, and as with the previous experiment, random walk and active SLAM (Algorithm 1) are compared.

这个试验展示了在一个机械臂上的主动SLAM算法。图1展示了用于主动SLAM的Kinova Mico臂。在机械臂上安装了一个ASUS RGB-D相机，与之前的试验一样，比较了random walk和主动SLAM算法（算法1）。

The experiments were done in four different environments, labelled as window, table, wall, and carpet. In each environment, each algorithm was run 10 times. Repeated experiments serve as a measure of the robustness of the algorithm in dealing with uncertainties rising from minor changes in illumination, or inaccuracies of the response of the controller or actuator to the commands.

试验在四种不同的环境中进行，标记为window, table, wall和carpet。在每种环境中，每种算法运行10次。重复的试验的作用是，可以衡量算法在处理各种不确定性的稳健性，比如光照的微小变化，controller或actuaor对命令的响应的不精确性等。

For the random walks, different initial seeds were used everytime. Due to the lack of ground truth information from the real environments, the consistency of the generated map was evaluated manually as either a success or failure of SLAM. If duplicates of one object were present in the map, it was considered as failure. The generated maps are available for inspection. Fig. 9 shows these results. As the figure demonstrates, in all four cases, active SLAM performs better than random walk. Particular performance difference is noted in the carpet experiment, where random walk failed in all 10 tries, and active SLAM succeeded in five out of ten tries by moving in and out and maintaining smaller information divergence than random walk.

对于random walks, 每次都使用不同的初始种子。由于缺少真实环境的真值信息，生成的地图的一致性进行了手动评估，是成功还是失败。如果一个目标在地图中有重复，那么就认为是失败。生成的地图可用于检视。图9给出了这些结果。如图所示，在所有四种情况中，主动SLAM都比random walk效果要好。特定的性能差异在carpet试验中进行了注释，其中random walk在所有10种尝试中都失败了，而主动SLAM在10个中的5个中成功了，其信息散度比random walk要小。

## 7. Conclusion and Future Work

This paper introduced a new domain for the design space exploration of the SLAM problem, called Motion and Structure (MS) space. The new domain is represented by parameters, calculated using information divergence, that can be used to meet the desired performance metrics. An active SLAM algorithm was also developed based on the MS parameters, and we showed how our method can be used to guide camera motion optimally to ensure robust performance. We also presented a design space exploration experiment which demonstrated that suitable MS parameters can be incorporated with other design space parameters, to yield a Pareto front.

本文提出了SLAM问题的设计空间探索的新领域，称为运动和结构空间。新领域由参数表示，使用信息散度计算，可以用于达到期望的性能度量。基于MS参数，提出了一种主动SLAM算法，我们展示了，我们的方法怎样用于最优的引导相机运动，以确保稳健的性能。我们还提出了一种设计空间探索试验，证明了合适的MS参数可以与其他设计空间参数一起使用，得到一个Pareto front。

In future work, we propose to use the information divergence metric to evaluate several other real-world robotic applications, including run-time adaptation. Another direction to explore is adding global path planning constraints to the active SLAM algorithm, to enable autonomous navigation as well as ensuring robust performance. Additionally, we are exploring improvements to the divergence measure, such as introducing spatial windowing across the image for histogram generation, and using the Earth mover’s distance to provide tolerance to small illumination changes.

在未来的工作中，我们提出要使用信息散度度量来评估几种其他的真实世界机器人应用，包括运行时调整。另一个方向的探索是，对主动SLAM算法加入全局路径规划约束，使自动导航成为可能，并能确保稳健的性能。另外，我们还在探索对散度度量的改进，比如引入在图像中的空域窗，来生成直方图，使用earth mover距离来对小的光照变化提供容忍性。