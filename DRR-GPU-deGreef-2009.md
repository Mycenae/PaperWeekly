# Accelerated ray tracing for radiotherapy dose calculations on a GPU

M. de Greef et. al. University of Amsterdam

**Purpose**: The graphical processing unit (GPU) on modern graphics cards offers the possibility of accelerating arithmetically intensive tasks. By splitting the work into a large number of independent jobs, order-of-magnitude speedups are reported. In this article, the possible speedup of PLATO’s ray tracing algorithm for dose calculations using a GPU is investigated.

**目的**：现代显卡上的GPU可以加速进行数学运算量很大的任务。通过将任务分割成大量独立的任务，可以得到数量级级别的加速。本文中，我们对，使用GPU对进行剂量计算的PLATO射线追踪算法进行加速，进行了研究。

**Methods**: A GPU version of the ray tracing algorithm was implemented using NVIDIA’s CUDA, which extends the standard C language with functionality to program graphics cards. The developed algorithm was compared based on the accuracy and speed to a multithreaded version of the PLATO ray tracing algorithm. This comparison was performed for three test geometries, a phantom and two radiotherapy planning CT datasets (a pelvic and a head-and-neck case). For each geometry, four different source positions were evaluated. In addition to this, for the head-and-neck case also a vertex field was evaluated.

**方法**：使用NVIDIA CUDA实现了GPU版的射线追踪算法，CUDA拓展了标准C语言，可以对显卡进行编程。提出的算法，与多线程版本的PLATO射线追踪算法，进行了准确率和速度的比较。这种比较在三种测试几何中进行，一个模体和两个放射治疗计划CT数据集（一个盆部的，一个头颈部的）。对于每种几何关系，评估了四种不同的源的位置。除了这个，对于头颈部的案例，还评估了一个vertex field的情况。

**Results**: The GPU algorithm was proven to be more accurate than the PLATO algorithm by elimination of the look-up table for z indices that introduces discretization errors in the reference algorithm. Speedups for ray tracing were found to be in the range of 2.1–10.1, relative to the multithreaded PLATO algorithm running four threads. For dose calculations the speedup measured was in the range of 1.5–6.2. For the speedup of both the ray tracing and the dose calculation, a strong dependency on the tested geometry was found. This dependency is related to the fraction of air within the patient’s bounding box resulting in idle threads.

**结果**：GPU算法被证明比PLATO算法更准确，因为不用对z方向的索引进行查找表的操作，这会在参考算法中带来离散化的误差。射线追踪算法的加速效果，大约有2.1倍到10.1倍，这是与多线程（4线程）PLATO算法相比的。对于剂量计算，加速的效果大约是1.5倍到6.2倍。对于射线追踪和剂量计算的加速效果，我们发现与测试的几何关系依赖性性很强。这种依赖性与病人的边界框中的空气部分的比例相关，因为会导致空闲的线程。

**Conclusions**: With the use of a GPU, ray tracing for dose calculations can be performed accurately in considerably less time. Ray tracing was accelerated, on average, with a factor of 6 for the evaluated cases. Dose calculation for a single beam can typically be carried out in 0.6–0.9 s for clinically realistic datasets. These findings can be used in conventional planning to enable (nearly) real-time dose calculations. Also the importance for treatment optimization techniques is evident. © 2009 American Association of Physicists in Medicine. (DOI: 10.1118/1.3190156)

**结论**：在使用GPU的时候，剂量计算用的射线追踪可以在少的多的时间内，进行精确的计算。射线追踪得到了加速，对于评估的案例，平均有6倍的加速。单个射野的剂量计算，在临床的真实数据集中，一般可以在0.6s-0.9s内计算完。这些发现可以用于传统计划中，可以达到接近实时的剂量计算。治疗优化的重要性是很明显的。

**Key words**: radiotherapy, dose calculation, GPU, graphics card, treatment planning

## 1. Introduction

Ray tracing is an important and time consuming step in dose calculations for radiotherapy. For a 3D CT dataset, the relative electron density has to be line integrated from the source position to every volume element (voxel) in that dataset. Moreover, for radiotherapy treatment planning, this process must be repeated for each source position (gantry angle) to determine the total dose in each voxel.

射线追踪在放射治疗的剂量计算中是很重要，非常耗时的一步。对于一个3D的CT数据集，其相对电子密度需要从源位置到每个体素在这个数据集中进行line integrated。而且，对于放射治疗计划，这个过程必须对每个源位置（机架角度）重复进行，以在每个体素中确定总计的剂量。

The traditional way of designing a treatment plan consists of a manual determination of the optimal beam configuration iteratively. Every adjustment requires recalculation. Typically, calculating the contribution of a single beam to the dose distribution requires 15 – 60 s of the calculation time. If dose calculations could be performed real time, a major reduction in workload is achieved. Furthermore, optimization of intensity modulated radiotherapy (IMRT) plans with variable beam angles is a computationally intensive task. This applies especially to recent developments such as volumetric modulated arc therapy, intensity modulated arc therapy, and/or tomotherapy. Speed improvement is also important in view of the growing amount of data acquired just prior or during treatment. Image-guided radiotherapy can lead to adaptive procedures with possible replanning during the course of treatment.

设计一个治疗计划的传统方法，包括手动迭代的确定最优射野配置。每个调整都需要重新计算。一般来说，计算单个射野对剂量的贡献需要15-60s的计算时间。如果剂量计算可以实时进行的话，可以大幅降低工作量。另外，变射野角度的IMRT计划的优化，其计算量非常大。这在最近的进展上尤其是这样，如volumetric modulated arc therapy, intensity modulated arc therapy, and/or tomotherapy。在治疗前、治疗中都会获得大量的数据，这样计算速度也会越来越重要。图像引导的放射治疗IGRT，可以带来自适应的治疗过程，可以在治疗过程中进行重新计划。

Multicore processors have now become common. However, faster execution can only be achieved using multithreading, which requires rewriting of the algorithm. Moreover, the speedup is related to the number of cores, leading to a maximum speedup of four times in practice.

多核处理器现在很常见。但是，更快的执行只能用多线程获得，这需要重新编写代码。而且，加速与核数量有关，实际中最大只能带来4倍的加速。

Another, rather expensive, option is using large CPU clusters. However, the associated overhead, which comes with network data transfer and thread scheduling, is large compared to the present execution time.

另一种更昂贵的选项，是使用大型CPU集群。但是，相关的开销，网络数据传输和线程调度，与现在的执行时间来比较，是很大的。

The ray tracing algorithm currently used in our department (as implemented in PLATO) is based on the algorithm proposed by Siddon. This algorithm is widely used in various radiotherapy treatment planning systems. Since its publication, the Siddon algorithm has been optimized for dose calculations in PLATO. However, there is still room for improvement for accurate (quasi) real-time user interaction and the advanced planning techniques mentioned earlier.

目前我们部门使用的光线追踪算法（在PLATO中实现的）是基于Siddon提出的算法。这个算法在各种放射治疗计划系统中广泛的使用。自从发表后，Siddon算法在PLATO中已经进行了优化。但是，对于准确的（准）实时用户交互和之前提到的高级的计划技术来说，仍然有改进的空间。

Since about 2000, the use of commodity graphics hardware for high-performance computing has become a subject of research for an increasing number of research groups in various fields. Designed for processing large amounts of data in parallel, the graphical processing unit (GPU) is used to speed up arithmetically intensive computational tasks. This use of the GPU outside its original context of computer graphics is referred to as general purpose computation on the GPU. The reported speedups are achieved by separating an arithmetically intensive task in a large number of lightweight tasks, threads, which operate on their own data. It is important that threads are not dependent on the (intermediate) results of other threads; this is referred to as data independence. Ray tracing for dose calculations is a good candidate for execution on the GPU. A large number of rays have to be traced and there is no dependency between these rays. The aim of this study is to investigate the possible acceleration of the presently used ray tracing algorithm for radiotherapy dose calculation using a GPU.

自从2000年以来，商用图形学硬件的使用，可以进行高性能计算，这在各种领域的越来越多的研究小组中都成了研究课题。GPU设计就是用于大量数据的并行运算，因此可以用于加速大量计算任务。这种在计算机图形学领域以外使用的GPU，称为GPU上的通用目标计算。通过将计算量很大的任务分解成大量轻量级任务，线程在其自己的数据上运算，就可以得到加速的效果了。线程不依赖于其他线程的中间结果，这非常重要；这被称为数据独立性。剂量计算的光线追踪是在GPU上运行的很好的候选。需要跟踪大量的射线，而且这些射线之间并没有相关性。本研究的目的是，研究目前放射治疗剂量计算中，应用的光线追踪算法，在GPU上可能的加速效果。

## 2. Materials and Methods

### 2.1 General purpose computation on the GPU

Since about 2000, the field of general purpose computation on GPUs is growing. Modern graphics cards have become highly programable and are developing fast regarding their computational capacities. Traditionally, toolkits for computer graphics such as OpenGL and DirectX were used to perform calculations outside of the computer graphics context. This approach is referred to as compute by drawing: A computational problem is translated into a graphical/ visualization problem. This requires implementation of algorithms in a graphical programming language.

自从2000年以来，在GPU上的通用目标计算领域的研究一直在增长。现代显卡已经是高度可编程的了，按照其计算能力来说，发展很快。传统上来说，计算机图形学的工具箱如OpenGL和DirectX，也用于过进行计算机图形学之外的计算。这种方法被称为通过drawing来进行计算：一个计算问题，转化为图形学/可视化问题。这需要算法以一种图形学编程的语言进行实现。

Apart from the compute-by-drawing strategy, there are a number of toolkits that allow programming on a more abstract level, closer to high-level programming languages. Examples of these languages are Brook, RapidMind, and CUDA (compute unified device architecture) by NVIDIA. In this study the latter language was used. CUDA is an extension of standard C that is preprocessed by a compiler driver and compiled using a C compiler. CUDA uses a data-parallel programming model where a large number of parallel threads are organized in a two-dimensional grid. Within a grid, blocks are specified that serve as a unit in resource assignment (Fig. 1).

除了这种compute-by-drawing的策略，还有一些工具箱，可以在更抽象的层次来进行编程，与高层编程语言更接近。这种语言的例子有Brook，RapidMind和CUDA。本研究中，我们使用了最后一种语言。CUDA是对标准C语言的一种延伸，由一个编译驱动器来预处理，用C编译器进行编译。CUDA使用一种数据并行的编程模型，其中大量并行线程是以一种二维网格的形式组织起来的。在一个网格中，指定了blocks，就是资源指定的单元（如图1所示）。

FIG. 1. The principle of a grid used to organize a number of parallel threads. A grid consists of a number of blocks that are identified by a block index. Within a block, the thread index identifies threads.

### 2.2 Siddon’s algorithm for exact calculation of the radiological path

Since the GPU ray tracing algorithm for dose calculations will be based on the algorithm by Siddon, the original algorithm is briefly summarized here. The algorithm is based on the representation of a 3D dataset as three sets of orthogonal planes. For a ray, traveling from the source to a point in the volume, the voxels through which the ray travels can be determined with the aid of these planes. The radiological path length (RPL) is found by summation of the length traveled by this ray in each voxel, multiplied by the relative electron density of the voxel. Figure 2 shows this procedure schematically. It should be noted that this method is different from methods that use numerical integration by sampling of the density along the ray trajectory. In contrast to the method used in this study, such a technique can show irregular behavior since the sampling distance has an impact on which voxels are taken into account and which ones are not.

因为用于剂量计算的GPU光线追踪算法是基于Siddon提出的算法，因此这里对原始算法进行简要介绍。算法是基于将一个3D数据集表示为三个正交的平面的。对于一条射线，从源到体中的一个点，射线穿过的体素可以通过这些平面的帮助来确定。放射路径长度(radiological path length, RPL)可以通过射线穿过的在每个体素中的长度，乘以体素的相对电子密度，然后求和得到。图2给出了这个过程的示意图。应当说明的是，这种方法，与沿着射线轨迹对密度进行采样，计算数值积分，是不一样的。与本研究中使用的方法形成对比的是，这种技术会表现出异常行为，因为采样距离对已经考虑的体素和没有考虑到的体素都有一些影响。

FIG. 2. Schematic 2D illustration of ray tracing as used in radiotherapy dose calculations. A ray is traced from the source to all the voxels in the dataset. The voxels a ray travels through on its way to a certain voxel in the dataset are collected so that the length of the fragments can be density weighted.

The path of a ray is represented using the following parameterization: 用下面的参数化形式来表示射线的路径：

$$x = x_1 + α(x_2 − x_1)$$(1)
$$y = y_1 + α(y_2 − y_1)$$(2)
$$z = z_1 + α(z_2 − z_1)$$(1)

where index 1 refers to the source position, index 2 to the center of the voxel the ray is traced to, and α is the distance traveled divided by the source-voxel distance. Now the dataset can be expressed as 下标1表示源位置，下标2表示射线到的体素的中心，α是射线穿越的距离处以源-体素距离。现在数据集可以表示为

$$X_i = X_0 + iΔX$$(4)
$$Y_j = Y_0 + jΔY$$(5)
$$Z_k = z_{plane}[k]$$(6)

where the use of capital symbols refers to the fact that the coordinates indicate plane positions, ΔX and ΔY are the voxel dimensions in the x and y directions respectively, i, j, and k are plane indices, and $X_0$ and $Y_0$ are the x and y coordinates of the planes that are defined by i and j equal to zero, respectively. In contrast to the original algorithm, the dataset is allowed to be nonequidistant in the z direction. The reason for this is that CT scans for radiotherapy treatment planning may be acquired with varying slice thickness. The plane positions are stored in array $z_{plane}$. From Eqs. (1)–(6), the value of α at an intersection of the ray with a plane can be easily calculated. For the intersection of the ray with planes, i, j, and k, it is found that

其中使用大写字母指的是，坐标说明了平面的位置，ΔX和ΔY分别是x和y方向的体素维度，i，j和k是平面的索引，$X_0$和$Y_0$分别是i和j等于0所定义的平面的x和y坐标。与原始算法相比，数据集在z方向是可以非等距的。这是因为，放射治疗计划中的CT scans可以通过不同的层厚来获得。平面位置存储在$z_{plane}$阵列中。从式(1)–(6)中，射线与平面相交处的α的值可以很容易的计算得到。对于射线与平面相交，其i, j和k，可以发现：

$$α_x(i) = \frac {X_0+iΔX-x_1} {x_2-x_1}$$(7)
$$α_y(j) = \frac {Y_0+jΔY-y_1} {y_2-y_1}$$(8)
$$α_z(k) = \frac {Z_k-z_1} {z_2-z_1}$$(9)

for α_x(i) and α_y(j), this can be written in recursive form as

$$α_x(i) = α_x(i-1)+\frac{ΔX}{x_2-x_1} = α_x(i-1)+δα_x$$(10)
$$α_y(j) = α_y(j-1)+\frac{ΔY}{y_2-y_1} = α_y(j-1)+δα_y$$(10)

Note that since the distance between two planes in the z direction is allowed to vary, such a recursive relation cannot be formulated for this direction. Using Eqs. (9)–(11), the values of α for all the intersections between the source point and the voxel the ray travels to can be calculated. This has to be done for every orthogonal plane direction and the results are stored in three different arrays

注意，由于z方向上的两个平面的距离是可以变化的，所以对于这个方向不能进行这样的迭代计算。用式(9-11)，源点和射线穿过的体素交点处α的值都可以计算得到。对于每个正交的平面都要做这样的计划，得到的结果存储在三个不同的阵列中

$$\underline{α}_x = \{ α_x(i_{min}),..., α_x(i_{max})\}$$(12)

$$\underline{α}_y = \{ α_y(j_{min}),..., α_y(j_{max})\}$$(13)

$$\underline{α}_z = \{ α_z(k_{min}),..., α_z(k_{max})\}$$(14)

where subscripts “min” and “max” denote the first and last
intersections of the ray with the orthogonal planes when traveling from the source to the target voxel. When merging these three lists together with αmin and αmax, this results in

下标min和max表示当从源到目标体素的过程中，射线与正交平面的第一个和最后一个相交点。当这三个列表与αmin、αmax合并，这就得到

$$\underline{α} = \{ α_{min}, merge(\underline{α}_x,\underline{α}_y,\underline{α}_z), α_{max} \}$$(15)

where the merge operation as described by Siddon creates an array with all the elements of αx, αy, and αz in ascending order. Two consecutive elements from α are associated with entering and leaving a voxel. The length traveled in a voxel is therefore given by

其中Siddon描述的合并运算，会创建一个数组，其中以升序包含了αx, αy和αz的所有元素。α中的两个连续的元素，是进入和离开一个体素的对应值。在一个体素中穿越的长度，由下式给出

$$l(m)=(\underline{α}(m+1)-\underline{α}(m))d_{12}$$(16)

where l(m) is the length traveled in the mth voxel the ray passes when traveling from the source to the target voxel, $d_{12}$ is the source-voxel distance. The voxel indices p and q can be calculated by

其中l(m)是射线在从源到目标体素的过程中，在第m个体素中射线穿越的距离，$d_{12}$是源到体素的距离。体素的索引p和q可以计算为：

$$p(m) = ⌊\frac {x_1+α_{mid}(x_2-x_1)-X_0}{ΔX}⌋$$(17)
$$q(m) = ⌊\frac {y_1+α_{mid}(y_2-y_1)-Y_0}{ΔY}⌋$$(18)

where ⌊⋅⌋ denotes the truncation operator and $α_{mid}$ is given by

$$α_{mid} = \frac{\underline{α}(m)+\underline{α}(m-1)}{2}$$(19)

The index in the z direction follows from a look-up table. The extent in the z direction in discretized into N(= 2048) intervals and for every interval the corresponding slice index is stored in an array $\underline z_{index}$. Using this approach, the slice index for a given z coordinate can be calculated by

z方向的索引从查找表中获得。z方向的长度离散到N=2048个间距，对于每个间距，对应的slice索引存储在一个阵列$\underline z_{index}$中。使用这种方法，对于给定的z坐标轴的slice索引计算为

$$r(m) = \underline z_{index} [⌊\frac {z_1+α_{mid}(z_2-z_1)-Z_0}{(Z_{max}-Z_0)/N}⌋]$$(20)

where $Z_{max}$ is the last "Z-plane". With this method the slice index can be retrieved fast. However, this comes at the cost of errors due to the discretization. The RPL is then eventually found by

这里$Z_{max}$是最后一个Z平面。这种方法中，slice索引可以很快的得到。但是，其代价是离散会带来误差。其RPL最后由下式得到

$$RPL = d_{12} \sum_{m=1}^N ρ(p(m),q(m),r(m)) \underline{α}(m+1)-\underline{α}(m)$$(21)

where ρ is the relative electron density. 其中ρ是相对电子密度。

### 2.3 Method 1: Reference CPU algorithm

Siddon’s algorithm was modified in two respects for the purpose of optimization. This optimized algorithm is part of PLATO’s dose calculation. First of all, the volume that is ray traced can be limited to the volume inside the bounding box of the body contour because that is generally the volume of interest. As a side effect, the number of unnecessary loads from dynamic random access memory (DRAM) to cache memory is reduced.

Siddon的算法为了优化，有2个方面的改进。这个优化的算法，是PLATO的剂量计算的一部分。首先，光线追踪的体，可以局限在身体轮廓边界框之内，因为一般来说，这是感兴趣的体。一个副作用是，DRAM到缓存中不必要的负载数量减少了。

The second optimization is based on reusing values of α. For all target voxels in a plane for which x is constant, the values of $α_x$ for all “X planes” can be calculated once and reused for all voxels in this plane. The same holds for all voxels on a line in this particular plane for which y is constant. Here the values of $α_y$ can be calculated once for all Y planes and they can be reused for all voxels in this line. In this way the number of intersections for which the value of α has to be calculated can be largely reduced. Timing of both strategies showed a reduction of 40%–60% in the calculation time.

第二种优化是基于对α值的重用。对于x为常数的平面中所有的目标体素，对于所有X平面上的$α_x$都可以只计算一次，然后对这个平面上的所有体素进行重用。对这个特定平面上y值是常数的在一条线上的所有体素，也是一样的。这里$α_y$的值对于所有的Y平面都可以计算一次，对于这条线上的所有体素都可以进行重用。这样，α值需要计算的相交点的数量可以得到大幅减少。两种策略的耗时显示，计算时间减少了40%-60%。

The optimized CPU algorithm, using multithreading, will be referred to as the reference algorithm. This version is the current validated clinical standard as implemented (single threaded) in PLATO.

优化的CPU算法，使用了多线程，称为参考算法。这个版本是目前得到验证的临床标准算法，在PLATO中有单线程的实现。

### 2.4 Method 2: GPU algorithm

Data parallelism was exploited by making every thread responsible for ray tracing to a single voxel in the region of interest. Rays can be traced completely independent and therefore this approach fits CUDA’s programming model well.

通过在感兴趣区域中，每个线程负责单个体素的射线追踪，数据并行得到了利用。射线可以完全独立的追踪，因此这种方法可以很好的符合CUDA编程模型。

Implementing Siddon’s algorithm on the graphics card required rewriting that part of the algorithm where $\underline{α}_x$, $\underline{α}_y$, and $\underline{α}_z$ are computed and stored. For a CT dataset of size 512×512×100, the following worst case estimation can be made for the dimensions of $\underline{α}_x$, $\underline{α}_y$, and $\underline{α}_z$. If the source position is along the main diagonal of the CT volume and just outside the dataset, based on symmetry, the number of intersections is approximately half of the number of planes on average, i.e., 256, 256, and 50, respectively. Consequently, 2×562 floating-point numbers will be needed for every voxel, on average, to store the three lists and the merged result. This means that in total, approximately 100 Gbytes of memory is needed. Even though this is a very coarse estimation and largely overestimates the required amount of memory in practice, it illustrates that with this approach the available graphical memory will be insufficient. For this reason, the algorithm was rewritten by using a stepping approach. The algorithm in the Appendix describes the tracing procedure schematically.

在显卡上实现Siddon的算法，需要重写一部分算法，即$\underline{α}_x$, $\underline{α}_y$和$\underline{α}_z$进行计算和存储的部分。对于一个CT数据集，大小为512×512×100，对于$\underline{α}_x$, $\underline{α}_y$和$\underline{α}_z$的维度，可以做下面最坏的情况估计。如果源位置是沿着CT体的主对角线，而且在数据集的外面的，基于对称性，那么交点的数量大约是平面数量的一半，即分别是256, 256和50。结果是，对于每个体素，平均需要2×562个浮点数，以存储三个列表和合并的结果。这意味着总计大约需要100GB的内存。虽然这只是一个非常粗略的估计，很可能高估了实践中需要的内存，但这说明了，这种方法下，可用的显存数量是不够的。基于这个原因，算法以步进的方法进行了重写。附录中的算法描述了跟踪的过程。

After the determination of the value of α at the entrance of the volume ($α_{min}$), the potential next intersecting plane is calculated for three directions. These intersecting planes correspond to $i_{min}, j_{min}$, and $k_{min}$. Furthermore, the index of the entrance voxel in the linear array holding the density is determined.

在确定了α在进入体时的值($α_{min}$)后，在三个方向上要计算可能的下一个相交平面。这些相交平面对应着$i_{min}, j_{min}$和$k_{min}$。而且，保存了密度值的线性阵列的入口体素的索引就确定了。

After these initialization steps, the algorithm continues with a loop where the next intersecting plane is determined. This is achieved by computing the minimum of the value of α for the three potential next planes of intersection. This next intersecting plane now becomes the current plane of intersection and a new potential plane of intersection is calculated. This loop is continued until the value of α for the current intersecting plane is larger than or equal to unity. In order to access the the elements of $\underline z_{plane}$ fast, this array is stored in the constant memory of the graphics card.

在这些初始化步骤之后，算法继续下一个循环，确定下一个相交的平面。这通过计算，可能的下三个相交平面，来得到最小的α值。这下一个相交平面现在成为目前的相交平面，然后就可以计算新的可能的相交平面。这个循环继续下去，直到当前平面的α值大于或等于1。为快速访问$\underline z_{plane}$的元素，这个阵列存储在显存的constant内存中。

Voxel indices in the x and y directions are calculated from the value of $α_{mid}$. In the z direction the index is updated when leaving the current slice. Using this approach, the use of a look-up table in the z direction has become obsolete. This also eliminates the related discretization errors present in the reference algorithm.

x和y方向的体素索引是从$α_{mid}$值中计算得到的。在z方向，索引的更新是通过离开当前slice来得到的。使用这种方法，使用z方向的查找表的方法就可以废弃不用了。这还消除了参考算法中，相关的离散误差。

### 2.5 Performance and accuracy analysis

The GPU algorithm was tested for accuracy and performance on two systems. Their specifications are summarized in Table I. Both systems have 32-bit Linux as their operating system (Fedora 8). All tests were done under CUDA version 1.1. The block size was chosen to be 16×16 threads for all evaluated cases. Based on hardware specifications, CPU timings are considered interchangeable for systems I and II.

GPU算法在两个系统中测试了其精度和性能。其规格如表I所示。两个系统都是32位Linux操作系统(Fedora 8)。所有的测试都是在CUDA 1.1下进行的。模块大小对于所有的评估案例，都是16×16线程的。基于硬件规格，对于系统I和系统II，CPU的计时是可以交换的。

TABLE I. Specification of the hardware of systems I and II.

|| System I | System II
--- | --- | ---
CPU clock frequency(GHz) | 2.66 | 2.66
Processor type | Intel Xeon quad-core | Intel Q9400 quad-core
Graphics card type | NVIDIA GeForce 8800 GTS | NVIDIA GeForce GTX 280
No. Processor cores | 96 | 240
Memory bandwidth (Gbytes/s) | 64 | 141.7
Graphical memory(Mbyte) | 320 | 1024

A set of test geometries was defined. Table II gives a description of the used test geometries, and Table III describes the evaluated test cases. The different test cases were defined for resampled geometries with resampling factors ($Δ_r$) ranging from 1 to 4. The resampling factor is the reduction in resolution in the XY plane applied to both the X and Y directions. The resampling strategy, following the PLATO convention, is illustrated in Fig. 3.

定义了一系列测试几何。表II给出了使用的测试几何的描述，表III描述了评估的测试案例。不同的测试案例，是通过重采样的几何，和重采样的因子($Δ_r$)来定义的，$Δ_r$在1到4的范围内。重采样的因子是分辨率下降的因子，在XY平面上对X和Y方向都有应用。重采样的策略，遵循PLATO的规范，如图3所示。

TABLE II. Description of the different datasets that were used for performance and accuracy analysis. The bounding box is the smallest box that fits the patient. Original in-plane dimensions of the CT datasets are 512 􏴢 512. Geometry A is a homogeneous water phantom. Geometries B and D are radiotherapy planning CT datasets of the pelvic region and geometry C is a planning CT of the head-and-neck region. As an illustration, Fig. 4 shows the central transversal slice for every dataset.

Geometry | No. of slices | Bounding box(-) | Spacing(mm)
--- | --- | --- | ---
A | 31 | 452×452 | 0.7^2×10.0
B | 35 | 448×260 | 0.8^2×5.0
C | 103 | 499×292 | 0.9^2×3.0
D | 56 | 308×190 | 0.9^2×3.0

TABLE III. Definition of five test cases. Test cases I–IV have been evaluated for test geometries A, B, and C. Test case V corresponds to irradiation with a vertex field and is therefore evaluated for test geometry C only. Figure 4 illustrates cases I–IV. $Δ_r$ is the in-plane spacing relative to the original spacing.

Test case | $Δ_r$(-) | $ϕ_{gantry}$(deg) | $ϕ_{table}$(deg)
--- | --- | --- | ---
I | 1 | 0 | 0
II | 2 | 45 | 0
III | 4 | 90 | 0
IV | 4 | 270 | 0
V | 1 | 90 | 90

FIG. 3. Illustration of the resampling strategy, here for a resampling factor of 2. Four voxels, numbered from 0 to 3, are reduced to one voxel. The relative electron density is determined by the value of voxel 0. Resampling is applied in 2D, i.e., within the same CT slice.

FIG. 4. Transversal midplane for all test geometries. Roman symbols indicate test cases I–IV. For geometry D, the arc of source positions used for the dose calculation timings is indicated.

#### 2.5.1 Accuracy

Since the GPU ray tracing algorithm does not use a look-up table, the reference algorithm (described in Sec. II C) was extended with a more elaborate but exact algorithm to determine the corresponding slice index for a given z coordinate. This algorithm will be referred to as CPU hunt from here on and is used as an accuracy benchmark for the GPU algorithm.

由于GPU版光线追踪算法没有使用查找表，2.3节描述的参考算法进行了拓展，成为了一个更复杂但更精确的算法，可以对一个给定的z坐标轴确定对应的slice索引。这个算法称为CPU hunt，作为GPU算法的一个准确度基准。

In addition, the GPU algorithm was compared to the reference algorithm to assess the effect of discretization errors resulting from the look-up table implementation.

另外，GPU算法与参考算法进行了比较，以评估查找表实现导致的离散化误差的效果。

#### 2.5.2 Performance

Besides accuracy analysis, the performance of the GPU algorithm relative to the multithreaded CPU algorithm was investigated. Computation time was measured between the start of both ray tracing algorithms and the moment the results are stored in DRAM. For the GPU algorithm this means that the time needed to transfer the data from graphical memory to DRAM is included. Time required for initialization, e.g., transfer of data from DRAM to graphical memory (typically <20 ms) and driver initialization (typically 100–200 ms) was excluded. The rationale for this choice is that for one planning session this is only done once while the actual ray tracing is performed numerous times.

除了准确度分析，与多线程CPU算法相比较起来，GPU算法的性能也进行了研究。计算时间的计时，是从射线追踪算法的开始，与结果存储到DRAM为止。对于GPU算法，这意味着，将数据从显存转移到DRAM的时间也包括在内。初始化所需的时间，如，从DRAM到显存迁移所需的时间（一般<20ms），和驱动器初始化（一般100-200ms）是不在计时范围内的。这种选择的原理是，对于一个计划对话，这只进行一次，而真实的射线追踪算法要进行很多次。

As an additional performance benchmark, the gantry angle was rotated over 20° with 1° steps and for each source position the contribution to the dose distribution was calculated using both the reference and the GPU algorithm. Dose calculations were carried out for a 10×10 cm^2 field for two test geometries, C and D. Timings were performed using a resampling factor of both 1 and 2. In contrast to the GPU dose calculations, CPU dose calculations make a distinction between the beam volume extended with a margin (2cm) and the remaining volume. Inside the first the specified resampling factor is used, outside a resampling factor of 8 is used. For geometry C, independent of the gantry angle, the field will “cover” the patients’ head/neck. Here a starting angle of 0° was chosen. For geometry D, a pelvic case, a substantial part of the volume will be handled by the CPU algorithm using a downsampled geometry when starting with a gantry angle of 0°. For this reason, the arc was started at 45°.

作为额外的性能基准，机架角度要旋转20度，以1度为步长，对于每个源的位置，对剂量分布的贡献都用参考算法和GPU算法进行进行计算。剂量计算的进行是在10×10 cm^2的野中进行的，有2个测试几何，C和D。与GPU剂量计算相比，CPU剂量计算在射野体拓展的margin(2cm)中，和剩余的体中会不一样。在内部，使用第一个指定的重采样系数，在外部，使用重采样系数8。对于几何C，与机架角度独立，射野会覆盖患者的头/颈。这里选择的初始角度为0度。对于几何D，是一个盆部的案例，这个体中的大部分都是由CPU算法处理的，当从机架角0度开始时，使用的是下采样的几何。基于这个原因，圆弧是从45度开始的。

## 3. Results

### 3.1 Accuracy analysis

#### 3.1.1 GPU versus CPU hunt

Table IV shows the results of the comparison of the CPU-hunt algorithm with the presented GPU algorithm. For all evaluated test cases, the maximum difference found was 0.064 mm. The results given by the GPU algorithm for the two different graphics cards were compared and no differences were found. For this reason, the comparison between CPU-hunt and the GPU algorithm is reported for one system only.

表IV给出了CPU-hunt算法与给出的GPU算法的对比。对于所有评估的测试案例，最大差异为0.064mm。两个不同的显卡给出的GPU算法的结果进行了比较，没有发现什么差异。由于这个原因，CPU-hunt和GPU算法的比较，只在一个系统中给出。

TABLE IV. Maximum absolute difference in RPL mm between the CPU-hunt algorithm and the GPU algorithm.

Test case | Test Geometry A | B | C
--- | --- | --- | ---
I | 0.002 | 0.050 | 0.064
II | 0.001 | 0.009 | 0.045
III | 0.001 | 0.006 | 0.005
IV | 0.000 | 0.004 | 0.007
V | - | - | 0.011

#### 3.1.2 GPU versus reference CPU algorithm

Figure 5 shows a cumulative histogram of the RPL difference between the GPU and the reference CPU algorithm for test geometry C using a resampling factor of 2. Due to discretization errors introduced by the look-up table, it was found that the slice index is incidentally calculated incorrectly by the CPU algorithm. For this particular example in 0.1% of the volume, a deviation larger than 2.7 mm was found (for geometries A and B a deviation larger than 4.4×10^−4 and 1.5 mm, respectively, was found in 0.1% of the volume). As an illustration of the effect of discretization errors in the calculation of the slice index, Fig. 6 shows a slice of the absolute RPL difference map for geometry C, test case II.

图5给出了GPU算法和参考CPU算法在测试几何C中，使用重采样系数为2时的RPL差异的累计直方图。由于查找表引入的离散化误差，我们发现，CPU算法计算的slice索引偶尔计算的会有错误。对于这个在体中的0.1%的特殊的例子，会有超过2.7mm的差异（对于几何A和B，在体的0.1%中，分别会有超过4.4×10^−4和1.5mm的偏差）。在计算slice索引时的离散化误差。图6给出了一个展示，在测试案例II，几何C的情况下，给出了绝对RPL误差图的一个slice。

FIG. 6. An example absolute difference map of the RPL for geometry C with the corresponding CT slice (31) for test case II. White, gray, and black correspond to 0–0.9, 0.9–1.8, and >1.8 mm, respectively. Discretization errors at the body contour of the patient yield RPL differences projecting at the neighboring slices. Differences from multiple slices can project on the same slice as is clearly visible in this example. As can clearly be seen, there is a close resemblance between the body contour of this slice and the upper “contour” in the RPL difference map.

### 3.2 Performance analysis

The timings of ray tracing for the different test cases are presented in Tables V–VII. Although, the achieved speedup is observed to be strongly dependent on the geometry tested for, the GPU algorithm is always faster. The timings for dose calculations when radiating while rotating the gantry over 20° in 1° steps are presented in Table VIII. The timings are reported for resampling factors 1 and 2 for the multithreaded CPU algorithm and the GPU algorithm timed on both systems I and II. The tested geometries are C and D. Averaged over the 21 source positions, the times needed for a single dose calculation with a resampling factor of 2 were found to be 0.9 and 0.6 s, respectively.

不同测试案例的射线跟踪的计时，在表V-VII中。虽然，得到的加速效果，与测试的几何高度相关，但GPU算法永远更快一些。当旋转机架20度，以1度为步长，进行剂量计算的计时，如表VIII所示。给出的计时，包括重采样率为1和2的情况，包括多线程CPU算法和GPU算法，包括在系统I和II的算法。测试的几何为C和D的情况。在21个源的位置进行平均，对重采样系数为2的情况，进行一次剂量计算，分别需要0.9s和0.6s。

TABLE V. Timing of the reference (multithreaded CPU) and the GPU ray tracing algorithm on systems I and II for test geometry A. S denotes speedup, GPU execution time relative to CPU execution time.

Test case | CPU(s) | System I GPU(s) | System I S(-) | System II GPU(s) | System II S(-)
--- | --- | --- | --- | --- | ---
I | 7.52 | 1.91 | 3.9 | 0.85 | 8.9
II | 1.31 | 0.32 | 4.1 | 0.15 | 8.9
III | 0.21 | 0.06 | 3.8 | 0.02 | 10.1
IV | 0.18 | 0.05 | 3.6 | 0.02 | 8.4

TABLE VI. Timing of the reference (multithreaded CPU) and the GPU ray tracing algorithm on systems I and II for test geometry B. S denotes speedup, GPU execution time relative to CPU execution time.

Test case | CPU(s) | System I GPU(s) | System I S(-) | System II GPU(s) | System II S(-)
--- | --- | --- | --- | --- | ---
I | 2.22 | 1.08 | 2.1 | 0.47 | 4.7
II | 0.44 | 0.13 | 3.3 | 0.06 | 7.1
III | 0.1 | 0.03 | 3.4 | 0.01 | 8.1
IV | 0.1 | 0.03 | 3.4 | 0.01 | 9.0

TABLE VII. Timing of the reference (multithreaded CPU) and the GPU ray tracing algorithm on systems I and II for test geometry C. S denotes speedup, GPU execution time relative to CPU execution time.

Test case | CPU(s) | System I GPU(s) | System I S(-) | System II GPU(s) | System II S(-)
--- | --- | --- | --- | --- | ---
I | 5.98 | 3.88 | 1.5 | 1.82 | 3.3
II | 1.12 | 0.84 | 1.3 | 0.41 | 2.7
III | 0.18 | 0.12 | 1.5 | 0.06 | 2.8
IV | 0.17 | 0.11 | 1.5 | 0.08 | 2.1
V | 4.61 | 3.28 | 1.4 | 1.75 | 2.6

TABLE VIII. Timing of dose calculations for an arc of 21 source positions for resampling factors of 1 and 2 and two test geometries. S denotes speedup, GPU execution time relative to CPU execution time. $Δ_r$ is the in-plane spacing relative to the original spacing. The starting angles were set to be 0° and 45°, respectively.

Geometry | $Δ_r$ | CPU(s) | System I GPU(s) | System I S(-) | System II GPU(s) | System II S(-)
--- | --- | --- | --- | --- | --- | ---
C | 1 | 97 | 90 | 1.1 | 64 | 1.5
C | 2 | 32 | 18 | 1.8 | 18 | 1.8
D | 1 | 314 | 59 | 5.4 | 51 | 6.2
D | 2 | 55 | 15 | 3.7 | 11 | 4.8

## 4. Discussion and conclusions

The aim of this study was to investigate the possible acceleration of the presently used ray tracing algorithm for radiotherapy dose calculations using a GPU. Both accuracy and execution time of the GPU algorithm were compared to a CPU benchmark. It was shown that accurate ray tracing can be performed, on average, six times faster than with the multithreaded reference CPU algorithm.

本文的目标是，研究使用GPU对最近使用的射线追踪算法进行放射治疗剂量计算的可能加速。与CPU基准算法比较了，GPU算法的准确率和执行时间。与多线程参考CPU算法比较，准确的射线跟踪算法平均可以提速6倍。

The difference in radiological path length between the CPU-hunt and the GPU algorithm was found to be smaller than 0.1 mm, the precision with which the RPL is stored in practice. The present differences can be explained by differences in accumulation of round-off errors. In the GPU algorithm the values of α are calculated incrementally, whereas for the CPU algorithm for all the planes of intersection, the value of α is calculated independently using Eqs. (7)–(9).

CPU-hunt和GPU算法计算的放射路径长度差异，一般小于0.1mm，RPL的精度是存储起来的。目前的差异可以通过四舍五入的差异的累计来解释。在GPU算法中，α值是递增计算的，而CPU算法中，所有相交的平面，其α值是用式(7)-(9)独立计算的。

The look-up table introduces discretization errors that can lead to a difference in RPL up to 7.5 mm (for test geometry C). At a large angle of incidence (relative to the z direction) and high density contrast, the GPU algorithm is therefore more accurate than the reference CPU algorithm.

查找表引入了离散化误差，可以带来RPL高达7.5mm的误差（对于测试几何C来说）。在大角度入射（相对于z方向）和高密度差异时，GPU算法因此比参考的CPU算法更精确。

These differences however are found in a marginal fraction of the irradiated volume: A deviation larger than 2.7 mm is found in only 0.1% of the volume (Fig. 5). Figure 6 shows that discretization errors at the location of the body contour are projected onto neighboring slices. Together with discretization errors at other high density contrast regions, this explains the found differences between the reference CPU and the GPU algorithm. As shown, in case of irradiation with a vertex field, the maximum error found was 0.5 mm. This is consistent with the explanation for the deviation in the co-planar cases since the angle of incidence is close to normal in this particular case. In clinical practice, the differences will even be less significant since the location of the differences found will be dependent on the beam angle. Deviations in the dose distribution will therefore be much smaller for a multiple beam plan than for a single beam.

这些差异在辐射体中的影响其实很小：只在0.1%的体中发现了超过2.7mm的偏差（图5）。图6表明，在身体轮廓位置处的离散化误差，投影到了相邻的slice中。与离散化误差一起，在其他高密度对比度区域，这解释了参考CPU算法和GPU算法的差异。在vertex场的辐射的情况下，最大的误差是0.5mm。这与共面情况下的偏差的解释是一致的，因为入射角在这种特殊情况下几乎是垂直的。在临床实践中，这种差异甚至会更小，因为位置的差异与射野角度相关。剂量分布的偏差，在多射野的情况下，会比单射野的情况小很多。

The presented timings show that the GPU implementation can speed up the multithreaded CPU ray tracing algorithm considerably by up to a factor of 10. This speedup, however, is observed to be strongly dependent on the geometry. The speedup found for geometry C, the head-and-neck case, is relatively small. For this geometry, a relatively large fraction of voxels within the bounding box contains air and no tracing needs to be done to these voxels. Threads are started, however, for all voxels within the bounding box, ray tracing is only performed when the density is unequal to zero. When there are threads within a warp (a group of threads executed physically in parallel) that are tracing and threads that are idle, this effectively lowers the level of occupation of the GPU and therefore reduces performance. This could possibly be solved by compressing the grid by the removal of all threads that do not have to do any work.

给出的计时结果表明，GPU实现与多线程CPU射线追踪算法相比有显著的加速效果，最大可以到达10倍的加速。但这种加速，与几何关系有很大的关系。对于几何C的情况，即头颈部的例子，这种加速相对较小。对于这种几何情况，边界框内的体素的相对较大一部分都是空气，对这些体素不需要进行跟踪。但对边界框中的体素都开启了一个线程，射线追踪只有在密度不等于0的时候进行计算的。当有一些线程是在一个warp中（一群线程，物理上并行执行）进行追踪时，当线程空闲时，这会明显降低GPU占用的水平，因此降低性能。这通过去掉不进行任何工作的线程，从而压缩网格，很有可能解决这个问题。

The time required for a complete single beam dose calculation at a resampling factor of 2 was measured to be 0.6–0.9 s, depending on the geometry. The reported speed-ups for dose calculation are lower than the speedups reported for ray tracing only. The speedups measured for geometry C were 1.5 and 1.8 using resampling factors of 1 and 2, respectively. For ray tracing alone, speedups of 3.3 and 2.7 were measured. This can be explained by the fact the number of voxels for which ray tracing is carried out is larger for the GPU algorithm than that for the CPU algorithm since no distinction is made between being in- and outside the “beam volume.” Furthermore, although more than 50% of the dose calculation time is made up by ray tracing, the remaining calculation time is not influenced by the GPU acceleration making the overall reduction relatively smaller.

一个完整的单射野剂量计算，重采样因子为2，需要的时间为0.6-0.9s，其时间与几何有关。对剂量计算得到的加速效果，比只进行射线追踪得到的加速效果要低。对于几何C得到的加速效果，在重采样因子为1和2时，分别是1.5和1.8。对于射线追踪，测量出来的加速效果为3.3和2.7。射线追踪执行的时候，涉及到的体素数量，在GPU算法里，比CPU算法里要多，因为在射野体素内外，并没有区分。而且，虽然50%的剂量计算时间是由射线追踪组成的，剩余的计算时间并没有受到GPU加速影响，这使总体时间缩减相对较小。

It is inherently unknown which elements of the density array will be accessed during ray tracing. This results in a complex access pattern that is not consistent with the requirements for fast parallel accessing of data. The ray tracing algorithm could possibly be optimized using texture memory. Texture memory is read-only memory that is optimized to acces grid data fast.

在射线追踪时，密度阵列中哪个体素得到了访问，这是一个无法提前知道的问题。这导致访问模式就很复杂，快速并行数据访问的需求并不是一致的。这种射线追踪算法，很可能由纹理内存得到优化。纹理内存是只读内存，专为快速网格数据访问进行了优化。

Compared to the performance of the two different graphics cards, it is found that the GTX 280 card is on average 2.16 times faster than the 8800 GTS card (minimum of 1.4, maximum of 2.66). The first benefits from a larger number of processor cores (240 versus 96) and a higher theoretical memory bandwidth (141.7 versus 64 Gbytes/s).

与两种不同的显卡的性能相比，我们发现GTX 280显卡平均比8800 GTS显卡快了2.16倍（最小1.4，最大2.66）。前者处理器核心多(240 vs 96)，理论上的内存带宽也更高(141.7 vs 64 GB/s)。

The developed GPU algorithm now enables dose calculations at a speed that will be experienced as real time for conventional forward planning based on clinically relevant datasets. In addition, there is no longer a dependency between execution time and field size. As mentioned in Sec. I this can lead to a major reduction in the workload of radiotherapy treatment planning. Moreover, the presented GPU algorithm can be used to accelerate more advanced treatment planning optimization techniques.

先进的GPU算法进行剂量计算，现在达到的速度，在传统的正向计划中，在临床相关的数据集上，其体验是实时的。另外，其执行时间与射野大小现在是没有相关的。如第1部分所述，这可以极大的减少放射治疗计划的工作量。而且，提出的GPU算法可以用于加速更先进的治疗计划优化技术。

## Acknowledgments

For this work, the source code of the PLATO planning system was used within the framework of the AMC–Nucletron (Veenendaal) treatment planning collaboration. Financial support was given by the Dutch Cancer Society (Grant No. UVA-2006-3484).