# Real-Time Loop Closure in 2D LIDAR SLAM

Wolfgang Hess, Damon Kohler, Holger Rapp, Daniel Andor

## Abstract

Portable laser range-finders, further referred to as LIDAR, and simultaneous localization and mapping (SLAM) are an efficient method of acquiring as-built floor plans. Generating and visualizing floor plans in real-time helps the operator assess the quality and coverage of capture data. Building a portable capture platform necessitates operating under limited computational resources. We present the approach used in our backpack mapping platform which achieves real-time mapping and loop closure at a 5 cm resolution. To achieve real-time loop closure, we use a branch-and-bound approach for computing scan-to-submap matches as constraints. We provide experimental results and comparisons to other well known approaches which show that, in terms of quality, our approach is competitive with established techniques.

移动式激光测距，进一步被称为LIDAR，和SLAM，是获取平面图的一种高效方法。实时生成平面图并可视化，帮助操作员评估捕获数据的质量和覆盖。构建一个移动式捕获平台，使在有限的计算资源下进行操作称为必须。我们提出的方法在背包式地图构建平台中使用，以5cm的分辨率，获得了实时的地图构建和回环检验。为获得实时的回环检验，我们使用了一种branch-and-bound的方法，计算扫描对子地图的匹配作为约束。我们与其他著名的方法给出试验结果和比较，表明我们的方法在质量上与已有方法是有竞争力的。

## 1. Introduction

As-built floor plans are useful for a variety of applications. Manual surveys to collect this data for building management tasks typically combine computed-aided design (CAD) with laser tape measures. These methods are slow and, by employing human preconceptions of buildings as collections of straight lines, do not always accurately describe the true nature of the space. Using SLAM, it is possible to swiftly and accurately survey buildings of sizes and complexities that would take orders of magnitude longer to survey manually.

平面图在很多应用中都很有用。对建筑物管理任务，手工勘察收集这种数据，一般都会与带有激光胶带测量的CAD结合。这些方法很慢，人工对建筑物的预想是直线的集合，采用这种预想并没有准确的描述空间的真正本质。使用SLAM，可能快速准确的勘察建筑物的大小和复杂度，如果采用手工进行，会耗费高很多数量级的时长。

Applying SLAM in this field is not a new idea and is not the focus of this paper. Instead, the contribution of this paper is a novel method for reducing the computational requirements of computing loop closure constraints from laser range data. This technique has enabled us to map very large floors, tens-of-thousands of square meters, while providing the operator fully optimized results in real-time.

将SLAM应用在这个领域，并不是一个新想法，也并不是本文的关注点。本文的贡献是一种降低从激光距离数据中计算回环约束的计算需求的新方法。这种技术使我们可以对非常大的平面进行建图，成千上万平方米，以实时的速度对操作员给出完全优化的结果。

## 2. Related Work

Scan-to-scan matching is frequently used to compute relative pose changes in laser-based SLAM approaches, for example [1]–[4]. On its own, however, scan-to-scan matching quickly accumulates error.

扫描对扫描的匹配频繁用于在基于激光的SLAM方法中计算相对姿态变化，比如[1-4]。但是就起本身来说，扫描对扫描的匹配会迅速累积误差。

Scan-to-map matching helps limit this accumulation of error. One such approach, which uses Gauss-Newton to find local optima on a linearly interpolated map, is [5]. In the presence of good initial estimates for the pose, provided in this case by using a sufficiently high data rate LIDAR, locally optimized scan-to-map matching is efficient and robust. On unstable platforms, the laser fan is projected onto the horizontal plane using an inertial measurement unit (IMU) to estimate the orientation of gravity.

扫描对地图的匹配会使得误差累积更有限。一种这样的方法使用了Gauss-Newton法在线性插值的地图中来寻找局部最小值[5]。对姿态存在很好的初始估计，在这个案例中是用足够高数据率的LIDAR给出的，局部优化的扫描对地图匹配是高效稳健的。在不稳定的平台上，使用惯性测量单元(IMU)来估计重力方向，然后激光扇束投影到水平平面上。

Pixel-accurate scan matching approaches, such as [1], further reduce local error accumulation. Although computationally more expensive, this approach is also useful for loop closure detection. Some methods focus on improving on the computational cost by matching on extracted features from the laser scans [4]. Other approaches for loop closure detection include histogram-based matching [6], feature detection in scan data, and using machine learning [7].

像素级的扫描匹配方法，比如[1]，进一步降低了局部误差累积。虽然在计算上更加昂贵，这种方法对于回环检测也是有用的。一些方法聚焦在，通过在从激光扫描中提取出的特征进行匹配，来改进计算代价[4]。其他的回环检测方法包括，基于直方图的匹配[6]，扫描数据中的特征检测，以及使用机器学习的方法[7]。

Two common approaches for addressing the remaining local error accumulation are particle filter and graph-based SLAM [2], [8]. 处理剩余的局部误差累积的两种常用方法是，粒子滤波和基于图的SLAM[2,8]。

Particle filters must maintain a representation of the full system state in each particle. For grid-based SLAM, this quickly becomes resource intensive as maps become large; e.g. one of our test cases is 22,000 m^2 collected over a 3 km trajectory. Smaller dimensional feature representations, such as [9], which do not require a grid map for each particle, may be used to reduce resource requirements. When an up-to-date grid map is required, [10] suggests computing submaps, which are updated only when necessary, such that the final map is the rasterization of all submaps.

例子滤波必须在每个粒子中维护完全系统状态的表示。对于基于网格的SLAM，这在图变大时，迅速变得需要大量资源；如，我们的一个测试案例是22000 m^2，在3km的轨迹上进行收集的。更低维度的特征表示，比如[9]，不需要对每个例子的网格图，会用于降低资源需求。当需要一个最新的网格图，[10]建议计算子图，只在需要的时候进行更新，这样最终的图是所有子图的栅格化。

Graph-based approaches work over a collection of nodes representing poses and features. Edges in the graph are constraints generated from observations. Various optimization methods may be used to minimize the error introduced by all constraints, e.g. [11], [12]. Such a system for outdoor SLAM that uses a graph-based approach, local scan-to-scan matching, and matching of overlapping local maps based on histograms of submap features is described in [13].

基于图的方法需要表示姿态和特征的节点的集合。图中的边缘是从观察生成的约束。各种优化方法都会用于最小化由各种约束引入的误差，如[11,12]。[13]描述了室外SLAM的这样一个系统，使用一种基于图的方法，局部扫描到扫描的匹配，以及用基于子图特征直方图的方法匹配重叠的局部图。

## 3. System Overview

Google’s Cartographer provides a real-time solution for indoor mapping in the form of a sensor equipped backpack that generates 2D grid maps with a r = 5 cm resolution. The operator of the system can see the map being created while walking through a building. Laser scans are inserted into a submap at the best estimated position, which is assumed to be sufficiently accurate for short periods of time. Scan matching happens against a recent submap, so it only depends on recent scans, and the error of pose estimates in the world frame accumulates.

谷歌的Cartographer为室内建图提供了实时解决方案，用传感器装备的背包生成2D网格图，分辨率为r=5cm。系统操作员在步行穿过一栋建筑时，会看到图正被创建。激光扫描在估计的最好的位置插入到子图中，我们假设在较短的时间段内是足够准确的。扫描匹配会对一个最近的子图发生，所以只依赖于最近的扫描，在世界框架中的姿态估计的误差会进行累积。

To achieve good performance with modest hardware requirements, our SLAM approach does not employ a particle filter. To cope with the accumulation of error, we regularly run a pose optimization. When a submap is finished, that is no new scans will be inserted into it anymore, it takes part in scan matching for loop closure. All finished submaps and scans are automatically considered for loop closure. If they are close enough based on current pose estimates, a scan matcher tries to find the scan in the submap. If a sufficiently good match is found in a search window around the currently estimated pose, it is added as a loop closing constraint to the optimization problem. By completing the optimization every few seconds, the experience of an operator is that loops are closed immediately when a location is revisited. This leads to the soft real-time constraint that the loop closure scan matching has to happen quicker than new scans are added, otherwise it falls behind noticeably. We achieve this by using a branch-and-bound approach and several precomputed grids per finished submap.

为用中等硬件需求来获得好的性能，我们的SLAM方法没有用粒子滤波器。为应对误差累积，我们有规律的运行姿态优化。当完成一个子图时，就不会有新的扫描插入其中了，子图会参与用于回环检测的扫描匹配。所有完成的子图和扫描自动被回环检测考虑。如果基于目前的姿态估计，它们足够接近，扫描匹配器就尝试在子图中找到扫描。如果在目前估计的姿态中在搜索窗口中找到一个足够好的匹配，就将其加入到优化问题的回环约束中。每几秒我们就完成优化，一个操作员的经验是，当一个位置重新访问时，回环就立刻闭合了。这带来了软的实时约束，回环扫描匹配必须要比新的扫描加入的要快，否则就会很明显的拉到后面。我们使用一种branch-and-bound方法，和对每个完成的子图的几个预计算的网格，来达到这个目标。

## 4. Local 2D SLAM

Our system combines separate local and global approaches to 2D SLAM. Both approaches optimize the pose, ξ = (ξx, ξy, ξθ) consisting of a (x, y) translation and a rotation ξθ, of LIDAR observations, which are further referred to as scans. On an unstable platform, such as our backpack, an IMU is used to estimate the orientation of gravity for projecting scans from the horizontally mounted LIDAR into the 2D world.

我们的系统将局部方法和全局方法结合成2D SLAM。两种方法都对姿态ξ = (ξx, ξy, ξθ)进行优化，包含LIDAR观测的一个(x,y)平移，和一个旋转ξθ，后面进一步称之为扫描。在一个不稳定的平台，比如背包，使用IMU来估计重力的方向，将水平安装的LIDAR的扫描投影到2D世界中。

In our local approach, each consecutive scan is matched against a small chunk of the world, called a submap M, using a non-linear optimization that aligns the scan with the submap; this process is further referred to as scan matching. Scan matching accumulates error over time that is later removed by our global approach, which is described in Section V.

在局部方法中，每个连续的扫描与世界的一部分，称之为子图M，使用非线性优化进行匹配，将扫描与子图进行对齐；这个过程进一步称之为扫描匹配。扫描匹配随着时间累积误差，后面由全局方法去除误差，这在第5部分进行描述。

### 4.1 Scans

Submap construction is the iterative process of repeatedly aligning scan and submap coordinate frames, further referred to as frames. With the origin of the scan at 0 ∈ R^2, we now write the information about the scan points as $H = {h_k}_{k=1,...,K}, h_k ∈ R^2$. The pose ξ of the scan frame in the submap frame is represented as the transformation $T_ξ$, which rigidly transforms scan points from the scan frame into the submap frame, defined as

子图构建的过程，是将扫描与子图坐标系框架进行重复对齐的过程，子图坐标系框架我们又称之为框架。扫描的原点在0 ∈ R^2，我们现在将扫描点的信息写为$H = {h_k}_{k=1,...,K}, h_k ∈ R^2$。扫描框架在子图框架中的姿态ξ表示为变换$T_ξ$，将扫描点从扫描框架刚性变换到子图框架中，定义为

$$T_ξp = \left( \begin{matrix} cos ξ_θ && − sin ξ_θ \\ sin ξ_θ && cos ξ_θ \end{matrix} \right) p + \left( \begin{matrix} ξ_x \\ ξ_y \end{matrix} \right)$$(1)

where the former matrix is $R_ξ$, and the latter vector is $t_ξ$. 其中前者是旋转矩阵$R_ξ$，后者是平移向量$t_ξ$。

### 4.2 Submaps

A few consecutive scans are used to build a submap. These submaps take the form of probability grids M : rZ × rZ → [p_min, p_max] which map from discrete grid points at a given resolution r, for example 5 cm, to values. These values can be thought of as the probability that a grid point is obstructed. For each grid point, we define the corresponding pixel to consist of all points that are closest to that grid point.

用几个连续扫描就可以构建子图。这些子图是概率网格的形式M : rZ × rZ → [p_min, p_max]，在给定的分辨率r上，比如5cm，将离散点映射到值。这些值可以认为是，一个网格点被阻碍的概率。对每个网格点，我们定义对应的像素为，与那个网格点最接近的所有点。

Whenever a scan is to be inserted into the probability grid, a set of grid points for hits and a disjoint set for misses are computed. For every hit, we insert the closest grid point into the hit set. For every miss, we insert the grid point associated with each pixel that intersects one of the rays between the scan origin and each scan point, excluding grid points which are already in the hit set. Every formerly unobserved grid point is assigned a probability p_hit or p_miss if it is in one of these sets. If the grid point x has already been observed, we update the odds for hits and misses as

一个扫描被插入到概率网格中时，要计算击中的网格点集合，和一个不相交的未击中的集合。对每个击中，我们将最接近的网格点插入到击中集中。对每个未击中，我们将与每个像素相关的网格点插入，这些像素与从扫描原点和每个扫描点之间的射线相交，并排除了已经在击中集中的网格点。每个之前未观测到的网格点，如果在击中或未击中集中，都指定了一个概率p_hit或p_miss。如果网格点x已经被观测过了，我们将击中和未击中的概率更新为

$$odds(p) = p/(1-p)$$(2)
$$M_{new}(x) = clamp(odd^{-1} (odds(M_{old}(x))· odds(p_{hit})))$$(3)

and equivalently for misses. 对未击中的也是类似。

### 4.3 Ceres scan matching

Prior to inserting a scan into a submap, the scan pose ξ is optimized relative to the current local submap using a Ceres-based [14] scan matcher. The scan matcher is responsible for finding a scan pose that maximizes the probabilities at the scan points in the submap. We cast this as a nonlinear least squares problem

在将一个扫描插入到子图中时，扫描姿态ξ要相对于目前的局部子图进行优化，使用一个基于Ceres[14]的扫描匹配器。扫描匹配器负责找到一个扫描姿态，将扫描点在子图中的概率进行最大化。我们将其表述为一个非线性最小二乘问题

$$argmin_ξ \sum_{k=1}^K (1-M_{smooth}(T_ξh_k))^2$$(CS)

where $T_ξ$ transforms $h_k$ from the scan frame to the submap frame according to the scan pose. The function $M_{smooth}: R^2 → R$ is a smooth version of the probability values in the local submap. We use bicubic interpolation. As a result, values outside the interval [0, 1] can occur but are considered harmless.

其中$T_ξ$将$h_k$，根据扫描姿态，从扫描框架中变换到子图框架中。函数$M_{smooth}: R^2 → R$是在局部子图中的概率值的平滑版本。我们使用双三次插值。结果是，可能会出现[0, 1]范围之外的值，但是是无害的。

Mathematical optimization of this smooth function usually gives better precision than the resolution of the grid. Since this is a local optimization, good initial estimates are required. An IMU capable of measuring angular velocities can be used to estimate the rotational component θ of the pose between scan matches. A higher frequency of scan matches or a pixel-accurate scan matching approach, although more computationally intensive, can be used in the absence of an IMU.

这个平滑函数的数学优化通常会给出比网格分辨率的更好精度。由于这是一个局部优化，所以需要很好的初始估计。一个能够测量角速度的IMU，可以用于估计扫描匹配之间的旋转分量θ。在没有IMU的情况下，可以用更高频率的扫描匹配，或像素精度的扫描匹配方法，但是这样计算量就会大很多。

## 5. Closing Loops

As scans are only matched against a submap containing a few recent scans, the approach described above slowly accumulates error. For only a few dozen consecutive scans, the accumulated error is small.

由于扫描只与子图进行匹配，而子图只包含最近的几个扫描，上述的方法会缓慢的累积误差。对几十个连续的扫描，累积的误差会很小。

Larger spaces are handled by creating many small submaps. Our approach, optimizing the poses of all scans and submaps, follows Sparse Pose Adjustment [2]. The relative poses where scans are inserted are stored in memory for use in the loop closing optimization. In addition to these relative poses, all other pairs consisting of a scan and a submap are considered for loop closing once the submap no longer changes. A scan matcher is run in the background and if a good match is found, the corresponding relative pose is added to the optimization problem.

通过创建很多小的子图，来处理更大的空间。我们的方法按照稀疏姿态调整[2]对所有的扫描和子图进行姿态优化。扫描插入处的相对姿态存储在内存中，在回环检测优化的时候使用。除了这些相对姿态，所有其他的扫描和子图对，一旦子图不再变化后，都用于回环闭合。扫描匹配器在后台运行，如果发现好的匹配，对应的相对姿态就加入到优化问题中。

### 5.1. Optimization problem

Loop closure optimization, like scan matching, is also formulated as a nonlinear least squares problem which allows easily adding residuals to take additional data into account. Once every few seconds, we use Ceres [14] to compute a solution to

回环闭合优化，就像扫描匹配一样，也表述成了一个非线性最小二乘问题，可以很容易的加入残差，以将更多数据纳入考虑。每几秒一次，我们使用Ceres来计算下式的分辨率

$$argmin_{Ξm,Ξs} \sum_{ij} ρ(E^2(ξ^m_i , ξ^s_j; Σ_{ij}, ξ_{ij}))/2$$(SPA)

where the submap poses $Ξ^m = \{ξ^m_i \}_{i=1,...,m}$ and the scan poses $Ξs = \{ξ^s_j\}_{j=1,...,n}$ in the world are optimized given some constraints. These constraints take the form of relative poses $ξ_{ij}$ and associated covariance matrices $Σ_{ij}$. For a pair of submap i and scan j, the pose $ξ_{ij}$ describes where in the submap coordinate frame the scan was matched. The covariance matrices $Σ_{ij}$ can be evaluated, for example, following the approach in [15], or locally using the covariance estimation feature of Ceres [14] with (CS). The residual E for such a constraint is computed by

其中子图的姿态$Ξ^m = \{ξ^m_i \}_{i=1,...,m}$，和在这个世界中扫描的姿态$Ξs = \{ξ^s_j\}_{j=1,...,n}$，在给定的一些约束中进行优化。这些约束的形式包括相对姿态$ξ_{ij}$，和相关的协方差矩阵$Σ_{ij}$。对于子图i和扫描j的对，姿态$ξ_{ij}$描述的子图坐标系框架中的哪里与扫描进行的匹配。协方差矩阵$Σ_{ij}$可以按照[15]中的方法进行评估，或用[14]中的协方差估计特征局部的进行。对这样的约束的残差E由下式计算

$$E^2(ξ^m_i, ξ^s_j; Σ_{ij}, ξ_{ij}) = e(ξ^m_i, ξ^s_j; ξ_{ij})^T Σ^{−1}_{ij} e(ξ^m_i, ξ^s_j; ξ_{ij})$$(4)

$$e(ξ^m_i, ξ^s_j; ξ_{ij}) = ξ_{ij} − \left( \begin{matrix} R^{-1}_{ξ^m_i}(t_{ξ^m_i} - t_{ξ^s_j}) \\ ξ^m_{i;θ}-ξ^s_{j;θ} \end{matrix} \right)$$(5)

A loss function ρ, for example Huber loss, is used to reduce the influence of outliers which can appear in (SPA) when scan matching adds incorrect constraints to the optimization problem. For example, this may happen in locally symmetric environments, such as office cubicles. Alternative approaches to outliers include [16].

一个损失函数ρ，比如Huber损失函数，用于降低外点的影响，当扫描匹配对优化问题加入了不正确的约束，这就会在SPA中出现。比如，这在局部对称的环境中就会发生，比如办公室小隔间。[16]是另一种处理外点的方法。

### 5.2. Branch-and-bound scan matching

We are interested in the optimal, pixel-accurate match 我们对最优的，像素级精确度的匹配感兴趣

$$ξ^⋆ = argmax_{ξ∈W} \sum_{k=1}^K M_{nearest} (T_ξh_k)$$(BBS) 

where W is the search window and M_nearest is M extended to all of R^2 by rounding its arguments to the nearest grid point first, that is extending the value of a grid points to the corresponding pixel. The quality of the match can be improved further using (CS).

其中W是搜索窗口，M_{nearest}是M拓展到了R^2，首先将其参数四舍五入到最近的网格点，即将一个网格点上的值拓展到对应的像素上。进一步用CS，可以改进匹配的质量。

Efficiency is improved by carefully choosing step sizes. We choose the angular step size $δ_θ$ so that scan points at the maximum range d_max do not move more than r, the width of one pixel. Using the law of cosines, we derive

通过仔细选择步长，可以改进效率。我们选择角度步长$δ_θ$，使最大距离d_max处的扫描点移动不会超过r，即一个像素的宽度。使用cosine定律，我们推导出

$$d_{max} = max_{k=1,...,K} ||h_k||$$(6)

$$δ_θ = arccos (1-\frac{r^2}{2d^2_{max}})$$(7)

We compute an integral number of steps covering given linear and angular search window sizes, e.g., W_x = W_y = 7m and W_θ = 30◦,

我们计算覆盖给定的线性和角度搜索窗口大小的整数步数，比如，W_x = W_y = 7m和W_θ = 30◦

$$w_x = ⌈W_x​/r⌉, w_y = ⌈W_y​/r⌉, w_θ = ⌈W_θ​/δ_θ⌉$$(8)

This leads to a finite set W forming a search window around an estimate $ξ_0$ placed in its center, 这带来了一个有限集合W，形成了一个搜索窗口，令$ξ_0$是在其中央的一个估计

$$\overline W = \{−w_x, . . ., w_x\} × \{−w_y, . . ., w_y\} × \{−w_θ, . . ., w_θ\}$$(9)

$$W = \{ξ_0 + (rj_x, rj_y, δ_θj_θ): (j_x, j_y, j_θ) ∈ \overline W \}$$(10)

A naive algorithm to find ξ⋆ can easily be formulated, see Algorithm 1, but for the search window sizes we have in mind it would be far too slow.

找到ξ⋆的最优朴素算法可以很容易表述，见算法1，但是对于这样的搜索窗口，这个算法会非常慢。

```
Algorithm 1 Naive algorithm for (BBS)
best score ← −∞
for jx = −wx to wx do
    for jy = −wy to wy do
        for jθ = −wθ to wθ do
            score ← Σ_{k=1}^K M_{nearest}(T_{ξ_0+(rj_x,rj_y,δ_θj_θ)}h_k)
            if score > best score then
                match ← ξ0 + (rj_x, rj_y, δ_θj_θ)
                best score ← score
            end if
        end for
    end for
end for
return best score and match when set.
```

Instead, we use a branch and bound approach to efficiently compute ξ⋆ over larger search windows. See Algorithm 2 for the generic approach. This approach was first suggested in the context of mixed integer linear programs [17]. Literature on the topic is extensive; see [18] for a short overview.

我们会使用替代的branch-and-bound方法来高效的在更大的搜索窗口中计算ξ⋆，见算法2的通用方法。这种方法首先是在混合整数线性规划问题[17]中提出来的。这个话题的文献非常多，见[18]。

```
Algorithm 2 Generic branch and bound
best_score ← −∞
C ← C0
while C ̸= ∅ do
    Select a node c ∈ C and remove it from the set.
    if c is a leaf node then
        if score(c) > best_score then
            solution ← n
            best_score ← score(c)
        end if
    else
        if score(c) > best_score then
            Branch: Split c into nodes Cc
            C ← C ∪ Cc
        else
            Bound.
        end if
    end if
end while
return best_score and solution when set.
```

The main idea is to represent subsets of possibilities as nodes in a tree where the root node represents all possible solutions, W in our case. The children of each node form a partition of their parent, so that they together represent the same set of possibilities. The leaf nodes are singletons; each represents a single feasible solution. Note that the algorithm is exact. It provides the same solution as the naive approach, as long as the score(c) of inner nodes c is an upper bound on the score of its elements. In that case, whenever a node is bounded, a solution better than the best known solution so far does not exist in this subtree.

其主要思想是，将子集的可能性表示为树中的节点，其中根节点表示所有可能的解，在我们的情况中是W。每个节点的子节点形成了其父节点的分割，这样他们共同表示了同样的可能性集合。叶节点是单元素集合；每个都表示了单个的可行解。注意这个算法是精确的。会与朴素方法得到同样的解，只要内节点c的score(c)是其元素的分数的上界。在这种情况下，只要一个节点被限定了，在这个子树上就不存在比现在已知的最优解更好的解。

To arrive at a concrete algorithm, we have to decide on the method of node selection, branching, and computation of upper bounds.

为得到具体的算法，我们必须确定节点选择，分支和计算上界的方法。

*1) Node selection*: Our algorithm uses depth-first search (DFS) as the default choice in the absence of a better alternative: The efficiency of the algorithm depends on a large part of the tree being pruned. This depends on two things: a good upper bound, and a good current solution. The latter part is helped by DFS, which quickly evaluates many leaf nodes. Since we do not want to add poor matches as loop closing constraints, we also introduce a score threshold below which we are not interested in the optimal solution. Since in practice the threshold will not often be surpassed, this reduces the importance of the node selection or finding an initial heuristic solution. Regarding the order in which the children are visited during the DFS, we compute the upper bound on the score for each child, visiting the most promising child node with the largest bound first. This method is Algorithm 3.

节点选择。在没有更好的选择的情况下，我们的算法使用深度有限搜索(DFS)作为默认选项：算法的效率大部分依赖于被剪枝掉的树。这依赖于两件事：一个好的上限，和一个好的当前解。后者由DFS帮助得到，可以很迅速的计算很多叶节点。由于我们不想加入不好的匹配作为回环闭合约束，我们还引入了一个分数阈值，在这个阈值之下，我们就不认为是最优解。因为在实践中，不会经常超过阈值，这就降低了节点选择，或找到初始的启发式解的重要性。鉴于DFS访问子节点的顺序的关系，我们计算每个子节点的上限，访问最有希望的最大界限的子节点。如算法3所示。

```
Algorithm 3 DFS branch and bound scan matcher for (BBS)
best score ← score threshold
Compute and memorize a score for each element in C0.
Initialize a stack C with C0 sorted by score, the maximum score at the top.
while C is not empty do
    Pop c from the stack C.
    if score(c) > best score then
        if c is a leaf node then
            match ← ξc
            best score ← score(c)
        else
            Branch: Split c into nodes Cc.
            Compute and memorize a score for each element in Cc.
            Push Cc onto the stack C, sorted by score, the maximum score last.
        end if
    end if
end while
return best score and match when set.
```

*2) Branching rule*: Each node in the tree is described by a tuple of integers $c = (cx, cy, cθ, ch) ∈ Z^4$. Nodes at height $c_h$ combine up to $2^{c_h}×2^{c_h}$ possible translations but represent a specific rotation:

分支规则：树中的每个节点都由整数元组$c = (cx, cy, cθ, ch) ∈ Z^4$来描述。在高度$c_h$的节点，结合到一起，在旋转上有$2^{c_h}×2^{c_h}$种可能，但是表示一种特定的旋转：

$$\overline{\overline W} = (\{ (j_x, j_y) ∈ Z^2:  c_x ≤ j_x < c_x + 2^{c_h}, c_y ≤ j_y < c_y + 2^{c_h} \} × \{c_θ\})$$(11)

$$\overline W_c = \overline{\overline W} ∩ \overline W$$(12)

Leaf nodes have height $c_h$ = 0, and correspond to feasible solutions $W ∋ ξ_c = ξ_0 + (rc_x, rc_y, δ_θc_θ)$.

叶节点的高度$c_h$ = 0，对应的可行解为$W ∋ ξ_c = ξ_0 + (rc_x, rc_y, δ_θc_θ)$。

In our formulation of Algorithm 3, the root node, encompassing all feasible solutions, does not explicitly appear and branches into a set of initial nodes $C_0$ at a fixed height $h_0$ covering the search window

在我们对算法3的表述中，根节点包含所有的可行解，但并不是显示的出现的，在一个固定的高度$h_0$覆盖了搜索窗口分支成一些初始节点的集合$C_0$

$$\begin{matrix} \overline W_{0,x} = \{ -w_x+2^{h_0}j_x: j_x∈Z, 0≤2^{h_0}j_x≤2w_x \} \\ \overline W_{0,y} = \{ -w_y+2^{h_0}j_y: j_y∈Z, 0≤2^{h_0}j_y≤2w_y \} \\ \overline W_{0,θ} = \{ j_θ∈Z: -w_θ≤j_θ≤w_θ \} \\ C_0 = \overline W_{0,x} × \overline W_{0,y} × \overline W_{0,θ} × \{ h_0 \} \end{matrix}$$(13)

At a given node c with $c_h > 1$, we branch into up to four children of height $c_h − 1$

在给定的高度为$c_h > 1$的节点c，我们分支成4个高度为$c_h − 1$的子节点

$$C_c = ((\{c_x, c_x + 2^{c_h−1}\} × \{c_y, c_y + 2^{c_h−1}\} × c_θ) ∩ \overline W) × \{c_h − 1\}$$(14)

*3) Computing upper bounds*: The remaining part of the branch and bound approach is an efficient way to compute upper bounds at inner nodes, both in terms of computational effort and in the quality of the bound. We use

计算上限。分支界定算法的剩下部分，是计算在内部节点上的上限的高效方法，计算效率要高，界限的质量也要高。我们使用的是

$$score(c) = \sum_{k=1}^K max_{j∈\overline{\overline W_c}} M_{nearest}(T_{ξ_j}h_k) ≥ \sum_{k=1}^K max_{j∈\overline W_c} M_{nearest}(T_{ξ_j}h_k) ≥ max_{j∈\overline W_c} \sum_{k=1}^K M_{nearest}(T_{ξ_j}h_k)$$(15)

To be able to compute the maximum efficiently, we use precomputed grids $M^{c_h}_{precomp}$. Precomputing one grid per possible height $c_h$ allows us to compute the score with effort linear in the number of scan points. Note that, to be able to do this, we also compute the maximum over $\overline{\overline W_c}$ which can be larger than $\overline W_c$ near the boundary of our search space.

为高效的计算最大值，我们使用预计算的网格$M^{c_h}_{precomp}$。在每一个可能的高度$c_h$预计算一个网格，使我们计算分数的工作量与扫描点数量呈线性关系。注意，为能够做这个，我们还计算了$\overline{\overline W_c}$的最大值，这会比$\overline W_c$在我们搜索空间的边缘要更大。

$$score(c) = \sum_{k=1}^K M^{c_h}_{precomp} (T_{ξ_c}h_k)$$(16)

$$M^h_{precomp} (x,y) = max_{x'∈[x,x+r(2^h−1)], y′∈[y,y+r(2^h−1)]} M_{nearest}(x', y')$$(17)

with $ξ_c$ as before for the leaf nodes. Note that $M^h_{precomp}$ has the same pixel structure as $M_{nearest}$, but in each pixel storing the maximum of the values of the $2^h × 2^h$ box of pixels beginning there. An example of such precomputed grids is given in Figure 3.

$ξ_c$像以前一样是叶节点。注意，$M^h_{precomp}$与$M_{nearest}$的像素结构是一样的，但在每个像素中，存储了从这里开始的$2^h × 2^h$个最大值。这样的预计算网格的一个例子如图3所示。

To keep the computational effort for constructing the precomputed grids low, we wait until a probability grid will receive no further updates. Then we compute a collection of precomputed grids, and start matching against it.

为保持构建预计算网格的计算工作量比较低，我们等待，直到概率网格不会收到任何更新。然后我们计算预计算网格的集合，然后对其进行匹配。

For each precomputed grid, we compute the maximum of a $2^h$ pixel wide row starting at each pixel. Using this intermediate result, the next precomputed grid is then constructed.

对每个预计算的网格，我们计算从每个像素起使，最大$2^h$个像素宽。使用这种中间结果，然后就可以构建下一个预计算网格。

The maximum of a changing collection of values can be kept up-to-date in amortized O(1) if values are removed in the order in which they have been added. Successive maxima are kept in a deque that can be defined recursively as containing the maximum of all values currently in the collection followed by the list of successive maxima of all values after the first occurrence of the maximum. For an empty collection of values, this list is empty. Using this approach, the precomputed grids can be computed in O(n) where n is the number of pixels in each precomputed grids.

变化的集合的值的最大值，要保持更新，如果值移除的顺序是加入的顺序，那么复杂度为摊销的O(1)。后续的最大值保存在双队列中，这个队列是迭代定义来的，首先包含目前集合中所有值的最大值，然后是在出现的第一个最大值之后，所有值的后续最大值的列表。对于值的空集合，这个列表是空的。使用这种方法，预计算的网格可以以O(n)的复杂度计算，其中n是每个预计算网格的像素数量。

An alternative way to compute upper bounds is to compute lower resolution probability grids, successively halving the resolution, see [1]. Since the additional memory consumption of our approach is acceptable, we prefer it over using lower resolution probability grids which lead to worse bounds than (15) and thus negatively impact performance.

计算上限的另外一种方式，是计算低分辨率概率网格，然后再将分辨率减半，见[1]。由于我们方法可以接受额外的内存消耗，所以我们不推荐使用更低的分辨率概率网格，会比(15)得到更坏的界限，因此对性能有负面影响。

## 6. Experimental Results

In this section, we present some results of our SLAM algorithm computed from recorded sensor data using the same online algorithms that are used interactively on the backpack. First, we show results using data collected by the sensors of our Cartographer backpack in the Deutsches Museum in Munich. Second, we demonstrate that our algorithms work well with inexpensive hardware by using data collected from a robotic vacuum cleaner sensor. Lastly, we show results using the Radish data set [19] and compare ourselves to published results.

本节中，我们给出我们的SLAM算法在记录的传感器数据上计算得到的一些结果，同样的在线算法也在背包中互动使用。首先，我们展示我们的Cartographer背包在慕尼黑Deutsches博物馆收集的数据结果。第二，我们证明了我们的算法在并不昂贵的硬件上工作效果很好，使用的数据是从机器人真空清洁工传感器收集来的。最后，我们展示使用Radish数据集的结果，与发表的结果进行了比较。

### 6.1. Real-World Experiment: Deutsches Museum

Using data collected at the Deutsches Museum spanning 1,913 s of sensor data or 2,253 m (according to the computed solution), we computed the map shown in Figure 4. On a workstation with an Intel Xeon E5-1650 at 3.2 GHz, our SLAM algorithm uses 1,018 s CPU time, using up to 2.2 GB of memory and up to 4 background threads for loop closure scan matching. It finishes after 360 s wall clock time, meaning it achieved 5.3 times real-time performance.

使用在Deutsches博物馆收集的1913s传感器数据或2253米的数据（根据计算的分辨率），我们计算得到的地图如图4所示。

The generated graph for the loop closure optimization consists of 11,456 nodes and 35,300 edges. The optimization problem (SPA) is run every time a few nodes have been added to the graph. A typical solution takes about 3 iterations, and finishes in about 0.3s.

对回环闭合优化生成的图，包含11456个节点和35300个边。每有几个节点加入到图中，就运行一下优化问题(SPA)。一个典型的解大概需要3次迭代得到，经过0.3秒得到结果。

### 6.2. Real-World Experiment: Neato’s Revo LDS

Neato Robotics uses a laser distance sensor (LDS) called Revo LDS [20] in their vacuum cleaners which costs under 30 dollars. We captured data by pushing around the vacuum cleaner on a trolley while taking scans at approximately 2 Hz over its debug connection. Figure 5 shows the resulting 5 cm resolution floor plan. To evaluate the quality of the floor plan, we compare laser tape measurements for 5 straight lines to the pixel distance in the resulting map as computed by a drawing tool. The results are presented in Table I, all values are in meters. The values are roughly in the expected order of magnitude of one pixel at each end of the line.

Neato机器人使用了一个激光距离传感器(LDS)称为Revo LDS[20]，用在了真空扫地机器人上，价格低于30美元。我们将真空扫地机器人放在一个手推车上推，通过debug连接以大约2Hz的频率进行扫描，收集数据。图5展示了得到的5cm分辨率的平面图。为评估平面图的质量，我们用5线激光带测量的结果，与得到的地图中用画图工具计算的像素距离进行比较。结果如表1所示，所有值的单位都是米。

Table 1. Quantitative Errors with Revo LDS

Laser Tape | Cartographer | Error (absolute) | Error (relative)
--- | --- | --- | ---
4.09 | 4.08 | −0.01 | −0.2%
5.40 | 5.43 | +0.03 | +0.6%
8.67 | 8.74 | +0.07 | +0.8%
15.09 | 15.20 | +0.11 | +0.7%
15.12 | 15.23 | +0.11 | +0.7%

### 6.3. Comparisons using the Radish data set

We compare our approach to others using the benchmark measure suggested in [21], which compares the error in relative pose changes to manually curated ground truth relations. Table II shows the results computed by our Cartographer SLAM algorithm. For comparison, we quote results for Graph Mapping (GM) from [21]. Additionally, we quote more recently published results from [9] in Table III. All errors are given in meters and degrees, either absolute or squared, together with their standard deviation.

我们使用[21]建议的基准测试测量，来比较我们的方法与其他的方法，基准测试中比较的是相对误差变化对手工维护的真值的误差。表2展示的是由我们的Cartographer SLAM算法计算得到的结果。为进行比较，我们引用了[21]中图映射(Graph Mapping, GM)的结果。另外，我们在表3中引用了最近发表的结果[9]。所有的误差都是以米和度给出的，要么是绝对值或平方，与其标准偏差一起给出。

Each public data set was collected with a unique sensor configuration that differs from our Cartographer backpack. Therefore, various algorithmic parameters needed to be adapted to produce reasonable results. In our experience, tuning Cartographer is only required to match the algorithm to the sensor configuration and not to the specific surroundings.

每个公开数据集都是用独特的传感器配置收集的，与我们的Cartographer背包是不同的。因此，各种算法参数需要调整，以生成合理的结果。以我们的经验，调节Cartographer只需要将算法与传感器配置进行匹配，不需要对具体的环境匹配。

Since each public data set has a unique sensor configuration, we cannot be sure that we did not also fit our parameters to the specific locations. The only exception being the Freiburg hospital data set where there are two separate relations files. We tuned our parameters using the local relations but also see good results on the global relations. The most significant differences between all data sets is the frequency and quality of the laser scans as well as the availability and quality of odometry.

由于每个公开的数据集的传感器配置都很独特，我们不能确定，我们的参数与具体的位置不能适配。唯一的例外是Freiburg医院数据集，有两个不同的关联文件。我们使用其局部关系调节我们的参数，但也在全局关系中看到了很好的结果。所有数据集之间最显著的差异是，激光扫描的频率和质量，以及里程计的可用性和质量。

Despite the relatively outdated sensor hardware used in the public data sets, Cartographer SLAM consistently performs within our expectations, even in the case of MIT CSAIL, where we perform considerably worse than Graph Mapping. For the Intel data set, we outperform Graph Mapping, but not Graph FLIRT. For MIT Killian Court we outperform Graph Mapping in all metrics. In all other cases, Cartographer outperforms both Graph Mapping and Graph FLIRT in most but not all metrics.

尽管公开数据集中使用的传感器硬件略微陈旧，Cartographer SLAM的表现一直在我们的预期之内，即使在MIT CSAIL的情况下，我们的表现比图映射(GM)差了不少。对于Intel数据集，我们超过了GM的性能，但并不是Graph FLIRT。对于MIT Killian Court，我们在所有测量中都超过了GM的性能。在所有其他情况中，Cartographer在大部分但并不是所有的测量中，都超过了GM和Graph FLIRT。

Since we add loop closure constraints between submaps and scans, the data sets contain no ground truth for them. It is also difficult to compare numbers with other approaches based on scan-to-scan. Table IV shows the number of loop closure constraints added for each test case (true and false positives), as well as the precision, that is the fraction of true positives. We determine the set of true positive constraints to be the subset of all loop closure constraints which are not violated by more than 20 cm or 1◦ when we compute (SPA). We see that while our scan-to-submap matching procedure produces false positives which have to be handled in the optimization (SPA), it manages to provide a sufficient number of loop closure constraints in all test cases. Our use of the Huber loss in (SPA) is one of the factors that renders loop closure robust to outliers. In the Freiburg hospital case, the choice of a low resolution and a low minimum score for the loop closure detection produces a comparatively high rate of false positives. The precision can be improved by raising the minimum score for loop closure detection, but this decreases the solution quality in some dimensions according to ground truth. The authors believe that the ground truth remains the better benchmark of final map quality.

由于我们在子图和扫描之间加入了回环闭合的约束，数据集对其并没有真值。与其他基于扫描对扫描的方法，也很难进行比较。表4给出了对每个测试案例中加入的回环闭合约束数量(true and false positives)，以及精度，即true positives的比例。在我们计算SPA的时候，没有违反多余20cm或1◦的情况，这些构成的所有回环闭合约束的子集，我们认为是true positive约束集合。在我们的扫描对子图的匹配过程中，产生的false positives需要被优化过程SPA进行处理，但在所有的测试案例中，都产生了足够多的回环检测约束的数量。我们在SPA中使用了Huber损失，这是使得回环闭合对外点稳健的一个因素。在Freiburg医院的案例中，对回环检测选择了低分辨率，选择了低的最小分数，产生的false positives数量相对较多。通过提升回环检测的最低分，精度可以得到改进，但根据真值，这在一些维度中降低了解的质量。作者相信，真值仍然是最终的地图质量中的更好的基准。

The parameters of Cartographer’s SLAM were not tuned for CPU performance. We still provide the wall clock times in Table V which were again measured on a workstation with an Intel Xeon E5-1650 at 3.2 GHz. We provide the duration of the sensor data for comparison.

Cartographer的SLAM的参数，并没有对CPU的性能进行调整。我们在表5中给出了处理时间，这是在一个3.2GHz的Intel Xeon E5-1650上进行的测量。我们给出了传感器数据的持续时间以进行比较。

## 7. Conclusions

In this paper, we presented and experimentally validated a 2D SLAM system that combines scan-to-submap matching with loop closure detection and graph optimization. Individual submap trajectories are created using our local, grid-based SLAM approach. In the background, all scans are matched to nearby submaps using pixel-accurate scan matching to create loop closure constraints. The constraint graph of submap and scan poses is periodically optimized in the background. The operator is presented with an up-to-date preview of the final map as a GPU-accelerated combination of finished submaps and the current submap. We demonstrated that it is possible to run our algorithms on modest hardware in real-time.

本文中，我们提出了一种2D SLAM系统，将扫描对子图的匹配与回环检测与图优化结合到了一起，并通过试验进行了验证。使用我们局部的，基于网格的SLAM方法，可以创建单个子图轨迹。在后台，所有扫描都会与附近的子图进行匹配，使用像素级准确的扫描匹配方法来创建回环约束。子图和扫描姿态的约束图，在后台进行周期性优化。会给操作员的，是更新的最终图的预览，是已经完成的子图和目前子图的GPU加速的结合显示。我们证明了，在一般的硬件上，以实时的速度可以运行我们的算法。