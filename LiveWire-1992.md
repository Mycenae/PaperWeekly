# Adaptive boundary detection using "live-wire" two dimensional dynamic programming

Eric Mortensen et. al.

## 0. Abstract

An adaptive boundary detection algorithm that uses two-dimensional dynamic programming is presented. The algorithm is less constrained than previous one-dimensional dynamic programming algorithms and allows the user to interactively determine the mathmatically optimal boundary between a user-selected seed point and any other dynamically selected "free" point in the image.

本文提出了一种自适应的边缘检测算法，使用的是2D动态规划。算法比之前的1D动态规划算法约束更少，允许用户在图像中用户选择的种子点和任何其他动态选择的自由点之间，互动的确定数学上最优的边缘。

Interactive movement of the free point by the cursor causes the boundary to behave like a "live wire" as it adapts to the new minimum cost path between the seed point and the currently selected free point. The algorithm can also be adapted or customized to learn boundary-defining features for a particular class of images.

自由点通过鼠标的互动运动，导致边缘的表现就像live wire，因为其会适应新的在种子点和目前选择的自由点之间的最小代价路径。算法也可以经过调整或定制，对特定类别的图像，学习定义边缘的特征。

Adaptive 2D DP perfoms well on a variety of images (angiocardiograms, CT, MRI). In particular, it accurately detects the boundaries of low contrast objects, such as occur with intravenous injections, as well as those found in noisy, low SNR images.

自适应2D DP在很多图像中表现良好（心血管造影图像，CT，MRI）。特别是，其可以准确的检测低对比度物体的边缘，比如静脉注射中的图像，以及那些含噪的，低信噪比的图像。

## 1. Introduction

Defining an object's boundary is a general problem in medical imaging as well as many other imaging processing fields. The segmentation problem (i.e. defining the boundaries/areas of the objects of interest in an image) has not been solved in a fully automated way. Many current edge following techniques exist which employ local edge gradient and/or  orientation information combined, at times, with some idea of an object template. Such techniques are limited by the relative local strength of the edge "creteria" as compared to the creteria for neighboring edges and/or noise.

在医学影像，以及其他很多图像处理领域中，定义目标的边缘是一个很通用的问题。完全自动的分割问题（即，在图像中定义感兴趣目标的边缘/区域），目前并没有解决。很多目前存在的边缘追随技术，采用的都是局部边缘梯度和方向信息的结合，有时候会与目标模板的思想结合。这种技术，与邻域边缘和噪声的准则相比，受限于边缘准则的相对局部强度。

Dynamic programming (DP) attempts to overcome the problems associated with using only local information for edge following. It does this by employing both local gradient information with global "edge cost" information. In general, one-dimensional dynamic programming (1-D DP)[1-4,6] attemps to discover a globally optimal edge path but imposes directional sampling and searching constrains for two-dimensional (2-D) images: thus requiring 2-D bounary templates.

动态规划试图克服与只使用局部信息进行边缘追随的问题。通过采用局部梯度信息，和全局的“边缘代价”信息，可以达到这种效果。一般来说，1D动态规划(1D DP)试图找到全局最优的边缘图，但对于2D图像来说，提出了方向采样和搜索的约束，因此需要2D边缘模板。

This paper presents a new technique in dynamic programming which allows freedom in two variables (2-D) as compred to freedom in one variable for 1-D DP. Two-dimensional dynamic programming (2-D DP) may discover a globally optimal boundary path that is allowed to vary freely between any given starting seed point and any other free point in the image. Further, 2-D DP can be used to dynamically and interactively define the desired boundary using an active contour ("live-wire") with minimal user interaction -- typically two to four seed points per boundary.

本文提出了一种动态规划的新技术，与1D DP中的一个变量的自由度相比，可以使用2个变量的自由度。2D DP可能会找到全局最优的边缘路径，可以在任意给定的开始种子点和其他自由点之间变化。而且，2D DP可以使用最小用户交互的主动边缘(live-wire)，一般每个边缘2到4个种子点，用于动态的互动的定义期望的边缘。

## 2. 2-D dynamic programming

As with 1-D DP, 2-D dynamic programming can be formulated as a directed graph-searching problem. The goal is to find the globally "optimal" (least cost or greatest cost) path between start and end points or nodes in the directed graph.

与1D DP一样，2D DP可以表述为有向图搜索问题。目标是要找到在有向图中的开始点和最终点或节点之间的全局“最优”路径（最小代价或最大代价）。

Formulating dynamic programming as a graph-searching problem requires nodes and arcs between nodes. For 2-D images, pixels become nodes with (initial) local costs calculated for each pixel and its neighbors. Arcs are defined as connections between a pixel and its neighbors. We define the globally "optimal" path as the minimum cumulative cost path between the start and end points.

将动态规划表述为图搜索问题，需要节点之间的节点和连接弧。对于2D图像，像素就很自然的成为了节点，对每个像素及其邻域进行计算，就可以得到（初始）局部代价。连接弧可以定义为像素及其邻域间的连接。我们定义全局“最优”路径为，开始点和终点之间的最小累积代价。

### 2.1 Connectivity

The basic difference between 1-D and 2-D dynamic programming lies in the defining the connectivity between nodes/pixels. 1-D DP constrains node connections to be in the approximate direction of the node. That is, node connections must be "in front" of each current node towards the end node. Thus, 1-D DP constrains the path to freedom in only one variable (y).

1D和2D动态规划之间的基本区别是，在节点/像素之间定义连接性的不同。1D DP将节点的连接，限制在节点的大致方向上。即，节点的连接必须要在每个当前节点朝向最终节点的方向的前面。因此，1D DP将路径约束为只有一个变量的自由度(y)。

However, 2-D DP allows freedom in two variables (x and y). Thus, node connections exist for a node's entire neighborhood (i.e., in front, to the side, and in back). Figure 1 illustrates the differences between 1-D and 2-D connectivity.

但是，2D DP允许两个变量的自由度(x,y)。因此，节点的连接在节点的整个邻域中都是可能存在的（即，在前方，在侧方，在后方）。图1描述了1D和2D连接性的差异。

These connectivity differences can result in a different globally optimal 1-D DP path than for a 2-D DP path. Figure 2 shows how a globally optimal 1-D DP path can differ from a 2-D DP path. As can be seen, 1-D DP must cut across the peninsula since it can only search forward (in the x direction).

这些连接性差异，对于1D DP路径和2D DP路径，会得到不同的全局最优路径。图2展示了，全局最优的1D DP和2D DP之间会有什么样的差异。可以看到，1D DP必须穿越半岛结构，因此只能往前搜索（在x方向）。

### 2.2 Cost matrix

Initally, we generate a two-dimensional "cost matrix" where every element in the matrix corresponds to an image pixel with the cost defined by the pixel's local boundary criteria. The cost matrix is generated as a function of the image's gradient magnitute and gradient orientation. Letting $G_x$ and $G_y$ represent the horizontal and vertical gradients of the image, then the cost matrix c(x,y) is generated as follows:

初始，我们生成一个2D代价矩阵，矩阵中的每个元素是一个像素点的代价，定义为像素的局部边缘准则。代价矩阵生成为图像梯度幅值和梯度方向的函数。令$G_x$和$G_y$表示图像的水平方向和垂直方向梯度，代价矩阵c(x,y)按照下式生成：

$$G(x,y) = \sqrt{G_x^2(x,y)+G_y^2(x,y)}$$
$$O(x,y) = tan^{-1}(\frac{G_x(x,y)}{G_y(x,y)})$$
$$c(x,y)= [max_{x,y}(G(x,y))-G(x,y)] + α|O(x,y)*f - O(x,y)|$$

where G(x,y) and O(x,y) are the gradient magnitude and orientation images respectively, f is an averaging or gaussian filter and α is a scaling factor. Note that the gradient magnitude image is subtracted from the maximum gradient magnitude value so that strong gradients represents low costs. Further, the absolute difference between the low-pass filtered orientation image and itself will also be low if a pixel's neighborhood contains gradient orientations that are similar to its neighbors. Figure 3a is an example cost matrix.

其中G(x,y)和O(x,y)分别是图像的梯度幅度和方向，f是一个平均滤波或或高斯滤波，α是一个标量。注意梯度幅度由最大梯度幅度值减去，这样强梯度代表低代价。而且，低通滤波的方向图及方向图的差的绝对值，如果像素邻域的梯度方向与其邻域类似，那么其值就会很低。图3a是代价矩阵的一个例子。

Associated with the cost matrix is a connection weighting matrix. Each connection is "weighted" such that diagonal connections have higher cost than horizontal or vertical connections. The horizontal and vertical connections have unity weights whereas the diagonal connections are weighted by $\sqrt 2$, thus maintain Euclidean distance weighting for the neighborhood.

与代价矩阵相关的是，连接加权矩阵。每个连接都是加权的，这样对角连接的代价比水平连接或垂直连接更大。水平和垂直连接加权为1，而对角连接的加权为$\sqrt 2$，因此对邻域中保持欧式距离加权。

### 2.3 Optimal path generation

Since the cost matrix and graph connections are defined in image space terms, image space terms will be used to describe the 2-D DP algorithm. 由于代价矩阵和图连接都是在图像空间中定义的，图像空间的术语就用于描述2D DP算法。

Unlike most other dynamic programming or graph-searching algorithms, we do not define both a start point and end point. Rather, we calculate the globally optimal path from a start or seed point to all other points/pixels in the image in an incremental fashion. The algorithm is similar to a well-known heuristic search algorithm[5] and is as follows:

与多数其他的动态规划或图搜索算法不同，我们不会同时定义开始点与终点。我们是从一个起始点或种子点开始，到所有其他点/像素，以递增的方式计算全局最优路径。算法与著名的启发式搜索算法[5]很类似，如下所述：

```
c_x -> local cost for point x.
p_x -> optimal path pointer from point x.
w_xy -> connection weigh between points x and y.
T_x -> total accumulated cost to point x.
L -> list of "active" points sorted by total costs (initially empty).

add seed point to L;
while L not empty do begin
    x <- min(L)
    for each y in the neighborhood of x do
        if y not processed then begin
            T_y <- T_x + w_xy c_y;
            add y to L;
            p_y <- x; (set pointer from y to x)
            if y is a diagonal then mark p_y as diagonal
        else if p_y is marked as diagonal and y is not a diagonal then begin
            if T_x + w_xy c_y < T_y then begin
                remove y from L;
                T_y <- T_x + w_xy c_y;
                add y to L;
            end
        end
    end
end
```

Starting with a user-defined seed point, the algorithm places that point on an initially empty sorted list, L. The point, x, with minimum total cost, T_x (initially on the seed point), is then removed from the list and checked for unprocessed neighbors. A total cost, T_y, is then computed for each unprocessed point y in the neighborhood of x. T_y is computed as the sum of the total cost to x, T_x, and the weighted local cost from x to y, w_xy*c_y. Each neighboring point, y, along with its associated total cost, is added to the sorted list for later processing and the optimal path pointer for the neighbor is set to the point being processed. If the neighboring point is a diagonal then it is marked as such since it may be necessary to recompute the total cost to that point.

从一个用户定义的种子点开始，算法将这个点放入初始是空的排序列表L中。最小总计代价T_x的点x（最开始是种子点），从列表中去掉，然后检查未处理的邻域。对x的邻域中的每个未处理的点y，计算其总计代价。T_y计算为，x的总计代价T_x，加上从x到y的加权的局部代价w_xy*c_y。每个邻域点y，与其相关的总计代价，加入到排序列表中，以进行后续的处理，邻域的最优路径指针设置为正处理的点。如果邻域点是对角点，那么将其进行标记，因为可能需要重新计算到这个点的总计代价。

Figure 3 gives an example of how the total cost and the least cost path is computed. Figure 3a is the initial local cost matrix with the seed point circled. Figure 3b shows the total cost/path matrix after the seed point has been processed. Figure 3c shows the matrix after processing 2 points - the seed point and the next lowest total cost point on the sorted list. Notice how the points diagonal to the seed point have changed total cost and direction pointers. The Euclidean weighting between the seed and diagonal points make them more expensive than non-diagonal paths. Figure 3d, 3e, and 3f show the matrix at various stages of completion. Note how the algorithm produces a "wave-front" of active points and that the wave-front grows out faster were there are lower costs. Thus the wave-front grows out more quickly along edges in the image.

图3给出了一个计算总计代价和最低代价路径的例子。图3a是初始代价矩阵，并圈出了种子点。图3b是处理了种子点后的总计代价/路径矩阵。图3c展示了处理了2个点后的矩阵，种子点和排序列表中的下一个最低总计代价点。注意与种子点为对角关系的点是怎样改变总结代价和方向指针的。种子点和对角点的欧式加权，使其比非对角路径代价更加昂贵。图3d, 3e和3f展示了在不同完成阶段的矩阵。注意算法是怎样生成一个活跃点的波前的，如果有更低的代价，那么波前就会增长的更快。因此波前在图像中沿着边缘增长的特别快。

### 2.4 Live-wire 2-D DP

Once the path matrix is finished, a boundary path can be chosen dynamically via a "free" point. Interactive movement of the free point by cursor position causes the boundary to behave like a live-wire as it adapts to the new minimum cost path. Thus, by constraining the seed point and the free point to lie near a given edge, the user is able to interactively "warp" the live-wire boundary around the object of interest. When movement of the free point causes the boundary to digress from the desired object edge, input of a new seed point prior to the point of departure reinitiates the 2-D DP boundary detection. This causes potential paths to be recomputed from the new seed point, while effectively "tieing off" the boundary computed up to the new seed point. Figures 4 and 5 show two example images (an MRI scan and a left ventriculogram) and indicate how many seed points each outlined object required.

一旦路径矩阵计算结束，可以动态的通过一个自由点来选择一个边缘路径。自由点通过鼠标的互动运动，会导致边缘的行为，在其适应到新的最小代价路径时，就像一个带电导线。因此，通过约束种子点和自由点在一个给定的边缘附近，用户可以互动的将带电导线的边缘变形到感兴趣目标的周围。当自由点的运动，导致边缘偏离了期望的目标边缘，在偏移点之间要输入一个新的种子点，就会重新初始化这个2D DP边缘检测过程。这导致潜在的路径从新的种子点处进行重新计算，同时将计算出来的边缘与新的种子点联系到一起。图4和5展示了两幅图像的例子（一个MRI和一个左心室造影），指示出了勾画出目标需要多少自由点。

## 3. Results

Adaptive 2-D DP performs well on a variety of images (angiocardiograms, CT, MRI). In particular, it accurately detects the boundaries of low contrast objects, such as occur with intravenous injections, as well as those found in noisy, low SNR images. Boundaries are typically detected with two to four seed points.

自适应2D DP在很多图像上表现良好（心血管造影，CT，MRI）。特别是，它可以准确的检测到低对比度目标的边缘，比如在静脉注射时的图像，以及在含噪、低SNR图像中的边缘。一般用2到4个种子点，就可以找到边缘。

The algorithm's computational complexity for a N image pixels is O(N). This can be seen by examining the algorithm in a worst case situation. Suppose first that all the weights were unity, then once a total cost to a point is computed, it is not computed again since that total cost already represents the minimum cost to that pixel. Thus the total cost is computed only once for each of the N pixels. There is also some computation required to add the point to a sorted list. But the unique conditions of this algorithm allow us to use a sort algorithm that requires only an array indexing operation (indexed by total cost) and changing two pointers. Thus, the computation complexity for sorting N points is N. Now, since the diagonal weights are not unity, then it may become necessary to recompute the cost to those points processed initally as diagonal neighbors. In the worst case, this will have to be done for half the points in the matrix since half the points will be initially processed as diagonal neighbors. This means in the worst case that total costs for N/2 points will have to be recomputed and those points will have to be re-added to the sorted list again. The algorithm's computational complexity is therefore N (to calculate the total costs) + N (to add N points to the sorted list) + N (to recompute the costs for diagonals and add them again to the list) = 3N, or O(N). This is comparable to the complexity for the more restricted 1-D DP algorithm.

算法的计算复杂度，对于N个像素，为O(N)。这可以通过在最坏的情况下检查算法得到。假设开始情况下，所有的权重都是1，那么一旦计算了一个点的总计代价，那么就不会再一次计算了，因为总计代价已经表示了到这个像素的最低代价。因此总计代价对于N个像素都计算了一遍。将点加入到排序列表中，也需要一些计算量。但这个算法的唯一条件，使我们可以使用排序算法，只需要一个阵列索引运算（由总计代价索引），以及改变两个指针。。因此，排序N个点的计算复杂度是N。现在，由于对角权重不是1，因此可能需要重新计算这些指针的代价，它们初始的时候是按照对角邻域计算的。在最坏的情况下，对矩阵中一半的点都需要进行计算，因此一半的点在开始时都会按照对角邻域进行计算。这意味着，在最坏的情况下，N/2个点的总计代价需要进行重新计算，这些点需要重新加入到排序列表。算法的计算复杂度因此是N（计算总计代价）+N（将N个点加入到排序列表）+N（重新计算对角点的代价，并将其重新加入到列表中）=3N，或O(N)。这与约束更多的1D DP算法相比，是差不多的。

The algorithm was implemented on a IBM compatible 33MHz 386 with 387 co-processor and a hardware imaging board. Generating the cost matrix for a 512x512 image requires approximately 45 seconds but only needs to be done once per image. The user then selects a seed point interactively with the mouse. "Growing" the optimal path map requires up to one and a half minutes per seed point for a 512x512 image but this process is usually interrupted after 15 to 20 seconds when the DP wave-front encloses the desired object or point. The user can then use the path matrix to interactively wrap a boundary around the desired objct. Though this process requires a small delay for each seed point, we are currently porting this algorithm to HP workstations where we expect to generate the optimal path matrix at interactive speeds.

算法在IBM兼容机33MHz 386上进行了实现，带有387协处理器，有一个硬件成像板。生成512x512的代价矩阵，大约需要45s，但每幅图像只需要计算一次。用户然后用鼠标选择一个种子点。生长出最佳路径图，在512x512的图像中，每个种子点最多需要1分半钟，但这个过程通常在15到20s后就打断了，这时DP波前包围了期望的目标或点。用户然后使用路径矩阵，互动的在期望的目标周围变形出一个边缘。虽然这个过程对每个种子点有一个小延迟，我们正在将这个算法移植到HP工作站中，可以以互动的速度生成最优路径矩阵。

## 4. Conclusions

An algorithm has been presented for iterative determination of globally optimal paths derived from local gradient magnitude and orientation information. The algorithm uses two-dimensional dynamic programming and can be applied to a variety of image types and anatomy. 2-D DP performs well based on visual comparison and is the same order of complexity as the more restrictive 1-D DP.

提出了一种算法，可以互动的从局部梯度幅度和方向信息中确定全局最优路径。算法使用2D DP，可以应用于很多图像类型，很多解剖结构中。2D DP基于视觉比较表现良好，与更多约束的1D DP算法复杂度类似。

By calculating the optimal path from all points to a seed point, 2-D DP accommodates interactive selection of the desired optimal path via a "live-wire", making it a valuable interactive tool for defining an object's boundaries.

通过计算所有点到一个种子点的最优路径，2D DP可以通过live-wire进行互动选择期望的最优路径，成为定义目标边缘的一个重要互动工具。