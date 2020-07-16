# Adaptive boundary detection using "live-wire" two dimensional dynamic programming

Eric Mortensen et. al.

## 0. Abstract

An adaptive boundary detection algorithm that uses two-dimensional dynamic programming is presented. The algorithm is less constrained than previous one-dimensional dynamic programming algorithms and allows the user to interactively determine the mathmatically optimal boundary between a user-selected seed point and any other dynamically selected "free" point in the image.

Interactive movement of the free point by the cursor causes the boundary to behave like a "live wire" as it adapts to the new minimum cost path between the seed point and the currently selected free point. The algorithm can also be adapted or customized to learn boundary-defining features for a particular class of images.

Adaptive 2D DP perfoms well on a variety of images (angiocardiograms, CT, MRI). In particular, it accurately detects the boundaries of low contrast objects, such as occur with intravenous injections, as well as those found in noisy, low SNR images.

## 1. Introduction

Defining an object's boundary is a general problem in medical imaging as well as many other imaging processing fields. The segmentation problem (i.e. defining the boundaries/areas of the objects of interest in an image) has not been solved in a fully automated way. Many current edge following techniques exist which employ local edge gradient and/or  orientation information combined, at times, with some idea of an object template. Such techniques are limited by the relative local strength of the edge "creteria" as compared to the creteria for neighboring edges and/or noise.

Dynamic programming (DP) attempts to overcome the problems associated with using only local information for edge following. It does this by employing both local gradient information with global "edge cost" information. In general, one-dimensional dynamic programming (1-D DP)[1-4,6] attemps to discover a globally optimal edge path but imposes directional sampling and searching constrains for two-dimensional (2-D) images: thus requiring 2-D bounary templates.

This paper presents a new technique in dynamic programming which allows freedom in two variables (2-D) as compred to freedom in one variable for 1-D DP. Two-dimensional dynamic programming (2-D DP) may discover a globally optimal boundary path that is allowed to vary freely between any given starting seed point and any other free point in the image. Further, 2-D DP can be used to dynamically and interactively define the desired boundary using an active contour ("live-wire") with minimal user interaction -- typically two to four seed points per boundary.

## 2. 2-D dynamic programming

As with 1-D DP, 2-D dynamic programming can be formulated as a directed graph-searching problem. The goal is to find the globally "optimal" (least cost or greatest cost) path between start and end points or nodes in the directed graph.

Formulating dynamic programming as a graph-searching problem requires nodes and arcs between nodes. For 2-D images, pixels become nodes with (initial) local costs calculated for each pixel and its neighbors. Arcs are defined as connections between a pixel and its neighbors. We define the globally "optimal" path as the minimum cumulative cost path between the start and end points.

### 2.1 Connectivity

The basic difference between 1-D and 2-D dynamic programming lies in the defining the connectivity between nodes/pixels. 1-D DP constrains node connections to be in the approximate direction of the node. That is, node connections must be "in front" of each current node towards the end node. Thus, 1-D DP constrains the path to freedom in only one variable (y).

However, 2-D DP allows freedom in two variables (x and y). Thus, node connections exist for a node's entire neighborhood (i.e., in front, to the side, and in back). Figure 1 illustrates the differences between 1-D and 2-D connectivity.

These connectivity differences can result in a different globally optimal 1-D DP path than for a 2-D DP path. Figure 2 shows how a globally optimal 1-D DP path can differ from a 2-D DP path. As can be seen, 1-D DP must cut across the peninsula since it can only search forward (in the x direction).

### 2.2 Cost matrix

Initally, we generate a two-dimensional "cost matrix" where every element in the matrix corresponds to an image pixel with the cost defined by the pixel's local boundary criteria. The cost matrix is generated as a function of the image's gradient magnitute and gradient orientation. Letting $G_x$ and $G_y$ represent the horizontal and vertical gradients of the image, then the cost matrix c(x,y) is generated as follows:

$$G(x,y) = \sqrt{G_x^2(x,y)+G_y^2(x,y)}$$
$$O(x,y) = tan^{-1}(\frac{G_x(x,y)}{G_y(x,y)})$$
$$c(x,y)= [max_{x,y}(G(x,y))-G(x,y)] + α|O(x,y)*f - O(x,y)|$$

where G(x,y) and O(x,y) are the gradient magnitude and orientation images respectively, f is an averaging or gaussian filter and α is a scaling factor. Note that the gradient magnitude image is subtracted from the maximum gradient magnitude value so that strong gradients represents low costs. Further, the absolute difference between the low-pass filtered orientation image and itself will also be low if a pixel's neighborhood contains gradient orientations that are similar to its neighbors. Figure 3a is an example cost matrix.

Associated with the cost matrix is a connection weighting matrix. Each connection is "weighted" such that diagonal connections have higher cost than horizontal or vertical connections. The horizontal and vertical connections have unity weights whereas the diagonal connections are weighted by $\sqrt 2$, thus maintain Euclidean distance weighting for the neighborhood.

### 2.3 Optimal path generation

Since the cost matrix and graph connections are defined in image space terms, image space terms will be used to describe the 2-D DP algorithm.

Unlike most other dynamic programming or graph-searching algorithms, we do not define both a start point and end point. Rather, we calculate the globally optimal path from a start or seed point to all other points/pixels in the image in an incremental fashion. The algorithm is similar to a well-known heuristic search algorithm[5] and is as follows:

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

Figure 3 gives an example of how the total cost and the least cost path is computed. Figure 3a is the initial local cost matrix with the seed point circled. Figure 3b shows the total cost/path matrix after the seed point has been processed. Figure 3c shows the matrix after processing 2 points - the seed point and the next lowest total cost point on the sorted list. Notice how the points diagonal to the seed point have changed total cost and direction pointers. The Euclidean weighting between the seed and diagonal points make them more expensive than non-diagonal paths. Figure 3d, 3e, and 3f show the matrix at various stages of completion. Note how the algorithm produces a "wave-front" of active points and that the wave-front grows out faster were there are lower costs. Thus the wave-front grows out more quickly along edges in the image.

### 2.4 Live-wire 2-D DP

Once the path matrix is finished, a boundary path can be chosen dynamically via a "free" point. Interactive movement of the free point by cursor position causes the boundary to behave like a live-wire as it adapts to the new minimum cost path. Thus, by constraining the seed point and the free point to lie near a given edge, the user is able to interactively "warp" the live-wire boundary around the object of interest. When movement of the free point causes the boundary to digress from the desired object edge, input of a new seed point prior to the point of departure reinitiates the 2-D DP boundary detection. This causes potential paths to be recomputed from the new seed point, while effectively "tieing off" the boundary computed up to the new seed point. Figures 4 and 5 show two example images (an MRI scan and a left ventriculogram) and indicate how many seed points each outlined object required.

## 3. Results

Adaptive 2-D DP performs well on a variety of images (angiocardiograms, CT, MRI). In particular, it accurately detects the boundaries of low contrast objects, such as occur with intravenous injections, as well as those found in noisy, low SNR images. Boundaries are typically detected with two to four seed points.

The algorithm's computational complexity for a N image pixels is O(N). This can be seen by examining the algorithm in a worst case situation. Suppose first that all the weights were unity, then once a total cost to a point is computed, it is not computed again since that total cost already represents the minimum cost to that pixel. Thus the total cost is computed only once for each of the N pixels. There is also some computation required to add the point to a sorted list. But the unique conditions of this algorithm allow us to use a sort algorithm that requires only an array indexing operation (indexed by total cost) and changing two pointers. Thus, the computation complexity for sorting N points in N. Now, since the diagonal weights are not unity, then it may become necessary to recompute the cost to those points processed initally as diagonal neighbors. In the worst case, this will have to be done for half the points in the matrix since half the points will be initially processed as diagonal neighbors. This means in the worst case that total costs for N/2 points will have to be recomputed and those points will have to be re-added to the sorted list again. The algorithm's computational complexity is therefore N (to calculate the total costs) + N (to add N points to the sorted list) + N (to recompute the costs for diagonals and add them again to the list) = 3N, or O(N). This is comparable to the complexity for the more restricted 1-D DP algorithm.

The algorithm was implemented on a IBM compatible 33MHz 386 with 387 co-processor and a hardware imaging board. Generating the cost matrix for a 512x512 image requires approximately 45 seconds but only needs to be done once per image. The user then selects a seed point interactively with the mouse. "Growing" the optimal path map requires up to one and a half minutes per seed point for a 512x512 image but this process is usually interrupted after 15 to 20 seconds when the DP wave-front encloses the desired object or point. The user can then use the path matrix to interactively wrap a boundary around the desired objct. Though this process requires a small delay for each seed point, we are currently porting this algorithm to HP workstations where we expect to generate the optimal path matrix at interactive speeds.

## 4. Conclusions

An algorithm has been presented for iterative determination of globally optimal paths derived from local gradient magnitude and orientation information. The algorithm uses two-dimensional dynamic programming and can be applied to a variety of image types and anatomy. 2-D DP performs well based on visual comparison and is the same order of complexity as the more restrictive 1-D DP.

By calculating the optimal path from all points to a seed point, 2-D DP accommodates interactive selection of the desired optimal path via a "live-wire", making it a valuable interactive tool for defining an object's boundaries.