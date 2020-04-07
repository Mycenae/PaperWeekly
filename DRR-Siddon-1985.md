# Fast calculation of the exact radiological path for a three-dimensional CT array

Robert L. Siddon Harvard Medical School

## 0. Abstract

Ready availability has prompted the use of computed tomography (CT) data in various applications in radiation therapy. For example, some radiation treatment planning systems now utilize CT data in heterogeneous dose calculation algorithms. In radiotherapy imaging applications, CT data are projected onto specified planes, thus producing "radiographs", which are compared with simulator radiographs to assist in proper patient positioning and delineation of radiological path through the CT array. Due to the complexity of the three-dimensional geometry and enormous amount of CT data, the exact evaluation of the radiological path has proven to be a time consuming and difficult problem. This paper identifies the inefficient aspect of the traditional exact evaluation of the radiological path as that of treating the CT data as individual voxels. Rather than individual voxels, a new exact algorithm is presented that considers the CT data as consisting of the intersection volumes of three orthogonal set of equally spaced, parallel planes. For a three-dimensional CT array of N^3 voxels, the new exact algorithm scales with 3N, the number of planes, rather than N^3, the numbers of voxels. Coded in FORTRAN-77 on a VAX 11/780 with a floating point option, the algorithm requires approximately 5ms to calculate an average radiological path in a 100^3 voxel array.

CT数据的即时可用性，推动了放射治疗中CT数据的各种应用。比如，一些放射治疗计划系统，在各种剂量计算算法中，现在使用CT数据。在放射治疗成像应用中，CT数据投影到特定平面上，因此产生了radiographs，与模拟器的radiographs进行比较，以辅助病人摆位和在CT阵列中的放射路径的轮廓勾画。由于3D几何的复杂性和CT数据量非常大，放射路径的精确评估，是一个非常耗时和困难的问题。本文指出了传统放射路径的精确评估的低效方面，这主要是因为将CT数据看作是独立的体素。我们没有将其视作独立体素，而是提出了一种新的精确算法，认为CT数据是由三个等间距、平行的平面集合的交集体组成的。对于一个三维CT阵列，有N^3个体素，新的精确算法，其计算量随着平面数量，即3N变化，而不是随着体素数量N^3变化。我们用FORTRAN-77在一台VAX 11/780机器上用浮点数精度编程，算法在100^3体素阵列中，计算放射路径，平均只需要大约5ms。

Key words: radiological path, inhomogeneity correction, CT

## 1. Introduction

In radiation therapy applications, computer tomography (CT) data are utilized in various dose calculation and imaging algorithms. For example, some radiation treatment planning systems now utilize two-dimensional CT data for pixel-based heterogeneous dose calculations. Other systems forward project three-dimensional CT data onto specified planes, thus forming "radiographs", which are compared with simulator radiographs to assist in proper patient positioning and delineation of target volumes. All such applications whether in inhomogeneity calculations or imaging applications, essentially reduce to the same geometric problem: that of calculating the radiological path for a specified ray through the CT array.

在放射治疗应用中，CT数据在各种剂量计算和成像算法中都有应用。比如，一些放射治疗计划系统现在利用2D CT数据进行基于像素的异质剂量计算。其他系统会将3D CT数据前向投射到特定的平面上，因此形成了radiographs，与模拟器radiographs相比较，以辅助进行病人摆位和目标体的勾画。所有这种应用，不管是非均一性计算，或成像应用，都可以归结到相同的几何问题：对于特定的射线穿过CT阵列，计算其放射路径。

Although very simple in principle, elaborate computer algorithms and a significant amount of computer time is required to evaluate the exact radiological path. The amount of detail involved was recently emphasized by Harauz and Ottensmeyer, who stated that even for the two-dimensional case, their algorithm for calculating the exact radiological path grew more and more unwieldy and time consuming, while remaining unreliable. For three dimensions, they concluded that determining the exact radiological path is not viable. This paper describes an exact, efficient, and reliable algorithm for calculating the radiological path through a three-dimensional CT array.

虽然原则上非常简单，但是要评估精确的放射路径，需要精细的计算机算法和非常多的计算时间。最近Harauz和Ottensmeyer详述了其细节，给出了在2D情况下的例子，其计算精确放射路径的算法，变得越来越笨拙并耗费大量时间，同时还不太稳定。对于3D情况，他们得出结论，确定精确的放射路径并不可行。本文描述了一种精确、高效、可靠的在3D CT阵列中计算放射路径的方法。

Denoting a particular voxel density as ρ(i,j,k) and the length contained by that voxel as l(i,j,k), the radiological path may be written as

令特定的体素灰度为ρ(i,j,k)，体素包含的长度为l(i,j,k)，那么放射路径可以写为

$$d = \sum_i \sum_j \sum_k l(i,j,k) ρ(i,j,k)$$(1)

Direct evaluation of Eq. (1) entails an algorithm which scales with the number of terms in the sums, that is, the number of voxels in the CT array. The following describes an algorithm that scales with the sum of linear dimensions of the CT array.

直接计算(1)式需要一个算法，随着求和项中的数量变化，即随着CT阵列中的体素数量变化。下面描述一种算法，随着CT阵列中的线性维度的数量变化。

## 2. Method

Rather than independent elements, the voxels are considered as the intersection volumes of orthogonal sets of equally spaced, parallel planes. Without loss of generality, Fig. 1 illustrates the two-dimensional case, where pixels are considered as the intersection area of orthogonal sets of equally spaced, parallel lines. The intersections of the ray with the lines are calculated, rather than intersections of the ray with the individual pixels. Determining the intersections of the ray with the equally spaced, parallel lines is a particularly simple problem. As the lines are equally spaced, it is only necessary to determine the first intersection and generate all the others by recursion. As shown on the right illustration of Fig. 1, the intersections consist of two sets, one set for the intersections with the horizontal lines (closed circles) and one set for the intersections with the vertical lines (open circles). Comparing the left and right illustrations of Fig. 1, it is clear that the intersections of the ray with the pixels is a subset of the intersections with the lines. Identifying that subset allows radiological path to be determined. The extension to the three-dimensional CT array is straightforward.

我们没有将体素视为独立的元素，而是将其视为等间距的、平行的平面的正交集合的相交体。不失一般性，图1展示了2D的情况，其中像素被视为等间距的平行线的正交集的相交区域。计算的是射线与直线的相交，而不是射线与单个像素的相交。确定射线与等间距的、平行线的交点是非常容易的事。因为线是等间距的，只需要确定第一个交点，然后通过递归来生成其他的交点。如图1右所示的，交点可以分为两个集合，一个集合是于水平线的交点（实心圆点），一个集合是与垂直线的交点（空心圆点）。比较图1的左右，很明显，射线与像素的交点，就是与线段交点的子集。确认这个子集，就可以计算确定放射路径。拓展到3D CT阵列是很直观的事。

Fig. 1. The pixels of the CT array (left) may be considered as the intersection areas of orthogonal sets of equally spaced, parallel lines (right). The intersections of the ray with the pixels are a subset of the intersections of the ray with the lines. The intersections of the ray with the lines are given by two equally spaced sets: one set for the horizontal lines (filled circles) and one set for the vertical lines (open circles). The generalization to a three-dimensional CT array is straightforward.

The ray from point 1 to point 2 may be represented parametrically as

从点1到点2的射线可以参数化的表示为

$$\begin{matrix} X(α) = X_1+α(X_2-X_1) \\ Y(α) = Y_1+α(Y_2-Y_1) \\ Z(α) = Z_1+α(Z_2-Z_1) \end{matrix}$$(2)

where the parameter α is zero at point 1 and unity at point 2. The intersection of the ray with the sides of the CT array are shown in Fig. 2. If both points 1 and 2 are outside the array [Fig. 2(a)], then the parametric values corresponding to the two intersection points of the ray with the sides are given by $α_{min}$ and $α_{max}$. All intersections of the ray with individual planes must have parametric values which lie in the range $(α_{min}, α_{max})$. For the case illustrated in Fig. 2(b), where the point 1 is inside the array, the value of $α_{min}$ is zero. Likewise, for Fig. 2(c), if point 2 is inside, then $α_{max}$ is one. For both points 1 and 2 inside the array [Fig. 2(d)], then $α_{min}$ is zero and $α_{max}$ is one. The solution to the intersection of the ray with CT voxels follows immediately: Determine the parametric intersection values, in  the range $(α_{min}, α_{max})$, of the ray with each orthogonal set of equally spaced, parallel planes. Merge the three sets of parametric values into one set; for example, merging of the sets (1,4,7), (2,5,8) and (3,6,9) results in the merged set (1,2,3,4,5,6,7,8,9). The length of the ray contained by a particular voxel, in units of the ray length, is simply the difference between two adjacent parametric values in the merged set. For each voxel intersection length, the corresponding voxel indices are obtained, and the products of the length and density are summed over all intersections to yield the radiological path. A more detailed description of the algorithm is given in the following section.

其中参数α在点1处为0，在点2处为1。射线与CT阵列的边的交点，如图2所示。如果点1和点2都在阵列之外（图2a），那么射线与阵列相交的两个点的参数由$α_{min}$和$α_{max}$给出。射线与单个平面的交点的参数值都在$(α_{min}, α_{max})$范围内。如图2b所示的情况，其中点1是在阵列中的，那么$α_{min}$的值为0。类似的，如图2c，如果点2在内部，那么$α_{max}$为1。若点1和点2都在阵列内部（图2d），那么$α_{min}$为0，$α_{max}$为1。射线与CT体素的交点的解立刻就可以求的了：确定射线与每个等间距、平行平面的正交集，相交点的参数值，在范围$(α_{min}, α_{max})$内。将三个集合的参数值融合成一个集合，如，将集合(1,4,7), (2,5,8)和(3,6,9)合并为集合(1,2,3,4,5,6,7,8,9)。射线包含在某一特定体素中的长度，以射线波长为单位，就是在合并的集合中，两个相邻参数值的差异。对每个体素相交的长度，得到对应的体素索引，长度与密度的乘积进行求和，得到放射路径。算法详述见下节。

Fig. 2. The quantities $α_{min}$ and $α_{max}$ define the allowed range of parametric values for the intersections of the ray with the sides of the CT array: (a)both 1 and 2 outside the array, (b)1 inside and 2 outside, (c)1 outside and 2 inside, and (d) 1 inside and 2 inside.

## 3. Algorithm

For a CT array of $(N_x-1, N_y-1, N_z-1)$ voxels, the orthogonal sets of equally spaced, parallel planes may be written as

对于一个有$(N_x-1, N_y-1, N_z-1)$个体素的CT阵列，等间距的平行平面的正交集可以写为

$$\begin{matrix} X_{plane}(i) = X_{plane}(1)+(i-1)d_x & (i=1,...,N_x) \\ Y_{plane}(j) = Y_{plane}(1)+(j-1)d_y & (j=1,...,N_y) \\ Z_{plane}(k) = Z_{plane}(1)+(k-1)d_z & (k=1,...,N_z) \end{matrix}$$(3)

where $d_x, d_y$ and $d_z$ are the distances between the x, y and z planes, respectively. The quantities $d_x, d_y$ and $d_z$ are also the lengths of the sides of the voxel. the parametric values $α_{min}$ and $α_{max}$ are obtained by intersecting the ray with the sides of the CT array. From Eqs. (2) and (3), the parametric values corresponding to the sides are given by the following:

其中$d_x, d_y$和$d_z$是分别x, y和z各个平面之间的距离。$d_x, d_y$和$d_z$量也是体素的边的长度。参数值$α_{min}$和$α_{max}$是通过将射线与CT阵列的边相交得到的。从式(2)和(3)中，对应这些边的参数值如下式给出：

$$\begin{matrix} If (X_2-X_1) \neq 0, \\ α_x(1) = [X_{plane}(1) - X_1]/(X_2-X_1), \\ α_x(N_x) = [X_{plane}(N_x) - X_1]/(X_2-X_1) \end{matrix}$$(4)

with similar expressions for $α_y(1), α_y(N_y), α_z(1)$ and $α_z(N_z)$. If the denominator $(X_2-X_1)$ in Eq.(4) is equal to zero, then the ray is perpendicular to the x axis, and the corresponding values of $α_x$ are undefined, and similarly for $α_y$ and $α_z$. If the $α_x, α_x$ and $α_z$ values are undefined, then those values are simply excluded in all the following discussion.

对于$α_y(1), α_y(N_y), α_z(1)$和$α_z(N_z)$表达式类似。如果(4)式中的分母$(X_2-X_1)$等于0，那么射线就与x轴垂直了，那么$α_x$的对应值就是未定的，对于$α_y$和$α_z$也是类似的。如果$α_x, α_x$和$α_z$值是未定的，那么这些值在下面的讨论中，只要排除掉就可以了。

In terms of the parametric values given above, the quantities $α_{min}$ and $α_{max}$ are given by 上述的参数值，$α_{min}$和$α_{max}$由下式给出

$$α_{min} = max\{ 0, min[α_x(1), α_x(N_x)], min[α_y(1), α_y(N_y)], min[α_z(1), α_z(N_z)] \}$$(5)
$$α_{max} = min\{ 0, max[α_x(1), α_x(N_x)], max[α_y(1), α_y(N_y)], max[α_z(1), α_z(N_z)] \}$$(5)

where the functions min and max select from their argument list, the minimum and maximum terms, respectively. If $α_{max}$ is less than or equal to $α_{min}$, then the ray does not intersect the CT array.

其中函数min和max从其变量表中分别选择其最大值和最小值。如果$α_{max}$小于或等于$α_{min}$， 那么射线就没有与CT阵列相交。

From all intersected planes, there are only certain intersected planes which will have parametric values in the range $(α_{min}, α_{max})$. From Eqs. (2), (3), and (5), the range of indices ($i_{min}, i_{max}$), ($j_{min}, j_{max}$), and ($k_{min}, k_{max}$), corresponding to these particular planes, are given by the following: If $(X_2 - X_1) ≥ 0$,

从所有相交的平面中，只有特定的相交平面会有参数值，在$(α_{min}, α_{max})$范围内。从式(2), (3)和(5)，索引范围($i_{min}, i_{max}$), ($j_{min}, j_{max}$)和($k_{min}, k_{max}$)，对应着这些特定的平面，按如下式给出，如果$(X_2 - X_1) ≥ 0$，

$$i_{min} = N_x - [X_{plane}(N_x)-α_{min}(X_2-X_1)-X_1]/d_x$$(6)
$$i_{max} = 1 + [X_1+α_{max}(X_2-X_1)-X_{plane}(1)]/d_x$$(6)

if $(X_2 - X_1) ≤ 0$

$$i_{min} = N_x - [X_{plane}(N_x)-α_{max}(X_2-X_1)-X_1]/d_x$$(6)
$$i_{max} = 1 + [X_1+α_{min}(X_2-X_1)-X_{plane}(1)]/d_x$$(6)

with similar expressions for $j_{min}, j_{max}, k_{min}$ and $k_{max}$.

For a given range of indicies ($i_{min}, i_{max}$), ($j_{min}, j_{max}$) and ($k_{min}, k_{max}$), the sets parametric values {α_x}, {α_y}, and {α_z}, corresponding to the intersections of the ray with the planes are given by the following: If (X_2 - X_1)>0,

{α_x} = {α_x(i_{min}), ..., α_x(i_{max})}; (7)

if (X_2 - X_1)<0,

{α_x} = {α_x(i_{max}), ..., α_x(i_{min})}; (7)

where

$$α_x(i) = [X_{plane}(i) - X_1]/(X_2 - X_1) = α_x(i-1)+[d_x/(X_2-X_1)]$$(7)

with similiar expressions for {α_y} and {α_z}.

As given by Eq.(7), the sets {α_x}, {α_y}, and {α_z} are each arranged in ascending order. Each term in each set corresponds to an intersection of the ray with a particular plane. The intersections of the ray with the voxels are found by merging the sets {α_x}, {α_y}, and {α_z} into one set. To include the case where one or both of the endpoints of the ray may be inside the CT array, the parametric values $α_{min}$ and $α_{max}$ are appended to the merged parametric sets. The terms $α_{min}, α_{max}$ and the merged sets {α_x}, {α_y}, and {α_z} are denoted by the set {α}:

{α} = {α_{min}, merge[{α_x}, {α_y}, {α_z}], α_{max}} = {α(0), ..., α(n)} (8)

where the last term has an index n given by

$$n = (i_{max}-i_{min} + 1)+(j_{max}-j_{min} + 1)+(k_{max}-k_{min} + 1)+1$$(9)

two adjacent terms in the set {α} refer to the intersections of the ray with a particular voxel. For two intersections m and m-1, the voxel intersection length l(m) is given by

$$l(m) = d_{12} [α(m)-α(m-1)], (m=1,...,n)$$(10)

where the quantity $d_{12}$ is the distance from point 1 to point 2,

$$d_{12} = [(X_2-X_1)^2+(Y_2-Y_1)^2+(Z_2-Z_1)^2]^{1/2}$$(11)

the voxel [i(m), j(m), k(m)], which corresponds to intersections m and m-1, is that which contains the midpoint of the two intersections. From Eqs. (2), (3) and (5), the indices [i(m), j(m), k(m)] are given by

$$i(m) = 1+[X_1+α_{mid}(X_2-X_1)-X_{plane}(1)]/d_x$$(12)
$$j(m) = 1+[Y_1+α_{mid}(Y_2-Y_1)-Y_{plane}(1)]/d_y$$(12)
$$k(m) = 1+[Z_1+α_{mid}(Z_2-Z_1)-Z_{plane}(1)]/d_z$$(12)

where $α_{mid}$ is given by

$$α_{mid} = [α(m)+α(m-1)]/2$$(13)

the radiological path d [Eq. (1)] may now be written as

$$d = \sum_{m=1}^{m=n} l(m) ρ[i(m),j(m),k(m)] = d_{12} \sum_{m=1}^{m=n} [α(m)-α(m-1)]ρ[i(m),j(m),k(m)]$$(14)

where n is given by Eq.(9), l(m) is given by Eq.(10), and the indices [i(m), j(m), k(m)] are given by Eq.(12).

## 4. Discussion

The new radiological path algorithm is summarized in the block diagram in Fig. 3. For a typical problem, the relative amount of computation time required in each section of the algorithm is given by the respective percentages to the right of each descriptive block. The new algorithm is coded in FORTRAN-77 and is run on a VAX 11/780 with a floating point option. At present, no attempt has been made to optimize the code in machine language or adapt the algorithm to run on an array processor. Rather, the algorithm has been coded in the straightforward manner described in the text.

start -> calculate range of parametric values (α_{min}, α_{max}) -> calculate range of indices ($i_{min}, i_{max}$), ($j_{min}, j_{max}$), and ($k_{min}, k_{max}$) -> calculate parametric sets {α_x}, {α_y}, and {α_z} -> merge sets to form set {α} -> calculate voxel lengths -> calculate voxel indices -> stop

Fig 3. Block diagram of the new algorithm to calculate the radiological path for a three-dimensional CT array. The percentages indicate the relative amount of computational time spent in various portions of the algorithm.

The performance of the algorithm is illustrated for a typical dose calculation problem shown in Fig. 4. The CT array is taken to be a cube with N^3 voxels. Point 1 of the ray path is centered above the arry. An internal calculation grid of 21^3 points, corresponding to point 2 of the ray path, is distributed uniformly within the CT array. The mean calculation time per point, t, is obtained as a function of the array size N (Fig.5). The mean time t is the total time to calculated the radiological path at all 21^3 points divided by the number of points. As illustrated in Fig 5, the new algorithm scales with N for a CT array of N^3 voxels.

Fig. 4. The performance of the new algorithm is illustrated for the problem of a CT array of N^3 voxels. Point 1 of the ray path is centered above the CT array. A calculation grid of 21^3 points, corresponding to point 2 of the ray path, is uniformly distributed within the CT array.

Fig. 5. The mean computational time per point, t, for the example in Fig. 4 as a function of the array size N. Note that the new algorithm scales with the linear size N and not the number of voxels N^3.

## 5. Conclusion

An algorithm has been developed which evaluates the exact radiological path of a ray through a three-dimensional CT array. Rather than consider individual voxels of the CT array, the algorithm calculates the intersections of the ray with orthogonal sets of equally spaced, parallel planes. For an array of N^3 voxels, considering the planes rather than the voxels allows the algorithm to scale with the number of planes (propotinal to N), rather than the number of voxels (proportional to N^3). The intersections are described as parametric values along the ray. The intersections of the ray with the voxels are obtained as a subset of the intersections of the ray with the planes. For each voxel intersection length, the corresponding voxel indices are obtained and the products of the intersected length particular voxel density are summed over all intersections to yield the radiological path. The algorithm is exact, efficient, reliable, and particularly straightforward to implement in computer code.