# The Image Foresting Transform: Theory, Algorithms, and Applications

## 0. Abstract

The image foresting transform (IFT) is a graph-based approach to the design of image processing operators based on connectivity. It naturally leads to correct and efficient implementations and to a better understanding of how different operators relate to each other. We give here a precise definition of the IFT, and a procedure to compute it—a generalization of Dijkstra’s algorithm—with a proof of correctness. We also discuss implementation issues and illustrate the use of the IFT in a few applications.

IFT是一种基于图的方法，基于连通性来设计图像处理算子。它会很自然的带来正确、高效的实现，还可以更好的理解不同的算子怎样相互关联。我们这里给出IFT的准确定义，以及计算的程序，是Dijkstra算法的一种泛化，并证明来正确性。我们还讨论了实现的问题，描述了IFT在几种应用的使用。

**Index Terms** — Dijkstra’s algorithm, shortest-path problems, image segmentation, image analysis, regional minima, watershed transform, morphological reconstruction, boundary tracking, distance transforms, and multiscale skeletonization.

## 1. Introduction

THE image foresting transform (IFT), which we describe here, is a general tool for the design, implementation, and evaluation of image processing operators based on connectivity. The IFT defines a minimum-cost path forest in a graph, whose nodes are the image pixels and whose arcs are defined by an adjacency relation between pixels. The cost of a path in this graph is determined by an application-specific path-cost function, which usually depends on local image properties along the path—such as color, gradient, and pixel position. The roots of the forest are drawn from a given set of seed pixels. For suitable path-cost functions, the IFT assigns one minimum-cost path from the seed set to each pixel, in such a way that the union of those paths is an oriented forest, spanning the whole image. The IFT outputs three attributes for each pixel: its predecessor in the optimum path, the cost of that path, and the corresponding root (or some label associated with it). A great variety of powerful image operators, old and new, can be implemented by simple local processing of these attributes.

我们这里描述的IFT，是一种通用工具，可以设计、实现和评估基于连通性的图像处理算子。IFT定义了图中的一个最小代价路径森林，其节点是图像像素，其弧是像素间的相邻关系。图中的路径的代价，是由具体应用的路径代价函数确定的，这通常依赖于沿着路径的图像局部性质，比如色彩，梯度和像素位置。森林的根是从给定集合的种子像素开始的。对合适的路径代价函数，从每个像素的种子集，IFT指定了一个最小代价路径，以这样的方式，这些路径的并集是一个有向森林，张成了整个图像。IFT对每个像素输出三个属性：在最优路径下的其前序节点，路径的代价，以及对应的根节点（或与之相关的一些标签）。非常多的强力图像算子，老的和新的，都可以用这些属性的简单局部处理来实现。

We describe a general algorithm for computing the IFT, which is essentially Dijkstra’s shortest-path algorithm [1], [2], [3], slightly modified for multiple sources and general path-cost functions. Since our conditions on the path costs apply only to optimum paths, and not to all paths, we found it necessary to rewrite and extend the classical proof of correctness of Dijkstra’s algorithm. In many practical applications, the path costs are integers with limited increments and the graph is sparse; therefore, the optimizations described by Dial [4] and Ahuja et al. [5] will apply and the running time will be linear on the number of pixels.

我们描述了一种通用算法来计算IFT，这实质上是Dijkstra的最短路径算法，对多源的情况和通用路径代价函数进行了简单的修改。由于我们在路径代价上的条件只应用到最优路径上，而并不是到所有路径上，我们发现有必要重写并拓展经典Dijkstra算法的正确性证明。在很多实际的应用中，路径代价都是整数，增量有限，图也是稀疏的；因此，由Dial[4]和Ahuja等[5]描述的其优化会得到应用，运行时间是像素数量的线性关系。

The IFT unifies and extends many image analysis techniques which, even though based on similar underlying concepts (ordered propagation, flooding, geodesic dilations, dynamic programming, region growing, A* graph search, etc.), are usually presented as unrelated methods. Those techniques can all be reduced to a partition of the image into influence zones associated with a given seed set, where the zone of each seed consists of the pixels that are “more closely connected” to that seed than to any other, in some appropriate sense. These influence zones are simply the trees of the forest defined by the IFT. Examples are watershed transforms [6], [7] and fuzzy-connected segmentation [8], [9], [10], [11]. The IFT also provides a mathematically sound framework for many image-processing operations that are not obviously related to image partition, such as morphological reconstruction [12], distance transforms [13], [14], multiscale skeletons [14], shape saliences and multiscale fractal dimension [15], [16], and boundary tracking [17], [18], [19].

IFT统一并拓展了很多图像分析技术，甚至是基于类似的潜在概念的（有序传播，洪水填充法，测地线膨胀，动态规划，区域增长，A*图搜索，等），这些通常都认为是不相关的方法。这些技术都可以认为是，图像分割成一个给定的种子集的相关影响区域，其中每个种子的区域都是由与那个种子联系比其他像素，在某种意义上更紧密的像素组成的。这种影响力区域就是IFT定义森林的树。例子有，分水岭变换，模糊连通的分割。IFT还对很多图像处理操作给出了一个数学上合理的框架，这些操作与图像分割关系并不特别相关，比如形态学重建，距离变换，多尺度骨架，形状显著性和多尺度分形维度，和边缘追踪。

By separating the general forest computation procedure from the application-specific path-cost function, the IFT greatly simplifies the implementation of image operators, and provides a fair testbed for their evaluation and tuning. For many classical operators, the IFT-based implementation [6] is much closer to the theoretical definition than the published algorithms [20], [21]. Indeed, many algorithms which have been used without proof [20], [21], [22], [23], [24] have their correctness established by being reformulated in terms of the IFT [6], [12]. By clarifying the relationship between different image transforms [12], the IFT approach often leads to novel image operators and considerably faster (but still provably correct) algorithms [14], [25], [26].

通过将通用森林计算过程与具体应用相关的路径代价函数分离，IFT极大的简化了图像算子的实现，提供了一个公平的测试平台进行评估和调节。对很多经典的算子，基于IFT的实现与发表的算法相比，更加接近理论上的定义。确实，很多没有经过证明就使用的算法，通过使用IFT重新表述，确定了其正确性。澄清了不同图像变换之间的关系，IFT方法通常会带来新的图像算子，以及相对快很多的算法（但仍然可以证明是正确的）。

The IFT definition and algorithms are independent of the nature of the pixels and of the dimension of the image, and therefore they apply to color and multispectral images, as well as to higher-dimensional images such as video sequences and tomography data [26].

IFT的定义和算法与像素的本质、图像的大小是无关的，因此可以应用于彩色图像和多光谱图像，以及更高维度的图像，比如视频序列和断层数据。

Section 2 reviews the previous related work. We define the IFT and related concepts in Section 3. Section 4 presents the basic IFT algorithm, a proof of its correctness and optimization hints. Tie-breaking policies are discussed in Section 5. In Section 6, we illustrate the use of the IFT in a few selected applications. Section 7 contains the conclusions and current research on the IFT.

第2部分回顾了之前相关的工作。我们在第3部分定义了IFT和相关的概念。第4部分给出了基本的IFT算法，其正确性和优化的证明。第5部分讨论了Tie-breaking的策略。第6部分，我们描述了IFT在一些选择的应用中的使用。第7部分是结论和目前在IFT上的研究。

## 2. Related Work

### 2.1 Shortest-Path Problems in General Graphs

The problem of computing shortest paths in a general graph has an enormous bibliography [27]. Most authors dispose of the multiple-source problem by trivial reduction to the single-source version. Solutions to the latter were first proposed by Moore in 1957 [1] and Bellman in 1958 [2]. The running time of their algorithms was O(mn), in the worst case, for a graph with n nodes and m arcs. An improved algorithm was given by Dijkstra in 1959 [3]; its running time of O(n^2) can be reduced to O(m+nlogn) through the use of a balanced heap data structure [5]. In 1969, Dial published a variant of Dijkstra’s algorithm [4], [5], for the special case where the arc costs are integers in the range [0..K], which uses bucket sorting to achieve running time O(m+nK).

计算通用图中的最短路径的问题，有很多参考文献。多数作者处理多源问题时，只是简单的将其用单源版本解决。对后者的解，首先由Moore[1]和Bellman[2]在1957、1958年提出。对于一个图有n个节点和m个弧，在最坏的情况下，其算法的运行时间为O(mn)。Dijkstra[3]在1959年给出了一个改进的算法；其运行时间O(n^2)，在使用均衡堆数据结构的情况下，可以降低到O(m+nlogn)。在1969年，Dial对弧代价是[0..k]范围内的整数的特殊情况，发表了Dijkstra算法的一个变体，使用了bucket排序，获得了O(m+nK)的运行时间。

The standard path-cost function is the sum of arc costs. Extensions to more general cost models fall in two major classes. The algebraic or “semiring” model considers fixed arc costs that are combined by an associative operation, that distributes over the min function (or, more generally, over a second associative operator which is used to summarize all paths leading to a given node) [28], [29]. Another model does not assume associativity or fixed arc costs, but requires that the cost of a path be computed incrementally from the cost of its prefixes through a composition function that satisfies certain monotonicity constraints [30]. We note that, in both approaches, the constraints on the path-cost functions are required to hold for all paths, not just for optimum ones.

标准的路径代价函数，是弧代价的和。拓展到更加通用的代价模型，主要落入两种类别。代数的，或半环模型，考虑的是固定的弧代价，通过一个关联的操作结合到一起，在min函数上分布（或者更通用的，在另一个相关算子上分布，用于总结带到给定节点的所有路径）。另一种模型并不假设连通性，或固定的弧代价，但要求一条路径的代价的计算，要从其前序节点，通过满足一定单调性约束的结合函数，进行增量计算得到。我们注意到，在两种方法中，路径代价函数的约束需要对所有路径都适用，而并不是只对最优的路径适用。

### 2.2 Image Processing Using Graph Concepts

In image processing, Montanari [31] and Martelli [32], [33] were the first to formulate a boundary finding problem as a shortest-path problem in a graph. Montanari [31] required the boundary to be star-shaped [34]. Martelli [32], [33] considered only a selected subset of the arcs and, therefore, his algorithm may fail in some situations. These restrictions have since been lifted by Falcao et al. [17] and Mortensen and Barrett [35]. In region-based image segmentation, Udupa and Samarasekera [8], and Saha and Udupa [9] proposed the fuzzy connectedness theory for object definition, which was efficiently implemented by Nyul et al. using Dial’s bucket queue [10]. Verwer et al. [36] and Sharaiha and Christofides [37] used the Dial’s algorithm to compute weighted distance and chamfer distance transforms. Meyer extended their work to some variations of the watershed transform based on the eikonal equation [38]. Dial’s bucket queue became the core of fast ordered propagation algorithms for various applications, including Euclidean distance transform [39], [40], watershed transforms [20], [21], [41], and morphological reconstructions [22], [23], [24].

在图像处理中，Montanari [31]和Martelli [32], [33]是第一个将寻找边缘的问题表述成一个图中的最短路径的。Montanari [31]要求边缘是星形的[34]。Martelli [32], [33]只考虑弧的有选择的子集，因此，其算法在一些情况中会失败。这些限制由Falcao等[17]和Mortensen and Barrett [35]自从那时就lifted。在基于区域的图像分割中，Udupa and Samarasekera [8], 和Saha and Udupa [9]提出了模糊连通性理论，可用于目标定义，这由Nyul等使用Dial的bucket queue进行了高效实现。Verwer等[36]和Sharaiha and Christofides [37]使用Dial的算法来计算加权距离和chamfer距离变换。Meyer拓展了其工作，到基于eikonal方程的分水岭变换的一些变体。Dial的bucket queue成为了快速有序传播算法的核心，可用于各种应用，包括欧式距离变换[39,40]，分水岭变换[20,21,41]，和形态学重建[22,23,24]。

### 2.3 The PDE Approach

There is a certain formal resemblance between the IFT and the partial differential equations (PDE) approach to image processing [42], since both involve the sweeping of the image by a propagating front— which, in the IFT approach, is the boundary between the sets S and $\overline S$ of Dijkstra’s original algorithm [5]. The eikonal problem, specifically, asks for the minimum-time path from the initial front to each point of the domain. The reciprocal of the local speed of propagation is analogous to the arc costs of the IFT model, and the Fast Marching method is seen as the analog of Dijkstra’s algorithm [43].

图像处理的IFT和PDE方法有一定程度的相似，因为两者都涉及到用一个传播的波前来扫过图像，在IFT方法中，是在集合S和Dijkstra的原始算法[5]中的$\overline S$。Eikonal问题，具体的，要求的是从初始波前到这个域中的每个点的最小时间路径。以局部速度进行传播的相应研究，与IFT模型的中弧代价是类似的，Fast Marching方法与Dijkstra算法是类似的。

To a large extent, the comparison of the two approaches is just another instance of the continuous versus discrete question. The discreteness of the IFT allows rigorous proofs and predictable algorithms, free from approximation errors and numerical instabilities that often affect the PDE. The IFT model also allows the cost of extending a path to depend on the whole path and not just on its current cost. On the other hand, the PDE approach allows the propagation front to move back and forth, or with speed that depends on its local shape—features that are used in contour smoothing, contour detection, and other applications. It is not obvious whether similar effects can be achieved in the IFT model, where the “shape of the front” is not a well-defined concept.

在很大程度上，这两种方法的比较，是连续与离散问题的另一个例子。IFT的离散性允许严格的证明和可预测的算法，近似误差和数值稳定性经常影响PDE，但IFT则没有这方面的问题。IFT模型还允许拓展一条路径并依赖于整条路径的代价，而不仅仅是其目前代价。另一方面，PDE方法可以使传播的波前前后运动，或速度依赖于其局部形状特征，在轮廓平滑、轮廓检测和其他应用中使用了这些特征。IFT模型是否可以取得类似的效果，这并不明显，其中波前的形状并不是一个定义明确的概念。

## 3. Notation and Definitions

An image I is a pair (II,I) consisting of a finite set II of pixels (points in Z^2), and a mapping I that assigns to each pixel t in II a pixel value I(t) in some arbitrary value space.

一幅图像I是一个对(II,I)，包含一个像素的有限集II（Z^2中的点），和一个映射I，对II中的每个像素t都指定了一个像素值I(t)，属于任意值空间。

An adjacency relation A is an irreflexive binary relation between pixels of II. Once the adjacency relation A has been fixed, the image I can be interpreted as a directed graph whose nodes are the image pixels and whose arcs are the pixel pairs in A. In what follows, n = |II| is the number of nodes (pixels) and m = |A| is the number of arcs.

一个相邻关系A，是对II中的像素之间的一种不可逆的二值关系。一旦相邻关系A固定，图像I可以解释为一个有向图，其节点是像素，弧是A中的像素对。在本文后续中，n = |II|是节点（像素）数量，m = |A|是弧的数量。

In most image-processing applications, the adjacency relation A is translation-invariant, meaning that sAt depends only on the relative position t-s of the pixels. For example, one often takes A to consist of all pairs of distinct pixels (s,t) ∈ I × I such that d(s,t) ≤ ρ, where d(s,t) denotes the Euclidean distance and ρ is a specified constant. Figs. 1a, 1b, and 1c show the adjacent pixels of a fixed central pixel s in this Euclidean adjacency relation, when ρ = 1 (the 4-connected adjacency), ρ = $\sqrt 2$ (8-connected), and ρ = $\sqrt 5$, respectively.

在多数图像处理应用中，相邻关系A是平移不变的，意味着sAt只依赖于相对位置t-s个像素。比如，通常使A包含所有不同像素对(s,t) ∈ I × I，满足d(s,t) ≤ ρ，其中d(s,t)表示欧式距离，ρ是一个指定的常数。图1a,1b和1c展示了一个固定中央像素s在欧式相邻关系中的相邻像素，分别是ρ = 1（4邻域），ρ = $\sqrt 2$（8邻域），和ρ = $\sqrt 5$。

More generally, we can take a finite subset M of Z^2 \ {(0,0)}; for example, M = {(-1, -1), (1, 1)} and define A as all pairs of pixels (s,t), where t-s∈M. See Fig.1d. Note that the adjacency relation does not have to be symmetric.

更一般的，我们取Z^2 \ {(0,0)}的一个有限子集M，比如，M = {(-1, -1), (1, 1)}，定义A为所有像素对(s,t)，其中t-s∈M。见图1d。注意相邻关系并不一定是对称的。

A path is a sequence of pixels π = $<t_1,t_2,...,t_k>$, where (t_i, t_i+1)∈A for 1≤i≤k-1. We denote the origin t_1 and the destination t_k of π by org(π) and dst(π), respectively. The path is trivial if k=1. If π and τ are paths such that dst(π) = org(π) = t, we denote by π·τ the concatenation of the two paths, with the two joining instances of t merged into one.

一条路径是一个像素序列，π = $<t_1,t_2,...,t_k>$，其中(t_i, t_i+1)∈A，1≤i≤k-1。我们表示π的起始点t_1和终点t_k分别为org(π)和dst(π)。如果k=1，那么路径就是无意义的。如果π和τ是路径，而dst(π) = org(π) = t，我们就用π·τ表示两条路径的拼接，而相交的点t融合成一个。

### 3.1 Path Costs

We assume given a function f that assigns to each path π a path cost f(π), in some totally ordered set V of cost values. Without loss of generality, we can always assume that V contains a maximum element, which we denote by +∞. Usually, the path cost depends on local properties of the image I—such as color, gradient, and pixel position—along the path.

我们假设给定了一个函数f，给每个路径π指定了一个路径代价f(π)，在一些完全有序的代价值集合V中。不失一般性，我们能假设，V包含一个最大的元素，我们表示为+∞。通常，路径代价依赖于图像I中沿着路径的一些局部性质，比如色彩，梯度和像素位置。

A popular example is the additive path-cost function, which satisfies 一个流行的例子是加性路径代价函数，满足下面的条件

$$f_{sum} (<t>) = h(t), \\ f_{sum} (π·<s,t>) = f_{sum}(π)+w(s,t)$$(1)

where (s,t) ∈ A, π is any path ending at s, h(t) is a fixed handicap cost for any paths starting at pixel t, and w(s,t) is a fixed nonnegative weight assigned to the arc (s,t).

其中(s,t) ∈ A，π是终点在s任意路径，h(t)是一个固定的handicap损失，对任意起始于像素t的路径的函数，w(s,t)是一个固定的非负权重，指定给弧(s,t)的。

Another important example is the max-arc path-cost function $f_{max}$, defined by 另一个重要的例子是max-arc路径损失函数

$$f_{max}(<t>) = h(t), \\ f_{max}(π·<s,t>) = max\{ f_{max}(π), w(s,t) \}$$(2)

where h(t) and w(s,t) are fixed but arbitrary handicap and arc weight functions. Note that $f_{sum}$ and $f_{max}$ are distinct models: Since the incremental cost $f_{max} (π·<s,t>)-f_{max} (π)$ depends on the cost of π, one cannot, in general, redefine $f_{max}$ as the sum of fixed weights w'(s,t).

其中h(t)和w(s,t)是固定但是任意的handicap和弧权重函数。注意$f_{sum}$和$f_{max}$是不同的模型：由于递增的代价$f_{max} (π·<s,t>)-f_{max} (π)$依赖于π的代价，一般我们不能重新定义$f_{max}$作为固定权重w'(s,t)的和。

### 3.2 Seed Pixels

In typical applications of the IFT, we would like to use a predefined path-cost function f but constrain the search to paths that start in a given set S ⊆ I of seed pixels. We can model this constraint by defining a new path-cost function f^S(π), which is equal to f(π) when org(π) ∈ S, and +∞ otherwise. In particular, for the f_{sum} and f_{max} functions, this is equivalent to setting h(t) = +∞ for pixels t $\notin$ S.

在IFT的典型应用中，我们要使用一个预定义的路径损失函数f，但路径的搜索会局限在开始于给定集合S ⊆ I的种子像素中的。我们可以对这个约束进行建模，定义一个新的路径代价函数f^S(π)，当org(π) ∈ S时与f(π)相等，否则就等于+∞。特别是，对于f_{sum}和f_{max}函数，这等价于对t $\notin$ S的像素设h(t) = +∞。

### 3.3 Optimum Paths

We say that a path π is optimum if f(π) ≤ f(π') for any other path π' with dst(π') = dst(π), irrespective of its starting point. In that case, f(π) is by definition the cost of pixel t = dst(π), denoted by $\hat f(t)$. Note that a trivial path is not necessarily optimum: even for f_{sum} or f_{max}, the handicaps of two pixels s and t may be such that a nontrivial path from s to t is cheaper than the trivial path $<t>$.

我们说一条路径π是最优的，那么对于任意其他路径π'，终点一致dst(π') = dst(π)，要有f(π) ≤ f(π')，而不论其起始点是什么。在这种情况下，f(π)在定义上就是t = dst(π)的代价，表示为$\hat f(t)$。注意，一条无意义的路径并不一定是最优的：即使对于f_{sum}或f_{max}，两个像素s和t的hadicaps可能是这样一种情况，即从s到t的一条有意义的路径，是比无意义的路径$<t>$要更加廉价。

### 3.4 Spanning Forests

A predecessor map is a function P that assigns to each pixel t in II either some other pixel in II, or a distinctive marker nil not in II — in which case t is said to be a root of the map. A spanning forest is a predecessor map which contains no cycles—in other words, one which takes every pixel to nil in a finite number of iterations. For any pixel t ∈ II, a spanning forest P defines a path $P^*(t)$ recursively as $<t>$ if P(t) = nil, and $P^*(s)·<s,t>$ if $P(t)=s \neq nil$. We will denote by $P^0(t)$ the initial pixel of $P^*(t)$ (see Fig. 2a).

一个前序图是一个函数P，对II中的每个像素t，指定了II中的其他像素，或不在II中的一个区别性的标记nil，这种情况下说明t是图的根。一个张树是一个前序图，其中没有任何循环，换句话说，对每个像素，都会在有限步迭代中带到nil。对于任意像素t ∈ II，一个张树P递归的定义了一个路径$P^*(t)$，如果P(t) = nil，那么就是$<t>$，如果$P(t)=s \neq nil$，那么就是$P^*(s)·<s,t>$。$P^*(t)$的初始像素，我们定义为$P^0(t)$（见图2a）。

### 3.5 The Image Foresting Transform

The image foresting transform (IFT) takes an image I, a path-cost function f and an adjacency relation A, and returns an optimum-path forest — a spanning forest P such that $P^*(t)$ is optimum, for every pixel t. See Figs. 2b and 2c.

IFT的输入为一幅图像I，一个路径代价函数f和一个相邻关系A，返回一个最优路径森林，即一个张树P，对于每个像素t，$P^*(t)$都是最优的。见图2b和图2c。

Note that, in an optimum-path forest P for a seed-restricted cost function $f^S$, any pixel t with finite cost $f^S(P^*(t))$ will belong to a tree whose root is a seed pixel; however, some seeds may not be roots of the forest because they may be more cheaply sourced by another root seed.

注意，对于一个种子约束的代价函数$f^S$，在一个最优路径森林P中，有限代价$f^S(P^*(t))$的任意像素t，都会属于一个树，其根是一个种子像素；但是，一些种子可能不会是森林的根，因为另一个根种子可能会是更便宜的源。

In general, there may be many paths of minimum cost leading to a given pixel; only the pixel costs $\hat f(t)$ are uniquely defined. Observe that, if we independently pick an optimum path for each pixel, the union of those paths may not be a forest. Indeed, certain graphs and cost functions may not even admit any optimum-path forest. Sufficient conditions for the existence of the IFT will be given in Section 4.3.

总体上，到一个给定的像素，会有很多最低代价路径；只有像素代价$\hat f(t)$是唯一定义的。观察到这个现象，如果我们对每个像素独立的选择一个最优路径，这些路径的并集可能不是一棵树。确实，特定图和代价函数并不一定会得到一个最优路径森林。IFT存在的充分条件会在4.3节中给出。

## 4. Algorithm

For suitable path-cost functions, the IFT can be computed by Algorithm 1 below — which is essentially Dijkstra’s procedure for computing minimum-cost paths from a single source in a graph [3], [5], slightly modified to allow multiple sources and more general cost functions. This variant was chosen for maximum flexibility and to simplify the proof of correctness. In practice, Algorithm 1 can be optimized in a number of ways (see Section 4.5).

对于合适的路径代价函数，IFT可以通过下面的算法1计算，其实就是在一个图中在单源的情况下计算最低代价的Dijkstra过程，进行了略微的修改，以允许多个源，以及更加一般的代价函数。选择了这个变体，可以允许最大的灵活性，并可以简化正确性的证明。在实践中，算法1可以以几种方法进行优化（见4.5节）。

**Algorithm 1**. Input: An image I = (II, I); an adjacency relation A ⊂ II × II; and a path-cost function f. Output: An optimum-path forest P. Auxiliary Data Structures: Two sets of pixels F, Q whose union is II.

算法1。输入：一幅图像I = (II, I)，相邻关系A ⊂ II × II，路径代价函数f。输出：一个最优路径森林P。辅助数据结构：两个像素集合F，Q，其并集是II。

1. Set F ← {}, Q ← II. For all t ∈ II, set P(t) ← nil.

2. While Q $\neq$ {}, do

2.1. Remove from Q a pixel s such that f(P^*(s)) is minimum, and add it to F.

2.2. For each pixel t such that (s,t)∈A, do

2.2.1. If $f(P^*(s)·<s,t>) < f(P^*(t))$, set P(t)←s.

### 4.1 General Properties

Some basic facts about Algorithm 1 are easily established by induction on the number of steps. Since every iteration of the main loop removes from Q exactly one pixel (which is never returned) and each arc of A is examined exactly once in Step 2.2, it follows that

算法1的一些基本事实，可以通过步骤数量的推理很容易的确立。因为主循环的每次迭代都从Q中移除一个像素（而且从未归还），A的每个弧在步骤2.2中都精确的检查一次，所以有

Lemma 1. Algorithm 1 terminates in O(n) iterations of the outer loop, and O(m) total iterations of the inner loop.

引理1. 算法1的外循环在O(n)次迭代后会终止，内循环会在O(m)次迭代后终止。

Moreover, each predecessor P(t) is initially nil and is modified only by the assignment P(t)←s in Step 2.2.1 — when t is still in Q but s is in F. Lemma 2 follows 而且，每个前任节点P(t)初始时是nil，由步骤2.2.1中的赋值P(t)←s进行修改，当t仍然在Q但s在F中时。引理2有

Lemma 2. The predecessor map P computed by Algorithm 1 is always a spanning forest.

引理2. 由算法计算得到的前任节点图P永远是一个张树。

Note that Lemmas 1 and 2 hold for any path-cost function f. This is more of a curse than a blessing, because it tempts people into using Algorithm 1 even when its result is not an optimum-path forest. The optimality depends on f being sufficiently well-behaved.

注意引理1和2对于任意的路径代价函数f都是成立的。这更多的是一个诅咒，而不是一个祝福，因为会引诱人们使用算法1，即使其结果并不是一种最优路径森林。其最优性依赖于f的性质要足够好。

### 4.2 Monotonic-Incremental Cost Functions

When the path-cost function is additive, the correctness of Algorithm 1 is established by the standard proof of Dijkstra’s method [28], [5]. Note that the extension to multiple starting pixels is trivial since it is equivalent to adding extra arcs from a dummy starting pixel u $\notin$ II to all pixels in II and setting w(u,t) = h(t) for each new arc (u, t). This remains true even when the arc costs and handicaps are allowed to be +∞.

当路径代价函数是加性的，算法1的正确性通过标准Dijkstra方法的证明就可以确立。注意，拓展到多个起始点是很简单的，因为这等价于，从一个dummy起始点u $\notin$ II增加额外的弧到II中的所有像素，并对每个新的弧(u, t)设w(u,t) = h(t)。这甚至在弧代价和hadicaps等于+∞时，也是正确的。

In fact, as shown by Frieze [30], the original proof of Dijkstra’s algorithm is easily generalized to monotonic-incremental (MI) path-cost functions, which satisfy

实际上，原始Dijkstra算法的证明也表明，很容易泛化到单调递增(MI)的路径代价函数，这满足下面条件

$$f(<t>)=h(t), f(π·<s,t>)=f(π)⊙(s,t)$$(3)

where h(t) is an arbitrary handicap cost, and ⊙: V × A → V is a binary operation that satisfies the conditions 其中h(t)是任意一个handicap代价，⊙: V × A → V是一个二值运算，满足下面的条件

M1. x' ≥ x ⇒ x'⊙(s,t) ≥ x⊙(s,t),

M2. x⊙(s,t) ≥ x.

for any x, x' ∈ V and any (s,t) ∈ A. An essential feature of this model is that ⊙ depends only on the cost of π, and not on any other property of π. Both the additive cost $f_{sum}$ (with nonnegative arc weights) and the max-arc cost $f_{max}$ are monotonic-incremental. It turns out that most image processing problems that we have successfully reduced to the IFT require MI path-cost functions, and therefore can be solved by Algorithm 1. In fact, Condition M2 can be weakened to f(π·κ)≥f(π) for any cycle κ [30].

对于任意的x, x' ∈ V和任意的(s,t) ∈ A。这个模型的一个基本特征是，⊙只依赖于π的代价，并不依赖于π的任意其他性质。$f_{sum}$的加性代价（带有非负弧权重），和$f_{max}$的max-arc代价，都是单调递增的。结果是，多数图像处理问题，我们都可以成功的表述为IFT，需要单调递增的路径代价函数，因此，可以用算法1进行求解。实际上，条件M2可以弱化为对任意循环κ，要有f(π·κ)≥f(π)。

On the other hand, it is easy to find counterexamples of path-cost functions—not MI, of course—which cause Algorithm 1 to fail. A textbook counterexample is the additive path-cost $f_{sum}$ when the arc weights w(s,t) are allowed to be negative. The algorithm may also fail for generalizations of $f_{sum}$ or $f_{max}$ where the arc weight w(s,t) is allowed to depend on the path already chosen for s. Unfortunately, this applies to several path-cost functions that would seem reasonable for image processing. For example, in multi-seeded region-based segmentation, it may seem reasonable to use the function $f_{abs}(π)$, defined as the maximum of |I(t)-I(org(π))| for any pixel t along the path π. Fig. 3 shows a situation where Dijkstra’s algorithm will fail (and, in fact, where the optimum paths do not form a forest). The same image graph is a counterexample when the path cost f(π) is defined as the variance of the pixel values along π. Another counterexample is the region-growing criterion proposed by Bischof and Adams [44], where each candidate pixel is ranked by the absolute difference between its value and the mean value of all pixels in each region.

另一方面，很容易找到路径代价函数并不是单调递增的反例，这会导致算法1失败。一个教科书上的反例是加性路径代价函数$f_{sum}$，而弧权重w(s,t)可以是负的。对于下面的泛化，即$f_{sum}$或$f_{max}$中，其弧权重w(s,t)允许依赖于已经为s而选的路径时，算法也会失败。不幸的是，在图像处理算法的几个路径代价函数中，这已经似乎是合理的。比如，在多种子基于区域的分割中，使用函数$f_{abs}(π)$似乎是合理的，这个函数的定义为，对于沿着路径π的任意像素t，|I(t)-I(org(π))|的最大值。图3展示了一种情况，其中Dijkstra算法会失败（实际上，最优路径并没有形成一个森林）。当路径代价f(π)定义为沿着π的像素值的方差时，同样的图像图是一个反例。另一个反例是，[44]中提出的区域增长规则，其中每个候选像素是根据，像素值和每个区域的所有像素的平均值的绝对差来排序的。

### 4.3 Smooth Path-Cost Functions

Algorithm 1 does work for certain cost functions which are not MI, or even monotonic. An example is the 4-connected adjacency and the function $f^S_{euc} (π)$ defined as the Euclidean distance between the endpoints of π, restricted to paths that start at a given seed set S. Algorithm 1 will correctly find an optimum-path forest as long as |S| ≤ 2, even though Condition M2 is violated when |S| = 1, and both M1 and M2 are violated when |S| = 2. Consider, for instance, three pixels t = r_1 = (0, 0),s = (2, 0), r_2 = (3, 0): We have $f_{euc}(<r_1, s>) > f_{euc}(<r_2, s>)$, but $f_{euc}(<r_1,s>·<s,t>)<f_{euc}(<r_2,s>·<s,t>)$; and, also, $f_{euc}(<s,t>·<t,s>)<f_{euc}(<s,t>)$. (The algorithm may fail, however, if |S| ≥ 3; see Section 6.4.)

算法1对于一些并不是单调递增，或甚至单调的函数，也是可以的。一个例子是，4连通邻域和函数$f^S_{euc} (π)$，定义为π的端点的欧式距离，但是限制于起始于给定种子集S的路径。只要|S| ≤ 2，算法1就会正确的找到一个最优路径森林，当|S| = 1，即使条件M2有冲突时，还有|S| = 2，条件M1和M2都有冲突时，都能得到正确的答案。比如，考虑三个像素t = r_1 = (0, 0),s = (2, 0), r_2 = (3, 0): 我们有$f_{euc}(<r_1, s>) > f_{euc}(<r_2, s>)$，但$f_{euc}(<r_1,s>·<s,t>)<f_{euc}(<r_2,s>·<s,t>)$; 而且，$f_{euc}(<s,t>·<t,s>)<f_{euc}(<s,t>)$。（如果|S| ≥ 3，算法会失败；见6.4节）

Examples like this one led us to search for conditions on the path-cost function f that are more general than M1 and M2 but still strong enough to ensure the correctness of Algorithm 1. Specifically, we claim that the algorithm will work if, for any pixel t ∈ II, there is an optimum path τ ending at t which either is trivial, or has the form $τ·<s,t>$, where

这样的例子使我们去搜索路径代价函数f的条件，比M1和M2更加通用，但仍然足够强健，以确保算法1的正确性。具体的，如果对于任意像素t ∈ II，有一个最优路径τ的末端在t，要么是无意义的，要么有$τ·<s,t>$的形式，满足下列条件，算法就可以工作：

C1. f(τ)≤f(π)
C2. τ is optimum, and
C3. for any optimum path τ' ending at s, $f(τ'·<s,t>) = f(π)$.

These conditions seem to capture the essential features of the path-cost function that are used in the classical proofs (cf. Bellman’s optimality principle [45]). Observe that Conditions C1, C2, and C3 are not required to hold for all paths ending at t, but only for some path π that is optimum. We say that a path-cost function f satisfying Conditions C1, C2, and C3 is smooth.

It can be checked that any MI path-cost function satisfies Conditions C1, C2, and C3. Also, if f is an MI cost function, then its restriction f^S to an arbitrary seed set S will be MI, and hence smooth too. (Unfortunately, this is not necessarily true if f is smooth, but not MI.)

For an example of a smooth function that is not MI, let f be an MI function and define f'(π) = f(π) + g(π), where g(π) is zero if π is optimum for f, and an arbitrary positive value otherwise. The function f' satisfies Conditions C1, C2, and C3, even though it may fail M1 for arbitrary paths. For a more realistic example, it can be checked that f^S_{euc} is smooth when |S| ≤ 2. Two key observations are that the influence zones of the seed pixels are 4-connected and that any path π with minimum number of arcs connecting S to t satisfies Conditions C1, C2, and C3.

## 7. Conclusions and current research

We gave a precise definition of the image foresting transform, a proof of correctness of its basic algorithm (for a fairly general class of cost functions), and an efficient implementation (Algorithm 2). We pointed out the importance of the tie-breaking policy, especially for the watershed transform, and introduced the LIFO variant (Algorithm 3), which led to an efficient way of locating regional minima. We also presented cost functions for watershed transforms, morphological reconstructions, boundary tracking, and EDT-related operators.

The IFT provides a common and well-founded framework for the design of image processing operators, which leads to correct, efficient, and flexible implementations. Algorithms 2 and 3 together with the examples of this paper are available through the Internet [50].

We are currently exploiting Algorithm 2 and variants to improve the efficiency of watershed transforms [25]. We are also developing novel IFT-based algorithms for automatic image segmentation [51], and paradigms for 3D segmentation of medical images at interactive speeds, such as the differential IFT [26], [52]. Finally, we are investigating parallel and hardware implementations of the IFT and its possible application to 3D multiscale skeletonization.