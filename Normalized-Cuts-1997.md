# Normalized Cuts and Image Segmentation

Jianbo Shi and Jitendra Malik, University of California

## 0. Abstract

we propose a novel approach for solving the perceptual grouping problem in vision. Rather than focusing on local features and their consistencies in the image data, our approach aims at extracting the global impression of an image. We treat image segmentation as a graph partitioning problem and propose a novel global criterion, the normalized cut, for segmenting the graph. The normalized cut craterion measures both the total dissimilarity between the different groups as well as the total similarity within the groups. We show that an eficient computational technique based on a generalized eigenvalue problem can be used to optimize this criterion. we have applied this approach to segmenting static images and found results very encouraging.

我们提出一种新的方法，求解视觉中的感知分组问题。我们关注的并不是图像数据中的局部特征及其连续性，我们方法的目标是提取图像的全局印象。我们将图像分割作为一种图分割问题来求解，提出一种新的全局标准，归一化分割，以分割图。归一化分割标准度量的不同分组之间的差异大小，以及分组内的相似性。我们证明了基于一般性的特征值问题的高效计算技术，可以用于优化这种标准。我们将这种方法应用于分割静态图像，发现结果非常不错。

## 1. Introduction

Nearly 75 years ago, Wertheimer[17] launched the Gestalt approach which laid out the importance of perceptual grouping and organization in visual perception. For our purposes, the problem of grouping can be well motivated by considering the set of points shown in the figure (1).

接近75年前，Wertheimer[17]启动了Gestalt方法，确定了视觉感知中感知分组和组织的重要性。就我们的目标而言，分组的问题是受到图1中点集问题的推动。

Typically a human observer will perceive four objects in the image - a circular ring with a cloud of points inside it, and two loosely connected clumps of points on its right. However this is not the unique partitioning of the scene. One can argue that there are three objects - the two clumps on the right constitute one dumbbell shaped object. Or there are only two objects, a dumb bell shaped object on the right, and a circular galaxy like structure on the left. If one were perverse, one could argue that in fact every point was a distinct object.

一般来说，一个人类观察者可以从图像中观察到四个目标，一个圆形的环，内部有一簇点集，右边有两簇点集。但是这并不是整个场景的唯一分割。也可以说这有三个目标，右边的两簇组成了一个哑铃状的目标。或者有两个目标，一个哑铃状的目标在右边，一个环形的银河状的结构在左边。如果一个人不太合常理，可以认为实际上每个点都是一个不同的目标。

This may seem to be an artificial example, but every attempt at image segmentation ultimately has to confront a similar question - there are many possible partitions of the domain D of an image into subsets D_i (including the extreme one of every pixel being a separate entity). How do we pick the "right" one? We believe the Bayesian view is appropriate - one wants to find the most probable interpretation in the context of prior world knowledge. The difficulty, of course, is in specifying the prior world knowledge - some of it is low level such as coherence of brightness, color, texture, or motion, but equally important is mid- or high-level knowledge about symmetries of objects or object models.

这似乎可能是一个人造的例子，但在图像分割中的每个尝试，都最终遇到了类似的问题，一幅图像D分割成子集D_i有很多可能的方式（包含极端的分割方式，将每个像素都作为一个分离的实体）。我们怎样选择正确的分割？我们相信，Bayesian视角是很合理的，我们想要找到的是在先验知识上下文中最可能的解释。当然，其困难在于指定世界的先验知识，其中一些是低层的知识，如亮度、色彩、纹理或运动的连续性，但同样重要的是中层知识或高层知识，如有关目标或目标模型的对称性。

This suggests to us that image segmentation based on low level cues can not and should not aim to produce a complete final "correct" segmentation. The objective should instead be to use the low-level coherence of brightness, color, texture or motion attributes to sequentially come up with candidate partitions. Mid and high level knowledge can be used to either confirm these groups or select some for further attention. This attention could result in further repartitioning or grouping. The key point is that image partitioning is to be done from the big picture downwards, rather like a painter first marking out the major areas and then filling in the details.

这说明，基于低层特征的图像分割，不能也不应当将目标定为生成完整的最终的正确分割。其目标应当是，使用亮度、色彩、纹理或运动属性的低层连续性，然后得到候选的分割。中层和高层知识，可以用于确认这些分组，或选择一些分组进行进一步的研究。这种研究应当得到进一步的重新分割或分组。关键点是，图像分组是从大局开始往下进行的，很像一个画家，首先标记出主要区域，然后将细节充满。

Prior literature on the related problems of clustering, grouping and image segmentation is huge. The clustering community[9] has offered us agglomerative and divisive algorithms; in image segmentation we have region-based merge and split algorithms. The hierarchical divisive approach that we are advocating produces a tree, the dendrogram. While most of these ideas go back to the 70s (and earlier), the 1980s brought in the use of Markov Random Fields[7] and variational formulations[13,2, 11]. The MRF and variational formulations also exposed two basic questions (1) What is the criterion that one wants to optimize? and (2) Is there an efficient algorithm for carrying out the optimization? Many an attractive criterion has been doomed by the inability to find an effective algorithm to find its minimum-greedy or gradient descent type approaches fail to find global optima for these high dimensional, nonlinear problems.

有关问题，如聚类，分组和图像分割，之前的文献是非常多的。聚类团体[9]提供了合并算法和分离算法；在图像分割中，我们有基于区域的合并和分裂算法。我们支持的层次式的分离方法，生成了一棵树，即树状图。这些想法大多数可以追溯到70s（以及更早的时候），1980s带来了马尔可夫随机场的使用，和变分方程。MRF和变分方程也提出了两个最基本的问题：(1)想要优化的准则是什么？(2)是否存在一种高效的算法进行这种优化？很多很吸引人的准则都肯定是不行的，因为不能找到有效的算法来找到其最小贪婪或梯度下降类型方法，不能为这些高维的非线性问题找到全局最优点。

Our approach is most related to the graph theoretic formulation of grouping. The set of points in an arbitrary feature space are represented as a weighted  undirected graph G = (V, E), where the nodes of the graph are the points in the feature space, and an edge is formed between every pair of nodes. The weight on each edge, w(i,j), is a function of the similarity between nodes i and j.

我们的方法与图论里分组的表述最为相关。在任意特征空间中的点集表示为一个加权无向图G = (V, E)，其中图的节点是特征空间中的点，在每一对节点之间都形成一条边缘。在每条边上的权重，w(i,j)，都是节点i和j之间的相似度。

In grouping, we seek to partition the set of vertices into disjoint sets V1,V2, ...,Vm,, where by some measure the similarity among the vertices in a set Vi is high and across different sets Vi,Vj is low.

在分组中，我们将顶点集分割成不相交的点集V1,V2, ...,Vm，通过某种相似性度量，集合Vi中的顶点相似度很高，在不同集合Vi,Vj之间相似性很低。

To partition a graph, we need to also ask the following questions: 为分割一个图，我们需要问下面的问题：

1. What is the precise criterion for a good partition? 一个好的分割，怎样才是精确的准则？

2. How can such a partition be computed efficiently? In the image segmentation and data clustering community, there has been much previous work using variations of the minimal spanning tree or limited neighborhood set approaches. Although those use efficient computational methods, the segmentation criteria used in most of them are based on local properties of the graph. Because perceptual grouping is about extracting the global impressions of a scene, as we saw earlier, this partitioning criterion often falls short of this main goal. 这样一个分割怎样才能高效的计算得到？在图像分割和数据聚类团体中，之前有很多的工作，使用最小张树的变体或有限邻域集的方法。虽然这些使用了高效的计算方法，在多数方法中使用的分割准则，是基于图的局部性质。因为感知分组是关于提取一个场景的全局印象，我们在之前看到过，这种分割准则通常因为全局目标缺少而失败。

In this paper we propose a new graph-theoretic criterion for measuring the goodness of an image partition - the normalized cut. We introduce and justify this criterion in section 2. The minimization of this criterion can be formulated as a generalized eigenvalue problem; the eigenvectors of this problem can be used to construct good partitions of the image and the process can be continued recursively as desired (section 3). In section 4 we show experimental results. The formulation and minimization of the normalized cut criterion draws on a body of results, theoretical and practical, from the numerical analysis and theoretical computer science communities - section 5 discusses previous work on the spectral partitioning problem. We conclude in section 6.

本文中，我们提出一种新的图论准则来衡量图像分割的好坏，归一化图割。我们在第2部分中提出并证明此准则的正当性。这种规则的最小化可以表述为一个通用的特征值问题；这个问题的特征向量可以用于构建好的图像分割，这个过程可以迭代进行，这在第3部分进行进行描述。第4部分中，我们给出试验结果。归一化图割的表述和最小化，得到了一系列结果，包括理论的和实践的，从数值分析和理论计算机科学团体 - 第5部分讨论了在谱分割问题上之前的工作。在第6部分中，我们进行总结。

## 2. Grouping as graph partitioning

A graph G = (V,E) can be partitioned into two disjoint sets, A, B, A ∪ B = V, A ∩ B = 0, by simply removing edges connecting the two parts. The degree of dissimilarity between these two pieces can be computed as total weight of the edges that have been removed. In graph theoretic language, it is called the cut:

一个图G = (V,E)可以分割成两个不相交的集合，A, B, A ∪ B = V, A ∩ B = 0，只要简单的将连接两个部分的边去掉。这两部分的不相似程度，可以计算为移除掉的边的总权重。在图论的语言里，这称为割：

$$cut(A,B) = \sum_{u∈A, v∈B} w(u,v)$$

The optimal bi-partitioning of a graph is the one that minimizes this cut value. Although there are exponential number of such partitions, finding the minimum cut of a graph is a well studied problem, and there exist efficient algorithms for solving it.

一个图的最佳双割是使这个割的值最小的那个。虽然这种分割的数量有指数级的数量，但找到一个图的最小割，是一个研究很透彻的问题，有很高效的算法进行求解。

Wu and Leahy [18] proposed a clustering method based on this minimum cut criterion. In particular, they seek to partition a graph into k-subgraphs, such that the maximum cut across the subgroups is minimized. This problem can be efficiently solved by recursively finding the minimum cuts that bisect the existing segments. As shown in Wu & Leahy's work, this globally optimal criterion can be used to produce good segmentation on some of the images.

Wu and Leahy [18]基于这个最小割的原则，提出了一种聚类方法。特别是，他们试图将一个图分割成k个子图，这样在子分组中最大的割得到最小化。这个问题可以通过迭代找到将现有的片段进行二等分的最小割来高效实现。如Wu & Leahy的工作所示，这种全局最优的准则可以用于生成部分图像的好的分割。

However, as Wu and Leahy also noticed in their work, the minimum cut criteria favors cutting small sets of isolated nodes in the graph. This is not surprising since the cut defined in (1) increases with the number of edges going across the two partitioned parts. Figure (2) illustrates one such case. Assuming the edge weights are inversely proportional to the distance between the two nodes, we see the cut that partitions out node n1 or n2 will have a very small value. In fact, any cut that partitions out individual nodes on the right half will have smaller cut value than the cut that partitions the nodes into the left and right halves.

但是，Wu & Leahy在其工作中也注意到了，最小割的准则倾向于在图中割出小的集合，包含孤立的节点。这并不令人惊讶，因为(1)中定义的割随着连接两个分割的部分的边的增加而增加。图2描述了一种这样的情况。假设边的权重与两个节点之间的距离成反比，我们会看到将节点n1或n2分割出来的割，会有一个很小的值。实际上，任何将右边一半的单个节点分割出来的割，其割值会小于那些，将节点分割到左边一半或右边一半的。

To avoid this unnatural bias for partitioning out small sets of points, we propose a new measure of dis-association between two groups. Instead of looking at the value of total edge weight connecting the two partitions, our measure computes the cut cost as a fraction of the total edge connections to all the nodes in the graph. We call this disassociation measure the normalized cut (Ncut):

为防止这种分割出小型点集的不自然偏移，我们提出一种新的度量两个分组之间的分离情况。我们没有关注连接两个分割的总计边缘权重，我们的度量计算的割的代价，是总结边缘连接在图中所有节点的占比。我们称这种分离度量为，归一化割裂(Ncut)：

$$Ncut(A,B) = \frac {cut(A,B)}{asso(A,V)} + \frac {cut(A,B)}{asso(B,V)}$$(2)

where $asso(A,V) = \sum_{u∈A, t∈V} w(u,t)$ is the total connection from nodes in A to all nodes in the graph, and asso(B,V) is similarly defined. With this definition of the disassociation between the groups, the cut that partitions out small isolated points will no longer have small Ncut value, since the cut value will almost certainly be a large percentage of the total connection from that small set to all other nodes. In the case illustrated in figure 2, we see that the cut1 value across node n1 will be 100% of the total connection from that node.

其中$asso(A,V) = \sum_{u∈A, t∈V} w(u,t)$，是从A中的节点到图中所有节点的总连接，asso(B,V)的定义类似。与分组间分离度量的定义一起，分割出小的孤立点的割将不会有很小的Ncut值，因为这个割值几乎肯定会是总连接的很大占比，即小型集合到所有其他节点的距离。在图2中所描述的情况中，我们看到节点n1的cut1值，是从这个节点的所有连接的100%。

In the same spirit, we can define a measure for total normalized association within groups for a given partition:

以同样的思想，我们可以对一个给定的分割的分组定义总计的归一化连接的一个度量：

$$Nasso(A,B) = \frac {asso(A,A)} {asso(A,V)} + \frac {asso(B,B)} {asso(B,V)}$$(3)

where asso(A,A) and asso(B,B) are total weights of edges connecting nodes within A and B respectively. We see again this is an unbiased measure, which reflects how tightly on average nodes within the group are connected to each other.

其中asso(A,A)和asso(B,B)分别是A和B内部连接节点的边的总计权重。我们再一次看到，这是一个无偏的度量，反应了节点在分组中连接到彼此的紧密平均程度。

Another important property of this definition of association and disassociation of a partition is that they are naturally related:

一个分割的连接和分离的这个定义另一个重要的性质是，他们在天然上就是关联在一起的：

$$Ncut(A,B) = \frac {cut(A,B)} {asso(A,V)} + \frac {cut(A,B)} {asso(B,V)} = \frac {asso(A,V)-asso(A,A)} {asso(A,V)} + \frac {asso(B,V)-asso(B,B)} {asso(B,V)} = 2 - (\frac {asso(A,A)} {asso(A,V)} + \frac {asso(B,B)} {asso(B,V)}) = 2 - Nasso(A,B)$$

Hence the two partition criteria that we seek in our grouping algorithm, minimizing the disassociation between the groups and maximizing the association within the group, are in fact identical, and can be satisfied simultaneously. In our algorithm, we will use this normalized cut as the partition criterion.

所以，我们在分组算法中追寻的这两个分割准则，最小化分组之间的分离，和最大化分组内的关联，实际上是等价的，可以同时满足。在我们的算法中，我们会使用归一化的割作为分割准则。

Having defined the graph partition criterion that we want to optimize, we will show how such an optimal partition can be computed efficiently.

定义了我们要优化的图割的准则，我们接下来展示一下，怎样高效的计算得到一个最佳的分割。

### 2.1 Computing the optimal partition

Given a partition of nodes of a graph, V, into two sets A and B, let x be an N=|V| dimensional indicator vector, $x_i=1$ if node i is in A, and -1 otherwise. Let $d(i)=\sum_j w(i,j)$, be the total connection from node i to all other nodes. With the definition x and d we can rewrite Ncut(A,B) as:

将一个图V的节点分割成两个集合A和B，令x为一个N=|V|维的指示向量，如果节点i在A中，$x_i=1$，否则就-1。令$d(i)=\sum_j w(i,j)$。令$d(i)=\sum_j w(i,j)$为从节点i到其他所有节点的总连接。有了x和d的定义，我们可以将Ncut(A,B)重写为：

$$Ncut(A,B) = \frac {cut(A,B)}{asso(A,V)} + \frac {cut(B,A)}{asso(B,V)} = \frac {\sum_{(x_i>0, x_j<0)}-w_{ij}x_ix_j} {\sum_{x_i>0}d_i} + \frac {\sum_{(x_i<0, x_j>0)}-w_{ij}x_ix_j} {\sum_{x_i<0}d_i}$$

Let D be an NxN diagonal matrix with d on its diagonal, W be an NxN symmetrical matrix with $W(i,j)=w_{ij}$, $k=\frac {\sum_{x_i>0}d_i} {\sum_i d_i}$, and 1 be an Nx1 vector of all ones. Using the fact $\frac {1+x} {2}$ and $\frac {1-x} {2}$ are indicator vectors for $x_i>0$ and $x_i<0$ respectively, we can rewrite 4[Ncut(x)] as:

令D为一个NxN的对角矩阵，对角线元素为d，W为一个NxN的对称矩阵，$W(i,j)=w_{ij}$，$k=\frac {\sum_{x_i>0}d_i} {\sum_i d_i}$，1为一个Nx1的1值向量。$\frac {1+x} {2}$和$\frac {1-x} {2}$分别是$x_i>0$和$x_i<0$的指示向量，我们可以将4[Ncut(x)]重写为：

$$=\frac {(1+x)^T (D-W) (1+x)} {k1^TD1} + \frac {(1-x)^T (D-W) (1-x)} {(1-k)1^TD1} = \frac {(x^T(D-W)x+1^T(D-W)1)} {k(1-k)1^TD1} + \frac {2(1-2k)1^T(D-W)x} {k(1-k)1^TD1}$$

Let $α(x)=x^T(D-W)x, β(x)=1^T(D-W)x, γ=1^T(D-W)1$, and $M=1^TD1$, we can then further expand the above equation as:

$$=\frac {(α(x)+γ)+2(1-2k)β(x)} {k(1-k)M} = \frac {(α(x)+γ)+2(1-2k)β(x)} {k(1-k)M} - \frac {2(α(x)+γ)} {M} + \frac {2α(x)} {M} + \frac {2γ} {M}$$

dropping the last constant term, which in this case equals 0, we get

... a lot of equations ...

Putting everything together we have,

$$min_x Ncut(x) = min_y \frac {y^T(D-W)y}{y^TDy}$$(5)

where $y = (1+x)-b(1-x)$, with condition $y_i∈${1, -b} and $y^TD1=0$.

Note that the above expression is the Rayleigh quotient[8]. If y is relaxed to take on real values, we can minimize equation (5) by solving the generalized eigenvalue system,

注意上述表达式是Rayleigh商。如果y取实值的话，我们可以求解一般化特征值系统，将式(5)最小化

$$(D-W)y = λDy$$(6)

However, we have two constraints on y, which come from the condition on the corresponding indicator vector x. First consider the constraint $yTD1 = 0$. We can show this constraint on y is automatically satisfied by the solution of the generalized eigensystem. We will do so by first transforming equation (6) into a standard eigensystem, and show the corresponding condition is satisfied there. Rewrite equation (6) as

但是，y还有两个约束，这来自于对应的指示向量x的条件。首先考虑约束$yTD1 = 0$。我们可以证明，y的这个约束，在求解一般化的特征系统问题中自动满足。我们首先将式(6)转变成一个标准的特征系统，证明对应的条件在那里是满足的。重写式(6)为：

$$D^{-\frac{1}{2}}(D-W)D^{-\frac{1}{2}}z=λz$$(7)

where $z=D^{\frac{1}{2}}y$. One can easily verify that $z_0=D^{\frac{1}{2}}1$ is an eigenvector of equation (7) with eigenvalue of 0. Furthermore, $D^{-\frac{1}{2}}(D-W)D^{-\frac{1}{2}}$ is symmetrical semi-positive definite, since (D-W), also called the Laplacian matrix, is known to be semi-positive definite[1]. Hence $z_0$ is in fact the smallest eigenvector of equation (7), and all eigenvectors of equation (7) are perpendicular to each other. In particular, $z_1$ the second smallest eigenvector is perpendicular to $z_0$. Translating this statement back into the general eigensystem (6), we have 1) $y_0 = (0,1)$ is the smallest eigenvector, and 2) $0=z_1Tz_0=y_1^TD_1$, where $y_1$ is the second smallest eigenvector of (6).

其中$z=D^{\frac{1}{2}}y$。可以很容易的核验，$z_0=D^{\frac{1}{2}}1$是(7)的一个特征向量，对应的特征值为0。而且，$D^{-\frac{1}{2}}(D-W)D^{-\frac{1}{2}}$是一个对称的半正定矩阵，由于(D-W)也称为Laplacian矩阵，是半正定的。因此$z_0$实际上是式(7)的最小特征向量，(7)式的所有特征向量都是彼此垂直的。特别是，$z_1$第二小的特征向量是垂直于$z_0$的。将这个翻译到一般的特征系统(6)中，我们有1)$y_0 = (0,1)$是最小的特征向量，2)$0=z_1Tz_0=y_1^TD_1$，其中$y_1$是(6)的第二最小的特征向量。

Now recall a simple fact that about the Rayleigh quotient[8]: 现在回想Rayleigh商的简单事实：

Let A be a real symmetric matrix. Under the constraint that x is orthogonal to the j-1 smallest eigenvectors $x_1, ..., x_{j-1}$, the quotient $\frac {x^TAx}{x^Tx}$ is minimized by the next smallest eigenvector $x_j$, and its minimum value is the corresponding eigenvalue $λ_j$.

令A是一个实值对称矩阵。在x与j-1个最小的特征向量$x_1, ..., x_{j-1}$都垂直的约束下，商$\frac {x^TAx}{x^Tx}$由下一个最小的特征向量$x_j$最小化，其最小值就是对应的特征值$λ_j$。

As a result, we obtain: 结果，我们得到

$$z_1 = argmin_{z^Tz_0=0} \frac {z^TD^{-\frac{1}{2}}(D-W)D^{-\frac{1}{2}}z}{z^Tz}$$(8)

and consequently,

$$y_1 = argmin_{y^TD1=0} \frac {y^T(D-W)y} {y^TDy}$$(9)

Thus the second smallest eigenvector of the generalized eigensystem (6) is the real valued solution to our normalized cut problem. The only reason that it is not necessarily the solution to our original problem is that the second constraint on y that $y_i$ takes on two discrete values is not automatically satisfied. In fact relaxing this constraint is what makes this optimization problem tractable in the first place. We will show in section (3) how this real valued solution can be transformed into a discrete form.

因此，一般化的特征系统(6)的第二小的特征向量，就是我们的归一化割问题的实值解。这个解并不一定是我们原始问题的解的唯一原因，是y的第二个约束，即$y_i$只取两个离散值，并不会自动满足。实际上，约束条件的松弛，是首先使这个优化问题容易处理的。我们会在第(3)部分展示，这个实值解怎样变换为离散形式的。

A similar argument can also be made to show that the eigenvector with the third smallest eigenvalue is the real valued solution that optimally sub-partitions the first two parts. In fact this line of argument can be extended to show that one can sub-divide the existing graphs, each time using the eigenvector with the next smallest eigenvalue. However, in practice because the approximation error from the real valued solution to the discrete valued solution accumulates with every eigenvector taken, and all eigenvectors have to satisfy a global mutual orthogonal constraint, solutions based on higher eigenvectors become unreliable. It is best to restart solving the partitioning problem on each subgraph individually.

可以得到一个类似的观点，证明第三最小的特征值对应的特征向量，是对前两个部分进行子分割的实值最优解。实际上，这种观点可以进行拓展，即可以对现有的图进行子分割，每次都使用下一个最小特征值对应的特征向量。但是，在实际中，因为从实值解到离散值解的近似误差会随着每个特征向量累积，所有的特征向量都需要满足全局相互正交的约束，所以基于更高层的特征向量的解变得不太可靠。在每个子图中，最好独立重新开始求解分割问题。

In summary, we propose using the normalized cut criteria for graph partitioning, and we have shown how this criteria can be computed efficiently by solving a generalized eigenvalue problem.

总结起来，我们提出了使用归一化割的准则进行图分割，我们证明了这个准则通过求解一个一般化的特征值问题来进行高效的计算。

## 3 The grouping algorithm 分组算法

As we saw above, the generalized eigensystem in (6) can be transformed into a standard eigenvalue problem. Solving a standard eigenvalue problem for all eigenvectors takes O(n^3) operations, where n is the number of nodes in the graph. This becomes impractical for image segmentation applications where n is the number of pixels in an image. Fortunately, our graph partitioning has the following properties: 1) the graphs often are only locally connected and the resulting eigensystems are very sparse, 2) only the top few eigenvectors are needed for graph partitioning, and 3) the precision requirement for the eigenvectors is low, often only the right sign bit is required. These special properties of our problem can be fully exploited by an eigensolver called the Lanczos method. The running time of a Lanczos algorithm is O(mn)+O(m M(n))[8], where m is the maximum number of matrix-vector computations allowed, and M(n) is the cost of a matrix-vector computation. In the case where (D - W) is sparse, matrix-vector takes only O(n) time. The number m depends on many factors[8]. In our experiments on image segmentations, m is typically less than $O(n^{1/2})$.

我们在上面看到了，(6)式中的一般化的特征系统可以变换成一个标准的特征值系统。求解一个标准的特征值问题，得到所有特征向量，其运算复杂度为O(n^3)，其中n是图中的节点数量。这对于图像分割应用来说，n就是图像中像素的数量，就有些很不实际。幸运的是，我们的图分割有以下的性质：1)图经常是局部连接的，得到的特征系统非常稀疏；2)只需要最高的少数几个特征向量来进行图割，3)特征向量的精度需求很低，通常只需要正确的符号位就可以了。我们的问题的这些特殊的性质，可以由一个特征求解器充分利用，称为Lanczos方法。Lanczos算法的运行时间是O(mn)+O(m M(n))[8]，其中m是允许的矩阵-向量计算的最大值，M(n)是一个矩阵-向量计算的代价。在(D-W)为稀疏的情况下，矩阵-向量运算复杂度为O(n)。m依赖于很多因素[8]。在我们的图像分割试验中，m通常小于$O(n^{1/2})$。

Once the eigenvectors are computed, we can partition the graph into two pieces using the second smallest eigenvector. In the ideal case, the eigenvector should only take on two discrete values, and the signs of the values can tell us exactly how to partition the graph. However, our eigenvectors can take on continuous values, and we need to choose a splitting point to partition it into two parts. There are many different ways of choosing such splitting point. One can take 0 or the median value as the splitting point, or one can search for the splitting point such that the resulting partition has the best Ncut(A,B) value. We take the latter approach in our work. Currently, the search is done by checking l evenly spaced possible splitting points, and computing the best Ncut among them. In our experiments, the values in the eigenvectors are usually well separated, and this method of choosing a splitting point is very reliable even with a small l.

一旦计算了特征向量，我们可以使用第二最小的特征向量来将图分割成两份。在理想的情况下，特征向量应当只取两个离散值，这些值的符号会告诉我们怎样分割这个图。但是，我们的特征向量的取值为连续的，我们需要选择一个分裂点，来将其分割成两部分。有很多不同的方法来选择这样的分裂点。可以选择0或中值点作为分裂点，或可以搜索分裂点，这样得到的分割有最佳的Ncut(A,B)值。我们在本文中选择后者。目前，搜索是这样进行的，检查l个平均间隔的可能分裂点，在其中计算最佳的Ncut值。在我们的试验中，特征向量的值通常都是很好的分隔的，即使只选择一个很小的l，这个方法也可以选择出一个很可靠的分裂点。

After the graph is broken into two pieces, we can recursively run our algorithm on the two partitioned parts. Or equivalently, we could take advantage of the special properties of the other top eigenvectors as explained in previous section to subdivide the graph based on those eigenvectors. The recursion stops once the Ncut value exceeds certain limit.

在图分裂成两段以后，我们可以在两个分割好的部分上递归的运行我们的算法。或等价的，我们可以利用其他特征向量的特殊性质，以基于这些特征向量来对图进行进一步的分割。一旦Ncut值超过了特定的极限，这个递归过程就停止了。

We also impose a stability criterion on the partition, rather analogous to a localization criterion in edge detection. In edge detection, we can distinguish a real edge from a region of high shading gradient by the criterion that varying the position of a true edge changes its strength, while in a smoothly shaded region varying the position of the putative edge does not effect its strength. In the current context, we regard a cut as unstable if varying the set of graph edges forming the cut, the Ncut value does not change much. To compute this stability measure, we vary the value of splitting point around the optimal value, and induce two different partitions, P1 = (A1,B1) and P2 = (A2,B2). The stability measure is the ratio $\frac {δcut(P1,P2)}{δD(P1,P2)}$, where $δD(P1,P2) = \sum_{i∈(A1/A2)} d_i$.

我们还对这个分割加上了一个稳定准则，与边缘检测中的局部准则非常类似。在边缘检测中，我们区分一个真实的边缘和一个具有高阴影梯度的区域，是通过如下的准则，即变换一个真实边缘的位置，会使其强度变化，而在一个平滑阴影的区域，变换假定的边缘的位置，不会影响其强度。在目前的上下文中，如果对形成一个割的图边缘进行变换，其Ncut值变化不大，我们认为这个割是不稳定的。为计算这个稳定性的度量，我们在最优值附近对分裂点的值进行变化，并引入两个不同的分割，P1 = (A1,B1)和P2 = (A2,B2)。稳定性度量是比值$\frac {δcut(P1,P2)}{δD(P1,P2)}$，其中$δD(P1,P2) = \sum_{i∈(A1/A2)} d_i$。

Our grouping algorithm can be summarized as follows: 我们的分组算法可以总结如下：

1. Given a set of features, set up a weighted graph G = (V,E), compute the weight on each edge, and summarize the information into W, and D. 给定一个特征集合，设置一个加权图G=(V,E)，计算每个边上的权重，并将信息总结进入W和D中。

2. Solve (D-W)z = λDZ for eigenvectors with the smallest eigenvalues. 对最小的特征值，求解(D-W)z = λDZ得到特征向量。

3. Use the eigenvector with second smallest eigenvalue to bipartition the graph by finding the splitting point such that Ncut is maximized, 用第二小的特征值对应的特征向量，通过找到分裂点，使得Ncut值最大化，从而对图进行二分；

4. Decide if the current partition should be subdivided by checking the stability of the cut, and make sure Ncut is below pre-specified value, 检查割的稳定的，确定Ncut小于预先指定的值，确定目前的分割是否还需要进一步进行分割；

5. Recursively repartition the segmented parts if necessary. 如果需要，递归的对分割的部分进行再次分割。

The number of groups segmented by this method is controlled directly by the maximum allowed Ncut. 通过这种方法分割出来的分组数量，是由最大允许的Ncut直接控制的。

## 4 Experiments

We have applied our grouping algorithm to monocular image segmentation based on brightness, color, or texture information. In each case, we construct the graph G = (V,E) by taking each pixel as a node, and define the edge weight $W_{ij}$ between node i and j as the product of a feature similarity term and spatial proximity term:

我们将分组算法应用到了单目图像分割中，基于亮度、色彩或纹理信息。在每种情况下，我们构建图G=(V,E)的方式是，将每个像素作为一个节点，定义节点i和j之间的边的权重$W_{ij}$，为特征相似性和空间接近程度的乘积：

$$w_{ij} = e^{-\frac {-||F(i)-F(j)||_2}{σ_I}} * \left\{ \begin{matrix} e^{-\frac {-||X(i)-X(j)||_2}{σ_X}} & if ||X(i)-X(j)||_2<r \\ 0 & otherwise \end{matrix} \right.$$

where X(i) is the spatial location of node i, and F(i) is the feature vector based on intensity, color, or texture information at that node defined as:

其中X(i)是节点i的空间位置，F(i)是在那个节点上基于亮度、色彩或纹理信息的特征向量，定义为：

- F(i)= 1, in the case of segmenting point sets, 在分割点集的情况下；

- F(i) = I(i),the intensity value, for segmenting brightness images, 亮度值，对于分割亮度图像；

- F(i) = $[v, v⋅s⋅sin(h), v⋅s⋅cos(h)](i)$, where h,s,v are the HSV values, for color segmentation, 其中h,s,v是HSV值，对于色彩分割；

- F(i) = $[|I*f_1|, ..., |I*f_n|](i)$, where the f_i are DOOG filters at various scales and orientations as used in [12], in the case of texture segmentation. 其中f_i是在不同尺度和方向上的DOOG滤波器，在[12]中使用，在纹理分割的情况下。

Note that the weight $w_{ij} = 0$ for any pair of nodes i and j that are more than r pixels apart. 注意对于超过r个像素的两个节点i和j，权重$w_{ij} = 0$。

We first tested our grouping algorithm on spatial point sets similar to the one shown in figure (2). Figure (3) shows the point set and the segmentation result. As we can see from the figure, the normalized cut criterion is indeed able to partition the point set in a desirable way as we have argued in section (2).

我们首先在空间点集中测试了我们的分组算法，与在图(2)中展示的类似。图(3)展示了点集和分割结果。我们可以从图中看出来，归一化割的准则确实可以对点集进行很理想的分割。

Figures (4), (5), and (6) shows the result of our segmentation algorithm on various brightness images. Figure (4) is an synthetic image with added noise. Figure (5) and (6) are natural images. Note that the “objects” in figure (6) have rather ill-defined boundary which would make edge detection perform poorly. Figure (7) shows the segmentation on a color image, reproduced in gray scale in these proceedings. The original image and many other examples can be found at web site http://www.cs.berkeley.edu/~jshi/Grouping.

图(4)(5)(6)在各种亮度的图像中展示了分割算法的结果。图(4)是一幅合成图像，并有噪声。图(5)(6)是自然图像。注意图(6)中的目标其边缘是颇为ill-defined，会使边缘检测的表现非常差。图(7)展示了在彩色图像中的分割结果，这里以灰度图像进行复现。原始图像和很多其他的例子可以在网站上找到。

Note that in all these examples the algorithm is able to extract the major components of scene, while ignoring small intra-component variation. As desired, recursive partitioning can be used to further decompose each piece.

注意，在所有这些例子中，算法可以提取场景的主要部分，而忽略小的部分内的变化。如同预期的一样，递归分割可用于进一步分解每个部分。

Finally, we conclude with preliminary results on texture segmentation for a natural image of a zebra against a background, see figure (8). Note that the measure we have used is orientation-variant, and therefore parts of the zebra skin with different stripe orientation should be marked as seperate regions.

最后，我们对一幅自然图像进行纹理分割，是带有一匹斑马的背景的图像，如图(8)所示。注意，我们所使用的度量是与随方向变化的，因此带有不同条纹方向的斑马皮肤的部分，会被标记为不同的区域。

Note that in these examples, we have considered segmentation based on brightness, color, and texture in isolation. Clearly these can be combined, as also with disparity and motion infomation. Preliminary work in this direction may be found in [15].

注意在这些例子中，我们考虑了基于灰度、色彩和纹理单独进行分割。很明显，这些可以综合起来，以及差异信息和运动信息。在这个方向的初步的工作可以在[15]中找到。

## 5 Related graph partition algorithms

The idea of using eigenvalue problems for finding partitions of graphs originated in the work of Donath & Hoffman[4], and Fiedler[G]. Fiedler suggested that the eigenvector with the second smallest eigenvalue of the system (D- W)x = λx could be used to split a graph. In fact the second smallest eigenvalue is called the Fiedler value, and corrsponding eigenvector the Fiedler vector. This spectral partitioning idea has been revived and further developed by several other researchers, and recently popularized by the work of [1], particularly in the area of parallel scientific computing.

使用特征值问题来找到图分割的思想，源于Donath & Hoffman[4], 和Fiedler[G]的工作。Fiedler建议，系统(D- W)x = λx第二小的特征值对应的特征向量，可以用于分割一个图。实际上，第二最小特征值称为Fiedler值，对应的特征向量称为Fiedler向量。这种谱分割思想有复兴，由几位其他的研究者进行了发展，[1]的工作非常流行，尤其是在并行科学计算的领域。

In applications to several different areas, many authors have noted that the spectral partition method indeed provides good partitions of graphs [1]. Most of the theoretical work done in this area has been focused on the connection between the ratio of cut and the Fiedler value. A ratio of cut of a partition of V, P = (A, V-A) is defined as $\frac {cut(A, V-A)} {min(|A|, |V-A|)}$. It was shown that if the Fiedler value is small, partitioning graph based on the Fiedler vector will lead to good ratio of cut[16]. Our derivation in section 2.1 can be adapted (by replacing the matrix D in the denominators by the identity matrix I) to show that the Fiedler vector is a real valued solution to the problem of $min_{A⊂V} \frac {cut(A,V-A)} {|A|} + \frac {cut(V-A,A)} {|V-A|}$, which we can call the average cut.

在几种不同的应用领域中，很多作者都指出，谱分割方法确实对图给出了很好的分割[1]。这个领域的多数理论工作关注的都是割率与Fiedler值的关系。V的分割P = (A, V-A)的割率定义为，$\frac {cut(A, V-A)} {min(|A|, |V-A|)}$。已经证明了，如果Fiedler值很小，基于Fiedler向量的分割图会得到很好的割率。我们在2.1节的推导可以进行修改，（将矩阵D替换为单位矩阵I），可以发现Fiedler向量是对问题$min_{A⊂V} \frac {cut(A,V-A)} {|A|} + \frac {cut(V-A,A)} {|V-A|}$的实值解，我们称之为平均割。

Although average cut looks similar to the normalized cut, average cut does not have the important property of having a simple relationship to the average association, which can be analogously defined as $\frac {asso(A,A)} {|A|} + \frac {asso(V-A, V-A)} {|V-A|}$. Consequently, one can not simultaneously minimize the disassociation across the partitions, while maximizing the association within the groups. When we applied both techniques to the image segmentation problem, we found that the normalized cut produces better results in practice.

虽然平均割看起来与归一化割很相似，平均割与平均关联没有很简单的关系，这个重要性质是没有的，平均关联是类似定义为$\frac {asso(A,A)} {|A|} + \frac {asso(V-A, V-A)} {|V-A|}$。结果是，在最大化分组内的关联的同时，不能同时最小化分割之间的分离。当我们对图像分割问题应用同样的技术，我们发现，归一化割在实践中会得到更好的结果。

The generalized eigenvalue approach was first applied to graph partitioning by [5] for dynamically balancing computational load in a parallel computer. Their algorithm is motivated by [1O]’s paper on representing a hypergraph in a Euclidean Space.

一般化的特征值方法第一次应用到图割问题，是[5]在一个并行计算机中动态平衡计算负载。其算法受到[10]的文章的启发，在欧几里得空间中表示一个超图。

In the computer vision community, there are a few related approaches for image segmentation. Wu&Leahy[18] use the minimum cut criterion for their segmentation. Cox et.al. [3] seek to minimize the ratio $\frac {cut(A,V-A)}{weight(A)}, A⊂V$, where weight(A) is some function of the set A. When weight(A) is taken to the be the sum of the elements in A, we see that this criterion becomes one of the terms in the definition of average cut above. Cox et. al. use an efficient discrete algorithm to solve their optimization problem assuming the graph is planar.

在计算机视觉中，有几个相关的方法进行图像分割。Wu&Leahy[18]使用最小割准则进行分割。Cox et.al. [3]最小化比率$\frac {cut(A,V-A)}{weight(A)}, A⊂V$，其中weight(A)是集合A的某个函数。当weight(A)是A中的元素之和，我们会看到，这个规则就成为了上面的平均割的定义的其中一项。Cox et. al.使用了一种高效的离散算法来求解其优化问题，假设图是平面的。

Sarkar & Boyer[14] use the eigenvector with the largest eigenvalue of the system Wx = λx for finding the most coherent region in an edge map. Although their eigensystem is not directly related to the graph partitioning problem, using a similar derivation as in section (2.1), we can see that their system approximate $min_{A⊂V} \frac {asso(A,A)} {|A|}$.

Sarkar & Boyer[14]使用了系统Wx = λx的最大特征值的特征向量，来找到一个边图的最连贯区域。虽然其特征系统与图割问题并不相关，使用2.1节类似的推导，我们可以看到其系统近似的是$min_{A⊂V} \frac {asso(A,A)} {|A|}$。

## 6 Conclusion

In this paper, we developed a grouping algorithm based on the view that perceptual grouping should be a process that aims to extract global impressions of a scene, and that provides a hierarchical description of it. By treating the grouping problem as a graph partitioning problem, we proposed the normalized cut criteria for segmenting the graph. Normalized cut is an unbiased measure of disassociation between subgroups of a graph, and it has the nice property that minimizing normalized cut leads directly to maximizing the normalized association which is an unbiased measure for total association within the sub-groups. In finding an efficient algorithm for computing the minimum normalized cut, we showed that a generalized eigenvalue system provides a real valued solution to our problem.

本文中，基于视觉分组应当是提取一个场景的全局印象的视角，应当给出场景的层次化描述，我们提出了一种分组算法。将分组问题视为一个图割问题，我们提出归一化割的原则来分割图。归一化割是图中的分组之间的不相关性的无偏估计，其有很好的性质，最小化归一化的割会直接带来最大化相关的效果，这个相关是子组之间的总相关的无偏估计。在找到高效的算法来计算最小归一化割上，我们展示了，一般化的特征值的系统，对我们的问题可以给出实值的解。

A computational method based on this idea has been developed, and applied to segmentation of brightness, color, and texture images. Results of experiments on real and synthetic images are very encouraging, and illustrate that the normalized cut criterion does indeed satisfy our initial goal of extracting the “big picture” of a scene.

基于这种思想，提出了一种计算方法，应用到了亮度、色彩和纹理图像的分割中。在真实和合成图像上的试验结果非常鼓舞人心，说明归一化割的准则确实可以实现我们的初始目标，即提取出一个场景的大局。