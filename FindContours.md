# Topological Structural Analysis of Digitized Binary Images by Border Following

Satoshi Suzuk, Keiichi Abe, Shizuoka University, Japan

Two border following algorithms are proposed for the topological analysis of digitized binary images. The first one determines the surroundness relations among the borders of a binary image. Since the outer borders and the hole borders have a one-to-one correspondence to the connected components of 1-pixels and to the holes, respectively, the proposed algorithm yields a representation of a binary image, from which one can extract some sort of features without reconstructing the image. The second algorithm, which is a modified version of the first, follows only the outermost borders (i.e., the outer borders which are not surrounded by holes). These algorithms can be effectively used in component counting, shrinking, and topological structural analysis of binary images, when a sequential digital computer is used.

对二值图像的拓扑分析，提出了两个边缘跟随算法。第一个确定了二值图像中的边缘的附近关系。由于外轮廓和孔轮廓分别与1-像素的连通部分和孔有一对一的对应关系，提出的算法得到了二值图像的一个表示，从中可以提取出一些特征，而不用重建图像。第二个算法，是第一个算法的修正版，只跟随外轮廓（即，没有被孔包围的外轮廓）。在使用一个序列数字计算机时，这些算法可以在二值图像的部件计数、收缩和拓扑结构分析中很好的应用。

## 1. Introduction

Border following is one of the fundamental techniques in the processing of digitized binary images. It derives a sequence of the coordinates or the chain codes from the border between a connected component of 1-pixels (1-component) and a connected component of 0-pixels (background or hole). The border following technique has been studied deeply, because it has a large variety of applications, including picture recognition, picture analysis, and image data compression [1-10].

边缘跟随是二值图像处理中的基础技术。这种技术，从1-像素和0-像素的连通部分（背景或洞）中，推导出一个坐标序列或链码。边缘跟随技术已经研究的很多了，因为有很多应用，包括图像识别，图像分析和图像数据压缩技术。

The purpose of this paper is to propose border following algorithms with a sort of topological analysis capability. If one wants to convert a binary picture into the border representation, then he can extract the topological structure of the image with little additional effort by using the algorithms presented here. The information to be extracted is the surroundness relation among the two types of borders: the outer borders and the hole borders. Since there exists one-to-one correspondence between an outer border and a 1-component, and between a hole border and a 0-component, the topological structure of a given binary image can be determined.

本文的目的是提出有拓扑分析能力的边缘跟随算法。如果一个人想要将二值图像转化到边缘表示，那么他可以使用这里的算法就可以提取出图像的拓扑结构。要提取的信息是两种边缘的周围关系：外边缘与洞边缘。由于外边缘与1-部分有一对一的对应关系，洞边缘与0-部分有一对一的对应关系，所以可以确定给定二值图像的拓扑关系。

Several works have been reported on the topological structural analysis of binary pictures using raster scan and labeling [1, 10, 11]. An alternative for such analysis is to use border following. If an image processing system utilizes the border following for some purpose and at the same time needs to analyze the topological structure of an input image, then this approach would be attractive. Few studies, however, have been devoted to this subject. The existing methods [6, 7] are rather complicated and time consuming, so that they do not seem advantageous over other structural analysis methods [1, 10, 11] which do not use border following.

在二值图像的拓扑结构分析上，使用光栅扫描和标记有几种方法。这种分析的另一种方法是使用轮廓追随。如果图像分析系统利用轮廓追随来实现一些目的，同时需要分析输入图像的拓扑结构，那么这种方法就很吸引人了。在这个主题上很少有研究。现有的方法非常复杂耗时，所以与其他没有使用轮廓跟随的结构分析方法相比，没有什么优势。

In this paper, we first present an algorithm which can extract the topological structure of a given binary image. This algorithm is an extended version of the border following algorithm [1] which discriminates between outer borders and hole borders. The extensions are: (1) to put a unique mark on each border rather than to adopt the same marking procedure for every border (border labeling); and (2) to add a procedure for obtaining the parent border of the currently followed border (see Definition 4). With this algorithm we can extract the surroundness relation among the borders, which corresponds to the surroundness relation among the connected components. If a binary image is stored in the form of the borders and the surroundness relation is extracted by this algorithm, some simple image processing can be done without restoring the original image. Thus the method offers an effective way of storing binary image data.

本文中，我们首先提出了一种算法，可以对一幅给定的二值图像提取其拓扑结构。这个算法是轮廓跟随算法[1]的一种拓展，区分了外轮廓和洞轮廓。其拓展包括：(1)对每个轮廓给出一个唯一的标记，而不是对每个轮廓采用相同的标记过程（轮廓标记）；(2)对每个正在跟随的轮廓，加入一个过程，得到其父轮廓（见定义4）。用这个算法，我们可以提取轮廓之中的周围关系，这对应着连通部分的周围关系。如果一个二值图像是用轮廓的形式存储，用这个算法提取其周围关系，可以进行一些简单的图像处理，而不用恢复原始图像。因此这种方法也给出了存储二值图像数据的有效方式。

Next we show a modified version of the first algorithm which follows only the outermost borders of a binary image (i.e., the outer borders which are not surrounded by holes). When we want to ignore the 1-components which are surrounded by other 1-components, this modified algorithm gives us a quick, sequential method of counting the 1-components or shrinking each 1-component to one point.

下面，我们给出了第一种算法的修改版本，只跟随了二值图像的外轮廓（即，没有被洞包围的外轮廓）。当我们忽略被其他1-部分包围的1-部分，这种修正的算法给出了一种快速的序列的方法来对1-部分进行计数，或将每个1-部分收缩到一个点。

## 2. Basic Concept and Notations

In this paper only digital binary pictures sampled at points of rectangular grids are considered. Though we will follow the general terminology and notations such as in [1], we would like to define and clarify some concepts and notations not so widely established.

本文中只考虑矩形网格点上的二值图像。虽然我们也参考了[1]中的通用术语和记号，但我们也定义了和澄清了一些其他概念和记号。

The uppermost row, the lowermost row, the leftmost column, and the rightmost column of a picture compose its frame. Pixels with densities 0 and 1 are called the 0-pixel and the 1-pixel, respectively. Without loss of generality, we assume that 0-pixels fill the frame of a binary picture. We assume also that we can assign any integer value to a pixel during the processing. The pixel located in the i th row and the j th column is represented by the row number and the column number (i, j). We adopt the following coordinate system: the row number i increases from top to bottom; the column number j from left to right. A picture having the density value f_ij at a pixel (i,j) is denoted by F={f_ij}.

一幅图像由其最上面的行，最下面的行，最左边的列，最右边的列框成。灰度为0和1的像素分别称为0-像素和1-像素。不失一般性，我们假设0像素充满了二值图像。我们还假设，在处理的过程中，可以给像素赋任意整数值。第i行第j列的像素表示为(i,j)。我们采用下面的坐标系统：行数i从上往下增长；列数j从左往右增长。在像素(i,j)处有灰度值f_ij的图像，记为F={f_ij}。

A 1-component and a 0-component are the connected components of 1-pixels and of O-pixels, respectively. If a 0-component S contains the frame of the picture, we call S the background; otherwise, a hole.

1-部分和0-部分分别是1-像素和0-像素的连通部分。如果0-部分S包含了图像了框架，我们称S为背景；否则称之为一个洞。

It is well known that in order to avoid a topological contradiction 0-pixels must be regarded as 8- (4-) connected if 1-pixels are dealt with as 4- (8-) connected. We will say “in the 4- (8-) connected case” when we deal with 1-pixels as 4- (8-) connected and 0-pixels as 8- (4-) connected. When we do not specify the type of the connectivity explicitly, we mean that the argument is valid for both types of the connectivity.

大家都知道，为避免拓扑收缩，如果1-像素是用4-(8-)连通处理的，那么0-像素就必须认为是8- (4-)连通的。当我们以4- (8-)连通处理1-像素，以8- (4-)连通处理0-像素时，我们说在“4- (8-)连通的情况下”。当我们不显式的指定连通性类型时，我们的意思是，参数对两种类型的连通性都是有效的。

The border and the surroundness among connected components are defined as follows. 连通部分的边缘和围绕性定义如下。

**Definition 1** (border point). In the 4- (8-) connected case, a 1-pixel (i, j) having a 0-pixel (p, q) in its 8- (4-) neighborhood is called a border point. It is also described as “a border point between a 1-component S1, and a 0-components S2,” if (i, j) is a member of S1, and (p, q) is a member of S2.

**定义1**（边缘点）。在4-(8-)连通的情况下，一个1-像素(i,j)在其8-(4-)邻域中有一个0-像素(p,q)，那么就称这个1-像素为一个边缘点。如果(i,j)是S1的一部分，(p,q)是S2的一部分，也可以描述为“在1-部分S1和0-部分S2之间的边缘点”。

**Definition 2** (surroundness among connected components). For given two connected components S1, and S2, in a binary picture, if there exists a pixel belonging to S2 for any 4-path from a pixel in S1 to a pixel on the frame, we say that S2 surrounds S1. If S2 surrounds S1 and there exists a border point between them, then S2 is said to surround S1 directly.

**定义2** （连通部分的包围性）。对二值图像中两个给定的连通部分S1和S2，如果对于S1中的一个像素到框架上的一个像素的任意4-路径中，都存在一个像素属于S2，我们就说S2包围了S1。如果S2包围了S1，并在其之间存在一个边缘点，那么就称S2直接包围S1。

**Definition 3**  (outer border and hole border). An outer border is defined as the set of the border points between an arbitrary 1-component and the 0-component which surrounds it directly. Similarly, we refer to the set of the border points between a hole and the 1-component which surrounds it directly as a hole border. We use the term “border” for either an outer border or a hole border. Note that the hole border is defined as a set of 1-pixels (not 0-pixels) as well as the outer border.

**定义3** （外轮廓和洞轮廓）。外轮廓定义为在任意1-部分和直接包围的0-部分之间的边界点结。类似的，我们称在一个洞和直接围绕其的1-部分之间的边界点集为洞边界。外边界和洞边界，我们都称之为边界。注意，洞边界和外边界，我们都定义为1-像素的集合（而不是0-像素）。

The following property holds for the connected components and the borders. 连通部分和边界有以下性质。

**Property 1**. For an arbitrary 1-component of a binary picture its outer border is one and unique. For any hole its hole border (the border between that hole and the l-component which surrounds it directly) is also unique.

**性质1**。对于二值图像中的任意1-部分，有一条外边界且是唯一的。对于任意的洞，其洞边界（在洞和直接包围洞的1-部分之间的边界）也是唯一的。

We define the parent border and the surroundness among borders. 下面我们定义父边界和边界之间的包围性。

Definition 4 (parent border). The parent border of an outer border between a 1-component S1 and the 0-component S2 which surrounds S1 directly is defined as:

**定义4** （父边界）。在1-部分S1和直接包围S1的0部分S2之间的外边界的父边界定义为：

(1) the hole border between S2 and the 1-component which surrounds S2 directly, if S2 is a hole; 如果S2是一个洞，那么就是S2和直接包围S2的1-部分的洞边界；

(2) the frame of the picture, if S2 is the background. 如果S2是背景，那么就是图像的框架。

The parent border of a hole border between a hole S3 and the 1-component S4 which surrounds S3 directly is defined as the outer border between S4 and the 0-component which surrounds S4 directly.

洞S3和直接包围S3的1-部分之间的洞边界，其父边界定义为S4和直接包围S4的0-部分的外边界。

**Definition 5** (surroundness among borders). For two given borders B0 and Bn of a binary picture, we say that Bn surrounds B0 is there exists a sequence of border B0, B1, . . . , Bn such that Bk is the parent border of Bk-1 for all k (1≤k≤n).

**定义5** 边界之间的包围性。对于二值图像中两个给定的边界B0和Bn，如果有一个边界序列B0, B1, . . . , Bn，对于所有的k(1≤k≤n)，Bk是Bk-1的父边界，我们说Bn包围B0。

From Definitions 2 through 5, we see that the following property holds between the surroundness among connected components and the surroundness among borders (Fig. 1).

从定义2到定义5，我们可以看出，在连通部分和边界之间的包围性上，有以下性质（图1）。

**Property 2**. For any binary picture, the above two surroundness relations are isomorphic with the following mapping:

**性质2**。对于任意二值图像，上述两种包围关系是同构的，有下列映射：

a 1-component ↔ its outer border; 一个1-部分↔其外轮廓；

a hole ↔ its hole border between the hole and the 1-component surrounding it directly; 一个洞↔其洞轮廓（在洞和直接包围其的1-部分之间）

the background ↔ the frame. 背景↔框架。

## 3. The Border Following Algorithm for Topological Analysis

We present a border following algorithm for topological structural analysis. This extracts the surroundness relation among the borders of a binary picture. First we give an informal explanation of the algorithm.

我们为拓扑结构分析提出一个边界跟随算法。该算法提取了二值图像中边界的包围关系。首先，我们给出该算法的非正式解释。

**Algorithm 1**. Scan an input binary picture with a TV raster and interrupt the raster scan when a pixel (i, j) is found which satisfies the condition for the border following starting point of either an outer border (Fig. 2a) or a hole border (Fig. 2b). If the pixel (i, j) satisfies both of the above conditions, (i, j) must be regarded as the starting point of the outer border. Assign a uniquely identifiable number to the newly found border. Let us call it the sequential number of the border and denote it by NBD.

**算法1**。用TV光栅扫描输入二值图像，当像素(i,j)发现满足边界跟踪起始点的条件时中止光栅，可以是外边界点或洞边界点。如果像素(i,j)满足上述两个条件，(i,j)必须认为是外边界的起始点。对新发现的边界指定一个唯一可识别数。我们称之为边界序列数，将之表示为NBD。

Determine the parent border of the newly found border as follows. During the raster scan we also keep the sequential number LNBD of the (outer or hole) border encountered most recently. This memorized border should be either the parent border of the newly found border or a border which shares the common parent with the newly found border. Therefore, we can decide the sequential number of the parent border of the newly found border with the types of the two borders according to Table 1.

确定新发现的边界的父边界的方法如下。在光栅扫描过程中，我们还记录了边界（外边界或洞边界）遇到的最多的序列数LNBD。这些记住的边界，应当要么是新发现的边界的父边界，或与新发现的边界有共同的父边界。因此，我们可以根据表1来确定新发现的边界的父边界的序列数。

Follow the found border from the starting point, marking the pixels on the border. The border following scheme is the classical one [1-3]; the distinguished feature of our algorithm is the marking policy.

从起始点跟随发现的边界，在边界上标记像素。边界跟踪方案是经典的方法；我们算法的特征是标记策略。

(a) If the current following border is between the 0-component which contains the pixel ( p, q + 1) and the 1-component which contains the pixel (p, q), change the value of the pixel (p, q) to -NBD.

(a) 如果目前正在跟踪的边界是在包含像素(p,q+1)的0-部分和包含(p,q)的1-部分之间的，将像素(p,q)的值更改到-NBD。

(b) Otherwise, set the value of the pixel (p, q) to NBD unless (p, q) is on an already followed border.

(b) 否则，将像素(p,q)的值设为NBD，除非(p,q)已经在跟踪的像素中了。

The conditions (a) and (b) prohibit the pixel (p, q) from being the border following starting points of the already followed hole border and outer border, respectively. The positive value NBD and the negative value, -NBD, correspond to the labels “1” and “r” of the border following algorithm in [1], respectively.

条件(a)和(b)分别避免了像素(p,q)成为已经跟踪的洞边界和外边界的跟随起点。正值NBD和负值-NBD，分别对应着[1]中跟随算法的标签“1”和“r”。

After following and marking the entire border, resume the raster scan. When the scan reaches the lower right corner of the picture, the algorithm stops.

在跟随并标记整个边界之后，恢复光栅扫描。当扫描到达图像的下右角后，算法停止。

Let us explain the process of the algorithm by an example depicted in Fig. 3. The input binary picture is given in (a). When the raster scan reaches the circled 1-pixel, we find that it satisfies the condition for the starting point of an outer border (Fig. 2a). The sequential number 2 is assigned to the border, because the number 1 is set aside for the frame of the picture. We also find out from Table 1 that its parent border is the frame.

让我们用图3中的例子来解释算法的过程。输入的二值图像如(a)所示。当光栅扫描到达圈住的1-像素时，我们发现其满足了外边界起始点的条件（图2a）。序列数2指定到这个边界上，因为数字1已经保留给图像的框架。我们还从表1发现，其父边界是框架。

Then the border following starts and all the pixel values on that border are changed to either 2 or - 2, as shown in (b). After following the entire border, raster scan resumes.

然后开始边界跟踪，这个边界上的所有像素值都变为2或-2，如图(b)所示。在跟踪了整个边界后，恢复光栅扫描。

The next border following starts at the circled pixel in (b), which satisfies the condition for the starting point of a hole border (Fig. 2b). Note that a point having a negative value cannot be a starting point of border following. The parent border of this border (with the number 3) is found to be the outer border with the number 2. During the border following, two pixels with value 2 are changed to -3 by the marking policy (a), and two pixels with value 1 are changed to 3 by the policy (b). All the other already visited pixels (with value 2) are unchanged. The steps go on in a similar manner and we obtain the final results (e). The formal description of Algorithm 1 is given in Appendix I.

下个边界跟踪从(b)中圈住的像素开始，这个点满足了一个洞边界的起始点条件（图2b）。注意带负值的点不能是边界跟踪的起始点。这个边界的父边界（数字3）是带有数字2的外边界。在边界跟踪的过程中，有两个值为2的像素，根据标记策略(a)变为-3，两个值为1的像根据策略(b)变为3。所有其他已访问的像素（值为2）值没有变。这个步骤以类似的方式继续，我们得到最后结果(e)。算法1的正式描述见附录1。

The following properties show the validity of Algorithm 1. 算法1有以下性质。

**Property 3**. The leftmost pixel (i, j) of the uppermost row of any 1-component S1 satisfies the condition shown in Fig. 2a; the border B followed from the pixel (i, j) is an outer border (i.e., the 0-component S2 which (i, j-1) belongs to surrounds S1); after this outer border has been followed, the same outer border is never followed again.

**性质3**。任意1-部分S1的最上面行的最左边像素(i,j)满足图2a的条件；像素(i,j)跟踪得到的边界B是一个外边界（即，包围S1的0-部分，(i,j-1)属于边界）；在这个外轮廓已经被跟随后，相同的外轮廓就不会再被跟随了。

**Property 4**. The pixel (i, j) whose right-hand neighbor (i, j + 1) is the leftmost
pixel of the uppermost row of any hole S1 satisfies the condition shown in Fig. 2b; the border B followed from the pixel (i, j) is a hole border (i.e., the 1-component S2 which contains (i, j) surrounds S1); after this border has been followed, the same hole border is never followed again.

**性质4**。任意洞S1的最上面行的最左边像素(i,j+1)，其左边的像素(i,j)满足图2b所示的条件；从像素(i,j)跟随得到的边界B，是一个洞边界（即，包围S1的1-部分S2，包含(i,j)）；在这个边界跟随得到后，不会再跟随同一个洞。

**Property 5**. The border following starting points of Algorithm 1 have a one-to-one correspondence to the 1-components or the holes.

**性质5**。算法1的起始点跟随得到的边界，与1-部分或洞有一对一的对应性。

**Property 6**. When the parent border of a particular border B is to be determined, the variable LNBD contains the sequential number of either the parent border of B or a border which shares the same parent with B.

**性质6**。当特定边界B的父边界待定时，变量LNBD包含的序列数，要么是B的父边界，要么是与B有着同样父边界的边界。

Algorithm 1, as well as the border following algorithm [1] which discriminates between outer borders and hole borders, has the following advantages in the applications as suggested by Properties 2 through 5.

算法1，以及[1]中的边界跟踪算法，区分了外边界和洞边界，在应用中有如下优势，这由性质2-5可以得到。

(1) We can count the 1-components and the holes in a binary picture and find its Euler number. 我们可以对二值图像中的1-部分和洞进行计数，找到其欧拉数。

(2) We can shrink every 1-component or every hole into one pixel, by representing a 1-component by the border following starting point of its outer border and a hole by the right-hand neighbor of the starting point of its hole border. The former and the latter locate at the leftmost in the uppermost row of the 1-components and the hole, respectively. 我们可以将每个1-部分或洞收缩到一个像素，将1-部分用其外边界的边界跟随起始点来表示，将洞用其洞边界的边界跟随起始点的右邻像素来表示。前者和后者分别位于，1-部分和洞的最上面行的最左边像素。

(3) We can perform (1) or (2) depending on the features derived from the border: for example, we can shrink only the 1-components whose perimeters are greater than a given threshold, or which have their framing rectangles with the areas larger than a specified value. 我们可以从边界的特征中进行(1)或(2)：比如，我们可以只对周长大于给定阈值的1-部分进行收缩，或其框矩形的面积大于指定值的1-部分进行收缩。

Algorithm 1 also has the following advantages as seen from Properties 2 and 6. 算法1还有下面的性质，这可以从性质2-6看出。

(4) We can extract the surroundness relation among the connected components in a binary picture. It can be applied to picture search in an image database or feature extraction of pictures. 我们可以提取二值图中连通部分的包围关系。这可以用于图像数据库中的图像搜索，或图像的特征提取。

(5) The border representation with topological structure derived by Algorithm 1 would be effective for a storing method of pictures. This is because some simple image processing can be executed without restoring the original picture. Such image processing includes: to obtain some features from the borders, e.g., the perimeters and the areas of the components; to analyze the topological structure, e.g., whether a 1-component is adjacent to the background or not, or whether it has more than n holes or not; and to extract or delete connected components or holes depending on their geometrical or topological features. 算法1推导得到的带有拓扑结构的边界表示，对于图像的存储是非常有效的。这是因为，在不恢复出原始图像的情况下，一些简单的图像处理是可以进行的。这些图像处理包括：从边界处得到一些特征，如一些部分的面积和周长；分析拓扑结构，如，1-部分是否与背景相邻接，或是否包含多余n个洞；根据连通部分或洞的几何特征或拓扑特征，提取或删除这些连通部分或洞。

Figure 4 shows an example of the representation of topological structure among borders derived by Algorithm 1. 图4展示了算法1推导得到的边界的拓扑结构的表示的一个例子。

## 4. The Border Following Algorithm for Extracting Only the Outermost Borders

We can modify the algorithm shown above so that it follows only the outermost borders of a binary picture (i.e., the outer border between the background and a 1-component). 我们可以修改上述算法，使其只跟随二值图最外层的边界（即，在背景与1-部分的外边界）。

**Algorithm 2**. We explain here only the difference of the algorithm from Algorithm 1. (1) We start the border following only at the points such that the condition for the border following starting point of an outer border (Fig. 2a) holds and LNBD ≤  0 when the raster scan reaches there. (2) The marking policy is the same as that of Algorithm 1 except that the values “2” and “ - 2” are substituted for the values “NBD” and “-NBD,” respectively. (3) We keep the value LNBD of the nonzero pixel encountered most recently during the raster scan. Every time we begin to scan a new row of the picture, reset LNBD to zero.

**算法2**。我们这里只解释与算法1的差异。(1)我们只在一些点上开始边界跟随，这些点药满足图2a的一个外边界的起始点的条件，当光栅扫描达到这里时LNBD≤0；(2)标记策略与算法1相同，但是值"2"和"-2"分别替换成了“NBD”和“-NBD”，(3)我们保持在光栅扫描中最近遇到的非零值像素的LNBD值。每次我们开始扫描图像的一个新行，将LNBD重置为0。

The rationale for Algorithm 2 is as follows. The parent border of an outermost border is the frame of the picture. Therefore, the border point (i, j) immediately at the right of a 0-pixel is on an outermost border if and only if (i, j) satisfies either of the following conditions (1) or (2).

算法2的原理如下。最外层边界的父边界为图像的框架。因此，在0-像素紧邻右边的边界点(i,j)，是一个最外层的边界，当且仅当(i,j)满足(1)或(2)的一个条件。

(1) All the pixels (i, l), (i, 2), . . . , (i, j - 1) are 0-pixels. 所有像素(i, l), (i, 2), . . . , (i, j - 1)都是0-像素。

(2) The border point (i, h) which has been encountered most recently during the TV raster scan is on an outer border and the pixel (i, h + 1) belongs to the background. 在TV光栅扫描时最近遇到的最多的边界点(i,h)是一个外边界点，且像素(i,h+1)属于背景。

Since we kept the value LNBD of the nonzero pixel encountered most recently during the TV raster scan, the above conditions (1) and (2) can be checked by the conditions “LNBD = 0” and “LNBD = -2,” respectively, without following the other borders than the outermost ones. 由于我们保留了在TV光栅扫描时最近遇到的最多的非零像素点的LNBD值，上述条件(1)和(2)可以分别通过条件“LNBD = 0”和“LNBD = -2”进行检查，而不用跟随除了最外层的其他边界。

**Property 7**. Algorithm 2 follows only the outermost borders in an input binary picture and each outermost border is followed only once. 算法2只跟随输入二值图像中的最外层边界，而且每个最外层边界只跟随一次。

Algorithm 2 is effectively used for 1-component counting and shape analysis if there exists no 1-component surrounded by another 1-component, because it does not follow any hole border and consequently works faster. It is also useful if all the 1-components surrounded by other 1-components are noise or the elements not to be processed. In Table 2 we compare the processing time of several algorithms for 1-component shrinking [1, 12, 13] when a general-purpose sequential digital computer is used. The input pictures, which have no 1-component surrounded by other 1-components, are shown in Fig. 5: one is blobs with optional holes and the other is a line pattern. The shrinking method by Algorithm 2 is almost as fast as the component labeling. If all the 1-components surrounded by other 1-components should be disregarded, it would be most suitable, as the component labeling cannot discriminate in itself the outermost components from the components to be ignored.

如果1-部分不会被另一个1-部分包围，算法2可以高效的用于1-部分技术和形状分析，因为不会跟随任何洞边界，结果工作更快。如果所有被其他1-部分包围的1-部分是噪声，或这些元素不用进行处理，算法也是有用的。在表2中，我们比较了几种1-部分收缩的算法的运行时间。输入图像没有被其他1-部分包围的1-部分，如图5所示：一个是带有可选洞的blobs，另外的是线状模式。算法2的收缩方法与component labeling几乎一样快。如果所有被其他1-部分包围的1-部分应当被抛弃，那么就会更合适，因为component labeling本身不能区分最外层的components和要忽略的components。

## 5. Conclusion

In this paper, we showed a way to analyze the topological structure of binary images by border following. This is an extension of the border following algorithm which discriminates between the outer borders and the hole borders of a binary picture. We also presented a method for counting the 1-components and extracting the borders of a binary picture when it is desirable to disregard all the 1-components but the outermost 1-components. This is a modified version of the first algorithm. These methods are quick and effective if a sequential digital computer is used. We are now investigating their application to document processing, such as flowchart recognition.

本文中，我们展示了一种通过边界跟随来分析二值图像的拓扑结构的方法。这是其他边界跟随算法的拓展，拓展了区分外边界和洞边界的功能。我们还提出了一种方法，当需要抛弃所有1-部分除了最外层的1-部分时，来对1-部分进行计数，提取一个二值图像的边界。这是第一种算法的修正。这些方法运行速度很快。我们正在研究其在文档处理中的应用，比如流程图识别。