# A combined corner and edge detector

Chris Harris, Mike Stephens Plessey Research Roke Manor, United Kingdom

Consistency of image edge filtering is of prime importance for 3D interpretation of image sequences using feature tracking algorithms. To cater for image regions containing texture and isolated features, a combined corner and edge detector based on the local auto-correlation function is utilised, and it is shown to perform with good consistency on natural imagery.

图像边缘滤波的一致性，对于利用特征追踪算法来进行图像序列的3D解释来说，是非常重要的。为满足包含纹理和孤立特征的图像区域，我们利用了一个角点和边缘的综合检测器，这是基于局部的自相关函数的，在自然图像中表现出了很好的一致性。

## 1. Introduction

The problem we are addressing in Alvey Project MMI149 is that of using computer vision to understand the unconstrained 3D world, in which the viewed scenes will in general contain too wide a diversity of objects for topdown recognition techniques to work. For example, we desire to obtain an understanding of natural scenes, containing roads, buildings, trees, bushes, etc., as typified by the two frames from a sequence illustrated in Figure 1. The solution to this problem that we are pursuing is to use a computer vision system based upon motion analysis of a monocular image sequence from a mobile camera. By extraction and tracking of image features, representations of the 3D analogues of these features can be constructed.

我们在Alvey Project MMI149中要处理的问题是，使用计算机视觉来理解无限制的3D世界，其中场景一般很包含非常多种类的目标，以进行自上而下的识别技术来工作。比如，我们期望得到自然场景的理解，包含道路，建筑，树木，灌木，等等，典型的如图1中描述的一个序列中的两帧。我们追求的这个问题的解，是用基于运动分析的计算机视觉系统，图像是从一个移动相机得到的单目图像序列。通过提取和追踪图像特征，可以构建出这些特征的3D类似物的表示。

To enable explicit tracking of image features to be performed, the image features must be discrete, and not form a continuum like texture, or edge pixels (edgels). For this reason, our earlier work has concentrated on the extraction and tracking of feature-points or corners, since they are discrete, reliable and meaningful. However, the lack of connectivity of feature-points is a major limitation in our obtaining higher level descriptions, such as surfaces and objects. We need the richer information that is available from edges.

为显式的追踪要处理的图像特征，这些图像特征必须要是离散的，不能形成一种连续体，比如纹理，或边缘像素(edgels)。基于这些原因，我们之前的工作聚焦在提取和追踪特征点或角点，因为它们是离散的，可靠的，有意义的。但是，这些特征点缺乏连续性，是在我们得到更高层表示的一个主要局限，这些更高层表示比如表面和目标。我们需要从边缘中可用的更丰富的信息。

## 2. The Edge Tracking Problem

Matching between edge images on a pixel-by-pixel basis works for stereo, because of the known epi-polar camera geometry. However for the motion problem, where the camera motion is unknown, the aperture problem prevents us from undertaking explicit edgel matching. This could be overcome by solving for the motion beforehand, but we are still faced with the task of tracking each individual edge pixel and estimating its 3D location from, for example, Kalman Filtering. This approach is unattractive in comparison with assembling the edgels into edge segments, and tracking these segments as the features.

边缘图之间逐个像素的匹配对于立体视觉的情况是可以的，因为相机的对极几何是已知的。但是对于运动问题，相机的运动是未知的，aperture问题使我们不能进行显式的边缘匹配。这可以通过解决手头的运动问题克服，但是我们仍然面临着追踪每个单独的边缘像素，并用Kalman滤波来估计其3D位置的任务。这种方法与将edgel组合成边缘片段，并将这些片段作为特征进行追踪相比，不是很吸引人。

Now, the unconstrained imagery we shall be considering will contain both curved edges and texture of various scales. Representing edges as a set of straight line fragments, and using these as our discrete features will be inappropriate, since curved lines and texture edges can be expected to fragment differently on each image of the sequence, and so be untrackable. Because of ill-conditioning, the use of parametrised curves (eg. circular arcs) cannot be expected to provide the solution, especially with real imagery.

现在，我们所考虑的无约束的图像，会包含弯曲的边缘和各种尺度的纹理。将边缘表示为线段的集合，并使用这些作为我们的离散特征，是不太合适的，因为曲线和纹理边缘在这些图像序列的每一幅中的表现不太一样，所以并不能进行追踪。因为这种病态的条件，参数化曲线的使用（如，圆弧）应当不能解决这个问题，尤其是在真实图像上。

Having found fault with the above solutions to the problem of 3D edge interpretation, we question the necessity of trying to solve the problem at all! Psychovisual experiments (the ambiguity of interpretation in viewing a rotating bent coat-hanger in silhouette), show that the problem of 3D interpretation of curved edges may indeed be effectively insoluble. This problem seldom occurs in reality because of the existence of small imperfections and markings on the edge which act as trackable feature-points.

在对3D边缘解释问题的解决方案上，我们找到了错误，然后我们就会对尝试解决这个问题的必要性进行质疑！心理视觉试验表明，曲线边缘3D解释的问题，可能真的是无法有效解决的。这个问题在现实中很少发生，是因为在边缘上存在小的不完美处和标记，这是可追踪的特征点。

Although an accurate, explicit 3D representation of a curving edge may be unobtainable, the connectivity it provides may be sufficient for many purposes - indeed the edge connectivity may be of more importance than explicit 3D measurements. Tracked edge connectivity, supplemented by 3D locations of corners and junctions, can provide both a wire-frame structural representation, and delimited image regions which can act as putative 3D surfaces.

虽然可能无法获得一个曲线边缘的准确、显式的3D表示，其给出的连接线可能对一些目的是足够的，确实，边缘的连接性比显式的3D测量要更加重要。跟踪边缘的连接性，附以角点和连接点的3D位置，可以提供结构表示和有限的图像区域，可以作为推定的3D表面。

This leaves us with the problem of performing reliable (ie. consistent) edge filtering. The state-of-the-art edge filters, such as, are not designed to cope with junctions and corners, and are reluctant to provide any edge connectivity. This is illustrated in Figure 2 for the Canny edge operator, where the above- and below-threshold edgels are represented respectively in black and grey. Note that in the bushes, some, but not all, of the edges are readily matchable by eye. After hysteresis has been undertaken, followed by the deletion of spurs and short edges, the application of a junction completion algorithm results in the edges and junctions shown in Figure 3, edges being shown in grey, and junctions in black. In the bushes, very few of the edges are now readily matched. The problem here is that of edges with responses close to the detection threshold: a small change in edge strength or in the pixellation causes a large change in the edge topology. The use of edges to describe the bush is suspect, and it is perhaps better to describe it in terms of feature-points alone.

这留给我们的问题成为了，进行可靠的（一致的）边缘滤波。目前最好的边缘滤波器并不是设计用于处理角点和连接点的，也不能提供任何边缘的连接性。这在图2中的Canny边缘算子进行了展示，其中上阈值和下阈值的edgel分别用黑色和灰色进行表示。注意在灌木中，边缘中的一部分（但不是所有）是可以通过人眼得到匹配的。在采用了滞后后，然后删除了spurs和短的边缘，使用连接点补全算法得到了图3中的边缘和连接点，边缘用灰色表示，连接点用黑色表示。在灌木中，很少有边缘是能匹配的上的。这里的问题是，接近于检测阈值的有响应的边缘：边缘强度或像素化很小的变化，就会导致边缘拓扑的很大变化。使用边缘来表示灌木是值得怀疑的，可能用特征点来表示更好一些。

The solution to this problem is to attempt to detect both edges and corners in the image: junctions would then consist of edges meeting at corners. To pursue this approach, we shall start from Moravec's corner detector.

这个问题的解决方案是，在图像中同时检测边缘和角点：连接点就由边缘与角点连接的地方组成。为研究这种方法，我们要从Moravec的角点检测器开始。

## 3. Moravec Revisited

Moravec's corner detector functions by considering a local window in the image, and determining the average changes of image intensity that result from shifting the window by a small amount in various directions. Three cases need to be considered:

Moravec角点检测器在图像中考虑一个局部窗口，将窗口在各种方向上偏移一个小量，确定图像灰度的平均变化。需要考虑三种情况：

A. If the windowed image patch is flat (i.e. approximately constant in intensity), then all shifts will result in only a small change; 如果图像块是平坦的（即，在灰度上几乎是常数），那么所有的偏移都只会得到很小的变化；

B. If the window straddles an edge, then a shift along the edge will result in a small change, but a shift perpendicular to the edge will result in a large change; 如果窗口横跨了一个边缘，那么沿着边缘的偏移会得到很小的变化，而垂直于边缘的偏移会得到很大的变化；

C. If the windowed patch is a corner or isolated point, then all shifts will result in a large change. A corner can thus be detected by finding when the minimum change produced by any of the shifts is large. 如果图像块是一个角点，或一个孤立的点，那么所有的偏移都会得到很大的变化。找到角点的方法，就可以通过寻找任意偏移得到的最小变化都很大的情况。

We now give a mathematical specification of the above. Denoting the image intensities by I, the change E produced by a shift (x,y) is given by: 我们现在给出上述的数学表达。用I表示图像，变化为E，偏移为(x,y)，那么：

$$E_{x,y} = \sum_{u,v} w_{u,v} |I_{x+u, y+v} - I_{u,v}|^2$$

where w specifies the image window: it is unity within a specified rectangular region, and zero elsewhere. The shifts, (x,y), that are considered comprise {(1,0), (1,1), (0,1), (-1,1)}. Thus Moravec's corner detector is simply this: look for local maxima in min{E} above some threshold value.

其中w指定了图像窗口：在指定的矩形区域中为1，其他地方为0。偏移(x,y)，是{(1,0), (1,1), (0,1), (-1,1)}。因此Moravec角点检测器就是：找到min{E}的局部极大值超过某一阈值的点。

## 4. Auto-Correlation Detector

The performance of Moravec's corner detector on a test image is shown in Figure 4a; for comparison are shown the results of the Beaudet and Kitchen & Rosenfeld operators (Figures 4b and 4c respectively). The Moravec operator suffers from a number of problems; these are listed below, together with appropriate corrective measures:

Moravec角点检测器在测试图像中的性能如图4a所示；我们还给出了Beaudet和Kitchen&Rosenfeld算子的结果，以进行比较（分别如图4b和4c）。Moravec算子有几个问题，如下所述，还有合适的修正措施：

1. The response is anisotropic because only a discrete set of shifts at every 45 degrees is considered - all possible small shifts can be covered by performing an analytic expansion about the shift origin:

响应是各向异性的，因为只考虑了偏移的离散集合，即每45度一个偏移；要覆盖所有可能的微小偏移，可以对偏移原点进行解析展开：

$$E_{x,y} = \sum_{u,v} w_{u,v} [I_{x+u, y+v} - I_{u,v}]^2 = \sum_{u,v} w_{u,v} [xX+yY+O(x^2,y^2)]^2$$

where the first gradients are approximated by 其中一阶梯度由下式进行近似

$$X = I⊗(-1, 0, 1) ≈ ∂I/∂x$$

$$Y = I⊗(-1, 0, 1)^T ≈ ∂I/∂y$$

Hence, for small shifts, E can be written 因此，对于小的偏移，E可以写为

$$E(x,y) = Ax^2 + 2Cxy +By^2$$

where

$$A = X^2⊗w, B=Y^2⊗w, C=(XY)⊗w$$

2. The response is noisy because the window is binary and rectangular - use a smooth circular window, for example a Gaussian: 响应含有噪声，这是因为窗口是二值和矩形的，我们使用了一个平滑的圆形窗口，比如一个高斯：

$$w_{u,v} = exp^{-(u^2+v^2)/2σ^2}$$

3. The operator responds too readily to edges because only the minimum of E is taken into account - reformulate the corner measure to make use of the variation of E with the direction of shift. 算子对边缘响应太好，因为只考虑了E的最小值 - 我们对角点度量重新用公式进行表达，使用了E对偏移方向的变化。

The change, E, for the small shift (x,y) can be concisely written as 对很小的偏移(x,y)得到的变化E，可以写为

$$E(x,y) = (x,y)M(x,y)^T$$

where the 2x2 symmetric matrix M is

$$M = [A,C; C,B]$$

Note that E is closely related to the local autocorrelation function, with M describing its shape at the origin (explicitly, the quadratic terms in the Taylor expansion). Let α, β be the eigenvalues of M. α and β will be proportional to the principal curvatures of the local auto-correlation function, and form a rotationally invariant description of M. As before, there are three cases to be considered:

注意E是局部自相关函数紧密相关，其中M描述了其原点处的形状（显式的，泰勒展开的平方项）。令α, β是M的特征值。α和β与局部自相关函数的主曲率成正比，形成了M的旋转不变描述。与之前一样，有三种情况要进行考虑：

A. If both curvatures are small, so that the local auto-correlation function is flat, then the windowed image region is of approximately constant intensity (ie. arbitrary shifts of the image patch cause little change in E); 如果两个曲率都很小，这样局部自相关函数很平坦，那么窗口图像区域近似于常数灰度（即，图像块的任意平移会导致E很小的变化）；

B. If one curvature is high and the other low, so that the local auto-correlation function is ridge shaped, then only shifts along the ridge (ie. along the edge) cause little change in E: this indicates an edge; 如果一个曲率很大，另一个很小，这样局部自相关函数是一个脊形的，那么只有沿着脊方向的偏移（即，沿着边缘）会导致E的变化很小：这指示了是一条边缘；

C. If both curvatures are high, so that the local auto-correlation function is sharply peaked, then shifts in any direction will increase E: this indicates a corner. 如果两个曲率都很大，那么局部自相关函数就是很高的陡峰，在任意方向的偏移都会使E增大：这就说明是一个角点。

Consider the graph of (α, β) space. An ideal edge will have α large and β zero (this will be a surface of translation), but in reality β will merely be small in comparison to α, due to noise, pixellation and intensity quantisation. A corner will be indicated by both α and β being large, and a flat image region by both α and β being small. Since an increase of image contrast by a factor of p will increase α and β proportionately by p^2, then if (α, β) is deemed to belong in an edge region, then so should (αp^2, βp^2), for positive values of p. Similar considerations apply to corners. Thus (α, β) space needs to be divided as shown by the heavy lines in Figure 5.

考虑(α, β)空间图。一个理想的边缘是α大，β小（这是一个平移的平面），但实际中β与α相比会很小，原因是噪声，像素化和灰度量化。角点的特征是α和β都很大，平坦的区域中α和β都很小。图像对比度增加p，那么α和β会成比例的增加p^2，那么如果(α, β)属于边缘区域，那么(αp^2, βp^2)也是，p值为正。类似的考虑也可用于角点。因此(α, β)空间可以如图5一样进行划分。

## 5. Corner/Edge Response Function

Not only do we need corner and edge classification regions, but also a measure of corner and edge quality or response. The size of the response will be used to select isolated corner pixels and to thin the edge pixels.

我们不止需要角点和边缘的分类区域，还需要角点和边缘质量或响应的度量。响应的大小会用于选择孤立的角点，以及使边缘像素细化。

Let us first consider the measure of corner response, R, which we require to be a function of α and β alone, on grounds of rotational invariance. It is attractive to use Tr(M) and Det(M) in the formulation, as this avoids the explicit eigenvalue decomposition of M, thus

我们首先考虑角点度量R，这需要是α和β的函数，并要表达旋转不变性。使用Tr(M)和Det(M)效果会很好，因为这避免了对M进行特征值分解，因此

$$Tr(M) = α + β = A + B$$

$$Det(M) = αβ = AB -C^2$$

Consider the following inspired formulation for the corner response 考虑下面的公式作为角点响应

$$R = Det - k Tr^2$$

Contours of constant R are shown by the fine lines in Figure 5. R is positive in the corner region, negative in the edge regions, and small in the flat region. Note that increasing the contrast (ie. moving radially away from the origin) in all cases increases the magnitude of the response. The flat region is specified by Tr falling below some selected threshold.

R为常数的轮廓，如图5中的细线所示。R在角点区域为正值，在边缘区域为负值，在平坦区域为很小的值。注意，增加对比度（即，从原点处很快移动离开）在所有情况下都是很快的增加响应的幅度。平坦区域是通过Tr在一些选择的阈值之下指定的。

A corner region pixel (ie. one with a positive response) is selected as a nominated corner pixel if its response is an 8-way local maximum: corners so detected in the test image are shown in Figure 4d. Similarly, edge region pixels are deemed to be edgels if their responses are both negative and local minima in either the x or y directions, according to whether the magnitude of the first gradient in the x or y direction respectively is the larger. This results in thin edges. The raw edge/corner classification is shown in Figure 6, with grey indicating corner regions, and white, the thinned edges.

角点区域像素（即，正响应的像素），如果其响应是一个8路局部极大值，那么就可以被选择为提名角点像素：这样检测得到的角点如图4d所示。类似的，边缘区域像素，如果其响应都是负的，其局部极小值要么是x方向，要么是y方向，就是edgel，根据其在x或y方向的一次梯度的幅度是更大。这可以得到很细的边缘。原始边缘/角点分类如图6所示，灰色的为角点区域，白色是细的边缘。

By applying low and high thresholds, edge hysteresis can be carried out, and this can enhance the continuity of edges. These classifications thus result in a 5-level image comprising: background, two corner classes and two edge classes. Further processing (similar to junction completion) will delete edge spurs and short isolated edges, and bridge short breaks in edges. This results in continuous thin edges that generally terminate in the corner regions. The edge terminators are then linked to the corner pixels residing within the corner regions, to form a connected edge-vertex graph, as shown in Figure 7. Note that many of the corners in the bush are unconnected to edges, as they reside in essentially textural regions. Although not readily apparent from the Figure, many of the corners and edges are directly matchable. Further work remains to be undertaken concerning the junction completion algorithm, which is currently quite rudimentary, and in the area of adaptive thresholding.

通过使用低阈值和高阈值，可以进行边缘滞后，这可以增强边缘的连续性。这些分类因此得到5级图像折中：背景，两个角点类别和两个边缘类别。进一步的处理（与连接点补全类似）会删除边缘spurs和短的孤立的边缘，将边缘中的短的断裂连接起来。这会得到连续的细边缘，在角点区域就终止了。边缘终止器然后与角点像素连接起来，这些角点像素在角点区域中，可以形成连接起来的边缘顶点图，如图7所示。注意，在灌木中的很多角点与边缘不是连接在一起的，因为它们处于纹理区域中。虽然在图中并不是特别的明显，很多角点和边缘都是可以直接匹配起来的。未来的工作将继续考虑连接点补全算法，目前还非常原始，还有自适应阈值的工作。