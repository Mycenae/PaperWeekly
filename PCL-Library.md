# 3D is here: Point Cloud Library (PCL)

## 0. Abstract

With the advent of new, low-cost 3D sensing hardware such as the Kinect, and continued efforts in advanced point cloud processing, 3D perception gains more and more importance in robotics, as well as other fields.

随着新的低成本的3D感知硬件的发展，如Kinect，以及在高级点云处理方面的持续努力，3D感知在机器人学以及其他领域中中得到了越来越多的重要性。

In this paper we present one of our most recent initiatives in the areas of point cloud perception: PCL (Point Cloud Library – http://pointclouds.org). PCL presents an advanced and extensive approach to the subject of 3D perception, and it’s meant to provide support for all the common 3D building blocks that applications need. The library contains state-of-the art algorithms for: filtering, feature estimation, surface reconstruction, registration, model fitting and segmentation. PCL is supported by an international community of robotics and perception researchers. We provide a brief walkthrough of PCL including its algorithmic capabilities and implementation strategies.

本文中，我们给出了在点云感知领域一个最新的方案：PCL点云库。PCL给出了3D感知的先进和大量的方法，要为所有应用需要的通用3D模块提供支持。这个库包括目前最好的算法：滤波，特征估计，平面重建，配准，模型拟合和分割。PCL由国际机器人和感知研究者团体支持。我们简单介绍了PCL，包括其算法能力和实现策略。

## 1. Introduction

For robots to work in unstructured environments, they need to be able to perceive the world. Over the past 20 years, we’ve come a long way, from simple range sensors based on sonar or IR providing a few bytes of information about the world, to ubiquitous cameras to laser scanners. In the past few years, sensors like the Velodyne spinning LIDAR used in the DARPA Urban Challenge and the tilting laser scanner used on the PR2 have given us high-quality 3D representations of the world - point clouds. Unfortunately, these systems are expensive, costing thousands or tens of thousands of dollars, and therefore out of the reach of many robotics projects.

对于在无结构环境中工作的机器人，他们需要感知世界。在过去20年中，我们走过了很长的路，从基于声纳或IR的简单的距离探测器，可以给出关于这个世界的几个bytes的信息，到无处不在的相机，到激光扫描仪。在过去几年中，传感器如在DARPA都市挑战赛中使用的Velodyne spinning LIDAR，和在PR2中使用的tilting激光扫描仪，都给我们提供了关于这个世界的高质量的3D表示，点云。不幸的是，这些系统都很昂贵，耗资几千上万美元，因此很多机器人工程无法使用。

Very recently, however, 3D sensors have become available that change the game. For example, the Kinect sensor for the Microsoft XBox 360 game system, based on underlying technology from PrimeSense, can be purchased for under 150 dollars, and provides real time point clouds as well as 2D images. As a result, we can expect that most robots in the future will be able to ”see” the world in 3D. All that’s needed is a mechanism for handling point clouds efficiently, and that’s where the open source Point Cloud Library, PCL, comes in. Figure 1 presents the logo of the project.

但是最近，3D传感器已经变得可用了，这就改变了游戏规则。比如，微软XBox 360游戏系统的Kinect传感器，基于PrimeSense的潜在技术，可用150美元的价格购买，可以给出实时的点云以及2D图像。结果是，我们可以期待多数机器人在未来都可以看到3D的世界。所有需要的，就是一种可以高效处理点云的机制，这就是为什么开源的PCL库会出现。图1给出了工程的logo。

PCL is a comprehensive free, BSD licensed, library for n-D Point Clouds and 3D geometry processing. PCL is fully integrated with ROS, the Robot Operating System (see http://ros.org), and has been already used in a variety of projects in the robotics community.

PCL是一个综合的n-D点云和3D几何处理库。PCL与ROS完全集成，已经在机器人团体中用于很多工程。

## 2. Architecture and Implementation

PCL is a fully templated, modern C++ library for 3D point cloud processing. Written with efficiency and performance in mind on modern CPUs, the underlying data structures in PCL make use of SSE optimizations heavily. Most mathematical operations are implemented with and based on Eigen, an open-source template library for linear algebra [1]. In addition, PCL provides support for OpenMP (see http://openmp.org) and Intel Threading Building Blocks (TBB) library [2] for multi-core parallelization. The backbone for fast k-nearest neighbor search operations is provided by FLANN (Fast Library for Approximate Nearest Neighbors) [3]. All the modules and algorithms in PCL pass data around using Boost shared pointers (see Figure 2), thus avoiding the need to re-copy data that is already present in the system. As of version 0.6, PCL has been ported to Windows, MacOS, and Linux, and Android ports are in the works.

PCL是一个完全模版化的现代C++库，进行3D点云处理。开发时就考虑了在现代CPUs上的效率和性能，PCL中潜在的数据结构都大量使用了SSE优化。多数数学运算都是使用或基于Eigen实现的，这是一个线性代数开源的模板库。另外，PCL为OpenMP和Intel Threading Building Blocks (TBB)库提供了支持，以进行多核并行。快速k近邻搜索运算的支柱，是由FLANN提供的。PCL中的所有模块和算法使用Boost的共享指针来传递数据（见图2），因此避免了重新拷贝数据的需要，因为数据本来就是在系统中存在的。作为0.6版本，PCL已经移植到了Windows, MacOS, and Linux，Android也在进行中。

From an algorithmic perspective, PCL is meant to incorporate a multitude of 3D processing algorithms that operate on point cloud data, including: filtering, feature estimation, surface reconstruction, model fitting, segmentation, registration, etc. Each set of algorithms is defined via base classes that attempt to integrate all the common functionality used throughout the entire pipeline, thus keeping the implementations of the actual algorithms compact and clean. The basic interface for such a processing pipeline in PCL is:

从算法的角度，PCL应当包含一些3D处理算法，对点云数据进行处理，包括：滤波，特征估计，表面重建，模型拟合，分割，配准等等。每组算法是通过基准类定义的，试图在整个流程中集成所有常见的功能，因此保持实际算法的实现紧凑干净。在PCL中这种处理流程的基本界面是：

- create the processing object (e.g., filter, feature estimator, segmentation); 创建处理目标（如，滤波，特征估计器，分割）等；

- use setInputCloud to pass the input point cloud dataset to the processing module; 使用setInputCloud来将输入点云数据集传入到处理模块中；

- set some parameters; 设置一些参数；

- call compute (or filter, segment, etc) to get the output. 调用compute（或filter，segment等）来得到输出。

The sequence of pseudo-code presented in Figure 2 shows a standard feature estimation process in two steps, where a NormalEstimation object is first created and passed an input dataset, and the results together with the original input are then passed together to an FPFH [4] estimation object.

图2中的伪码序列展示了一个标准的特征估计过程的两个步骤，其中NormalEstimation目标首先创建，传入一个输入数据集，结果与原始输入一起送入一个FPFH估计目标中。

To further simplify development, PCL is split into a series of smaller code libraries, that can be compiled separately: 为进一步简化开发，PCL分割成一些更小的代码库，可以分别编译：

- libpcl_filters: implements data filters such as downsampling, outlier removal, indices extraction, projections, etc; 实现了数据滤波器，如下采样，离群点去除，索引提取，投影等；

- libpcl_features: implements many 3D features such as surface normals and curvatures, boundary point estimation, moment invariants, principal curvatures, PFH and FPFH descriptors, spin images, integral images, NARF descriptors, RIFT, RSD, VFH, SIFT on intensity data, etc; 实现很多3D特征，如表面法向量和曲率，边缘点估计，动量不变量，主曲率，PFH和FPFH描述子，spin图像，积分图像，NARF描述子，在亮度数据上的RIFT，RSD，VFH，SIFT等；

- libpcl_io: implements I/O operations such as writing to/reading from PCD (Point Cloud Data) files; 实现I/O操作，如PCD文件的读写；

- libpcl_segmentation: implements cluster extraction, model fitting via sample consensus methods for a variety of parametric models (planes, cylinders, spheres, lines, etc), polygonal prism extraction, etc; 实现了群提取，通过样本一致性方法对很多参数化模型（平面，圆柱，球体，线段等）实现模型拟合，多边形棱镜提取，等；

- libpcl_surface: implements surface reconstruction techniques, meshing, convex hulls, Moving Least Squares, etc; 实现平面重建的技术，网格化，凸包，移动最小二乘等；

- libpcl_registration: implements point cloud registration methods such as ICP, etc; 实现点云配准方法，如ICP等；

- libpcl_keypoints: implements different keypoint extraction methods, that can be used as a preprocessing step to decide where to extract feature descriptors; 实现不同的关键点提取方法，可以用于预处理步骤来确定哪里来提取特征描述子；

- libpcl_range_image: implements support for range images created from point cloud datasets. 实现对从点云数据中创建的距离图像的支持。

To ensure the correctness of operations in PCL, the methods and classes in each of the above mentioned libraries contain unit and regression tests. The suite of unit tests is compiled on demand and verified frequently by a dedicated build farm, and the respective authors of a specific component are being informed immediately when that component fails to test. This ensures that any changes in the code are tested throughly and any new functionality or modification will not break already existing code that depends on PCL.

为确保PCL中运算的正确性，每个上述库中的方法和类都进行了单元测试和回归测试。单元测试是按需编译的，并频繁由一个细致的build farm来验证，当组件测试失败时，具体组件的各自的作者都会立刻得到通知。这确保了，代码中的任何修改都进行了彻底的测试，任何新的功能或修改都不会破坏依赖于PCL的已有代码。

In addition, a large number of examples and tutorials are available either as C++ source files, or as step-by-step instructions on the PCL wiki web pages. 此外，很多例子和教程都是可用的，要么是C++源代码，要么是在PCL维基页面上的一步一步的指引。

## 3. PCL and ROS

One of the corner stones in the PCL design philosophy is represented by Perception Processing Graphs (PPG). The rationality behind PPGs are that most applications that deal with point cloud processing can be formulated as a concrete set of building blocks that are parameterized to achieve different results. For example, there is no algorithmic difference between a wall detection algorithm, or a door detection, or a table detection – all of them share the same building block, which is in this case, a constrained planar segmentation algorithm. What changes in the above mentioned cases is a subset of the parameters used to run the algorithm.

PCL设计哲学中的一个基石，是感知处理图(PPG, Perception Processing Graphs)。PPGs之后的合理性是，多数处理点云数据的应用，可以表述为一些模块的集合，可以进行参数化的表达，以获得不同的结果。比如，墙的检测算法，门的检测算法，或桌子的检测之间，没有任何算法区别，所有都是一样的模块，在这种情况下，是一种受约束的平面分割算法。在上述提到的案例中，变化的部分是一些参数，用于运行这个算法。

With this in mind, and based on the previous experience of designing other 3D processing libraries, and most recently, ROS, we decided to make each algorithm from PCL available as a standalone building block, that can be easily connected with other blocks, thus creating processing graphs, in the same way that nodes connect together in a ROS ecosystem. Furthermore, because point clouds are extremely large in nature, we wanted to guarantee that there would be no unnecessary data copying or serialization/deserialization for critical applications that can afford to run in the same process. For this we created nodelets, which are dynamically loadable plugins that look and operate like ROS nodes, but in a single process (as single or multiple threads).

以这个为原则，并基于之前在设计其他3D处理库中的经验，以及在ROS上的经验，我们决定使PCL中的每个算法都作为一个单独的模块可用，这可以很容易的与其他模块连接起来，因此创建了处理图，与ROS生态系统中连接在一起的节点一样。而且，由于点云本身非常大，我们希望确保对于关键的应用，没有不必要的数据拷贝或序列化/反序列化。这样，我们创建了nodelets，这是可以动态加载的插件，与ROS节点外观和操作都类似，但是在单独的进程中（作为单线程或多线程）。

A concrete nodelet PPG example for the problem of identifying a set of point clusters supported by horizontal planar areas is shown in Figure 3. 一个具体的nodelet PPG例子，处理的是识别水平平面区域的点云集，如图3所示。

## 4. Visualization

PCL comes with its own visualization library, based on VTK [5]. VTK offers great multi-platform support for rendering 3D point cloud and surface data, including visualization support for tensors, texturing, and volumetric methods.

PCL有其自己的可视化库，基于VTK。VTK有很好的多平台支持，可以进行3D点云数据和表面数据的渲染，包括张量、纹理和体方法的可视化支持。

The PCL Visualization library is meant to integrate PCL with VTK, by providing a comprehensive visualization layer for n-D point cloud structures. Its purpose is to be able to quickly prototype and visualize the results of algorithms operating on such hyper-dimensional data. As of version 0.2, the visualization library offers:

PCL可视化库是将PCL与VTK整合到了一起，对n-D点云结构提供了很多可视化层。其目的是可以迅速的构建原型，对进行高维数据的算法结果进行可视化。在0.2版本中，可视化库给出了：

- methods for rendering and setting visual properties (colors, point sizes, opacity, etc) for any n-D point cloud dataset; 对任意n-D点云数据集渲染和设置可视化性质的方法（颜色，点的大小，透明度等）；

- methods for drawing basic 3D shapes on screen (e.g., cylinders, spheres, lines, polygons, etc) either from sets of points or from parametric equations; 基本3D形状在屏幕上的绘制方法（如，圆柱，球体，线段，多边形等），要么是点集，要么是参数化的公式；

- a histogram visualization module (PCLHistogramVisualizer) for 2D plots; 2D绘图的直方图可视化模块；

- a multitude of geometry and color handlers. Here, the user can specify what dimensions are to be used for the point positions in a 3D Cartesian space (see Figure 4), or what colors should be used to render the points (see Figure 5); 多个几何和颜色处理工具。这里，用户可以指定用什么维度来对点的位置在3D笛卡尔空间中（见图4），或应当用什么颜色来渲染这些点（见图5）；

- RangeImage visualization modules (see Figure 6). 距离图像可视化模块（见图6）。

The handler interactors are modules that describe how colors and the 3D geometry at each point in space are computed, displayed on screen, and how the user interacts with the data. They are designed with simplicity in mind, and are easily extendable. A code snippet that produces results similar to the ones shown in Figure 4 is presented in Algorithm 1.

Handler interactors是描述在空间中每个点上颜色和3D几何是怎样计算的，怎样展示在屏幕上的，以及用户与数据的交互。它们的设计要尽量简洁，还要容易拓展。与图4中功能类似的代码片段，如算法1所示。

The library also offers a few general purpose tools for visualizing PCD files, as well as for visualizing streams of data from a sensor in real-time in ROS. 库也有一些通用目标工具，对PCD文件进行可视化，以及对从ROS传感器输出的实时流数据进行可视化。

## 5. Usage examples

In this section we present two code snippets that exhibit the flexibility and simplicity of using PCL for filtering and segmentation operations, followed by three application examples that make use of PCL for solving the perception problem: i) navigation and mapping, ii) object recognition, and iii) manipulation and grasping.

本节中，我们给出两个代码片段，展现了使用PCL进行滤波和分割运算的灵活性和简洁性，随后是三个应用案例，使用PCL解决感知问题：i)导航和地图，ii)目标识别，iii)操作和控制。

Filtering constitutes one of the most important operations that any raw point cloud dataset usually goes through, before any higher level operations are applied to it. Algorithm 2 and Figure 7 present a code snippet and the results obtained after running it on the point cloud dataset from the left part of the figure. The filter is based on estimating a set of statistics for the points in a given neighborhood (k = 50 here), and using them to select all points within 1.0·σ distance from the mean distance μ, as inliers (see [6] for more information).

滤波是任何原始点云数据集通常都会经历的一种最重要运算，然后才是一些更高层的运算应用到其中。算法2和图7给出了一个代码片段和在图左的点云数据上运行之后得到的结果。这个滤波器是对在给定邻域中的点估计统计值的集合(k=50)，使用其来选择从平均距离μ距离1.0·σ之内的所有点，作为在群点（更多信息如[6]）。

The second example constitutes a segmentation operation for planar surfaces, using a RANSAC [7] model, as shown in Algorithm 3. The input and output results are shown in Figure 8. In this example, we are using a robust RANSAC estimator to randomly select 3 non-collinear points and calculate the best possible model in terms of the overall number of inliers. The inlier thresholding criterion is set to a maximum distance of 1cm of each point to the plane model.

第二个例子是，对于平面表面的分割操作，使用的是RANSAC模型，如算法3所示。输入和输出结果如图8所示。在这个例子中，我们使用一种稳健的RANSAC估计器来随机选择3个非共线的点，计算以总计的在群点计算可能最好的模型。在群阈值准则设置为每个点到平面模型最大1cm。

An example of a more complex navigation and mapping application is shown in the left part of Figure 9, where the PR2 robot had to autonomously identify doors and their handles [8], in order to explore rooms and find power sockets [9]. Here, the modules used included constrained planar segmentation, region growing methods, convex hull estimation, and polygonal prism extractions. The results of these methods were then used to extract certain statistics about the shape and size of the door and the handle, in order to uniquely identify them and to reject false positives.

一个更复杂的导航和地图应用的例子，如图9左所示，其中PR2机器人要自动识别出门和把手，以探索房间，找到能量槽。这里，使用的模块包含，受约束的平面分割，区域增长方法，凸包估计，和多边形棱镜提取。这些方法的结果然后用于，对门和把手的形状和大小，提取特定的统计值，以进行唯一的识别，并拒绝掉那些假阳性。

The right part of Figure 9 shows an experiment with real-time object identification from complex 3D scenes [10]. Here, a set of complex 3D keypoints and feature descriptors are used in a segmentation and registration framework, that aims to identify previously seen objects in the world.

图9的右边展示了，从复杂3D场景中进行实时目标识别的试验。这里，一系列复杂的3D关键点和特征描述子用于分割和配准框架，以识别之前看到的世界中的目标。

Figure 10 presents a grasping and manipulation application [11], where objects are first segmented from horizontal planar tables, clustered into individual units, and a registration operation is applied that attaches semantic information to each cluster found.

图10给出了一个抓去和操作的应用，其中目标首先从水平平面桌子上分割出来，聚类成单个的单元，然后应用配准运算，以将语义信息与找到的每个聚类联系起来。

## 6. Community and Future Plans

PCL is a large collaborative effort, and it would not exist without the contributions of several people. Though the community is larger, and we accept patches and improvements from many users, we would like to acknowledge the following institutions for their core contributions to the development of the library: AIST, UC Berkeley, University of Bonn, University of British Columbia, ETH Zurich, University of Freiburg, Intel Reseach Seattle, LAAS/CNRS, MIT, University of Osnabruck, Stanford University, University of Tokyo, TUM, Vienna University of Technolog, and Washington University in St. Louis.

Our current plan for PCL is to improve the documentation, unit tests, and tutorials and release a 1.0 version. We will continue to add functionality and make the system available on other platforms such as Android, and we plan to add support for GPUs using CUDA and OpenCL.

We welcome any new contributors to the project, and we hope to emphasize the importance of code sharing for 3D processing, which is becoming crucial for advancing the robotics field.