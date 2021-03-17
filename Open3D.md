# Open3D: A Modern Library for 3D Data Processing

Qian-Yi Zhou, Jaesik Park, Vladlen Koltun, Intel Labs

## 0. Abstract

Open3D is an open-source library that supports rapid development of software that deals with 3D data. The Open3D frontend exposes a set of carefully selected data structures and algorithms in both C++ and Python. The backend is highly optimized and is set up for parallelization. Open3D was developed from a clean slate with a small and carefully considered set of dependencies. It can be set up on different platforms and compiled from source with minimal effort. The code is clean, consistently styled, and maintained via a clear code review mechanism. Open3D has been used in a number of published research projects and is actively deployed in the cloud. We welcome contributions from the open-source community.

Open3D是快速开发处理3D数据软件的一个开源库。Open3D的前端以C++和Python给出了仔细选择的数据结构和算法集合，后端是高度优化的，是并行设计的。Open3D的开发基础是非常干净的，依赖很少，并都经过仔细选择。可以在不同平台上设置，从源进行编译的工作很少。代码很干净，风格一致，通过一种清晰代码分析机制进行维护。Open3D已经在一些发表的研究项目中使用，在云上部署活跃。我们欢迎开源团体的贡献。

## 1. Introduction

The world is three-dimensional. Systems that operate in the physical world or deal with its simulation must often process three-dimensional data. Such data takes the form of point clouds, meshes, and other representations. Data in this form is produced by sensors such as LiDAR and depth cameras, and by software systems that support 3D reconstruction and modeling.

世界是三维的。在物理世界中运行的系统，或进行其仿真的，通常必须处理三维数据。这样的数据的形式是点云，网格和其他表示。这种形式的数据是通过传感器产生的，如LiDAR和深度相机，以及支持3D重建和建模的软件系统。

Despite the central role of 3D data in fields such as robotics and computer graphics, writing software that processes such data is quite laborious in comparison to other data types. For example, an image can be efficiently loaded and visualized with a few lines of OpenCV code [3]. A similarly easy to use software framework for 3D data has not emerged. A notable prior effort is the Point Cloud Library (PCL) [18]. Unfortunately, after an initial influx of open-source contributions, PCL became encumbered by bloat and is now largely dormant. Other open-source efforts include MeshLab [6], which provides a graphical user interface for processing meshes; libigl [11], which supports discrete differential geometry and related research; and a variety of solutions for image-based reconstruction [13]. Nevertheless, there is currently no open-source library that is fast, easy to use, supports common 3D data processing workflows, and is developed in accordance with modern software engineering practices.

尽管3D数据在机器人和计算机图形学处于中心地位，与其他数据类型相比，编写软件处理这种数据是非常复杂的。比如，用几行OpenCV代码，图像可以很高效的加载并可视化。对3D数据来说，同样易用的软件还没有出现。值得注意的一个工作是PCL。不幸的是，在初始的开源贡献后，PCL已经比较膨胀，目前基本上没有什么维护。其他开源的工作包括MeshLab，这是处理网格的图形界面工具；libigl支持离散微分几何和相关的研究；还有很多基于图像的重建工作。尽管如此，快速易用，支持常见的3D数据处理流程，根据现代软件工程实践开发的开源库目前并不存在。

Open3D was created to address this need. It is an open-source library that supports rapid development of software that deals with 3D data. The Open3D frontend exposes a set of carefully selected data structures and algorithms in both C++ and Python. The Open3D backend is implemented in C++11, is highly optimized, and is set up for OpenMP parallelization. Open3D was developed from a clean slate with a small and carefully considered set of dependencies. It can be set up on different platforms and compiled from source with minimal effort. The code is clean, consistently styled, and maintained via a clear code review mechanism.

Open3D就是满足这种需求开发的。这是一个开源库，支持快速开发3D数据处理的软件。Open3D前端是仔细选择的数据结构和算法，包括C++和Python实现。Open3D的后端是用C++11实现的，高度优化，为OpenMP并行进行过设置。Open3D的开发非常干净，依赖很少，是经过仔细选择的。可以在不同的平台上设置，从源进行编译的工作很少。代码很干净，风格一致，是经过清晰的代码审查机制维护的。

Open3D has been in development since 2015 and has been used in a number of published research projects [22, 13, 15, 12]. It has been deployed and is currently running in the Tanks and Temples evaluation server [13]. Open3D is released open-source under the permissive MIT license and is available at http://www.open3d.org.

Open3D自从2015年开发，在很多已经发表的研究项目中进行了使用。目前在Tanks and Temples评估服务器中部署运行。Open3D已经开源。

We welcome contributions from the open-source community. 我们欢迎开源团体的贡献。

## 2. Design

Two primary design principles of Open3D are usefulness and ease of use [8]. Our key decisions can be traced to these two principles. Usefulness motivated support for popular representations, algorithms, and platforms. Ease of use served as a countervailing force that guarded against heavy-weight dependencies and feature creep.

Open3D的两个基础设计原则是有用性和易用性。我们的主要决策都可以追溯到这两个原则。有用性得到了流行的表示、算法和平台。易用性则是一种抵消的力量，避免非常多的依赖项，和特征蠕动。

Open3D provides data structures for three kinds of representations: point clouds, meshes, and RGB-D images. For each representation, we have implemented a complete set of basic processing algorithms such as I/O, sampling, visualization, and data conversion. In addition, we have implemented a collection of widely used algorithms, such as normal estimation [16], ICP registration [2, 4], and volumetric integration [7]. We have verified that the functionality of Open3D is sufficient by using it to implement complete workflows such as large-scale scene reconstruction [5, 15].

Open3D为三种表示提供了数据结构：点云，网格和RGBD图像。对每种表示，我们都实现了基础处理算法的完全几何，比如I/O，采样，可视化，数据转换。除此以外，我们还实现了很多广泛使用的算法，比如法线估计[16]，ICP配准[2,4]和体集成[7]。我们已经验证了Open3D的功能是足够的，使用其实现了完全的工作流，比如大规模场景重建[5,15]。

We carefully chose a small set of lightweight dependencies, including Eigen for linear algebra [9], GLFW for OpenGL window support, and FLANN for fast nearest neighbor search [14]. For easy compilation, powerful but heavyweight libraries such as Boost and Ceres are excluded. Instead we use either lightweight alternatives (e.g., pybind11 instead of Boost.Python) or in-house implementations (e.g., for Gauss-Newton and Levenberg-Marquardt graph optimization). The source code of all dependencies is distributed as part of Open3D. The dependencies can thus be compiled from source if not automatically detected by the configuration script. This is particularly useful for compilation on operating systems that lack package management software, such as Microsoft Windows.

我们仔细选择了很少依赖项集合，包括Eigen进行线性代数计算，GLFW支持OpenGL，FLANN进行快速最近邻搜索。为编译容易，我们并没有包括很强但是很重的库，比如Boost和Ceres。我们使用了轻量级的替代（如，用pybind11而没有使用Boost.Python）或内建的实现（如，对Gauss-Newton和Levenberg-Marquardt图优化）。所有依赖的源代码是Open3D的一部分。如果没有从配置脚本中自动检测到，这些依赖因此可以从源进行编译。这在缺少包管理软件的操作系统中的编译尤其有用，比如Windows。

The development of Open3D started from a clean slate and the library is kept as simple as possible. Only algorithms that solve a problem of broad interest are added. If a problem has multiple solutions, we choose one that the community considers standard. A new algorithm is added only if its implementation demonstrates significantly stronger results on a well-known benchmark.

Open3D的开发是从很干净的底板开始的，库保持尽可能的简单。只有解决很广泛的问题的算法加入到了库中。如果一个问题有多个解，我们选择团体认为是标准的那个。只有新算法的实现证明了在著名的benchmark中得到更强的结果，才被加入到库中。

Open3D is written in standard C++11 and uses CMake to support common C++ toolchains including 是用标准C++11写的，用CMake来支持通用的C++工具链，包括

- GCC 4.8 and later on Linux
- XCode 8.0 and later on OS X
- Visual Studio 2015 and later on Windows

A key feature of Open3D is ubiquitous Python binding. We follow the practice of successful computer vision and deep learning libraries [1, 3]: the backend is implemented in C++ and is exposed through frontend interfaces in Python. Developers use Python as a glue language to assemble components implemented in the backend.

Open3D的一个关键特征是全部都有Python binding。我们按照成功的计算机视觉和深度学习库的实践：后端是用C++实现的，前端用Python给出接口。使用Python作为glue语言的组装的部件都是进行的后端实现。

Figure 1 shows code snippets for a simple 3D data processing workflow. An implementation using the Open3D Python interface is compared to an implementation using the Open3D C++ interface and to an implementation based on PCL [18]. The implementation using the Open3D Python interface is approximately half the length of the implementation using the Open3D C++ interface, and about five times shorter than the implementation based on PCL. As an added benefit, the Python code can be edited and debugged interactively in a Jupyter Notebook.

图1给出了简单的3D数据处理流程的代码片段。Open3D Python接口的实现，与Open3D C++接口的实现，和PCL的实现进行了比较。使用Open3D Python接口的实现，是Open3D C++接口的实现的长度的大约一半，是PCL实现的1/5。而且，Python代码可以用Jupyter Notebook进行编辑和调试。

## 3. Functionality

Open3D has nine modules, listed in Table 1.

Module | Functionality
--- | ---
Geometry | Data structures and basic processing algorithms
Camera | Camera model and camera trajectory
Odometry |  Tracking and alignment of RGB-D images
Registration | Global and local registration
Integration | Volumetric integration
I/O | Reading and writing 3D data files
Visualization | A customizable GUI for rendering 3D data with OpenGL
Utility | Helper functions such as console output, file system, and Eigen wrappers
Python | Open3D Python binding and tutorials

### 3.1. Data

The Geometry module implements three geometric representations: PointCloud, TriangleMesh, and Image.

Geometry模块实现了三种几何表示：点云PointCloud，网格TriangleMesh和图像Image。

The PointCloud data structure has three data fields: PointCloud.points, PointCloud.normals, PointCloud.colors. They are used to store coordinates, normals, and colors. The master data field is PointCloud.points. The other two fields are considered valid only when they have the same number of records as PointCloud.points. Open3D provides direct memory access to these data fields via a numpy array. The following code sample demonstrates reading and accessing the point coordinates of a point cloud.

点云PointCloud数据结构有三个数据字段：PointCloud.points, PointCloud.normals, PointCloud.colors，用于存储坐标，法线和色彩。主数据字段是PointCloud.points。另外两个只有当与PointCloud.points有相同数量的记录时，才会认为是有效的。Open3D通过numpy阵列提供到这些数据字段的直接内存访问。下面的代码例子展示了点云点坐标的读取和访问。

```
from py3d import *
import numpy as np
pointcloud = read_point_cloud(’pointcloud.ply’)
print(np.asarray(pointcloud.points))
```

Similarly, TriangleMesh has two master data fields – TriangleMesh.vertices and TriangleMesh.triangles – as well as three auxiliary data fields: TriangleMesh.vertex_normals, TriangleMesh.vertex_colors, and TriangleMesh.triangle_normals.

类似的，TriangleMesh有两个主数据字段 - TriangleMesh.vertices和TriangleMesh.triangles，以及三个辅助数据字段：TriangleMesh.vertex_normals, TriangleMesh.vertex_colors, and TriangleMesh.triangle_normals。

The Image data structure is implemented as a 2D or 3D array and can be directly converted to a numpy array. A pair of depth and color images of the same resolution can be combined into a data structure named RGBDImage. Since there is no standard depth image format, we have implemented depth image support for multiple datasets including NYU [19], TUM [20], SUN3D [21], and Redwood [5]. The following code sample reads a pair of RGB-D images from the TUM dataset and converts them to a point cloud.

Image数据结构是用2D或3D阵列实现的，可以直接转换成numpy阵列。一对分辨率相同的深度图像和彩色图像，可以结合成为一个名为RGBDImage的数据结构。由于没有标准的深度图像格式，我们实现了深度图像支持多个数据及，包括NYU, TMU, SUN3D, 和Redwood。下面的代码例子从TUM数据集中读取一对RGB-D图像，将其转换成点云。

```
from py3d import *
import numpy as np
depth = read_image('TUM_depth.png')
color = read_image('TUM_color.jpg')
rgbd = create_rgbd_image_from_tum_format(color, depth)
pointcloud = create_point_cloud_from_rgbd_image(rgbd, PinholeCameraIntrinsic.prime_sense_default)
```

### 3.2. Visualization

Open3D provides a function draw_geometries() for visualization. It takes a list of geometries as input, creates a window, and renders them simultaneously using OpenGL. We have implemented many functions in the visualizer, such as rotation, translation, and scaling via mouse operations, changing rendering style, and screen capture. A code sample using draw_geometries() and its result are shown in Figure 2.

Open3D给出了一个函数draw_geometries()进行可视化。它以一些几何形状列表作为输入，创建一个窗口，使用OpenGL对其进行渲染。我们在可视化器中实现了很多函数，比如旋转，平移和通过鼠标操作的缩放，改变渲染类型，捕获屏幕。使用draw_geometries()的代码示例及其结果如图2所示。

```
from py3d import *
pointcloud = read_point_cloud('pointcloud.pcd')
mesh = read_triangle_mesh('mesh.ply')
mesh.computer_vertex_normals()
draw_geometries([pointcloud, mesh])
```

In addition to draw_geometries(), Open3D has a number of sibling functions with more advanced functionality. draw_geometries_with_custom_animation() allows the programmer to define a custom view trajectory and play an animation in the GUI. draw_geometries_with_animation_callback() and draw_geometries_with_key_callback() accept Python callback functions as input. The callback function is called in an automatic animation loop, or upon a key press event. The following code sample shows a rotating point cloud using draw_geometries_with_animation_callback().

除了draw_geometries()，Open3D还有一些函数，其功能更加高级。draw_geometries_with_custom_animation()使程序员可以定义一个定制的视角轨迹，在GUI中播放一个动画。draw_geometries_with_animation_callback()和draw_geometries_with_key_callback()接受Python回调函数作为输入。回调函数在一个自动动画循环中会调用，或在一个按键事件中调用。下面的代码片段展示了一个旋转的点云，使用的是draw_geometries_with_animation_callback()。

```
from py3d import *
pointcloud = read_point_cloud('pointcloud.pcd')
def rotate_view(vis):
  # Rotate the view by 10 degrees
  ctr = vis.get_view_control()
  ctr.rotate(10.0, 0.0)
  return False
draw_geometries_with_animation_callback([pointcloud], rotate_view)
```

In the backend of Open3D, these functions are implemented using the Visualizer class. This class is also exposed in the Python interface. 在Open3D的后端中，这些函数是使用Visualizer类别进行实现的。这个类也有Python接口。

### 3.3. Registration

Open3D provides implementations of multiple state-of-the-art surface registration methods, including pairwise global registration, pairwise local refinement, and multiway registration using pose graph optimization. This section gives an example of a complete pairwise registration workflow for point clouds. The workflow begins by reading raw point clouds, downsampling them, and estimating normals:

Open3D给出了多个目前最好的面配准算法的实现，包括成对全局配准，成对局部精调，和使用姿态图优化的多路配准。本节给出了点云成对配准的完整工作流程例子。工作流的开始是读取原始点云，降采样并估计法线：

```
from py3d import *
source = read_point_cloud('source.pcd')
target = read_point_cloud('target.pcd')
source_down = voxel_down_sample(source, 0.05)
target_down = voxel_down_sample(target, 0.05)
estimate_normals(source_down, KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
estimate_normals(target_down, KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
```

We then compute FPFH features and apply a RANSAC-based global registration algorithm [17]: 然后我们计算FPFH特征，使用了一个基于RANSAC的全局配准算法[17]：

```
source_fpfh = compute_fpfh_feature(source_down, KDTreeSearchParamHybrid(radius = 0.25, max_nn = 100))
target_fpfh = compute_fpfh_feature(target_down, KDTreeSearchParamHybrid(radius = 0.25, max_nn = 100))
result_ransac = registration_ransac_based_on_feature_matching(source_down, target_down, source_fpfh, target_fpfh, max_correspondence_distance = 0.075, TransformationEstimationPointToPoint(False), ransac_n = 4, [CorrespondenceCheckerBasedOnEdgeLength(0.9), CorrespondenceCheckerBasedOnDistance(0.075)], RANSACConvergenceCriteria(max_iteration = 4000000, max_validation = 500))
```

We used profiling tools to analyze the RANSAC-based algorithm and found that its most time-consuming part is the validation of matching results. Thus, we give the user an option to specify a termination criterion via the RANSAC-ConvergenceCriteria parameter. In addition, we provide a set of functions that prune false matches early, including CorrespondenceCheckerBasedOnEdgeLength and CorrespondenceCheckerBasedOnDistance.

我们使用性能分析工具来分析基于RANSAC的算法，发现其最耗时的部分是匹配结果的验证。因此，我们给了用户一个选项，通过RANSAC-ConvergenceCriteria参数来指定一个结束规则。另外，我们给出了一个函数集合，很早就可以将错误的匹配削减掉，包括CorrespondenceCheckerBasedOnEdgeLength和CorrespondenceCheckerBasedOnDistance。

The final part of the pairwise registration workflow is ICP refinement [2, 4], applied to the original dense pointclouds: 最终的成对匹配算法工作流是ICP精调的额，对原始的密集点云使用：

```
result_icp = registration_icp(source, target, max_correspondence_distance = 0.02, result_ransac.transformation, TransformationEstimationPointToPlane())
```

Here TransformationEstimationPointToPlane() invokes a point-to-plane ICP algorithm. Other ICP variants are implemented as well. Intermediate and final results of the demonstrated registration procedure are shown in Figure 3.

这里TransformationEstimationPointToPlane()调用了一个点对平面的ICP算法。也实现了其他的ICP变体。展示的配准过程的中间和最终结果，如图3所示。

### 3.4. Reconstruction

A sophisticated workflow that is demonstrated in an Open3D tutorial is a complete scene reconstruction system [5, 15]. The system is implemented as a Python script that uses many algorithms implemented in Open3D. It takes an RGB-D sequence as input and proceeds through three major steps.

在Open3D的教程中展示的一个复杂工作流是，一个完整的场景重建系统[5, 15]。系统是用Python实现的，使用了很多在Open3D中实现的算法。其以一个RGB-D序列作为输入，通过三个主要步骤来进行。

1. Build local geometric surfaces {Pi} (referred to as fragments) from short subsequences of the input RGB-D sequence. There are three substeps: matching pairs of RGB-D images, robust pose graph optimization, and volumetric integration. 从输入的RGB-D序列的短序列中构建局部几何面{Pi}（称之为片段）。有三个子步骤：成对RGB-D图像的匹配，稳健的姿态图优化，和体集成。

2. Globally align the fragments to obtain fragment poses {Ti} and a camera calibration function C(·). There are four substeps: global registration between fragment pairs, robust pose graph optimization, ICP registration, and global non-rigid alignment. 对这些片段进行全局对齐，以得到片段姿态{Ti}和相机标定函数C(·)。有4个子步骤：片段对的全局配准，稳健的姿态图优化，ICP配准，和全局非刚性对齐。

3. Integrate RGB-D images to generate a mesh model for the scene. 将RGB-D图像集成，为场景生成一个网格模型。

Figure 4 shows a reconstruction produced by this pipeline for a scene from the SceneNN dataset [10]. The visualization was also done via Open3D. 图4给出了用这个流程，对SceneNN数据集中的一个场景得到的重建。

## 4. Optimization

Open3D was engineered for high performance. We have optimized the C++ backend such that Open3D implementations are generally faster than their counterparts in other 3D processing libraries. For each major function, we used profiling tools to benchmark the execution of key steps. This analysis accelerated the running time of many functions by multiplicative factors. For example, our optimized implementation of the ICP algorithm is up to 25 times faster than its counterpart in PCL [18]. Our implementation of the reconstruction pipeline of Choi et al. [5] is up to an order of magnitude faster than the original implementation released by the authors.

Open3D的设计使其性能很高。我们优化了C++后端，这样Open3D的实现一般要比其他3D处理库的要快。对于每个主要的函数，我们使用性能分析工具来测试了主要步骤的执行。这种分析加速了很多函数的运行时间。比如，我们对ICP算法的优化时间比PCL的要快最多25倍。我们对Choi等[5]的重建流程的实现，比作者的原始实现要快一个数量级。

A large number of functions are parallelized with OpenMP. Many functions, such as normal estimation, can be easily parallelized across data samples using “#pragma omp for” declarations. Another example of consequential parallelization can be found in our non-linear least-squares solvers. These functions optimize objectives of the form

大量函数是用OpenMP进行并行实现的。很多函数，比如法线估计，可以在不同数据样本间使用“#pragma omp for”声明很简单的进行并行计算。另一个相应的并行化例子可以在我们的非线性最小二乘求解器中法线。这些函数优化下面形式的目标函数

$$L(x) = \sum_i r_i^2(x)$$(1)

where $r_i(x)$ is a residual term. A step in a Gauss-Newton solver takes the current solution x^k and updates it to 其中$r_i(x)$是一个残差项。在Gauss-Newton求解器中的一个步骤是以x^k为输入，将其更新成

$$x^{k+1} = x^{k} - (J_r^T r)^{-1} (J_r^T r)$$(2)

where $J_r$ is the Jacobian matrix for the residual vector r, both evaluated at x^k. In the Open3D implementation, the most time-consuming part is the evaluation of $J_r^T J_r$ and $J_r^T r$. These were parallelized using OpenMP reduction. A lambda function specifies the computation of $J_r^T J_r$ and $J_r^T r$ for each data record. This lambda function is called in a reduction loop that sums over the matrices.

其中$J_r$是残差向量r的Jacobian矩阵，都是在x^k上计算得到的。在Open3D的实现中，最耗时的部分是$J_r^T J_r$和$J_r^T r$的计算。它们使用OpenMP进行并行计算。一个lambda函数对每个数据记录指定了$J_r^T J_r$和$J_r^T r$的计算。这个lambda函数在reduction循环中进行调用，在矩阵上进行求和。

The parallelization of the Open3D backend accelerated the most time-consuming Open3D functions by a factor of 3-6 on a modern CPU.

Open3D后端的并行化加速了大多数耗时的Open3D函数，在CPU上加速了3-6倍。

## 5. Release

Open3D is released open-source under the permissive MIT license and is available at http://www.open3d.org. Ongoing development is coordinated via GitHub. Code changes are integrated via the following steps.

1. An issue is opened for a feature request or a bug fix.
2. A developer starts a new branch on a personal fork of Open3D, writes code, and submits a pull request when ready.
3. The code change is reviewed and discussed in the pull request. Modifications are made to address issues raised in the discussion.
4. One of the admins merges the pull request to the master branch and closes the issue.
   
The code review process maintains a consistent coding style and adherence to modern C++ programming guidelines. Pull requests are automatically checked by a continuous integration service.

We hope that Open3D will be useful to a broad community of developers who deal with 3D data.