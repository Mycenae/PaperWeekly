# ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras

Ra´ul Mur-Artal and Juan D. Tard´os  Spain

## 0. Abstract

We present ORB-SLAM2 a complete SLAM system for monocular, stereo and RGB-D cameras, including map reuse, loop closing and relocalization capabilities. The system works in real-time on standard CPUs in a wide variety of environments from small hand-held indoors sequences, to drones flying in industrial environments and cars driving around a city. Our back-end based on bundle adjustment with monocular and stereo observations allows for accurate trajectory estimation with metric scale. Our system includes a lightweight localization mode that leverages visual odometry tracks for unmapped regions and matches to map points that allow for zero-drift localization. The evaluation on 29 popular public sequences shows that our method achieves state-of-the-art accuracy, being in most cases the most accurate SLAM solution. We publish the source code, not only for the benefit of the SLAM community, but with the aim of being an out-of-the-box SLAM solution for researchers in other fields.

我们提出了ORB-SLAM2，对单目、立体和RGB-D相机的完整SLAM系统，包括地图重用，回环闭合和重定位能力。系统在标准CPUs上，可以在很多环境中实时运行，从小型手持室内序列，到在工业环境中飞行的无人机，和在城市中驾驶的车辆。我们的后端是基于BA的，带有单目和立体观察，可以进行精确的轨迹估计。我们的系统包含轻量级的定位模式，利用视觉里程计跟踪未建图的区域，与地图点进行匹配，得到零漂移的定位。在29个流行的公开序列中的评估表明，我们的方法取得了目前最好的准确率，在大多数情况下，是最精确的SLAM解决方案。我们公开了源码，不仅仅是为了SLAM团体，而且是为了给其他领域的研究者提供一个开箱即用的SLAM解决方案。

## 1. Introduction

Simultaneous Localization and Mapping (SLAM) has been a hot research topic in the last two decades in the Computer Vision and Robotics communities, and has recently attracted the attention of high-technological companies. SLAM techniques build a map of an unknown environment and localize the sensor in the map with a strong focus on real-time operation. Among the different sensor modalities, cameras are cheap and provide rich information of the environment that allows for robust and accurate place recognition. Therefore Visual SLAM solutions, where the main sensor is a camera, are of major interest nowadays. Place recognition is a key module of a SLAM system to close loops (i.e. detect when the sensor returns to a mapped area and correct the accumulated error in exploration) and to relocalize the camera after a tracking failure, due to occlusion or aggressive motion, or at system re-initialization.

SLAM在过去二十年中，是计算机视觉和机器人团体的热门研究课题，最近吸引了很多高科技公司的注意力。SLAM对未知环境进行建图，对传感器在地图中进行定位，而且聚焦在实时操作上。在不同的传感器模态中，相机非常便宜，包含了环境的丰富信息，可以进行稳健和准确的位置识别。因此视觉SLAM解决方案是主要的兴趣点，其中主要的传感器是相机。位置识别是SLAM系统进行回环闭合的一个关键模块（即，检测传感器回到了已经建图的区域，修正探索中的累积误差），也是在跟踪失败后重新定位相机的一个关键模块，这可能是因为遮挡，或激进的运动，或是系统在重新初始化。

Visual SLAM can be performed by using just a monocular camera, which is the cheapest and smallest sensor setup. However as depth is not observable from just one camera, the scale of the map and estimated trajectory is unknown. In addition the system bootstrapping require multi-view or filtering techniques to produce an initial map as it cannot be triangulated from the very first frame. Last but not least, monocular SLAM suffers from scale drift and may fail if performing pure rotations in exploration. By using a stereo or an RGB-D camera all these issues are solved and allows for the most reliable Visual SLAM solutions.

视觉SLAM可以只用一个单目相机进行，这是最便宜最小的传感器设置。但是，深度不能只用一个相机观察得到，地图的尺度和估计的轨迹都是未知的。而且，系统的提升需要多视角或滤波技术，以产生初始梯度，因为不能从第一帧中进行三角定位。最后，单目SLAM存在尺度漂移的问题，如果在探索中进行纯旋转，则会出现失败。通过使用立体相机，或RGB-D相机，所有这些问题都可以得到解决，可以得到最可靠的视觉SLAM解决方案。

In this paper we build on our monocular ORB-SLAM [1] and propose ORB-SLAM2 with the following contributions: 本文中，我们在单目ORB-SLAM的基础上，提出了ORB-SLAM2，有下列贡献：

• The first open-source SLAM system for monocular, stereo and RGB-D cameras, including loop closing, relocalization and map reuse. 第一个开源单目、立体和RGB-D相机SLAM系统，包括回环闭合，重定位和地图重用。

• Our RGB-D results show that by using Bundle Adjustment (BA) we achieve more accuracy than state-of-the-art methods based on ICP or photometric and depth error minimization. 我们的RGB-D结果表明，通过使用BA，我们比目前最好的方法得到了更多的准确率，它们使用的是ICP或光度和深度误差最小化。

• By using close and far stereo points and monocular observations our stereo results are more accurate than the state-of-the-art direct stereo SLAM. 通过使用近和远立体点，和单目观察，我们的立体结果比目前最好的直接立体SLAM准确率更高。

• A lightweight localization mode that can effectively reuse the map with mapping disabled. 一个轻量级定位模式，可以有效在建图禁用的时候的重用地图。

Fig. 1 shows examples of ORB-SLAM2 output from stereo and RGB-D inputs. The stereo case shows the final trajectory and sparse reconstruction of the sequence 00 from the KITTI dataset [2]. This is an urban sequence with multiple loop closures that ORB-SLAM2 was able to successfully detect. The RGB-D case shows the keyframe poses estimated in sequence fr1 room from the TUM RGB-D Dataset [3], and a dense pointcloud, rendered by backprojecting sensor depth maps from the estimated keyframe poses. Note that our SLAM does not perform any fusion like KinectFusion [4] or similar, but the good definition indicates the accuracy of the keyframe poses. More examples are shown on the attached video.

图1给出了ORB-SLAM2在立体和RGB-D输入时的例子。立体的情况给出了，KITTI数据集中序列00的最终轨迹和稀疏重建。这是一个市区序列，有多个回环闭合，ORB-SLAM2可以很成功的检测得到。RGB-D的情况，给出了TUM RGB-D数据集中fr1序列估计得到关键帧的姿态，和一个密集的点云，通过从估计的关键帧姿态反向投影传感器深度地图渲染得到的。注意，我们的SLAM并没有像KinectFusion或类似的进行任何融合，但很好的定义表明了关键帧姿态的准确率。更多的例子见附带的视频。

In the rest of the paper, we discuss related work in Section II, we describe our system in Section III, then present the evaluation results in Section IV and end with conclusions in Section V.

本文剩余部分，我们在第2部分中讨论相关工作，在第3部分中讨论了我们的系统，然后在第4部分给出评估结果，在第5部分给出结论。

## 2. Related Work

In this section we discuss related work on stereo and RGB-D SLAM. Our discussion, as well as the evaluation in Section IV is focused only on SLAM approaches.

在本节中，我们讨论在立体和RGB-D SLAM中的相关工作。我们的讨论，以及第4部分的评估，关注的都只是SLAM方法。

### 2.1. Stereo SLAM

A remarkable early stereo SLAM system was the work of Paz et al. [5]. Based on Conditionally Independent Divide and Conquer EKF-SLAM it was able to operate in larger environments than other approaches at that time. Most importantly, it was the first stereo SLAM exploiting both close and far points (i.e. points whose depth cannot be reliably estimated due to little disparity in the stereo camera), using an inverse depth parametrization [6] for the latter. They empirically showed that points can be reliably triangulated if their depth is less than ∼40 times the stereo baseline. In this work we follow this strategy of treating in a different way close and far points, as explained in Section III-A.

早期的一个极佳立体SLAM系统是Paz等[5]的工作。基于Conditionally Independent Divide and Conquer EKF-SLAM，在那时，比其他方法可以在更大的环境中进行操作。更重要的是，这是第一个同时利用近点和远点的立体SLAM系统（远点是其深度不能被可靠的估计，因为在立体相机中的视差很小），对远点使用了逆深度参数化。他们通过经验表明，如果点的深度小于立体相机基线的大约40倍，就可以进行可靠的三角剖分。在本文中，我们按照这个策略，对近点和远点区别对待，在3.1节进行了解释。

Most modern stereo SLAM systems are keyframe-based [7] and perform BA optimization in a local area to achieve scalability. The work of Strasdat et al. [8] performs a joint optimization of BA (point-pose constraints) in an inner window of keyframes and pose-graph (pose-pose constraints) in an outer window. By limiting the size of these windows the method achieves constant time complexity, at the expense of not guaranteeing global consistency. The RSLAM of Mei et al. [9] uses a relative representation of landmarks and poses
and performs relative BA in an active area which can be constrained for constant-time. RSLAM is able to close loops which allow to expand active areas at both sides of a loop, but global consistency is not enforced. The recent S-PTAM by Pire et al. [10] performs local BA, however it lacks large loop closing. Similar to these approaches we perform BA in a local set of keyframes so that the complexity is independent of the map size and we can operate in large environments. However our goal is to build a globally consistent map. When closing a loop, our system aligns first both sides, similar to RSLAM, so that the tracking is able to continue localizing using the old map and then performs a pose-graph optimization that minimizes the drift accumulated in the loop, followed by full BA.

多数现代的立体视觉SLAM系统是基于关键帧的[7]，在局部区域进行BA优化，以得到可扩展性。Strasdat等[8]对BA（点-姿态约束）在关键帧的内窗和姿态图（姿态-姿态约束）在外窗中进行联合优化。通过限制这些窗口的大小，该方法的时间复杂度为常数，但不能保证全局的一致性。Mei等[9]的RSLAM使用关键点和姿态的相对表示，在活跃区域中进行相对BA，可以约束在常数时间内完成。RSLAM能够闭合回环，可以在回环的两边都扩散活跃区域，但不能施加全局一致性。最近的Pire等[10]的S-PTAM进行局部BA，但是缺少大的回环闭合。与这些方法类似，我们在关键帧的局部集合中进行BA，这样复杂度与地图大小无关，我们可以在大型环境中进行操作。但是我们的目标是构建一个全局一致的地图。当闭合回环时，我们的系统对两边都进行对齐，与RSLAM类似，这样跟踪就可以继续使用老的地图进行定位，然后使用姿态图优化，对回环中的累积漂移进行最小化，然后进行完整的BA。

The recent Stereo LSD-SLAM of Engel et al. [11] is a semi-dense direct approach that minimizes photometric error in image regions with high gradient. Not relying on features, the method is expected to be more robust to motion blur or poorly-textured environments. However as a direct method its performance can be severely degraded by unmodeled effects like rolling shutter or non-lambertian reflectance.

最近Engel等[11]的立体LSD-SLAM是一个半密集的直接方法，在高梯度图像区域中最小化光度误差。该方法并不依赖于特征，对运动模糊或纹理很少的环境会更加稳健。但是，作为一种直接方法，其性能会被未建模的效果严重影响，如卷帘快门，或非lambertian反射率。

### 2.2. RGB-D SLAM

One of the earliest and most famed RGB-D SLAM systems was the KinectFusion of Newcombe et al. [4]. This method fused all depth data from the sensor into a volumetric dense model that is used to track the camera pose using ICP. This system was limited to small workspaces due to its volumetric representation and the lack of loop closing. Kintinuous by Whelan et al. [12] was able to operate in large environments by using a rolling cyclical buffer and included loop closing using place recognition and pose graph optimization.

最早最著名的一个RGB-D SLAM系统，是Newcombe等[4]的KinectFusion。该方法将传感器的所有深度数据融合成了一个体密集模型，使用这个模型用ICP跟踪相机姿态。系统局限于小型工作空间，因为体表示很大，还有缺少回环闭合。Whelan等[12]的Kintinuous可以在大型环境中操作，使用一个rolling cyclical buffer，使用位置识别和姿态图优化来包括回环闭合。

Probably the first popular open-source system was the RGB-D SLAM of Endres et al. [13]. This is a feature-based system, whose front-end computes frame-to-frame motion by feature matching and ICP. The back-end performs pose-graph optimization with loop closure constraints from a heuristic search. Similarly the back-end of DVO-SLAM by Kerl et al. [14] optimizes a pose-graph where keyframe-to-keyframe constraints are computed from a visual odometry that minimizes both photometric and depth error. DVO-SLAM also searches for loop candidates in a heuristic fashion over all previous frames, instead of relying on place recognition.

可能第一个流行的开源系统是Endres等[13]的RGB-D SLAM。这是一个基于特征的系统，其前端通过特征匹配和ICP来计算帧对帧的运动。后端进行带有回环闭合约束的姿态图优化，进行启发式搜索。类似的，Kerl等[14]的DVO-SLAM的后端对姿态图进行优化，其中关键帧到关键帧的约束是从视觉里程计进行计算的，对光度和深度误差进行最小化。DVO-SLAM还以启发式的方式对所有之前的帧进行回环候选搜索，而不是依赖于位置识别。

The recent ElasticFusion of Whelan et al. [15] builds a surfel-based map of the environment. This is a map-centric approach that forget poses and performs loop closing applying a non-rigid deformation to the map, instead of a standard pose-graph optimization. The detailed reconstruction and localization accuracy of this system is impressive, but the current implementation is limited to room-size maps as the complexity scales with the number of surfels in the map.

最近Whelan等的ElasticsFusion构建了环境的基于surfel的地图。这是一个以地图为中心的方法，没有姿态，对地图应用非刚性形变，来进行回环闭合，而不是标准的姿态图优化。这个系统的详细重建和定位准确率是令人印象深刻的，但当前的实现局限于房间大小的地图，因为复杂度随着地图中的surfel的数量增加。

As proposed by Strasdat et al. [8] our ORB-SLAM2 uses depth information to synthesize a stereo coordinate for extracted features on the image. This way our system is agnostic of the input being stereo or RGB-D. Differently to all above methods our back-end is based on bundle adjustment and builds a globally consistent sparse reconstruction. Therefore our method is lightweight and works with standard CPUs. Our goal is long-term and globally consistent localization instead of building the most detailed dense reconstruction. However from the highly accurate keyframe poses one could fuse depth maps and get accurate reconstruction on-the-fly in a local area or post-process the depth maps from all keyframes after a full BA and get an accurate 3D model of the whole scene.

Strasdat等[8]提出，我们的ORB-SLAM2使用深度信息来合成一个立体坐标，与图像中提取出的特征共同使用。使用这种方法，我们的系统就不需要知道输入是立体相机还是RGB-D。与上述所有方法不同，我们的后端是基于BA的，构建了一个全局一致的稀疏重建。因此我们的方法是轻量级的，用标准CPU就可以工作。我们的目标是长期的，全局一致的定位，而不是构建最详细的密集重建。但是，从高精度的关键帧姿态中，我们就可以融合深度图，在局部区域中得到在线的精确重建，或对所有关键帧在进行完全BA后，对深度图进行后处理，得到整个场景的精确3D模型。

## 3. ORB-SLAM2

ORB-SLAM2 for stereo and RGB-D cameras is built on our monocular feature-based ORB-SLAM [1], whose main components are summarized here for reader convenience. A general overview of the system is shown in Fig. 2. The system has three main parallel threads: 1) the tracking to localize the camera with every frame by finding feature matches to the local map and minimizing the reprojection error applying motion-only BA, 2) the local mapping to manage the local map and optimize it, performing local BA, 3) the loop closing to detect large loops and correct the accumulated drift by performing a pose-graph optimization. This thread launches a fourth thread to perform full BA after the pose-graph optimization, to compute the optimal structure and motion solution.

对于立体相机和RGB-D相机的ORB-SLAM2是在我们的单目的基于特征的ORB-SLAM基础之上构建起来的，我们在这里总结了其主要组成部分，方便读者。系统的总体概览如图2所示。系统有3个主要的并行线程：1) 跟踪用每一帧输入定位相机，找到与局部地图的匹配特征，使用只有运动的BA来最小化重投影误差，2) 局部建图，管理局部地图，并对其进行优化，进行局部BA，3) 回环闭合，检测大型回环，修正累积的漂移，进行姿态图优化。这个线程会发起第四个线程，在姿态图优化后进行完成的BA，计算最优的结构和运动解。

The system has embedded a Place Recognition module based on DBoW2 [16] for relocalization, in case of tracking failure (e.g. an occlusion) or for reinitialization in an already mapped scene, and for loop detection. The system maintains a covisibiliy graph [8] that links any two keyframes observing common points and a minimum spanning tree connecting all keyframes. These graph structures allow to retrieve local windows of keyframes, so that tracking and local mapping operate locally, allowing to work on large environments, and serve as structure for the pose-graph optimization performed when closing a loop.

系统嵌入了一个基于DBoW2[16]的位置识别模块，在跟踪失败（如，遮挡），或在一个已经建图的场景中进行重新初始化的时候，以及进行回环检测的时候，进行重定位。系统维护了一个互可见图[8]，将任意两个观察到共同点的关键帧连接到一起，和连接了所有关键帧的最小张树。这些图结构可以获取关键帧的局部窗口，这样跟踪和局部建图可以在局部进行，可以在大型环境中工作，可以在进行回环闭合的时候，作为姿态图优化所用的结构。

The system uses the same ORB features [17] for tracking, mapping and place recognition tasks. These features are robust to rotation and scale and present a good invariance to camera auto-gain and auto-exposure, and illumination changes. Moreover they are fast to extract and match allowing for real-time operation and show good precision/recall performance in bag-of-word place recognition [18].

系统对跟踪，建图和位置识别任务，使用同样的ORB特征。这些特征对旋转和尺度非常稳健，对相机自动增益和自动曝光，以及光照变化有很好的不变性。而且，提取和匹配的速度都很快，可以进行实时操作，在bag-of-word位置识别的时候展现出了很好的precision/recall性能。

In the rest of this section we present how stereo/depth information is exploited and which elements of the system are affected. For a detailed description of each system block, we refer the reader to our monocular publication [1].

在本节的剩余部分，我们给出了怎样利用立体信息、深度信息，系统的哪些元素受到影响。对于每个系统模块的详细描述，我们推荐读者去参考我们的单目文章[1]。

### 3.1. Monocular, Close Stereo and Far Stereo Keypoints

ORB-SLAM2 as a feature-based method pre-processes the input to extract features at salient keypoint locations, as shown in Fig. 2b. The input images are then discarded and all system operations are based on these features, so that the system is independent of the sensor being stereo or RGB-D. Our system handles monocular and stereo keypoints, which are further classified as close or far.

ORB-SLAM2作为一种基于特征的方法，对输入进行预处理，以在明显的关键点位置提取特征，如图2b所示。输入图像在预处理之后就丢弃不用了，所有的系统操作都是基于这些特征的，这样系统与传感器是立体相机还是RGB-D是无关的。我们的系统处理的是单目和立体关键点，然后进一步归类为近点或远点。

**Stereo keypoints** are defined by three coordinates $x_s = (u_L, v_L, u_R)$, being ($u_L, v_L$) the coordinates on the left image and $u_R$ the horizontal coordinate in the right image. For stereo cameras, we extract ORB in both images and for every left ORB we search for a match in the right image. This can be done very efficiently assuming stereo rectified images, so that epipolar lines are horizontal. We then generate the stereo keypoint with the coordinates of the left ORB and the horizontal coordinate of the right match, which is subpixel refined by patch correlation. For RGB-D cameras, we extract ORB features on the RGB image and, as proposed by Strasdat et al. [8], for each feature with coordinates ($u_L, v_L$) we transform its depth value d into a virtual right coordinate:

立体关键点由三个坐标定义$x_s = (u_L, v_L, u_R)$，($u_L, v_L$)是在左图中的坐标，$u_R$是右图中的水平坐标。对于立体相机，我们在两幅图像中提取ORB特征点，对左图中的每个ORB，我们都在右图中搜索得到一个匹配。假设立体图像经过了校正，这可以很高效的计算完成，这样对极线是水平的。然后我们用左ORB的坐标，和右边的匹配（通过块相关提炼得到亚像素的精度）的水平坐标，来生成立体关键点。对于RGB-D相机，我们在RGB图像上提取ORB特征，对每个坐标为($u_L, v_L$)的特征，我们将其深度值d转换成一个虚拟右坐标，Strasdat等[8]也这样建议：

$$u_R - u_L - \frac {f_xb}{d}$$(1)

where f_x is the horizontal focal length and b is the baseline between the structured light projector and the infrared camera, which we approximate to 8cm for Kinect and Asus Xtion.

其中f_x是水平焦点长度，b是结构光投影仪与红外相机之间的基准线，对于Kinect和Asus Xtion来说大约是8cm。

The uncertainty of the depth sensor is represented by the uncertainty of the virtual right coordinate. In this way, features from stereo and RGB-D input are handled equally by the rest of the system.

深度传感器的不确定性，是由虚拟右坐标的不确定性来表示的。这样，立体相机和RGB-D相机的输入的特征，也可以由系统的其他部分一样的处理。

A stereo keypoint is classified as close if its associated depth is less than 40 times the stereo/RGB-D baseline, as suggested in [5], otherwise it is classified as far. Close keypoints can be safely triangulated from one frame as depth is accurately estimated and provide scale, translation and rotation information. On the other hand far points provide accurate rotation information but weaker scale and translation information. We triangulate far points when they are supported by multiple views.

一个立体关键点被认为是近点，如果其相关的深度小于立体相机/RGB-D相机基线的40倍，否则就称为远点。近关键点可以从一帧中很安全的进行三角定位，因为深度可以得到很准确的估计，并给出尺度，平移和旋转信息。另一方面，远点可以给出准确的旋转信息，但是很弱的尺度和平移信息。如果有多视的支撑，我们就对远点进行三角测量。

Monocular keypoints are defined by two coordinates $x_m = (u_L, v_L)$ on the left image and correspond to all those ORB for which a stereo match could not be found or that have an invalid depth value in the RGB-D case. These points are only triangulated from multiple views and do not provide scale information, but contribute to the rotation and translation estimation.

单目关键点由左边图像中的两个坐标$x_m = (u_L, v_L)$定义，对应的是那些没有立体匹配的，或在RGB-D的情况下，深度值无效的那些所有ORB。这些点只是从多视中进行三角定位，不给出尺度信息，但对旋转和平移的估计有贡献。

### 3.2. System Bootstrapping

One of the main benefits of using stereo or RGB-D cameras is that, by having depth information from just one frame, we do not need a specific structure from motion initialization as in the monocular case. At system startup we create a keyframe with the first frame, set its pose to the origin, and create an initial map from all stereo keypoints.

使用立体相机或RGB-D相机的一个主要好处是，我们对一帧有了深度信息后，我们就不需要特定的SfM初始化，就像在单目的情况下。在系统启动的时候，我们用第一帧创建关键帧，设其姿态为原点，从所有立体关键点中就可以创建一个初始地图。

### 3.3. Bundle Adjustment with Monocular and Stereo Constraints

Our system performs BA to optimize the camera pose in the tracking thread (motion-only BA), to optimize a local window of keyframes and points in the local mapping thread (local BA), and after a loop closure to optimize all keyframes and points (full BA). We use the Levenberg–Marquardt method implemented in g2o [19].

我们的系统在跟踪进程中进行BA对相机姿态进行优化（只有运动的BA），优化关键帧的一个局部窗口，和局部建图线程的点（局部BA），在回环闭合后，优化所有的关键帧和点（完整BA）。我们使用g2o中实现的Levenberg–Marquardt方法[19]。

**Motion-only BA** optimizes the camera orientation R ∈ SO(3) and position $t ∈ R^3$, minimizing the reprojection error between matched 3D points $Xi ∈ R^3$ in world coordinates and keypoints $x^i_{(·)}$, either monocular $x^i_m ∈ R^2$ or stereo $x^i_s ∈ R^3$, with i ∈ X the set of all matches:

只有运动的BA优化相机的方向R ∈ SO(3)和位置$t ∈ R^3$，对匹配的在世界坐标系中的3D点$Xi ∈ R^3$和关键点$x^i_{(·)}$的重投影误差进行最小化，这里的关键点可以是单目的$x^i_m ∈ R^2$，或立体的$x^i_s ∈ R^3$，i ∈ X，X是所有匹配的集合：

$$\{ R, t\} = argmin_{R,t} \sum_{i∈X} ρ(||x^i_{(·)} - π_{(·)} (RX^i + t)||^2_Σ)$$(2)

where ρ is the robust Huber cost function and Σ the covariance matrix associated to the scale of the keypoint. The projection functions π_(·), monocular π_m and rectified stereo π_s, are defined as follows:

其中ρ是稳健Huber代价函数，Σ是与关键点尺度相关的协方差矩阵。投影函数π_(·)，单目π_m和校正立体π_s，定义如下：

$$π_m([X, Y, Z]^T) = [f_x X/Z + c_x, f_y Y/Z + c_y]^T, π_s ([X, Y, Z]^T) = [f_x X/Z + c_x, f_y Y/Z + c_y, f_x (X-b)/Z + c_x]^T$$(3)

where (f_x, f_y) is the focal length, (c_x, c_y) is the principal point and b the baseline, all known from calibration.

其中(f_x, f_y)是焦距，(c_x, c_y)是主点，b是基准，都是从校准中已知的。

**Local BA** optimizes a set of covisible keyframes K_L and all points seen in those keyframes P_L. All other keyframes K_F , not in K_L, observing points in P_L contribute to the cost function but remain fixed in the optimization. Defining X_k as the set of matches between points in P_L and keypoints in a keyframe k, the optimization problem is the following:

局部BA优化的是互可见的关键帧K_L的集合，和在这些关键帧中所有可看见的点P_L。所有其他的关键帧K_F，不在K_L之中，但可以看到P_L中的点，对代价函数有贡献，但在优化中保持固定。将P_L中的点与关键帧k中的关键点的匹配的集合，定义为X_k，优化问题如下：

$$\{ X^i, R_l, t_l | i∈P_L, l∈K_L \} = argmin_{X^i, R_l, t_l} \sum_{k∈K_L∪K_F} \sum_{j∈X_k} ρ (E_{kj}), E_{kj} = ||x^j_{(·)} - π_{(·)} (R_k X^j + t_k)||^2_Σ$$(4)

**Full BA** is the specific case of local BA, where all keyframes and points in the map are optimized, except the origin keyframe that is fixed to eliminate the gauge freedom.

完整BA是局部BA的特殊情况，其中地图中的所有关键帧和点都进行优化，除了初始关键帧是固定的，以消除测量自由度。

### 3.4. Loop Closing and Full BA

Loop closing is performed in two steps, firstly a loop has to be detected and validated, and secondly the loop is corrected optimizing a pose-graph. In contrast to monocular ORB-SLAM, where scale drift may occur [20], the stereo/depth information makes scale observable and the geometric validation and pose-graph optimization no longer require dealing with scale drift and are based on rigid body transformations instead of similarities.

回环闭合由两步进行，首先，必须检测到一个回环并进行验证，第二，优化姿态图修正回环。单目ORB-SLAM中尺度的漂移可能发生，与之相比，立体/深度信息使得尺度是可以观察到的，几何验证和姿态图优化不需要处理尺度漂移，可以基于刚体变换，而不用是相似度。

In ORB-SLAM2 we have incorporated a full BA optimization after the pose-graph to achieve the optimal solution. This optimization might be very costly and therefore we perform it in a separate thread, allowing the system to continue creating map and detecting loops. However this brings the challenge of merging the bundle adjustment output with the current state of the map. If a new loop is detected while the optimization is running, we abort the optimization and proceed to close the loop, which will launch the full BA optimization again. When the full BA finishes, we need to merge the updated subset of keyframes and points optimized by the full BA, with the non-updated keyframes and points that where inserted while the optimization was running. This is done by propagating the correction of updated keyframes (i.e. the transformation from the non-optimized to the optimized pose) to non-updated keyframes through the spanning tree. Non-updated points are transformed according to the correction applied to their reference keyframe.

在ORB-SLAM2中，我们在姿态图之后又进行了完全BA，以得到最优解。这个优化过程非常耗时，所以我们单独用一个进程处理，使系统可以继续创建地图并检测回环。但是，这带来了一个挑战，即将BA的输入与当前的地图状态进行融合。如果检测到了一个新回环，而优化又在进行，我们就放弃优化，继续回环闭合，这会再次发起完整BA优化。当完整BA结束，我们需要更新完整BA优化过的关键帧和点的子集，并与没更新的关键帧和点进行融合，优化在运行的时候同时插入。这通过将更新的关键帧的修正（即，从非优化的到优化的姿态的变换）通过张树传播到非更新的关键帧来完成。非更新的点要根据应用到参考关键帧的修正来进行变换。

### 3.5. Keyframe Insertion

ORB-SLAM2 follows the policy introduced in monocular ORB-SLAM of inserting keyframes very often and culling redundant ones afterwards. The distinction between close and far stereo points allows us to introduce a new condition for keyframe insertion, which can be critical in challenging environments where a big part of the scene is far from the stereo sensor, as shown in Fig. 3. In such environment we need to have a sufficient amount of close points to accurately estimate translation, therefore if the number of tracked close points drops below τ_t and the frame could create at least τ_c new close stereo points, the system will insert a new keyframe. We empirically found that τ_t = 100 and τ_c = 70 works well in all our experiments.

ORB-SLAM2按照单目ORB-SLAM提出的策略，非常经常的插入关键帧，然后再选择剔除冗余的关键帧。立体近点和远点的差异，使我们可以引入关键帧插入的新条件，这在部分场景远离立体相机的环境中，非常关键，如图3所示。在这种环境中，我们需要有足够多的近点，来准确的估计平移，因此如果跟踪的近点的数量低于τ_t，而这一帧可以创建至少τ_c个新的近立体点，系统就会插入一个新的关键帧。我们通过经验发现，τ_t = 100，τ_c = 70，在我们所有的试验中效果很好。

### 3.6. Localization Mode

We incorporate a Localization Mode which can be useful for lightweight long-term localization in well mapped areas, as long as there are not significant changes in the environment. In this mode the local mapping and loop closing threads are deactivated and the camera is continuously localized by the tracking using relocalization if needed. In this mode the tracking leverages visual odometry matches and matches to map points. Visual odometry matches are matches between ORB in the current frame and 3D points created in the previous frame from the stereo/depth information. These matches make the localization robust to unmapped regions, but drift can be accumulated. Map point matches ensure drift-free localization to the existing map. This mode is demonstrated in the accompanying video.

我们包括了一个定位模式，对在已经建图很好的区域中进行轻量级长期定位非常有用，只要环境中没有明显的变化。在这个模式中，局部建图和回环闭合线程处于非激活状态，相机通过跟踪，如果有必要的话就使用重新定位来持续的进行定位。在这个模式中，跟踪利用视觉里程计匹配，和与地图点的匹配。视觉里程计匹配，是当前帧的ORB，与之前帧从立体/深度信息创建的3D点，之间的匹配。这些匹配使定位对未建图的区域非常稳健，但仍然会累积漂移。地图点匹配，确保了与已有地图的无漂移定位。这个模式在附属的视频中有展示。

## 4. Evaluation

We have evaluated ORB-SLAM2 in three popular datasets and compared to other state-of-the-art SLAM systems, using always the results published by the original authors and standard evaluation metrics in the literature. We have run ORB-SLAM2 in an Intel Core i7-4790 desktop computer with 16Gb RAM. In order to account for the non-deterministic nature of the multi-threading system, we run each sequence 5 times and show median results for the accuracy of the estimated trajectory. Our open-source implementation includes the calibration and instructions to run the system in all these datasets.

我们在三个流行的数据集上评估了ORB-SLAM2，与其他目前最好的系统进行了对比。我们在Intel Core i7-4790，16Gb RAM的桌面电脑上运行ORB-SLAM2。为考虑多线程系统的非确定性本质，我们对每个序列运行5次，展示了估计的轨迹的准确率的中值。我们的开源实现，包括校准，和在所有这些数据集上运行系统的使用说明。

### 4.1. KITTI Dataset

The KITTI dataset [2] contains stereo sequences recorded from a car in urban and highway environments. The stereo sensor has a ∼54cm baseline and works at 10Hz with a resolution after rectification of 1240 × 376 pixels. Sequences 00, 02, 05, 06, 07 and 09 contain loops. Our ORB-SLAM2 detects all loops and is able to reuse its map afterwards, except for sequence 09 where the loop happens in very few frames at the end of the sequence. Table I shows results in the 11 training sequences, which have public ground-truth, compared to the state-of-the-art Stereo LSD-SLAM [11], to our knowledge the only stereo SLAM showing detailed results for all sequences. We use two different metrics, the absolute translation RMSE t_abs proposed in [3], and the average relative translation t_rel and rotation r_rel errors proposed in [2]. Our system outperforms Stereo LSD-SLAM in most sequences, and achieves in general a relative error lower than 1%. The sequence 01, see Fig. 3, is the only highway sequence in the training set and the translation error is slightly worse. Translation is harder to estimate in this sequence because very few close points can be tracked, due to high speed and low frame-rate. However orientation can be accurately estimated, achieving an error of 0.21 degrees per 100 meters, as there are many far point that can be long tracked. Fig. 4 shows some examples of estimated trajectories.

KITTI数据集[2]包含的立体序列，是用一辆车在城市和告诉环境中录制的。立体相机的基线为约54cm，工作频率10Hz，校正后分辨率为1240 × 376。序列00, 02, 05, 06, 07和09包含回环。我们的ORB-SLAM2检测到了所有回环，可以随后重用其地图，除了序列09，其回环是在序列最后的少数几帧中。表1给出了11个训练序列中的结果，有公开的真值，于目前最好的立体LSD-SLAM进行了对比，据我们所知，这是对所有序列都给出了详细结果的立体SLAM系统。我们使用了两个不同的度量系统，[3]中提出的绝对平移RMSE t_abs，[2]中提出的平均相对平移t_rel和旋转r_rel误差。我们的系统在多数序列中都超过了LSD-SLAM，总体上得到的相对误差低于1%。序列01是训练集中的唯一的高速序列，误差较差，见图3。平移在这个序列中很难估计，因为速度很快、帧率较低，只能跟踪很少的近点。但是可以很准确的估计方向，每100米的估计误差为0.21度，因为有很多远点可以长期追踪。图4给出了估计的轨迹的一些例子。

Compared to the monocular results presented in [1], the proposed stereo version is able to process the sequence 01 where the monocular system failed. In this highway sequence, see Fig. 3, close points are in view only for a few frames. The ability of the stereo version to create points from just one stereo keyframe instead of the delayed initialization of the monocular, consisting on finding matches between two keyframes, is critical in this sequence not to lose tracking. Moreover the stereo system estimates the map and trajectory with metric scale and does not suffer from scale drift, as seen in Fig. 5.

与[1]中给出的单目结果相比，我们提出的立体视觉可以处理序列01，而单目系统不能。在这个高速序列中，见图3，近点只在几帧中可见。立体视觉只从一个立体关键帧中创建点的能力，坚持在两个关键帧之中找到匹配，对于在这个序列中不丢失跟踪，是非常关键的，比较之下，单目系统的初始化则是被延迟的。而且，立体系统估计的地图和轨迹，都是有计量尺度的，不会有尺度漂移的问题，如图5所示。

### 4.2. EuRoC Dataset

The recent EuRoC dataset [21] contains 11 stereo sequences recorded from a micro aerial vehicle (MAV) flying around two different rooms and a large industrial environment. The stereo sensor has a ∼11cm baseline and provides WVGA images at 20Hz. The sequences are classified as easy, medium and difficult depending on MAV’s speed, illumination and scene texture. In all sequences the MAV revisits the environment and ORB-SLAM2 is able to reuse its map, closing loops when necessary. Table II shows absolute translation RMSE of ORB-SLAM2 for all sequences, comparing to Stereo LSD-SLAM, for the results provided in [11]. ORB-SLAM2 achieves a localization precision of a few centimeters and is more accurate than Stereo LSD-SLAM. Our tracking get lost in some parts of V2_03_difficult due to severe motion blur. As shown in [22], this sequence can be processed using IMU information. Fig. 6 shows examples of computed trajectories compared to the ground-truth.

最近的EuRoC数据集[21]包含11个立体序列，是一个微型飞行器(MAV)在两个不同的房间和一个大型工业环境中飞行录制的。立体传感器的基线为大约11cm，以20Hz的速度给出WVGA图像。根据MAV的速度，光照条件和场景纹理，这些序列分为容易，中等和困难三个级别。在所有序列中，MAV都会重新访问环境，ORB-SLAM2可以重用地图，在必要的时候闭合回环。表2给出了ORB-SLAM2对所有序列的绝对平移RMSE，与立体LSD-SLAM在[11]中给出的结果进行了比较。ORB-SLAM2得到的定位精度为几cm，比LSD-SLAM更加精确。我们的跟踪在V2_03_difficult序列中的一些部分丢失了，因为有严重的运动模糊。如[22]中所示，这个序列可以用IMU信息进行处理。图6展示了计算得到的轨迹与真值进行比较的例子。

### 4.3. TUM RGB-D Dataset

The TUM RGB-D dataset [3] contains indoors sequences from RGB-D sensors grouped in several categories to evaluate object reconstruction and SLAM/odometry methods under different texture, illumination and structure conditions. We show results in a subset of sequences where most RGB-D methods are usually evaluated. In Table III we compare our accuracy to the following state-of-the-art methods: ElasticFusion [15], Kintinuous [12], DVO-SLAM [14] and RGB-D SLAM [13]. Our method is the only one based on bundle adjustment and outperforms the other approaches in most sequences. As we already noticed for RGB-D SLAM results in [1], depthmaps for freiburg2 sequences have a 4% scale bias, probably coming from miscalibration, that we have compensated in our runs and could partly explain our significantly better results. Fig. 7 shows the point clouds that result from backprojecting the sensor depth maps from the computed keyframe poses in four sequences. The good definition and the straight contours of desks and posters prove the high accuracy localization of our approach.

TUM RGB-D数据集[3]包含RGB-D传感器的室内序列，分组在几个类别中，以评估在不同纹理、光照和结构条件下的目标重建和SLAM/里程计的方法。我们在一部分序列中展示了结果，多数RGB-D方法都在这上面进行评估。在表3中，我们比较了我们的方法与下面的目前最好的方法：ElasticFusion [15], Kintinuous [12], DVO-SLAM [14] 和 RGB-D SLAM [13]。我们的方法是唯一一个基于BA的方法，在多数序列中都超过了其他方法。我们在[1]中的RGB-D SLAM结果中就注意到了，freiburg2序列的深度图有4%的尺度偏差，可能是因为误校准，我们在我们的运行中进行了补偿，可以部分的解释我们明显更好的结果。我们将传感器的深度图从计算得到的关键帧姿态中进行重新投影，其中四个序列的点云如图7所示。桌子和海报的好的定义，和平直的轮廓，证明了我们方法的很高精度的定位。

### 4.4. Timing Results

In order to complete the evaluation of the proposed system, we present in Table IV timing results in three sequences with different image resolutions and sensors. The mean and two standard deviation ranges are shown for each thread task. As these sequences contain one single loop, the full BA and some tasks of the loop closing thread are executed just once and only a single time measurement is reported.The average tracking time per frame is below the inverse of the camera frame-rate for each sequence, meaning that our system is able to work in real-time. As ORB extraction in stereo images is parallelized, it can be seen that extracting 1000 ORB features in the stereo WVGA images of V2_02 is similar to extracting the same amount of features in the single VGA image channel of fr3_office.

为完成提出的系统的评估，我们在表4中给出了三个序列的计时结果，有着不同的图像分辨率和传感器。均值和两个标准偏差的范围对每个线程的任务进行了展示。这些序列包含单个回环，完整的BA和回环闭合的一些任务只执行了一次，所以只给出了单个时间测量。由于立体图像中的ORB提取是并行化的，可以看到，在V2_02的立体WVGA图像中提取1000个ORB特征，和在fr3_office中的单个VGA图像通道中提取相同数量的特征是类似的。

The number of keyframes in the loop is shown as reference for the times related to loop closing. While the loop in KITTI 07 contains more keyframes, the covisibility graph built for the indoor fr3_office is denser and therefore the loop fusion, pose-graph optimization and full BA tasks are more expensive. The higher density of the covisibility graph makes the local map contain more keyframes and points and therefore local map tracking and local BA are also more expensive.

我们给出了回环中的关键帧数量，作为回环闭合的相关时间的参考。在KITTI 07中的回环包含更多的关键帧，对室内fr3_office构建的互可见图更密集，因此回环融合，姿态图优化和完整BA任务更加昂贵。互可见图的更高的密度，使局部地图包含更多的关键帧和点，因此局部地图跟踪和局部BA的计算更加耗时。

## 5. Conclusion

We have presented a full SLAM system for monocular, stereo and RGB-D sensors, able to perform relocalization, loop closing and reuse its map in real-time on standard CPUs. We focus on building globally consistent maps for reliable and long-term localization in a wide range of environments as demonstrated in the experiments. The proposed localization mode with the relocalization capability of the system yields a very robust, zero-drift, and ligthweight localization method for known environments. This mode can be useful for certain applications, such as tracking the user viewpoint in virtual reality in a well-mapped space.

我们对单目、立体相机和RGB-D传感器提出了一个完整的SLAM系统，可以进行重定位，回环闭合和地图重用，在标准CPU上可以实时运行。我们聚焦在构建全局一致的地图，在各种环境中可以进行可靠的长期定位，在试验中进行了展示。提出的带有重定位能力的定位模式，对已知的环境得到了一个非常稳健、零漂移、轻量的定位方法。这个模式对特定的应用是有用的，比如在VR中在已经很好建图的空间中跟踪用户视角。

The comparison to the state-of-the-art shows that ORB-SLAM2 achieves in most cases the highest accuracy. In the KITTI visual odometry benchmark ORB-SLAM2 is currently the best stereo SLAM solution. Crucially, compared with the stereo visual odometry methods that have flourished in recent years, ORB-SLAM2 achieves zero-drift localization in already mapped areas.

与目前最好方法的比较展示了，ORB-SLAM2在多数情况下获得了最高的精确度。在KITTI视觉里程计的基准测试中，ORB-SLAM2是目前最好的立体视觉SLAM解决方案。关键是，与近几年蓬勃发展的立体视觉里程计方法相比，ORB-SLAM2在已经建图的区域，获得了零漂移的定位。

Surprisingly our RGB-D results demonstrate that if the most accurate camera localization is desired, bundle adjustment performs better than direct methods or ICP, with the additional advantage of being less computationally expensive, not requiring GPU processing to operate in real-time.

令人惊讶的是，我们的RGB-D结果证明了，如果想要最精确的相机定位，BA比直接方法或ICP性能都要好，而且在计算量上也没有那么大，不需要GPU处理就可以实时运行。

We have released the source code of our system, with examples and instructions so that it can be easily used by other researchers. ORB-SLAM2 is to the best of our knowledge the first open-source visual SLAM system that can work either with monocular, stereo and RGB-D inputs. Moreover our source code contains an example of an augmented reality application using a monocular camera to show the potential of our solution.

我们系统已经开源，有例子和指引，这样其他研究者也可以很容易的使用。据我们所知，ORB-SLAM2是第一个开源的能够以单目、立体相机和RGB-D输入工作的视觉SLAM系统。而且，我们的源码包含了AR应用的一个例子，使用的是单目相机，表明我们的解决方案非常有潜力。

Future extensions might include, to name some examples, non-overlapping multi-camera, fisheye or omnidirectional cameras support, large scale dense fusion, cooperative mapping or increased motion blur robustness.

未来的拓展可能包括，非重叠多相机，鱼眼或全向相机，大规模密集融合，合作建图或对更大的运动模糊增加稳健性。