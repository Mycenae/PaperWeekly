# ORB-SLAM: a Versatile and Accurate Monocular SLAM System

Ra´ul Mur-Artal*, J. M. M. Montiel, Juan D. Tard´os

## 0. Abstract

This paper presents ORB-SLAM, a feature-based monocular SLAM system that operates in real time, in small and large, indoor and outdoor environments. The system is robust to severe motion clutter, allows wide baseline loop closing and relocalization, and includes full automatic initialization. Building on excellent algorithms of recent years, we designed from scratch a novel system that uses the same features for all SLAM tasks: tracking, mapping, relocalization, and loop closing. A survival of the fittest strategy that selects the points and keyframes of the reconstruction leads to excellent robustness and generates a compact and trackable map that only grows if the scene content changes, allowing lifelong operation. We present an exhaustive evaluation in 27 sequences from the most popular datasets. ORB-SLAM achieves unprecedented performance with respect to other state-of-the-art monocular SLAM approaches. For the benefit of the community, we make the source code public.

本文给出了ORB-SLAM，这是一种基于特征的单目SLAM系统，在小型和大型，室内和室外环境中，都可以实时运行。系统对严重的运动杂乱非常稳健，允许很宽基准的回环闭合和重新定位，包括完整的自动的初始化。在近年来的优秀算法之上，我们从头设计了一个新的系统，对所有SLAM任务都使用同样的特征：跟踪，制图，重定位，和回环闭合。选择重建用的点和关键帧的最合适的策略的幸存者，有着非常好的稳健性，生成了一个紧凑、可追踪的图，只有在场景内容变化时，才会增加，可以进行终身的操作。我们对最流行的数据集的27帧中给出了完全的评估。ORB-SLAM与其他的目前最好的单目SLAM方法相比，获得了最好的性能。为使团体受益，我们将代码进行了开源。

**Index Terms** — Lifelong Mapping, Localization, Monocular Vision, Recognition, SLAM

## 1. Introduction

Bundle Adjustment (BA) is known to provide accurate estimates of camera localizations as well as a sparse geometrical reconstruction [1], [2], given that a strong network of matches and good initial guesses are provided. For long time this approach was considered unaffordable for real time applications such as Visual Simultaneous Localisation and Mapping (Visual SLAM). Visual SLAM has the goal of estimating the camera trajectory while reconstructing the environment. Nowadays we know that to achieve accurate results at non-prohibitive computational cost, a real time SLAM algorithm has to provide BA with:

如果给定了很强的匹配网络和很好的初始估计，光束平差(Bundle Adjustment, BA)法可以给出精确的相机位置估计，和稀疏的几何重建。在很长时间里，这种方法被认为无法实现实时应用，比如视觉SLAM。视觉SLAM的目的是，估计相机轨迹，同时重建环境。现在，我们知道了，为了在不算太大的计算代价下得到精确的结果，实时SLAM算法必须要给BA算法提供：

• Corresponding observations of scene features (map points) among a subset of selected frames (keyframes). 在选定的关键帧中，场景特征的对应观察；

• As complexity grows with the number of keyframes, their selection should avoid unnecessary redundancy. 因为复杂度随着关键帧数量的增加而增加，其选择应当避免不必要的冗余；

• A strong network configuration of keyframes and points to produce accurate results, that is, a well spread set of keyframes observing points with significant parallax and with plenty of loop closure matches. 关键帧和点的很强的网络配置，以产生精确的结果，即，分布良好的关键帧，观察的点有明显的视差，并有大量的回环闭合匹配。

• An initial estimation of the keyframe poses and point locations for the non-linear optimization. 关键帧和点位置的初始估计，用于非线性优化。

• A local map in exploration where optimization is focused to achieve scalability. 在探索的局部地图，在此优化关注的是获得可扩展性。

• The ability to perform fast global optimizations (e.g. pose graph) to close loops in real-time. 进行快速全局优化（如姿态图）的能力，以实时闭合回环。

The first real time application of BA was the visual odometry work of Mouragon et. al. [3], followed by the ground breaking SLAM work of Klein and Murray [4], known as Parallel Tracking and Mapping (PTAM). This algorithm, while limited to small scale operation, provides simple but effective methods for keyframe selection, feature matching, point triangulation, camera localization for every frame, and relocalization after tracking failure. Unfortunately several factors severely limit its application: lack of loop closing and adequate handling of occlusions, low invariance to viewpoint of the relocalization and the need of human intervention for map bootstrapping.

BA的首个实时应用是Mouragon等[3]的视觉里程计，然后是Klein和Murray等的SLAM奠基之作[4]，称之为PTAM。这个算法局限在小规模操作上，对每一帧给出简单但有效的关键帧选择，特征匹配，点三角测量，相机定位的方法，在跟踪失败之后重新进行定位。不幸的是，几个因素严重局限了其应用：缺少回环闭合和对遮挡的足够处理，对定位视角的变化缺少不变性，需要人类干预进行地图bootstrapping。

In this work we build on the main ideas of PTAM, the place recognition work of G´alvez-L´opez and Tard´os [5], the scale-aware loop closing of Strasdat et. al [6] and the use of covisibility information for large scale operation [7], [8], to design from scratch ORB-SLAM, a novel monocular SLAM system whose main contributions are:

本文中，我们在PTAM，[5]的位置识别工作，和[6]的感知尺度的回环闭合的主要思想之上，使用互可见信息进行大规模操作[7,8]，从头设计了ORB-SLAM，这是一个新的单目SLAM系统，其主要贡献包括：

• Use of the same features for all tasks: tracking, mapping, relocalization and loop closing. This makes our system more efficient, simple and reliable. We use ORB features [9] which allow real-time performance without GPUs, providing good invariance to changes in viewpoint and illumination. 对所有任务使用了相同的特征：跟踪，建图，重定位和回环闭合。这使整个系统更加高效，简单和可靠。我们使用ORB特征，在没有GPU的情况下可以得到实时性能，对视角和光照的变化有很好的不变性。

• Real time operation in large environments. Thanks to the use of a covisibility graph, tracking and mapping is focused in a local covisible area, independent of global map size. 在很大的环境中可以实时操作。多亏了使用互可见图，跟踪和建图都聚焦在一个局部互可见区域，与全局地图规模无关。

• Real time loop closing based on the optimization of a pose graph that we call the Essential Graph. It is built from a spanning tree maintained by the system, loop closure links and strong edges from the covisibility graph. 实时回环闭合，基于我们称之为Essential Graph的姿态图的优化。这是从系统维护的张树构建而来的，回环链接和很强的边缘由互可见图得到。

• Real time camera relocalization with significant invariance to viewpoint and illumination. This allows recovery from tracking failure and also enhances map reuse. 实时的相机重定位，对视角和光照有明显的不变性。这使得可以从跟踪失败中恢复，可以增强地图重用性。

• A new automatic and robust initialization procedure based on model selection that permits to create an initial map of planar and non-planar scenes. 新的自动稳健初始化过程，基于模型选择，允许创建平面和非平面的初始图。

• A survival of the fittest approach to map point and keyframe selection that is generous in the spawning but very restrictive in the culling. This policy improves tracking robustness, and enhances lifelong operation because redundant keyframes are discarded. 地图点和关键帧的选择的最合适方法的幸存者，在spawning时非常慷慨，在culling时非常有限制性。这种策略改进了跟踪的稳健性，增强了终生操作，因为冗余的关键帧被丢弃了。

We present an extensive evaluation in popular public datasets from indoor and outdoor environments, including hand-held, car and robot sequences. Notably, we achieve better camera localization accuracy than the state of the art in direct methods [10], which optimize directly over pixel intensities instead of feature reprojection errors. We include a discussion in Section IX-B on the possible causes that can make feature-based methods more accurate than direct methods.

我们在流行的公开数据集中给出了广泛的评估，包括室内和室外环境，包括手持，车载和机器人序列。值得注意的是，我们比目前最好的直接方法[10]获得了更好的相机定位准确率，[10]对像素灰度进行直接优化，而不是特征重定位的误差。我们在IX-B节中讨论了基于特征的方法比直接方法更加准确的可能原因。

The loop closing and relocalization methods here presented are based on our previous work [11]. A preliminary version of the system was presented in [12]. In the current paper we add the initialization method, the Essential Graph, and perfect all methods involved. We also describe in detail all building blocks and perform an exhaustive experimental validation.

这里提出的回环闭合和重定位方法是基于我们以前的工作[11]。[12]给出了系统的初级版本。本文中，我们加入了初始化方法，Essential Graph，和所有涉及到的方法。我们还详细描述了所有的建构模块，进行了详尽的试验验证。

To the best of our knowledge, this is the most complete and reliable solution to monocular SLAM, and for the benefit of the community we make the source code public. Demonstration videos and the code can be found in our project webpage.

据我们所知，这是最全面最可靠的单目SLAM解决方案，我们代码进行了开源。项目主页上有验证视频和代码。

## 2. Related Work

### 2.1. Place Recognition

The survey by Williams et al. [13] compared several approaches for place recognition and concluded that techniques based on appearance, that is image to image matching, scale better in large environments than map to map or image to map methods. Within appearance based methods, bags of words techniques [14], such as the probabilistic approach FAB-MAP [15], are to the fore because of their high efficiency. DBoW2 [5] used for the first time bags of binary words obtained from BRIEF descriptors [16] along with the very efficient FAST feature detector [17]. This reduced in more than one order of magnitude the time needed for feature extraction, compared to SURF [18] and SIFT [19] features that were used in bags of words approaches so far. Although the system demonstrated to be very efficient and robust, the use of BRIEF, neither rotation nor scale invariant, limited the system to in-plane trajectories and loop detection from similar viewpoints. In our previous work [11], we proposed a bag of words place recognizer built on DBoW2 with ORB [9]. ORB are binary features invariant to rotation and scale (in a certain range), resulting in a very fast recognizer with good invariance to viewpoint. We demonstrated the high recall and robustness of the recognizer in four different datasets, requiring less than 39ms (including feature extraction) to retrieve a loop candidate from a 10K image database. In this work we use an improved version of that place recognizer, using covisibility information and returning several hypotheses when querying the database instead of just the best match.

Williams等[13]的综述比较了几种位置识别的方法，得出结论，基于外观的技术，即要进行图像匹配的图像，在更大的环境中扩展效果比较好，这是与地图对地图，或图像到地图的方法。在基于外观的方法中，bags-of-words技术[14]，比如概率方法FAB-MAP[15]，因为其效率较高，是较为流行的。DBoW2[5]第一次使用从BRIEF描述子中得到的bags-of-binary-words，与非常高效的FAST特征检测器一起使用。与目前在bags-of-words中使用的SURF或SIFT特征相比，这降低了特征提取所需的时间一个量级以上。虽然系统非常高效、稳健，使用的BRIEF对旋转和缩放都不是不变的，限制了系统必须是平面内的轨迹，和类似视角的回环闭合。在之前的工作中[11]，我们提出了基于带有ORB的DBoW2的bag-of-words位置识别器。ORB是对旋转和缩放（在一定范围内）都不变的二值特征，得到的快速识别器，对视角变化有很好的不变性。我们在4个不同的数据集中，证明了识别器具有高召回率，非常稳健，从一个10K的图像数据库中检索一个回环候选只需要不到39ms（包括特征提取）。在本文中，我们使用这种位置识别器的改进版本，当查询数据库时，使用了互可见性信息，返回几个假设，而不只是最好的匹配。

### 2.2. Map Initialization

Monocular SLAM requires a procedure to create an initial map because depth cannot be recovered from a single image. One way to solve the problem is to initially track a known structure [20]. In the context of filtering approaches, points can be initialized with high uncertainty in depth using an inverse depth parametrization [21], which hopefully will later converge to their real positions. The recent semi-dense work of Engel et al. [10], follows a similar approach initializing the depth of the pixels to a random value with high variance.

单目SLAM需要一个过程来创建初始地图，因为不能从单幅图像中恢复出深度信息。一种解决这个问题的方法是，在开始的时候追踪一个已知的结构[20]。在滤波方法的上下文中，点的初始化，使用逆深度参数化[21]，在深度上会有很高的不确定性，这后来可能会收敛到其真实位置上。最近Eigel等[10]的半密集工作遵循类似的方法，初始化像素的深度为一个高方差的随机值。

Initialization methods from two views either assumes locally scene planarity [4], [22] and recover the relative camera pose from a homography using the method of Faugeras et. al [23], or compute an essential matrix [24], [25] that models planar and general scenes, using the five-point algorithm of Nister [26], which requires to deal with multiple solutions. Both reconstruction methods are not well constrained under low parallax and suffer from a twofold ambiguity solution if all points of a planar scene are closer to one of the camera centers [27]. On the other hand if a non-planar scene is seen with parallax a unique fundamental matrix can be computed with the eight-point algorithm [2] and the relative camera pose can be recovered without ambiguity.

两个视角的初始化方法，要么假设局部场景的平面性[4,22]，用Faugeras等[23]方法，从单映性中恢复相对相机姿态，或计算一个本质矩阵[24,25]，对平面场景和通用场景进行建模，使用的是Nister[26]的5点算法，需要处理多个解。这两种重建方法在视差不够的情况下，都会约束不够，如果一个平面场景中的所有点都与一个相机的中心接近时，都会有双重模糊解[27]。另一方面，如果带有视差来观察一个非平面场景，可以用8点算法[2]来计算一个唯一的基础矩阵，可以毫无疑义的恢复出相对相机姿态。

We present in Section IV a new automatic approach based on model selection between a homography for planar scenes and a fundamental matrix for non-planar scenes. A statistical approach to model selection was proposed by Torr et al. [28]. Under a similar rationale we have developed a heuristic initialization algorithm that takes into account the risk of selecting a fundamental matrix in close to degenerate cases (i.e. planar, nearly planar, and low parallax), favoring the selection of the homography. In the planar case, for the sake of safe operation, we refrain from initializing if the solution has a twofold ambiguity, as a corrupted solution could be selected. We delay the initialization until the method produces a unique solution with significant parallax.

我们在第IV部分提出一种基于模型选择的新的自动方法，包括平面场景的单映性，和非平面场景的基础矩阵。Torr等[28]提出了一种模型选择的统计方法。在类似的原理下，我们提出了一种启发式的初始化算法，考虑了选择基础矩阵在接近降质情况下的风险（即，平面的，接近平面的，和低视差的），倾向于选择单映性。在平面的情况下，为安全操作考虑，如果解有双重疑义，我们就不进行初始化了，因为可能会选择不好的解。我们将初始化推迟到，方法会产生有明显视差的唯一解。

### 2.3. Monocular SLAM

Monocular SLAM was initially solved by filtering [20], [21], [29], [30]. In that approach every frame is processed by the filter to jointly estimate the map feature locations and the camera pose. It has the drawbacks of wasting computation in processing consecutive frames with little new information and the accumulation of linearization errors. On the other hand keyframe-based approaches [3], [4] estimate the map using only selected frames (keyframes) allowing to perform more costly but accurate bundle adjustment optimizations, as mapping is not tied to frame-rate. Strasdat et. al [31] demonstrated that keyframe-based techniques are more accurate than filtering for the same computational cost.

单目SLAM开始是通过滤波进行求解的[20,21,29,30]。在那种方法中，每一帧都通过滤波器进行处理，对地图特征位置和相机姿态进行联合估计。这有一些缺点，在处理没有什么新信息的连续帧的时候，很浪费计算量，线性化误差也会累积。另一方面，基于关键帧的方法[3,4]只使用选定的帧（关键帧）对地图来进行估计，可以进行运算量很大，但是更精确的光束平差优化，因为建图与帧率不是完全捆绑的。Strasdat等[31]证明了，基于关键帧的技术在相同的计算代价下比滤波更加准确。

The most representative keyframe-based SLAM system is probably PTAM by Klein and Murray [4]. It was the first work to introduce the idea of splitting camera tracking and mapping in parallel threads, and demonstrated to be successful for real time augmented reality applications in small environments. The original version was later improved with edge features, a rotation estimation step during tracking and a better relocalization method [32]. The map points of PTAM correspond to FAST corners matched by patch correlation. This makes the points only useful for tracking but not for place recognition. In fact PTAM does not detect large loops, and the relocalization is based on the correlation of low resolution thumbnails of the keyframes, yielding a low invariance to viewpoint.

最有代表性的基于关键帧的SLAM系统可能就是Klein和Murray[4]的PTAM。该文章第一次提出了将相机跟踪和建图分到不同的线程中去的思想，在小型环境的AR应用中非常成功。原始版本后来用边缘的特征进行了改进，这是一种在跟踪时的旋转估计步骤，和更好的重定位方法[32]。PTAM的地图点，对应着由块关联匹配的FAST角点。这使得这些点对于跟踪非常有用，但对位置识别用处不大。实际上，PTAM并没有检测大型回环，重定位是基于关键帧的低分辨率缩略图的相关，对视角的不变性较低。

Strasdat et. al [6] presented a large scale monocular SLAM system with a front-end based on optical flow implemented on a GPU, followed by FAST feature matching and motion-only BA, and a back-end based on sliding-window BA. Loop closures were solved with a pose graph optimization with similarity constraints (7DoF), that was able to correct the scale drift appearing in monocular SLAM. From this work we take the idea of loop closing with 7DoF pose graph optimization and apply it to the Essential Graph defined in Section III-D.

Strasdat等[6]提出了一种大规模单目系统，前端是基于在GPU上实现的光流，然后是FAST特征匹配，和只有运动的BA，后端是基于滑窗的BA。回环闭合是用带有相似度约束(7DoF)的姿态图优化求解的，可以修正单目SLAM的尺度漂移。从这篇文章中，我们采用了其7DoF姿态图优化的回环闭合，将其应用到III-D中定义的Essential Graph中。

Strasdat et. al [7] used the front-end of PTAM, but performed the tracking only in a local map retrieved from a covisibility graph. They proposed a double window optimization back-end that continuously performs BA in the inner window, and pose graph in a limited-size outer window. However, loop closing is only effective if the size of the outer window is large enough to include the whole loop. In our system we take advantage of the excellent ideas of using a local map based on covisibility, and building the pose graph from the covisibility graph, but apply them in a totally redesigned front-end and back-end. Another difference is that, instead of using specific features for loop detection (SURF), we perform the place recognition on the same tracked and mapped features, obtaining robust frame-rate relocalization and loop detection.

Strasdat等[7]使用了PTAM的前端，但只在从互可见性图中得到的局部地图中进行跟踪。他们提出了一种双窗口优化的后端，在内部窗口中不断进行BA，以及在有限大小的外窗中的姿态图。但是，回环闭合的有效之处，只在外窗足够大，包含了整个回环时才可以。在我们的系统中，我们利用了使用基于互可见性的局部地图的优秀思想，从互可见图中构建姿态图，但在一个完全重新涉及的前端和后端中进行应用。另一个差异是，我们没有使用特定的特征进行回环检测(SURF)，我们在同样的跟踪和的映射的特征中进行位置识别，得到稳健的帧率级重定位和回环检测。

Pirker et. al [33] proposed CD-SLAM, a very complete system including loop closing, relocalization, large scale operation and efforts to work on dynamic environments. However map initialization is not mentioned. The lack of a public implementation does not allow us to perform a comparison of accuracy, robustness or large-scale capabilities.

Pirker等[33]提出了CD-SLAM，一个非常完备的系统，包括回环闭合，重定位，大规模操作，和在动态环境中工作的努力。但是地图重新初始化并没有提及。他们的工作缺少公开实现，使我们无法与其比较准确率，稳健性或大规模操作的能力。

The visual odometry of Song et al. [34] uses ORB features for tracking and a temporal sliding window BA back-end. In comparison our system is more general as they do not have global relocalization, loop closing and do not reuse the map. They are also using the known distance from the camera to the ground to limit monocular scale drift.

Song等[34]的视觉里程计，使用ORB特征来跟踪，BA后端使用了时域的滑窗。比较起来，我们的系统更加一般化，因为他们没有全局的重定位，回环闭合，还没有重新使用地图。他们还使用了已知的相机到地面的距离，来限制单目尺度漂移。

Lim et. al [25], work published after we submitted our preliminary version of this work [12], use also the same features for tracking, mapping and loop detection. However the choice of BRIEF limits the system to in-plane trajectories. Their system only tracks points from the last keyframe so the map is not reused if revisited (similar to visual odometry) and has the problem of growing unbounded. We compare qualitatively our results with this approach in section VIII-E.

Lim等[25]使用了一样的特征进行跟踪，建图和回环检测。但是选择了BRIEF，将系统限制在了平面轨迹中。其系统只跟踪最后一个关键帧中的点，这样如果重新访问的话，地图就没有重新使用（与视觉里程计类似），而且还有增长无限制的问题。我们在VIII-E中与该方法进行了量化对比。

The recent work of Engel et. al [10], known as LSD-SLAM, is able to build large scale semi-dense maps, using direct methods (i.e. optimization directly over image pixel intensities) instead of bundle adjustment over features. Their results are very impressive as the system is able to operate in real time, without GPU acceleration, building a semi-dense map, with more potential applications for robotics than the sparse output generated by feature-based SLAM. Nevertheless they still need features for loop detection and their camera localization accuracy is significantly lower than in our system and PTAM, as we show experimentally in Section VIII-B. This surprising result is discussed in Section IX-B.

Engel等[10]的最近工作，也称为LSD-SLAM，可以构建大规模半密集地图，使用了直接方法（即，对图像像素灰度进行直接优化），而不是在特征上进行BA。他们的结果非常令人印象深刻，因为系统可以实时运算，不需要GPU加速，构建了一个半密集地图，与基于特征的SLAM生成的稀疏输出相比，更能应用于机器人。尽管如此，他们仍然需要特征进行回环检测，其相机定位精度比我们的系统和PTAM明显更低，我们在VIII-B通过试验给出了结果，在IX-B中进行讨论。

In a halfway between direct and feature-based methods is the semi-direct visual odometry SVO of Forster et al. [22]. Without requiring to extract features in every frame they are able to operate at high frame-rates obtaining impressive results in quadacopters. However no loop detection is performed and the current implementation is mainly thought for downward looking cameras.

在直接法和基于特征方法之间，还有半直接的视觉里程计，即Forster等[22]的SVO。他们不需要在每一帧中提取特征，可以以很高的帧率进行运行，在无人机上得到令人印象深刻的结果。但是，没有进行回环检测，目前的实现主要是用于向下看的相机。

Finally we want to discuss about keyframe selection. All visual SLAM works in the literature agree that running BA with all the points and all the frames is not feasible. The work of Strasdat et al. [31] showed that the most cost-effective approach is to keep as much points as possible, while keeping only non-redundant keyframes. The PTAM approach was to insert keyframes very cautiously to avoid an excessive growth of the computational complexity. This restrictive keyframe insertion policy makes the tracking fail in hard exploration conditions. Our survival of the fittest strategy achieves unprecedented robustness in difficult scenarios by inserting keyframes as quickly as possible, and removing later the redundant ones, to avoid the extra cost.

最后，我们讨论一下关键帧选择。文献中所有的视觉SLAM都同意，对所有帧所有点都进行BA是不可行的。Strasdat等[31]表明，最有效的方法是保留尽可能多的点，单只保留非冗余的关键帧。PTAM方法是非常谨慎的插入关键帧，以避免计算复杂度的过度增长。这种有限制性的关键帧插入策略，在很难的探索条件下，会使得跟踪失败。我们存活下来的最适应策略，通过尽可能块的插入关键帧，并然后移除冗余的关键帧，以避免额外的代价，在困难场景中得到了前所未有的稳健性。

## 3. System Overview

### 3.1. Feature Choice

One of the main design ideas in our system is that the same features used by the mapping and tracking are used for place recognition to perform frame-rate relocalization and loop detection. This makes our system efficient and avoids the need to interpolate the depth of the recognition features from near SLAM features as in previous works [6], [7]. We requiere features that need for extraction much less than 33ms per image, which excludes the popular SIFT (∼ 300ms) [19], SURF (∼ 300ms) [18] or the recent A-KAZE (∼ 100ms) [35]. To obtain general place recognition capabilities, we require rotation invariance, which excludes BRIEF [16] and LDB [36].

我们系统中的一个主要涉及思想是，建图和追踪使用的特征，和进行帧率级重定位和回环检测的位置识别所使用的特征是一样的。这使我们的系统更高效，之前的工作[6,7]要从附近的SLAM特征对识别的特征的深度进行插值，我们的系统则没有这个必要。我们需要进行提取的特征每幅图像耗时小于33ms，这就排除了流行的SIFT (∼ 300ms) [19], SURF (∼ 300ms) [18] 或最近的 A-KAZE (∼ 100ms) [35]。为得到通用位置识别能力，我们需要旋转不变性，这就排除了BRIEF [16]和LDB [36]。

We chose ORB [9], which are oriented multi-scale FAST corners with a 256 bits descriptor associated. They are extremely fast to compute and match, while they have good invariance to viewpoint. This allows to match them from wide baselines, boosting the accuracy of BA. We already shown the good performance of ORB for place recognition in [11]. While our current implementation make use of ORB, the techniques proposed are not restricted to these features.

我们选择了ORB[9]，这是有向的多尺度FAST角点，有256 bits的描述子。它们计算和匹配起来非常快速，而且对视角有很好的不变性。这就可以从很宽的基准上对其进行匹配，提升BA的准确度。我们已经证明了ORB在位置识别中的好性能[11]。我们目前的实现是基于ORB的，但这里提出的技术并不局限于这些特征。

### 3.2. Three Threads: Tracking, Local Mapping and Loop Closing

Our system, see an overview in Fig. 1, incorporates three threads that run in parallel: tracking, local mapping and loop closing. The tracking is in charge of localizing the camera with every frame and deciding when to insert a new keyframe. We perform first an initial feature matching with the previous frame and optimize the pose using motion-only BA. If the tracking is lost (e.g. due to occlusions or abrupt movements), the place recognition module is used to perform a global relocalization. Once there is an initial estimation of the camera pose and feature matchings, a local visible map is retrieved using the covisibility graph of keyframes that is maintained by the system, see Fig. 2(a) and Fig. 2(b). Then matches with the local map points are searched by reprojection, and camera pose is optimized again with all matches. Finally the tracking thread decides if a new keyframe is inserted. All the tracking steps are explained in detail in Section V. The novel procedure to create an initial map is presented in Section IV.

我们系统的概览如图1所示，包括三个并行运行的线：跟踪，局部建图和回环闭合。跟踪负责用每一帧进行相机定位，并决定什么时候插入一个新的关键帧。我们首先进行与之前的帧的初始特征匹配，使用只有运动的BA进行姿态优化。如果跟踪丢失（如，因为遮挡或突然的运动），位置识别模块就用于进行全局重新定位。一旦有了相机姿态的初始估计和特征匹配，使用系统维护的关键帧的互可见图来获取一个局部可见地图，见图2a和图2b。然后通过重新投影来搜索与局部地图点的匹配，然后与所有的匹配对相机的姿态进行优化。最后，跟踪线程来决定是否要插入一个新的关键帧。所有的跟踪步骤在第5部分详细解释。创建初始地图的新过程在第4部分给出。

The local mapping processes new keyframes and performs local BA to achieve an optimal reconstruction in the surroundings of the camera pose. New correspondences for unmatched ORB in the new keyframe are searched in connected keyframes in the covisibility graph to triangulate new points. Some time after creation, based on the information gathered during the tracking, an exigent point culling policy is applied in order to retain only high quality points. The local mapping is also in charge of culling redundant keyframes. We explain in detail all local mapping steps in Section VI.

局部建图要处理新的关键帧，进行局部BA以对相机姿态的周围获得最优的重建。在新的关键帧中未匹配的ORB，在互可见图中的连接的关键帧中搜索新的对应性，以对新的点进行三角定位。在创建之后一段时间，基于跟踪时收集的信息，就应用一个苛刻的点挑选策略，以只保留最高质量的点。局部建图还负责挑选冗余的关键帧。我们在第6部分详细解释了所有的局部建图步骤。

The loop closing searches for loops with every new keyframe. If a loop is detected, we compute a similarity transformation that informs about the drift accumulated in the loop. Then both sides of the loop are aligned and duplicated points are fused. Finally a pose graph optimization over similarity constraints [6] is performed to achieve global consistency. The main novelty is that we perform the optimization over the Essential Graph, a sparser subgraph of the covisibility graph which is explained in Section III-D. The loop detection and correction steps are explained in detail in Section VII.

回环闭合对每个新的关键帧搜索回环。如果检测到一个回环，我们计算一个相似度变换，了解回环中积累的漂移。然后回环的两边进行对齐，并融合重复的点。最后，进行对相似性约束进行姿态图优化，以获得全局一致性。主要的创新是，我们在Essential Graph上进行优化，这是互可见图的一个更稀疏的子图，在3.4中进行解释。回环检测和修正步骤在第7部分进行详细解释。

We use the Levenberg-Marquardt algorithm implemented in g2o [37] to carry out all optimizations. In the Appendix we describe the error terms, cost functions, and variables involved in each optimization.

我们使用g2o中实现的Levenberg-Marquardt算法来进行所有优化。附录中我们描述了在每个优化中涉及到的误差项，代价函数，和变量。

### 3.3. Map Points, KeyFrames and their Selection

Each map point $p_i$ stores: 每个地图点$p_i$存储了：

• Its 3D position $X_{w,i}$ in the world coordinate system. 在世界坐标系中的3D位置。

• The viewing direction $n_i$, which is the mean unit vector of all its viewing directions (the rays that join the point with the optical center of the keyframes that observe it). 观察方向$n_i$，是所有观察方向的平均单位向量（连接这个点，和观察它的关键帧的光学中心的射线）。

• A representative ORB descriptor $D_i$, which is the associated ORB descriptor whose hamming distance is minimum with respect to all other associated descriptors in the keyframes in which the point is observed. 一个代表性的ORB描述子$D_i$，在这个点被观察的所有的关键帧中，与这些相关联的描述子的hamming距离是最小的关联ORB描述子。

• The maximum d_max and minimum d_min distances at which the point can be observed, according to the scale invariance limits of the ORB features. 根据ORB特征的尺度不变性限制，这个点被观察的最大距离d_max和最小距离d_min。

Each keyframe $K_i$ stores: 每个关键帧$K_i$存储的是：

• The camera pose $T_{iw}$, which is a rigid body transformation that transforms points from the world to the camera coordinate system. 相机姿态$T_{iw}$，就是将点从世界坐标系到相机坐标系的刚体变换。

• The camera intrinsics, including focal length and principal point. 相机内参，包括焦距和主点。

• All the ORB features extracted in the frame, associated or not to a map point, whose coordinates are undistorted if a distortion model is provided. 帧中提取的所有ORB特征，无论是否与地图点进行关联，如果给出了畸变模型，其坐标点就是无畸变的。

Map points and keyframes are created with a generous policy, while a later very exigent culling mechanism is in charge of detecting redundant keyframes and wrongly matched or not trackable map points. This permits a flexible map expansion during exploration, which boost tracking robustness under hard conditions (e.g. rotations, fast movements), while its size is bounded in continual revisits to the same environment, i.e. lifelong operation. Additionally our maps contain very few outliers compared with PTAM, at the expense of containing less points. Culling procedures of map points and keyframes are explained in Sections VI-B and VI-E respectively.

地图点和关键帧是用丰富的策略创建的，后来有一个非常苛刻的挑选机制，来负责检测冗余的关键帧和错误匹配或没有跟踪的地图点。这使得在探索的时候，地图扩展会非常灵活，在非常困难的条件下提升了跟踪的稳健性（如，旋转，快速运动），而其大小在对同样环境的连续重复访问时是有上限的，即，终生操作。另外，我们的地图与PTAM相比，外点很少，而其代价是包含的点更少。地图点和关键帧的挑选过程在6.2和6.5中分别进行解释。

### 3.4. Covisibility Graph and Essential Graph

Covisibility information between keyframes is very useful in several tasks of our system, and is represented as an undirected weighted graph as in [7]. Each node is a keyframe and an edge between two keyframes exists if they share observations of the same map points (at least 15), being the weight θ of the edge the number of common map points.

关键帧之间的互可见信息，在我们系统中的几个任务中都非常有用，表示为[7]中的一个无向加权图。每个节点都是一个关键帧，如果两个关键帧之间有共同的地图点（至少15个），那么这两个节点之间就有一条边，边的权重θ就是共同的地图点的数量。

In order to correct a loop we perform a pose graph optimization [6] that distributes the loop closing error along the graph. In order not to include all the edges provided by the covisibility graph, which can be very dense, we propose to build an Essential Graph that retains all the nodes (keyframes), but less edges, still preserving a strong network that yields accurate results. The system builds incrementally a spanning tree from the initial keyframe, which provides a connected subgraph of the covisibility graph with minimal number of edges. When a new keyframe is inserted, it is included in the tree linked to the keyframe which shares most point observations, and when a keyframe is erased by the culling policy, the system updates the links affected by that keyframe. The Essential Graph contains the spanning tree, the subset of edges from the covisibility graph with high covisibility (θ_min = 100), and the loop closure edges, resulting in a strong network of cameras. Fig. 2 shows an example of a covisibility graph, spanning tree and associated essential graph. As shown in the experiments of Section VIII-E, when performing the pose graph optimization, the solution is so accurate that an additional full bundle adjustment optimization barely improves the solution. The efficiency of the essential graph and the influence of the θ_min is shown at the end of Section VIII-E.

为修正回环，我们进行姿态图优化[6]，将回环闭合误差分布到整个图中。为不包含互可见图中的所有边，这可能是很密集的边，我们提出构建Essential Graph，保留了所有的节点（关键帧），但是边要更少一些，可以仍然保留很强的网络，得到准确的结果。这个系统从初始的关键帧中逐渐构建了一个张树，给出了互可见图的一个连接子图，边的数量最小。当插入一个新的关键帧时，它被包含在共享最多点的关键帧所项链的树中，当挑选策略去掉一个关键帧时，系统会更新受到这个关键帧所影响的连接。Essential Graph包含了张树，互可见图中边有最高互可见性(θ_min = 100)的子集，回环闭合边，得到相机的一个很强的网络。图2展示了互可见图、张树和关联的essential graph的一个例子。在8.5的试验中表明，当进行姿态图优化时，解非常的精确，额外的完全BA优化基本上对解没有改进。Essential Graph的效率和θ_min的影响在8.5的最后也会进行展示。

### 3.5. Bags of Words Place Recognition

The system has embedded a bags of words place recognition module, based on DBoW2 [5], to perform loop detection and relocalization. Visual words are just a discretization of the descriptor space, which is known as the visual vocabulary. The vocabulary is created offline with the ORB descriptors extracted from a large set of images. If the images are general enough, the same vocabulary can be used for different environments getting a good performance, as shown in our previous work [11]. The system builds incrementally a database that contains an invert index, which stores for each visual word in the vocabulary, in which keyframes it has been seen, so that querying the database can be done very efficiently. The database is also updated when a keyframe is deleted by the culling procedure.

系统还包含一个bag-of-words位置识别模块，基于DBoW2 [5]，以进行回环检测和重定位。视觉words只是描述子空间的一个离散化，也称之为视觉词典。词典是用从大量图像中提取出的ORB描述子中离线创建的。如果图像非常通用，同样的词典可以用于不同的环境，得到很好的性能，这在我们之前的工作中已有展示[11]。系统逐渐构建了一个数据库，包含了一个逆索引，对词典中的每个视觉word都进行了存储，在它观察过的每个关键帧中，这样查询数据库就可以非常高效的进行。当一个关键帧由挑选过程删除时，数据库也进行更新。

Because there exists visual overlap between keyframes, when querying the database there will not exist a unique keyframe with a high score. The original DBoW2 took this overlapping into account, adding up the score of images that are close in time. This has the limitation of not including keyframes viewing the same place but inserted at a different time. Instead we group those keyframes that are connected in the covisibility graph. In addition our database returns all keyframe matches whose scores are higher than the 75% of the best score.

因为在关键帧中存在视觉重叠，当查询数据库时，会有多个关键帧分数都很高。原始DBoW2将这个重叠纳入了考虑，将在时间上接近的图像的分数相加。这有一个局限，没有将在同一位置单不同时间观察的关键帧视角包括进来。我们则用互可见图中相连的关键帧来进行分组。此外，我们的数据库会返回所有的分数高于最高分75%的的所有关键帧。

An additional benefit of the bags of words representation for feature matching was reported in [5]. When we want to compute the correspondences between two sets of ORB features, we can constraint the brute force matching only to those features that belong to the same node in the vocabulary tree at a certain level (we select the second out of six), speeding up the search. We use this trick when searching matches for triangulating new points, and at loop detection and relocalization. We also refine the correspondences with an orientation consistency test, see [11] for details, that discards outliers ensuring a coherent rotation for all correspondences.

[5]中给出了进行特征匹配的Bag-of-words表示的另外的好处。当想要计算两个ORB特征集合的对应性时，我们会只对在特定级别（我们从6个中选了第二个）上在词典树种属于同样节点的特征进行暴力匹配，这样就加速的搜索。我们在搜索用于对新的点进行三角测量的匹配时，和在回环检测和重新定位时，使用这个技巧。我们还用方向一致性测试精炼了对应性，细节见[11]，丢弃外点则保证了对所有对应性的一致旋转。

## 4. Automatic Map Initialization

The goal of the map initialization is to compute the relative pose between two frames to triangulate an initial set of map points. This method should be independent of the scene (planar or general) and should not require human intervention to select a good two-view configuration, i.e. a configuration with significant parallax. We propose to compute in parallel two geometrical models, a homography assuming a planar scene and a fundamental matrix assuming a non-planar scene. We then use a heuristic to select a model and try to recover the relative pose with a specific method for the selected model. Our method only initializes when it is certain that the two-view configuration is safe, detecting low-parallax cases and the well-known twofold planar ambiguity [27], avoiding to initialize a corrupted map. The steps of our algorithm are:

地图初始化的目标是，计算两帧之间的相对姿态，以进行三角测量，得到地图点的初始集合。这个方法应当是独立于场景的（平面的或通用的），不应当需要人的干预，就可以选择很好的双视角配置，即，有明显视差的配置。我们提出并行计算两个几何模型，一个是单应矩阵，假设是平面场景，还有一个是基础矩阵，假设是非平面场景。我们然后使用启发式来选择一个模型，试图对选择的模型用特定的方法恢复相对姿态。我们的方法只在确定，双视角配置是安全的情况下（检测低视差情况和著名的双倍平面疑义性），才进行初始化。避免初始化到一个损坏的地图。我们算法的步骤如下：

1) Find initial correspondences: 找到初始的对应性：

Extract ORB features (only at the finest scale) in the current frame $F_c$ and search for matches $x_c ↔ x_r$ in the reference frame $F_r$. If not enough matches are found, reset the reference frame.

在当前帧$F_c$中，只在最精细的尺度提取ORB特征，在参考帧$F_r$中搜索匹配$x_c ↔ x_r$。如果没有找到足够的匹配，就重置参考帧。

2) Parallel computation of the two models: 两个模型的并行计算：

Compute in parallel threads a homography $H_{cr}$ and a fundamental matrix $F_{cr}$: 用并行的线程计算一个单应性$H_{cr}$和一个基础矩阵$F_{cr}$：

$$x_c = H_{cr} x_r, x^T_c F_{cr} x_r = 0$$(1)

with the normalized DLT and 8-point algorithms respectively as explained in [2] inside a RANSAC scheme. To make homogeneous the procedure for both models, the number of iterations is prefixed and the same for both models, along with the points to be used at each iteration, 8 for the fundamental matrix, and 4 of them for the homography. At each iteration we compute a score $S_M$ for each model M (H for the homography, F for the fundamental matrix):

分别使用的是[2]中带有RANSAC机制的归一化DLT和8点算法。为使得两个模型的计算过程同质，迭代次数预先固定，对两个模型都一样，在每次迭代中要使用的点也似乎一样的，计算基础矩阵是8个点，计算单应性是4个点。在每次迭代中，我们对每个模型M计算一个分数$S_M$（H代表单应性，F代表基础矩阵）。

$$S_M = \sum_i (ρ_M(d_{cr}^2(x_c^i, x_r^i, M)) + ρ_M(d_{rc}^2(x_c^i, x_r^i, M))), ρ_M(d^2) = \left \{ \begin{matrix} Γ−d^2 & if & d^2<T_M \\ 0 & if & d^2≥T_M \end{matrix} \right.$$(2)

where $d^2_{cr}$ and $d^2_{rc}$ are the symmetric transfer errors [2] from one frame to the other. $T_M$ is the outlier rejection threshold based on the $χ^2$ test at 95% (TH = 5.99, TF = 3.84, assuming a standard deviation of 1 pixel in the measurement error). Γ is defined equal to $T_H$ so that both models score equally for the same d in their inlier region, again to make the process homogeneous.

其中$d^2_{cr}$和$d^2_{rc}$是从一帧到另一帧的对称迁移误差[2]。$T_M$是基于$χ^2$测试在95%处的外点拒绝阈值（TH = 5.99, TF = 3.84，假设测量误差有1个像素的标准偏差）。Γ定义等于$T_H$，这样两个模型对内点区域中同样的d评分相同，再次使得这个过程同质化。

We keep the homography and fundamental matrix with highest score. If no model could be found (not enough inliers), we restart the process again from step 1.

我们保留最高分的单应性和基础矩阵。如果找不到模型（没有足够的内点），我们就从步骤1重启整个过程。

3) Model selection: 模型选择：

If the scene is planar, nearly planar or there is low parallax, it can be explained by a homography. However a fundamental matrix can also be found, but the problem is not well constrained [2] and any attempt to recover the motion from the fundamental matrix would yield wrong results. We should select the homography as the reconstruction method will correctly initialize from a plane or it will detect the low parallax case and refuse the initialization. On the other hand a non-planar scene with enough parallax can only be explained by the fundamental matrix, but a homography can also be found explaining a subset of the matches if they lie on a plane or they have low parallax (they are far away). In this case we should select the fundamental matrix. We have found that a robust heuristic is to compute:

如果场景是平面的，接近平面的，或视差很小，就可以用单应性来进行解释。但是，也可以找到一个基础矩阵，但是问题并不是良好约束的[2]，想要从基础矩阵中恢复出运动，都会得到错误的结果。我们应当选择单应性，因为重建方法会正确的从平面进行初始化，否则就会检测到低视差的情况，拒绝初始化。另一方面，有足够视差的非平面场景只能用基础矩阵进行解释，但也可以找到一个单应性，如果匹配的一个子集是在一个平面上，或视差很低，就可以用单应性解释这部分。在这种情况下，我们就应当选择基础矩阵。我们发现，一个稳健的启发式是，计算下式：

$$R_H = \frac {S_H} {S_H + S_F}$$(3)

and select the homography if $R_H$>0.45, which adequately captures the planar and low parallax cases. Otherwise, we select the fundamental matrix.

如果$R_H$>0.45，就选择单应性，这足可以捕获平面和低视差的情况。否则，我们就选择基础矩阵。

4) Motion and Structure from Motion recovery: 运动恢复得到运动和结构：

Once a model is selected we retrieve the motion hypotheses associated. In the case of the homography we retrieve 8 motion hypotheses using the method of Faugeras et. al [23]. The method proposes cheriality tests to select the valid solution. However these tests fail if there is low parallax as points easily go in front or back of the cameras, which could yield the selection of a wrong solution. We propose to directly triangulate the eight solutions, and check if there is one solution with most points seen with parallax, in front of both cameras and with low reprojection error. If there is not a clear winner solution, we do not initialize and continue from step 1. This technique to disambiguate the solutions makes our initialization robust under low parallax and the twofold ambiguity configuration, and could be considered the key of the robustness of our method.

一旦选择了一个模型，我们就获取关联的运动假设。在单应性的情况下，我们使用Faugeras等[23]的方法获取8运动假设。该方法提出cheriality测试来选择有效解。但是，如果视差很低，这个测试就会失败，因为点会很容易跑到相机的前面或后面，使得这个选择是一个错误的解。我们提出直接对这8个解进行三角定位，并检查是否有一个解，会使多数点都是有视差，在两个相机前，重投影误差比较低而观测得到的。如果没有明确的获胜的解，我们就不进行初始化，从步骤1再开始。这种去除解的疑义的技术，使我们的初始化在低视差和双重疑义配置下也很稳健，可以认为是我们方法稳健性的关键。

In the case of the fundamental matrix, we convert it in an essential matrix using the calibration matrix K: 在基础矩阵的情况下，我们使用校准矩阵K将其转换成本征矩阵：

$$E_{rc} = K^T F_{rc} K$$(4)

and then retrieve 4 motion hypotheses with the singular value decomposition method explained in [2]. We triangulate the four solutions and select the reconstruction as done for the homography.

然后用[2]中的SVD分解来得到4运动假设。我们对这4个解进行三角定位，用单应性中的方法选择重建结果。

5) Bundle adjustment: 光束平差

Finally we perform a full BA, see the Appendix for details, to refine the initial reconstruction. 最后我们进行完整的BA，详见附录，来提炼初始的重建。

An example of a challenging initialization in the outdoor NewCollege robot sequence [39] is shown in Fig. 3. It can be seen how PTAM and LSD-SLAM have initialized all points in a plane, while our method has waited until there is enough parallax, initializing correctly from the fundamental matrix.

图3给出了室外NewCollege机器人序列的一个有挑战的初始化的例子。可以看到，PTAM和LSD-SLAM将所有点都初始化到了一个平面上，而我们的方法一直在等待直到有了足够的视差，从基础矩阵中进行了正确的初始化。

## 5. Tracking

In this section we describe the steps of the tracking thread that are performed with every frame from the camera. The camera pose optimizations, mentioned in several steps, consist in motion-only BA, which is described in the Appendix.

本章中，我们描述了跟踪线程的步骤，这在相机的每一帧中都进行。在几个步骤中提到的相机姿态优化，只包含运动的BA，在附录中描述。

### 5.1. ORB Extraction

We extract FAST corners at 8 scale levels with a scale factor of 1.2. For image resolutions from 512 × 384 to 752 × 480 pixels we found suitable to extract 1000 corners, for higher resolutions, as the 1241 × 376 in the KITTI dataset [40] we extract 2000 corners. In order to ensure an homogeneous distribution we divide each scale level in a grid, trying to extract at least 5 corners per cell. Then we detect corners in each cell, adapting the detector threshold if not enough corners are found. The amount of corners retained per cell is also adapted if some cells contains no corners (textureless or low contrast). The orientation and ORB descriptor are then computed on the retained FAST corners. The ORB descriptor is used in all feature matching, in contrast to the search by patch correlation in PTAM.

我们在8个尺度级中提取FAST角点，缩放系数为1.2。对于从512 × 384到752 × 480分辨率的图像，我们发现提取1000个角点是合适的，对于更高的分辨率，比如KITTI数据集中的1241 × 376，我们提取2000个角点。为确保同质分布，我们将每个尺度级别分成一个网格，在每个单元中至少提取出5个角点。然后我们在每个单元中检测角点，如果没有找到足够的角点，就调整检测器的阈值。如果有的单元不包含角点（无纹理，或低对比度），那么每个单元中保留的角点数量，也进行调整。在保留的FAST角点上计算方向和ORB描述子。所有的特征匹配中都使用ORB描述子，比较之下，PTAM中是用块关联进行搜索。

### 5.2. Initial Pose Estimation from Previous Frame

If tracking was successful for last frame, we use a constant velocity motion model to predict the camera pose and perform a guided search of the map points observed in the last frame. If not enough matches were found (i.e. motion model is clearly violated), we use a wider search of the map points around their position in the last frame. The pose is then optimized with the found correspondences.

如果上一帧的跟踪是成功的，我们使用常数速度模型来预测相机姿态，然后对在上一帧中的地图点进行引导搜索。如果没有找到足够的匹配（即，明显违反了运动模型），我们在上一帧中地图点位置附近进行更宽的搜索。然后用找到的对应性，对姿态进行优化。

### 5.3. Initial Pose Estimation via Global Relocalization

If the tracking is lost, we convert the frame into bag of words and query the recognition database for keyframe candidates for global relocalization. We compute correspondences with ORB associated to map points in each keyframe, as explained in section III-E. We then perform alternatively RANSAC iterations for each keyframe and try to find a camera pose using the PnP algorithm [41]. If we find a camera pose with enough inliers, we optimize the pose and perform a guided search of more matches with the map points of the candidate keyframe. Finally the camera pose is again optimized, and if supported with enough inliers, tracking procedure continues.

如果跟踪丢失，我们将帧转换成bag-of-words，查询识别数据库，以求得到全局重定位的关键帧。我们用ORB计算与每个关键帧相关联的地图点的对应性，如3.5解释。然后我们对每个关键帧进行另外的RANSAC迭代，使用PnP算法来试图找到一个相机姿态[41]。如果我们找到了一个有足够内点的相机姿态，我们对姿态进行优化，用候选帧上的地图点搜索更多的匹配。最后相机姿态再次进行优化，如果有足够的内点支持，那么就继续进行跟踪。

### 5.4. Track Local Map

Once we have an estimation of the camera pose and an initial set of feature matches, we can project the map into the frame and search more map point correspondences. To bound the complexity in large maps, we only project a local map. This local map contains the set of keyframes K1, that share map points with the current frame, and a set K2 with neighbors to the keyframes K1 in the covisibility graph. The local map also has a reference keyframe Kref ∈ K1 which shares most map points with current frame. Now each map point seen in K1 and K2 is searched in the current frame as follows:

一旦我们对相机姿态有了估计，并有了初始的特征匹配集，我们可以将地图投影到帧中，并搜索更多的地图点对应性。为限制大型地图的复杂度，我们只投影一个局部地图。这个局部地图包含的关键帧K1的集合，与当前的帧贡献很多地图点，还有一个集合K2，是在互可见图中K1帧的邻域中。局部地图还有一个参考关键帧Kref ∈ K1，与当前帧共享的地图点最多。现在在K1和K2中看到的每个地图点，按照下面的过程在当前帧中进行搜索：

1) Compute the map point projection x in the current frame. Discard if it lays out of the image bounds. 计算在当前帧的地图点投影x，如果超出了图像的边界，就丢弃之；
2) Compute the angle between the current viewing ray v and the map point mean viewing direction n. Discard if v · n < cos(60◦). 计算当前的视线v和地图点平均观察方向n之间的夹角。如果v · n < cos(60◦)，就丢弃该点。
3) Compute the distance d from map point to camera center. Discard if it is out of the scale invariance region of the map point $d \notin [dmin, dmax]$. 计算从地图点到相机中心的距离d。如果超出了地图点的尺度不变性区域$d \notin [dmin, dmax]$，就抛弃这个点。
4) Compute the scale in the frame by the ratio d/dmin. 通过比率d/dmin来计算帧中的尺度。
5) Compare the representative descriptor D of the map point with the still unmatched ORB features in the frame, at the predicted scale, and near x, and associate the map point with the best match. 用帧中仍然未匹配的ORB特征，在预测的尺度，在x附近，计算地图点的代表性描述子D，并将地图点与最佳匹配进行关联。

The camera pose is finally optimized with all the map points found in the frame. 最后，相机姿态用帧中找到的所有地图点进行优化。

### 5.5. New Keyframe Decision

The last step is to decide if the current frame is spawned as a new keyframe. As there is a mechanism in the local mapping to cull redundant keyframes, we will try to insert keyframes as fast as possible, because that makes the tracking more robust to challenging camera movements, typically rotations. To insert a new keyframe all the following conditions must be met:

最后一步是，决定当前帧是否是一个新的关键帧。因为有一个在局部建图中选择冗余关键帧的机制，我们会决定尽快的插入关键帧，因为这会使得跟踪对有挑战的相机运动更加稳健，典型的是旋转。为插入新的关键帧，必须满足下面的所有条件：

1) More than 20 frames must have passed from the last global relocalization. 从最后一次全局重定位，已经过去了超过20帧。
2) Local mapping is idle, or more than 20 frames have passed from last keyframe insertion. 局部建图是空闲的，或从最后一次关键帧插入，已经过去了超过20帧。
3) Current frame tracks at least 50 points. 当前帧跟踪了至少50个点。
4) Current frame tracks less than 90% points than K_ref. 当前帧跟踪的关键点少于K_ref的90%。

Instead of using a distance criterion to other keyframes as PTAM, we impose a minimum visual change (condition 4). Condition 1 ensures a good relocalization and condition 3 a good tracking. If a keyframe is inserted when the local mapping is busy (second part of condition 2), a signal is sent to stop local bundle adjustment, so that it can process as soon as possible the new keyframe.

PTAM这样的系统使用的是对其他关键帧的距离的准则，我们则是加入了最小视觉变化（条件4）。条件1确保了很好的重定位，条件3确保了很好的跟踪。如果一个关键帧的插入，是在局部建图很忙的时候（条件2的第2部分），就发送一个信号，停止局部光束平差，这样就可以在新的关键帧可用的时候尽快进行处理。

## 6. Local Mapping

In this section we describe the steps performed by the local mapping with every new keyframe Ki. 本章中，我们描述了用每个新的关键帧Ki的局部建图所进行的步骤。

### 6.1. KeyFrame Insertion

At first we update the covisibility graph, adding a new node for Ki and updating the edges resulting from the shared map points with other keyframes. We then update the spanning tree linking Ki with the keyframe with most points in common. We then compute the bags of words representation of the keyframe, that will help in the data association for triangulating new points.

首先，我们更新互可见图，加入了一个新节点Ki，更新由于与其他关键帧有共享的地图点而导致的边。然后我们更新连接Ki与有最多公共点的关键帧之间的张树。然后我们计算关键帧的bags-of-words表示，在对新点进行三角定位时，对相关的数据关联会有帮助。

### 6.2. Recent Map Points Culling

Map points, in order to be retained in the map, must pass a restrictive test during the first three keyframes after creation, that ensures that they are trackable and not wrongly triangulated, i.e due to spurious data association. A point must fulfill these two conditions: 地图点要保留在地图中，在创建后必须在前3个关键帧中通过限制性测试，这确保了它们是可跟踪的，没有进行错误的三角定位，即，由于错误的数据关联所导致。一个点必须满足这两个条件：

1) The tracking must find the point in more than the 25% of the frames in which it is predicted to be visible. 在这个点被预测可见的帧中，至少25%的帧中要通过跟踪找到。
2) If more than one keyframe has passed from map point creation, it must be observed from at least three keyframes. 如果多于一个关键帧从地图点创建中通过，就必须从至少3个关键帧中观察到。

Once a map point have passed this test, it can only be removed if at any time it is observed from less than three keyframes. This can happen when keyframes are culled and when local bundle adjustment discards outlier observations. This policy makes our map contain very few outliers.

一旦一个地图点通过了这个测试，如果在任意时刻，在少于三个关键帧中观察到这个点，这是才能将其移除。这在关键帧被选择移除后可以发生，在局部光束平差抛弃了外点观察后也可以发生。这个策略使我们的地图包含非常少的外点。

### 6.3. New Map Point Creation

New map points are created by triangulating ORB from connected keyframes Kc in the covisibility graph. For each unmatched ORB in Ki we search a match with other unmatched point in other keyframe. This matching is done as explained in Section III-E and discard those matches that do not fulfill the epipolar constraint. ORB pairs are triangulated, and to accept the new points, positive depth in both cameras, parallax, reprojection error and scale consistency are checked. Initially a map point is observed from two keyframes but it could be matched in others, so it is projected in the rest of connected keyframes, and correspondences are searched as detailed in section V-D.

在互可见图中相连的关键帧Kc上，对ORB特征点进行三角测量，创建新的地图点。对Ki中每个未匹配ORB，我们在其他关键帧中与其他未匹配的点搜索一个匹配。这个匹配过程如3.5节解释，对不满足对极约束的匹配，就抛弃之。ORB对进行三角定位，要接受新的点，要检查在两个相机中的正深度，视差，重投影误差，和尺度一致性。开始的时候，一个地图点从两个关键帧中观测到，但也可以在其他关键帧中匹配到，所以要投影到剩余的连接的关键帧中，关联性的搜索在5.4节中详述。

### 6.4. Local Bundle Adjustment

The local BA optimizes the currently processed keyframe Ki, all the keyframes connected to it in the covisibility graph Kc, and all the map points seen by those keyframes. All other keyframes that see those points but are not connected to the currently processed keyframe are included in the optimization but remain fixed. Observations that are marked as outliers are discarded at the middle and at the end of the optimization. See the Appendix for more details about this optimization.

局部BA优化了目前处理的关键帧Ki，在互可见图Kc中相连的所有关键帧，和这些关键帧所看到的所有地图点。所有看到这点但并不连接到当前处理的关键帧的其他关键帧，包括在优化中，但保持固定。被标记为外点的观测在优化中和优化后被抛弃。附录有优化过程的更多详细描述。

### 6.5. Local Keyframe Culling

In order to maintain a compact reconstruction, the local mapping tries to detect redundant keyframes and delete them. This is beneficial as bundle adjustment complexity grows with the number of keyframes, but also because it enables lifelong operation in the same environment as the number of keyframes will not grow unbounded, unless the visual content in the scene changes. We discard all the keyframes in Kc whose 90% of the map points have been seen in at least other three keyframes in the same or finer scale. The scale condition ensures that map points maintain keyframes from which they are measured with most accuracy. This policy was inspired by the one proposed in the work of Tan et. al [24], where keyframes were discarded after a process of change detection.

为保持紧凑的重建，局部建图要检测冗余关键帧，并删除之。这是有好处的，因为BA的复杂度随着关键帧的数量而增加，还因为，这会使得在同样的环境中终生操作时，关键帧的数量不会无限制增长，除非场景中的视觉内容变化了。对于在同样尺度或更精细尺度的关键帧中，有至少3个其他关键帧有90%的相同地图点的关键帧，我们抛弃了Kc中所有的这样关键帧。尺度条件确保了，地图点维护的关键帧，其测量有最高的准确度。这个策略是受到Tan等[24]提出的策略所启发，其中关键帧在变化检测的过程后被抛弃。

## 7. Loop Closing

The loop closing thread takes Ki, the last keyframe processed by the local mapping, and tries to detect and close loops. The steps are next described. 回环闭合线程以Ki为输入，即局部建图所处理的最后一个关键帧，试图检测并闭合回环。步骤如下所述。

### 7.1. Loop Candidates Detection 回环候选检测

At first we compute the similarity between the bag of words vector of Ki and all its neighbors in the covisibility graph (θ_min = 30) and retain the lowest score s_min. Then we query the recognition database and discard all those keyframes whose score is lower than s_min. This is a similar operation to gain robustness as the normalizing score in DBoW2, which is computed from the previous image, but here we use covisibility information. In addition all those keyframes directly connected to Ki are discarded from the results. To accept a loop candidate we must detect consecutively three loop candidates that are consistent (keyframes connected in the covisibility graph). There can be several loop candidates if there are several places with similar appearance to Ki.

首先我们计算Ki和在互可见图(θ_min = 30)中所有相邻点的bag-of-words向量的相似性，保留最低分数s_min的点。然后我们查询识别数据库，抛弃所有分数低于s_min的关键帧。DBoW2中的归一化分数是从前一幅图像中计算得到的，用来获得稳健性，我们这也是一个类似的操作，这里我们使用互可见信息。另外，所有与Ki直接相连的关键帧都从结果中丢弃了。为接受一个回环候选，我们必须检测到连续三个一致的回环候选（在互可见图中连接在一起的关键帧）。如果有几个与Ki外观相似的位置，就可能有几个回环候选。

### 7.2. Compute the Similarity Transformation 计算相似性变换

In monocular SLAM there are seven degrees of freedom in which the map can drift, three translations, three rotations and a scale factor [6]. Therefore to close a loop we need to compute a similarity transformation from the current keyframe Ki to the loop keyframe Kl that informs us about the error accumulated in the loop. The computation of this similarity will serve also as geometrical validation of the loop.

在单目SLAM中，地图漂移有7个自由度，三个平移，三个旋转，和一个缩放因子[6]。因此，要闭合一个回环，我们需要计算当前关键帧Ki到回环关键帧Kl的相似度变换，告诉我们回环中积累的误差。这个相似度的计算，也会起到回环的几何验证的作用。

We first compute correspondences between ORB associated to map points in the current keyframe and the loop candidate keyframes, following the procedure explained in section III-E. At this point we have 3D to 3D correspondences for each loop candidate. We alternatively perform RANSAC iterations with each candidate, trying to find a similarity transformation using the method of Horn [42]. If we find a similarity S_il with enough inliers, we optimize it (see the Appendix), and perform a guided search of more correspondences. We optimize it again and, if S_il is supported by enough inliers, the loop with Kl is accepted.

我们首先计算，与当前关键帧和回环候选关键帧中的地图点相关联的ORB之间的对应性，这个过程在3.5节进行了解释。在这一点上，我们对每个回环候选有了3D到3D的对应性。我们然后用每个候选轮流进行RANSAC迭代，试图使用Horn[42]的方法找到一个相似性变换。如果我们用足够的内点找到了一个相似性S_il，我们对其进行优化（见附录），并引导搜索更多的对应性。我们再次对其进行优化，如果S_il受到更多的内点支持，那么就接受了对Kl的回环。

### 7.3. Loop Fusion 回环融合

The first step in the loop correction is to fuse duplicated map points and insert new edges in the covisibility graph that will attach the loop closure. At first the current keyframe pose T_iw is corrected with the similarity transformation S_il and this correction is propagated to all the neighbors of Ki, concatenating transformations, so that both sides of the loop get aligned. All map points seen by the loop keyframe and its neighbors are projected into Ki and its neighbors and matches are searched in a narrow area around the projection, as done in section V-D. All those map points matched and those that were inliers in the computation of S_il are fused. All keyframes involved in the fusion will update their edges in the covisibility graph effectively creating edges that attach the loop closure.

回环修正的第一步，是融合重复的地图点，在互可见图中插入新的边，附着到回环闭合上。首先，当前的关键帧姿态T_iw用相似度变化S_il进行修正，这个修正传播到所有Ki的邻居中，拼接这个变换，这样回环的两边就对齐了。回环关键帧和其邻居帧看到的所有地图点，投影到Ki和其邻居帧中，在投影附近很小的区域中搜索匹配，5.4进行的就是这个工作。所有匹配的地图点，和在S_il的计算中是内点的，都进行了融合。融合中涉及的所有关键帧会在互可见图中更新其边，高效的创建附着到回环闭合上的边。

### 7.4. Essential Graph Optimization

To effectively close the loop, we perform a pose graph optimization over the Essential Graph, described in Section III-D, that distributes the loop closing error along the graph. The optimization is performed over similarity transformations to correct the scale drift [6]. The error terms and cost function are detailed in the Appendix. After the optimization each map point is transformed according to the correction of one of the keyframes that observes it.

为高效的闭合回环，我们对Essential Graph进行一个姿态图优化，如3.4节所述，将回环闭合误差沿着图中进行分布。优化是对相似度变换之上进行的，以修正尺度漂移[6]。误差项和代价函数如附录详述。在优化过后，每个地图点都根据观察到的一个关键帧的修正进行变换。

## 8. Experiments

We have performed an extensive experimental validation of our system in the large robot sequence of NewCollege [39], evaluating the general performance of the system, in 16 handheld indoor sequences of the TUM RGB-D benchmark [38], evaluating the localization accuracy, relocalization and lifelong capabilities, and in 10 car outdoor sequences from the KITTI dataset [40], evaluating real-time large scale operation, localization accuracy and efficiency of the pose graph optimization.

我们对我们的系统，在大型机器人序列NewCollege上进行了广泛的试验验证，在TUM RGB-G基准测试的16个手持室内序列评估了系统的通用性能，在KITTI数据集的10个汽车室外序列中，评估了定位准确率，重定位准确率和终生能力，评估了实时大规模操作，定位准确率和姿态图优化的效率。

Our system runs in real time and processes the images exactly at the frame rate they were acquired. We have carried out all experiments with an Intel Core i7-4700MQ (4 cores @ 2.40GHz) and 8Gb RAM. ORB-SLAM has three main threads, that run in parallel with other tasks from ROS and the operating system, which introduces some randomness in the results. For this reason, in some experiments, we report the median from several runs.

我们的系统实时运行，处理图像的速度，和获取图像的帧率完全相等。我们进行所有试验的机器配置为，an Intel Core i7-4700MQ (4 cores @ 2.40GHz), 8Gb RAM。ORB-SLAM有3个主线程并行运行，还有ROS和操作系统的其他任务，这带来了结果的一些随机性。基于此，在试验中，我们给出几次运行结果的中值。

### 8.1. System Performance in the NewCollege Dataset

The NewCollege dataset [39] contains a 2.2km sequence from a robot traversing a campus and adjacent parks. The sequence is recorded by a stereo camera at 20 fps and a resolution 512×382. It contains several loops and fast rotations that makes the sequence quite challenging for monocular vision. To the best of our knowledge there is no other monocular system in the literature able to process this whole sequence. For example Strasdat et al. [7], despite being able to close loops and work in large scale environments, only showed monocular results for a small part of this sequence.

NewCollege数据集[39]包含2.2km的序列，机器人穿越了一个校园和临近的公园。一个立体相机以20fps的帧率，512×382的分辨率录制了序列。序列包含几个回环和快速旋转，使这个序列对于单目视觉非常有挑战。据我们所知，文献中尚没有其他单目视觉系统可以处理整个序列。比如Strasdat等[7]，尽管能够闭合回环，在大规模环境中工作，但只对这个序列的一小部分给出了单目的结果。

As an example of our loop closing procedure we show in Fig. 4 the detection of a loop with the inliers that support the similarity transformation. Fig. 5 shows the reconstruction before and after the loop closure. In red it is shown the local map, which after the loop closure extends along both sides of the loop closure. The whole map after processing the full sequence at its real frame-rate is shown in Fig. 6. The big loop on the right does not perfectly align because it was traversed in opposite directions and the place recognizer was not able to find loop closures.

作为我们回环闭合过程的一个例子，图4展示了有内点的检测，支持相似度变换。图5展示了回环闭合前后的重建结果。红色表示的是局部地图，回环闭合之后，延展于回环闭合的前后两边。以实时的帧率处理了完整的序列后，完整地图如图6所示。右边的大的回环没有完美的对齐，因为是沿着相反的方向行进的，所以位置识别器不能找到回环闭合。

We have extracted statistics of the times spent by each thread in this experiment. Table I shows the results for the tracking and the local mapping. Tracking works at frame-rates around 25-30Hz, being the most demanding task to track the local map. If needed this time could be reduced limiting the number of keyframes that are included in the local map. In the local mapping thread the most demanding task is local bundle adjustment. The local BA time varies if the robot is exploring or in a well mapped area, because during exploration bundle adjustment is interrupted if tracking inserts a new keyframe, as explained in section V-E. In case of not needing new keyframes local bundle adjustment performs a generous number of prefixed iterations.

我们对试验中的每个线程，提取了所耗费时间的统计结果。表1展示了跟踪和局部建图的结果。跟踪的帧率为大约25-30Hz，跟踪局部地图是最耗时的任务。如果需要的话，这个时间可以通过限制包括在局部地图中的关键帧的数量来降低。在局部建图线程中，最耗时的任务是局部光束平差。局部BA的时间有变化，取决于机器人是否在已经很好建图的区域进行探索，因为在探索中，如果跟踪插入了一个新的关键帧，会打断BA，这在5.5中进行了解释。在不需要新的关键帧的情况下，局部BA进行相当数量的预先固定的迭代次数。

Table II shows the results for each of the 6 loop closures found. It can be seen how the loop detection increases sublinearly with the number of keyframes. This is due to the efficient querying of the database that only compare the subset of images with words in common, which demonstrates the potential of bag of words for place recognition. Our Essential Graph includes edges around 5 times the number of keyframes, which is a quite sparse graph.

表2给出了找到的6个回环闭合的结果。可以看到，回环检测的时间随着关键帧的数量的增长亚线性的增长。这是因为对数据库的高效查询，只比较那些有共同words的图像子集，这说明bag-of-words对位置识别有相当的潜力。我们的Essential Graph包含的边的数量为关键帧数量的大约5倍，这是一个很稀疏的图。

### 8.2. Localization Accuracy in the TUM RGB-D Benchmark

The TUM RGB-D benchmark [38] is an excellent dataset to evaluate the accuracy of camera localization as it provides several sequences with accurate ground truth obtained with an external motion capture system. We have discarded all those sequences that we consider that are not suitable for pure monocular SLAM systems, as they contain strong rotations, no texture or no motion.

TUM RGB-D基准测试[38]是一个优秀的数据集，可以评估相机定位的准确度，因为有几个序列提供了准确的真值，是用外部的运动捕获系统得到的。我们抛弃了所有我们认为不适合于纯单目SLAM系统的序列，因为他们包含很强的旋转，没有纹理，或没有运动。

For comparison we have also executed the novel, direct, semi-dense LSD-SLAM [10] and PTAM [4] in the benchmark. We compare also with the trajectories generated by RGBD-SLAM [43] which are provided for some of the sequences in the benchmark website. In order to compare ORB-SLAM, LSD-SLAM and PTAM with the ground truth, we align the keyframe trajectories using a similarity transformation, as scale is unknown, and measure the absolute trajectory error (ATE) [38]. In the case of RGBD-SLAM we align the trajectories with a rigid body transformation, but also a similarity to check if the scale was well recovered. LSD-SLAM initializes from random depth values and takes time to converge, therefore we have discarded the first 10 keyframes when comparing with the ground truth. For PTAM we manually selected two frames from which we get a good initialization. Table III shows the median results over 5 executions in each of the 16 sequences selected.

为比较，我们也对基准测试运行了新的，直接的，半密集的LSD-SLAM和PTAM。我们还与RGBD-SLAM[43]生成的轨迹进行了比较，在基准测试网站上，一些序列提供了这个结果。为将ORB-SLAM，LSD-SLAM和PTAM与真值进行比较，我们将关键帧的轨迹用相似度变换进行对齐，因为尺度是未知的，并测量绝对轨迹误差(ATE)。在RGBD-SLAM的情况中，我们将轨迹用刚体变换进行对齐，还用相似度来检查，尺度是否得到了很好的恢复。LSD-SLAM从随机深度值进行初始化，需要一些时间进行收敛，因此我们在与真值进行比较时，抛弃了开始10个关键帧。对于PTAM，我们手工选择了两帧，从中可以得到一个很好的初始化。表3给出了16个选择的序列的5次执行的中间结果。

It can be seen that ORB-SLAM is able to process all the sequences, except for fr3_nostructure_texture_far (fr3_nstr_tex_far). This is a planar scene that because of the camera trajectory with respect to the plane has two possible interpretations, i.e. the twofold ambiguity described in [27]. Our initialization method detects the ambiguity and for safety refuses to initialize. PTAM initializes selecting sometimes the true solution and others the corrupted one, in which case the error is unacceptable. We have not noticed two different reconstructions from LSD-SLAM but the error in this sequence is very high. In the rest of the sequences, PTAM and LSD-SLAM exhibit less robustness than our method, loosing track in eight and three sequences respectively.

可以看出，ORB-SLAM可以处理所有序列，除了fr3_nostructure_texture_far (fr3_nstr_tex_far)。这是一个平面场景，因为相对于平面的相机轨迹有两个可能的解释，即，[27]中表述的双重疑义性。我们的初始化方法检测到了疑义性，为安全起见，拒绝进行初始化。PTAM的初始化，有时候选择了正确的解，有时候选择了错误的解，这种情况下，错误是不可接受的。我们从LSD-SLAM中没有注意到两个不同的重建，但在这个序列中的误差是非常高的。在这个序列中的剩余部分，PTAM和LSD-SLAM与我们的方法相比，稳健性没有那么好，分别对8个和3个序列丢失了跟踪。

In terms of accuracy ORB-SLAM and PTAM are similar in open trajectories, while ORB-SLAM achieves higher accuracy when detecting large loops as in the sequence fr3_nostructure_texture_near_withloop (fr3_nstr_tex_near). The most surprising results is that both PTAM and ORB-SLAM are clearly more accurate than LSD-SLAM and RGBD-SLAM. One of the possible causes can be that they reduce the map optimization to a pose-graph optimization where sensor measurements are discarded, while we perform bundle adjustment and jointly optimize cameras and map over sensor measurements, which is the gold standard algorithm to solve structure from motion [2]. We further discuss this result in Section IX-B. Another interesting result is that LSD-SLAM seems to be less robust to dynamic objects than our system as seen in fr2_desk_with_person and fr3_walking_xyz.

以准确率来讲，ORB-SLAM和PTAM在开放轨迹中是类似的，而在检测更大的回环时，如序列fr3_nostructure_texture_near_withloop (fr3_nstr_tex_near)，ORB-SLAM则得到了更高的准确率。最令人惊讶的结果是，PTAM和ORB-SLAM明显比LSD-SLAM和RGBD-SLAM要更精确。一种可能的原因可以是，他们将地图优化简化成了一个姿态图优化问题，抛弃了传感器的测量，而我们则进行光束平差，对相机和地图在传感器测量的基础上进行联合优化，这是求解SfM问题的金标准算法[2]。我们在9.2中进一步讨论这个结果。另一个有趣的结果是，LSD-SLAM与我们的系统相比，对动态目标没那么稳健，这可以在fr2_desk_with_person和fr3_walking_xyz中看到。

We have noticed that RGBD-SLAM has a bias in the scale in fr2 sequences, as aligning the trajectories with 7 DoF significantly reduces the error. Finally it should be noted that Engel et al. [10] reported that PTAM has less accuracy than LSD-SLAM in fr2_xyz with an RMSE of 24.28cm. However, the paper does not give enough details on how those results were obtained, and we have been unable to reproduce them.

我们注意到，RGBD-SLAM在fr2序列中有一些偏差，用7 DoF将轨迹进行对齐会显著降低误差。最后，要提到，Engel等[10]PTAM在fr2_xyz中的准确率比LSD-SLAM中要低，其RMSE有24.28cm。但是，文章没有给出这些结果如何获得的细节，我们也没能复现出结果。

### 8.3. Relocalization in the TUM RGB-D Benchmark

We perform two relocalization experiments in the TUM RGB-D benchmark. In the first experiment we build a map with the first 30 seconds of the sequence fr2_xyz and perform global relocalization with every successive frame and evaluate the accuracy of the recovered poses. We perform the same experiment with PTAM for comparison. Fig. 7 shows the keyframes used to create the initial map, the poses of the relocalized frames and the ground truth for those frames. It can be seen that PTAM is only able to relocalize frames which are near to the keyframes due to the little invariance of its relocalization method. Table IV shows the recall and the error with respect to the ground truth. ORB-SLAM accurately relocalizes more than the double of frames than PTAM. In the second experiment we create an initial map with sequence fr3_sitting_xyz and try to relocalize all frames from fr3_walking_xyz. This is a challenging experiment as there are big occlusions due to people moving in the scene. Here PTAM finds no relocalizations while our system relocalizes 78% of the frames, as can be seen in Table IV. Fig. 8 shows some examples of challenging relocalizations performed by our system in these experiments.

我们在TUM RGB-D基准测试中进行了两个重定位试验。在第一个试验中，我们用序列fr2_xyz的前30秒构建了一个地图，用每个后续的帧进行了全局重定位，评估了恢复的姿态的准确率。我们用PTAM进行了相同的试验，以进行比较。图7展示了用于创建初始地图的关键帧，重新定位的帧的姿态，和这些帧的真值。可以看到，PTAM只能重新定位在关键帧附近的帧，因为其重定位方法的不变性很小。表4展示了对真值的召回率和误差。ORB-SLAM与PTAM相比，准确的重定位的帧数量要多了2倍多。在第二个试验中，我们用序列fr3_sitting_xyz创建了一个初始图，然后从fr3_walking_xyz中对所有帧进行重定位。这是一个很有挑战的试验，因为人们在场景中移动，有很大的遮挡。这里PTAM没有找到准确的重定位，而我们的系统重新定位了78%的帧，如表4所示。图8展示了一些我们的系统在这些试验中进行的有挑战的重定位的例子。

### 8.4. Lifelong Experiment in the TUM RGB-D Benchmark

Previous relocalization experiments have shown that our system is able to localize in a map from very different viewpoints and robustly under moderate dynamic changes. This property in conjunction with our keyframe culling procedure allows to operate lifelong in the same environment under different viewpoints and some dynamic changes.

之前的重定位试验已经表明，我们的系统可以在地图中以非常不同的视角进行定位，在中等动态的变化中很稳健的定位。这个性质与我们的关键帧选择剔除过程一起，使得可以在相同的环境中以不同的视角和一些动态变化进行终身操作。

In the case of a completely static scenario our system is able to maintain the number of keyframes bounded even if the camera is looking at the scene from different viewpoints. We demonstrate it in a custom sequence where the camera is looking at the same desk during 93 seconds but performing a trajectory so that the viewpoint is always changing. We compare the evolution of the number of keyframes in our map and those generated by PTAM in Fig. 9. It can be seen how PTAM is always inserting keyframes, while our mechanism to prune redundant keyframes makes its number to saturate.

在完全静态的场景中，我们的系统可以保持关键帧数量是有限的，即使相机观察场景是从不同的视角。我们在一个定制序列中进行证明，其中相机在93秒中都在查看相同的桌子，但观察的轨迹使得视角一直在变化。我们在图9中比较了我们地图中关键帧数量的演化，和PTAM所生成的结果。可以看到，PTAM一直在插入关键帧，而我们的机制会修剪冗余的关键帧，使其数量会达到饱和。

While the lifelong operation in a static scenario should be a requirement of any SLAM system, more interesting is the case where dynamic changes occur. We analyze the behavior of our system in such scenario by running consecutively the dynamic sequences from fr3: sitting_xyz, sitting_halfsphere, sitting_rpy, walking_xyz, walking_halfspehere and walking_rpy. All the sequences focus the camera to the same desk but perform different trajectories, while people are moving and change some objects like chairs. Fig. 10(a) shows the evolution of the total number of keyframes in the map, and Fig. 10(b) shows for each keyframe its frame of creation and destruction, showing how long the keyframes have survived in the map. It can be seen that during the first two sequences the map size grows as all the views of the scene are being seen for the first time. In Fig. 10(b) we can see that several keyframes created during these two first sequences are maintained in the map during the whole experiment. During the sequences sitting_rpy and walking_xyz the map does not grow, because the map created so far explains well the scene. In contrast, during the last two sequences, more keyframes are inserted showing that there are some novelties in the scene that were not yet represented, due probably to dynamic changes. Finally Fig. 10(c) shows a histogram of the keyframes according to the time they have survived with respect to the remaining time of the sequence from its moment of creation. It can be seen that most of the keyframes are destroyed by the culling procedure soon after creation, and only a small subset survive until the end of the experiment. On one hand, this shows that our system has a generous keyframe spawning policy, which is very useful when performing abrupt motions in exploration. On the other hand the system is eventually able to select a small representative subset of those keyframes.

在静态场景中终身操作应该是任何SLAM系统必须具备的功能，但在动态变化发生的时候，就有更有趣的情况了。我们分析了我们系统在这样场景中的行为，对fr3中的动态序列连续运行：sitting_xyz, sitting_halfsphere, sitting_rpy, walking_xyz, walking_halfspehere和walking_rpy。所有序列都将相机对准同样的桌子聚焦，但进行不同的轨迹，同时人在移动，对改变一些目标，如椅子。图10a展示了关键帧数量在地图中的演化，图10b对每个关键帧展示了其创建和丢弃的过程，展示了关键帧在地图中存活的时长。可以看到，在前两个序列中，地图大小持续增加，因为场景的所有视角是第一次被看到。在图10b中，我们可以看到，在前两个序列中创建的几个关键帧，在整个试验过程中都保留在了地图中。在序列sitting_rpy和walking_xyz过程中，地图并没有增长，因为迄今为止创建的地图可以很好的解释整个场景。对比之下，在后两个序列时，插入了更多的关键帧，表明场景中有一些新东西，还有得到很好的表示，可能是因为动态的变化。图10c最终展示了关键帧的直方图，是关键帧存活的时间，对从其创建的序列的剩余时间。可以看到，多数关键帧都在创建后没多久由选择剔除过程摧毁了，只有很少一部分存活到了试验结束。一方面，这表明系统的关键帧生成策略很慷慨，这在探索过程中进行突然运动时非常有用。另一方面，系统最终还是能够选择这些关键帧中的小部分有代表性的子集。

In these lifelong experiments we have shown that our map grows with the content of the scene but not with the time, and that is able to store the dynamic changes of the scene which could be useful to perform some scene understanding by accumulating experience in an environment.

在这个终生试验中，我们证明了我们的地图是随着场景中的内容增长的，而不是随着时间增长的，而且能够存储场景的动态变化，这在进行场景理解的时候是有用的，可以累积环境中的一些经验。

### 8.5. Large Scale and Large Loop Closing in the KITTI Dataset

The odometry benchmark from the KITTI dataset [40] contains 11 sequences from a car driven around a residential area with accurate ground truth from GPS and a Velodyne laser scanner. This is a very challenging dataset for monocular vision due to fast rotations, areas with lot of foliage, which make more difficult data association, and relatively high car speed, being the sequences recorded at 10 fps. We play the sequences at the real frame-rate they were recorded and ORB-SLAM is able to process all the sequences by the exception of sequence 01 which is a highway with few trackable close objects. Sequences 00, 02, 05, 06, 07, 09 contain loops that were correctly detected and closed by our system. Sequence 09 contains a loop that can be detected only in a few frames at the end of the sequence, and our system not always detects it (the results provided are for the executions in which it was detected).

KITTI数据集的里程计基准测试，包含11个序列，是一个车在居住区附近驾驶，带有GPS和Velodyne激光雷达的精确真值。这对于单目视觉是一个非常有挑战的数据集，因为有快速的旋转，有很多树叶的区域，对于数据关联来说非常困难，车速也相对较高，序列的录制速度为10 fps。我们以录制的真实帧率来播放这个序列，ORB-SLAM可以处理绝大部分序列，但序列01无法处理，因为是一个高速路，没有可以追踪的较近的目标。序列00, 02, 05, 06, 07, 09包含回环，我们的系统可以正确的检测到并进行闭合。序列09包含了一个回环，只能在序列最后的几帧中检测到，我们的系统并不总是能够检测到（给出的结果是检测到的结果）。

Qualitative comparisons of our trajectories and the ground truth are shown in Fig. 11 and Fig. 12. As in the TUM RGB-D benchmark we have aligned the keyframe trajectories of our system and the ground truth with a similarity transformation. We can compare qualitatively our results from Fig. 11 and Fig. 12 with the results provided for sequences 00, 05, 06, 07 and 08 by the recent monocular SLAM approach of Lim et. al [25] in their figure 10. ORB-SLAM produces clearly more accurate trajectories for all those sequences by the exception of sequence 08 in which they seem to suffer less drift.

图11和12给出了我们的轨迹和真值轨迹的定性比较。就像在TUM RGB-D基准测试中一样，我们将我们系统的关键帧轨迹和真值用相似变换进行了对齐。我们将图11和图12的我们的结果，与最近Lim等[25]提出的单目SLAM方法处理的00, 05, 06, 07和08序列进行了比较。ORB-SLAM明显对所有这些序列得到了更精确的轨迹，但序列08中似乎有有一些漂移。

Table V shows the median RMSE error of the keyframe trajectory over five executions in each sequence. We also provide the dimensions of the maps to put in context the errors. The results demonstrate that our system is very accurate being the trajectory error typically around the 1% of its dimensions, sometimes less as in sequence 03 with an error of the 0.3% or higher as in sequence 08 with the 5%. In sequence 08 there are no loops and drift cannot be corrected, which makes clear the need of loop closures to achieve accurate reconstructions.

表5给出了每个序列在5次执行的过程的关键帧轨迹的中值RMSE误差。我们还给出了地图的维度，以知道误差的上下文。结果表明，我们的系统是非常精确的，轨迹误差一般是维度的1%，有时候更小，比如序列03，误差为0.3%，有时候更大一些，比如在序列08中，为5%。在序列08中，没有回环，漂移不能被修正，这就说明回环闭合对进行精确重建的必要性。

In this experiment we have also checked how much the reconstruction can be improved by performing 20 iterations of full BA, see the Appendix for details, at the end of each sequence. We have noticed that some iterations of full BA slightly improves the accuracy in the trajectories with loops but it has negligible effect in open trajectories, which means that the output of our system is already very accurate. In any case if the most accurate results are needed our algorithm provides a set of matches, which define a strong camera network, and an initial guess, so that full BA converge in few iterations.

在这个试验中，我们还核实了20次完整的BA可以改进重建结果多少，见附录的细节，在每个序列最后。我们注意到，完整BA的一些迭代，在带有回环的轨迹中，可以略微改进准确率，但在开放轨迹中的影响是可以忽略的，这说明我们系统的输出就已经非常精确了。在任意情况下，如果需要最精确的结果，我们算法会给出匹配的集合，这定义了一个很强的相机网络，和一个初始估计，这样完整的BA会在几次迭代中收敛。

Finally we wanted to show the efficacy of our loop closing approach and the influence of the θ_min used to include edges in the essential graph. We have selected the sequence 09 (a very long sequence with a loop closure at the end), and in the same execution we have evaluated different loop closing strategies. In table VI we show the keyframe trajectory RMSE and the time spent in the optimization in different cases: without loop closing, if we directly apply a full BA (20 or 100 iterations), if we apply only pose graph optimization (10 iterations with different number of edges) and if we apply pose graph optimization and full BA afterwards. The results clearly show that before loop closure, the solution is so far from the optimal, that BA has convergence problems. Even after 100 iterations still the error is very high. On the other hand essential graph optimization shows fast convergence and more accurate results. It can be seen that the choice of θ_min has not significant effect in accuracy but decreasing the number of edges the time can be significantly reduced. Performing an additional BA after the pose graph optimization slightly improves the accuracy while increasing substantially the time.

最后，我们希望展示我们的回环闭合方法的效用，以及用于在essential graph中包含边的θ_min的影响。我们选择了序列09（一个非常长的序列，在最后有一个回环闭合），在同样的执行中，我们评估了不同的回环闭合策略。在表6中，我们展示了关键帧轨迹的RMSE，和在不同情况中优化过程的时间耗费：没有回环闭合，我们直接应用一个完整的BA（20或100次迭代），我们只应用姿态图优化（10次迭代，有不同数量的边），我们进行姿态图优化和完整的BA。结果明显说明，在回环闭合前，结果与最佳值差距很大；BA则有收敛的问题，即使在100次迭代后，误差仍然非常高；另一方面，essential graph优化则可以快速收敛，得到更准确的结果。可以看到，θ_min的选择对准确率没有明显的影响，但减少边的数量，消耗的时间可以明显降低。在姿态图优化后进行额外的BA，会略微改进准确率，但处理时间会极大的增加。

## 9. Conclusions and Discussions

### 9.1. Conclusions

In this work we have presented a new monocular SLAM system with a detailed description of its building blocks and an exhaustive evaluation in public datasets. Our system has demonstrated that it can process sequences from indoor and outdoor scenes and from car, robot and hand-held motions. The accuracy of the system is typically below 1 cm in small indoor scenarios and of a few meters in large outdoor scenarios (once we have aligned the scale with the ground truth).

在本文中，我们提出了一种新的单目SLAM系统，详细叙述了其构建模块，在公开数据集中进行了详尽的评估。我们的系统证明了，可以处理室内和室外场景的序列，可以处理车载，机器人和手持运动的序列。我们系统的精确度，在室内场景一般是小于1cm，在大型室外场景中一般是几米（尺度要与真值进行对齐）。

Currently PTAM by Klein and Murray [4] is considered the most accurate SLAM method from monocular video in real time. It is not coincidence that the backend of PTAM is bundle adjustment, which is well known to be the gold standard method for the offline Structure From Motion problem [2]. One of the main successes of PTAM, and the earlier work of Mouragnon [3], was to bring that knowledge into the robotics SLAM community and demonstrate its real time performance. The main contribution of our work is to expand the versatility of PTAM to environments that are intractable for that system. To achieve this, we have designed from scratch a new monocular SLAM system with some new ideas and algorithms, but also incorporating excellent works developed in the past few years, such as the loop detection of G´alvez-L´opez and Tard´os [5], the loop closing procedure and covisibility graph of Strasdat et.al [6], [7], the optimization framework g2o by Kuemmerle et. al [37] and ORB features by Rubble et. al [9]. To the best of our knowledge, no other system has demonstrated to work in as many different scenarios and with such accuracy. Therefore our system is currently the most reliable and complete solution for monocular SLAM. Our novel policy to spawn and cull keyframes, permits to create keyframes every few frames, which are eventually removed when considered redundant. This flexible map expansion is really useful in poorly conditioned exploration trajectories, i.e. close to pure rotations or fast movements. When operating repeatedly in the same environment, the map only grows if the visual content of the scene changes, storing a history of its different visual appearances. Interesting results for long-term mapping could be extracted analyzing this history.

目前PTAM被认为是最准确的单目实时SLAM方法。PTAM的后端是BA，这并不是巧合，这是离线SfM问题的金标准，非常有名。PTAM的一个主要成功，以及Mouragnon[3]的早期工作，是将这个知识带入到机器人SLAM团体中，展示其实时性能。我们工作的主要贡献，是拓展PTAM到这个系统无法处理的环境。为获得这个成果，我们从头设计了一个新的单目SLAM系统，有新的思想和算法，但也使用了过去几年提出了一些精彩工作，比如回环检测，回环闭合过程，互可见图，g2o的优化框架，和ORB特征。据我们所知，还没有其他系统能够在这么多不同的场景中以这样的准确度工作。因此，我们的系统是目前单目SLAM最可靠和完整的解决方案。我们生成和选择剔除关键帧的新策略，可以每几帧就创建关键帧，但如果认为是冗余的话，就被移除了。这种灵活的地图扩张在条件不太好的探索轨迹中非常有用，即，接近纯旋转，或快速运动。当在同样的环境中重复操作时，地图只会在场景的视觉内容变化时增长，存储不同的视觉外观的历史。长期建图的有趣结果，可以通过分析其历史进行提取。

Finally we have also demonstrated that ORB features have enough recognition power to enable place recognition from severe viewpoint change. Moreover they are so fast to extract and match (without the need of multi-threading or GPU acceleration) that enable real time accurate tracking and mapping.

最后，我们证明ORB特征有足够的识别能力，可以在严重的视角变化下进行位置识别。而且，ORB提取和匹配的速度都很快（不需要多线程或GPU加速），可以进行实时精确跟踪和建图。

### 9.2. Sparse/Feature-based vs. Dense/Direct Methods

Recent real-time monocular SLAM algorithms such as DTAM [44] and LSD-SLAM [10] are able to perform dense or semi dense reconstructions of the environment, while the camera is localized by optimizing directly over image pixel intensities. These direct approaches do not need feature extraction and thus avoid the corresponding artifacts. They are also more robust to blur, low-texture environments and high-frequency texture like asphalt [45]. Their denser reconstructions, as compared to the sparse point map of our system or PTAM, could be more useful for other tasks than just camera localization.

最近的实时单目SLAM算法，比如DTAM和LSD-SLAM可以进行环境的密集或半密集重建，相机定位可以通过直接对像素灰度的优化来进行。这些直接方法不需要进行特征提取，因此避免了对应的失真。它们对模糊、低纹理环境和高频纹理更加稳健，比如沥青。其更密集的重建，与我们系统或PTAM的稀疏点地图相比，对除了相机定位的其他任务可能更有用。

However, direct methods have their own limitations. Firstly, these methods assume a surface reflectance model that in real scenes produces its own artifacts. The photometric consistency limits the baseline of the matches, typically narrower than those that features allow. This has a great impact in reconstruction accuracy, which requires wide baseline observations to reduce depth uncertainty. Direct methods, if not correctly modeled, are quite affected by rolling-shutter, auto-gain and auto-exposure artifacts (as in the TUM RGB-D Benchmark). Finally, because direct methods are in general very computationally demanding, the map is just incrementally expanded as in DTAM, or map optimization is reduced to a pose graph, discarding all sensor measurements as in LSD-SLAM.

但是，直接方法有其局限。首先，这些方法假设真实场景中的表面反射模型会产生其失真。光度学一致性限制了这些匹配的基准，一般比特征所允许的更加窄。这对重建准确率有很大影响，需要很宽的基准观测，以减低深度的不确定性。直接方法如果没有正确的建模，会受到卷帘快门、自动增益和自动曝光的失真的严重影响（就像在TUM RGB-D基准测试）。最后，因为直接方法一般来说计算量都非常大，地图都是像DTAM中逐渐扩张的，或地图优化缩减成了姿态图优化，像LSD-SLAM中丢弃了所有传感器测量。

In contrast, feature-based methods are able to match features with a wide baseline, thanks to their good invariance to viewpoint and illumination changes. Bundle adjustment jointly optimizes camera poses and points over sensor measurements. In the context of structure and motion estimation, Torr and Zisserman [46] already pointed the benefits of feature-based against direct methods. In this work we provide experimental evidence (see Section VIII-B) of the superior accuracy of feature-based methods in real-time SLAM. We consider that the future of monocular SLAM should incorporate the best of both approaches.

比较起来，基于特征的方法能够匹配很宽基准范围内的特征，多亏了对视角和光照变化的不变性。BA对相机姿态和点通过传感器测量进行联合优化。在结构和运动估计的上下文下，Torr and Zisserman [46]已经指出，基于特征的比直接方法的优点。在本文中，我们给出了基于特征的方法在实时SLAM中的优秀准确率的试验证据（见8.2）。我们认为，未来的单目视觉应当纳入两种方法的好的方面。

### 9.3. Future Work

The accuracy of our system can still be improved incorporating points at infinity in the tracking. These points, which are not seen with sufficient parallax and our system does not include in the map, are very informative of the rotation of the camera [21].

我们系统的准确率仍然可以得到改进，在跟踪中将无穷远处的点纳入进来。这些点并没有以足够的视差进行观察，我们的系统也没有将其纳入到地图中，包含很多相机旋转的信息[21]。

Another open way is to upgrade the sparse map of our system to a denser and more useful reconstruction. Thanks to our keyframe selection, keyframes comprise a compact summary of the environment with a very high pose accuracy and rich information of covisibility. Therefore the ORB-SLAM sparse map can be an excellent initial guess and skeleton, on top of which a dense and accurate map of the scene can be built. A first effort in this line is presented in [47].

另一种开放的路径，是升级我们系统的稀疏地图，称为更密集的，更有用的重建。多亏了我们的关键帧选择，关键帧包含了环境的紧凑总结，包括很高的姿态准确率，丰富的互可见信息。因此，ORB-SLAM稀疏图可以是一个很好的初始估计和骨架，在其之上，可以构建出场景的密集精确地图。这条线的一个努力是[47]。