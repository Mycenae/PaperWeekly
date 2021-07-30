# ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM

Carlos Campos∗, Richard Elvira∗, Juan J. G´omez Rodr´ıguez, Jos´e M.M. Montiel and Juan D. Tard´os, Spain

## 0. Abstract

This paper presents ORB-SLAM3, the first system able to perform visual, visual-inertial and multi-map SLAM with monocular, stereo and RGB-D cameras, using pin-hole and fisheye lens models.

本文提出了第一个可以用单目、立体相机和RGB-D相机进行视觉、视觉-惯性和多地图SLAM系统，使用针孔模型和鱼眼镜头模型，ORB-SLAM3。

The first main novelty is a feature-based tightly-integrated visual-inertial SLAM system that fully relies on Maximum-a-Posteriori (MAP) estimation, even during the IMU initialization phase. The result is a system that operates robustly in real time, in small and large, indoor and outdoor environments, and is two to ten times more accurate than previous approaches.

第一个主要的创新是基于特征的紧耦合的视觉-惯性SLAM系统，完全依赖于最大后验估计，即使是在IMU初始化的阶段。结果是得到了在小型和大型，室内和室外环境，都可以实时稳健运行的系统，比之前的方法准确了2到10倍。

The second main novelty is a multiple map system that relies on a new place recognition method with improved recall. Thanks to it, ORB-SLAM3 is able to survive to long periods of poor visual information: when it gets lost, it starts a new map that will be seamlessly merged with previous maps when revisiting mapped areas. Compared with visual odometry systems that only use information from the last few seconds, ORB-SLAM3 is the first system able to reuse in all the algorithm stages all previous information. This allows to include in bundle adjustment co-visible keyframes, that provide high parallax observations boosting accuracy, even if they are widely separated in time or if they come from a previous mapping session.

第二个主要的创新是多地图系统，这是依赖于新的位置识别方法，召回率得到了改进。多亏了这个，ORB-SLAM3在很长时间较差的视觉信息时也可以继续运行：当丢失的时候，其开启了一个新的地图，当重新访问已经建图的区域时，会无缝的与之前的地图融合到一起。有的视觉里程计系统只使用最后几秒的信息，与之相比，ORB-SLAM3是在所有算法阶段对所有之前的信息进行重用的系统。这使得在BA时包含了互可见关键帧，给出了大视差的观察，提升了准确率，即使在时间上是分隔的很长的，或是从之前的建图环节中得到的。

Our experiments show that, in all sensor configurations, ORB-SLAM3 is as robust as the best systems available in the literature, and significantly more accurate. Notably, our stereo-inertial SLAM achieves an average accuracy of 3.5 cm in the EuRoC drone and 9 mm under quick hand-held motions in the room of TUM-VI dataset, a setting representative of AR/VR scenarios. For the benefit of the community we make public the source code.

我们的试验表明，在所有传感器配置中，ORB-SLAM3与文献中可用的最好系统是一样稳健的，而且明显更加准确。值得注意的是，我们的立体-惯性SLAM在EuRoC无人机数据集上获得了3.5cm的平均精度，在TUM-VI数据集的快速手持室内运动中得到了9mm的平均精度，这是在AR/VR场景中的代表。我们的代码已经开源。

## 1. Introduction

Intense research on Visual Simultaneous Localization and Mapping systems (SLAM) and Visual Odometry (VO), using cameras either alone or in combination with inertial sensors, has produced during the last two decades excellent systems, with increasing accuracy and robustness. Modern systems rely on Maximum a Posteriori (MAP) estimation, which in the case of visual sensors corresponds to Bundle Adjustment (BA), either geometric BA that minimizes feature reprojection error, in feature-based methods, or photometric BA that minimizes the photometric error of a set of selected pixels, in direct methods.

在SLAM和VO上有很多研究，使用的是相机，或与惯性传感器一起，在过去二十年中，得到了非常好的系统，准确率和稳健性逐渐增加。现代系统依赖于最大后验概率估计，在视觉传感器的情况下，这对应着BA，在基于特征的方法中是几何BA，最小化特征重投影误差，在直接方法中是光度BA，对选定的像素集的光度误差进行最小化。

With the recent emergence of VO systems that integrate loop closing techniques, the frontier between VO and SLAM is more diffuse. The goal of Visual SLAM is to use the sensors on-board a mobile agent to build a map of the environment and compute in real-time the pose of the agent in that map. In contrast, VO systems put their focus on computing the agent’s ego-motion, not on building a map. The big advantage of a SLAM map is that it allows matching and using in BA previous observations performing three types of data association (extending the terminology used in [1]):

最近兴起了VO系统，整合了回环闭合技术，VO和SLAM的边界也越发模糊了。视觉SLAM的目标，是使用移动智能体上的传感器，来构建环境的地图，实时计算地图中的智能体的姿态。对比起来，VO系统的目的是，计算智能体的自我运动，而并不是构建地图。SLAM地图的一个巨大优势是可以与之前的观察进行匹配，在BA中使用，进行三种类型的数据关联：

• Short-term data association, matching map elements obtained during the last few seconds. This is the only data association type used by most VO systems, that forget environment elements once they get out of view, resulting in continuous estimation drift even when the system moves in the same area.

短期数据关联，与最后几秒获得地图元素进行匹配。多数VO系统都只使用这种数据关联，一旦脱离实现，就忘掉环境元素，会得到持续的估计漂移，即使系统在同样的区域中移动。

• Mid-term data association, matching map elements that are close to the camera whose accumulated drift is still small. These can be matched and used in BA in the same way than short-term observations and allow to reach zero drift when the systems moves in mapped areas. They are the key of the better accuracy obtained by our system compared against VO systems with loop detection.

中期数据关联，对与相机很接近的地图元素进行匹配，其累积漂移仍然很小。这可以与短期观察一样的进行匹配，在BA中使用，当系统移动到已建图区域中时，可以达到零漂移。与带有回环检测的VO系统相比，我们的系统可以得到更好的准确率，这是最关键的因素。

• Long-term data association, matching observations with elements in previously visited areas using a place recognition technique, regardless of the accumulated drift (loop detection), the current area being previously mapped in a disconnected map (map merging), or the tracking being lost (relocalization). Long-term matching allows to reset the drift and to correct the map using pose-graph (PG) optimization, or more accurately, using BA. This is the key of SLAM accuracy in medium and large loopy environments.

长期数据关联，与之前的访问区域中的观察元素，使用位置识别技术进行匹配，不管多少累积漂移（回环闭合），当前区域之前就在不连接的地图中进行建图过（地图合并），或追踪丢失了（重定位）。长期匹配可以重置漂移，使用姿态图优化来修正地图，或使用BA更准确的修正地图。这是SLAM在中型和大型回环环境中准确率的关键。

In this work we build on ORB-SLAM [2], [3] and ORB-SLAM Visual-Inertial [4], the first visual and visual-inertial systems able to take full profit of short-term, mid-term and long-term data association, reaching zero drift in mapped areas. Here we go one step further providing multi-map data association, which allows us to match and use in BA map elements coming from previous mapping sessions, achieving the true goal of a SLAM system: building a map that can be used later to provide accurate localization.

本文中，我们在ORB-SLAM和ORB-SLAM视觉-惯性的基础上，构建了第一个视觉和视觉-惯性系统，可以完整利用短期，中期和长期数据关联，在已建图区域中达到零漂移。这里，我们进一步给出了多地图数据关联，使我们对之前建图环节的地图元素进行匹配，和在BA中使用，获得SLAM系统的真正目标：构建一个地图，可以后来进行使用，以给出准确的定位。

This is essentially a system paper, whose most important contribution is the ORB-SLAM3 library itself [5], the most complete and accurate visual, visual-inertial and multi-map SLAM system to date (see table I). The main novelties of ORB-SLAM3 are:

这实际上是一篇系统文章，其最重要的贡献是ORB-SLAM3库，迄今为止最完整最准确的视觉，视觉-惯性，和多地图SLAM系统。ORB-SLAM3的主要创新是：

• A monocular and stereo visual-inertial SLAM system that fully relies on Maximum-a-Posteriori (MAP) estimation, even during the IMU (Inertial Measurement Unit) initialization phase. The initialization method proposed was previously presented in [6]. Here we add its integration with ORB-SLAM visual-inertial [4], the extension to stereo-inertial SLAM, and a thorough evaluation in public datasets. Our results show that the monocular and stereo visual-inertial systems are extremely robust and significantly more accurate than other visual-inertial approaches, even in sequences without loops.

单目和立体视觉-惯性SLAM系统，完全依赖于后验概率估计，即使是在IMU初始化的阶段。之前[6]提出了初始化方法。这里我们加入整合到了ORB-SLAM视觉-惯性中，这是立体-惯性SLAM的拓展，我们还加入了在公开数据集中的彻底评估。我们的结果表明，单目和立体视觉-惯性系统非常稳健，比其他的视觉-惯性方法要明显更加精确，即使是在没有回环的序列中。

• Improved-recall place recognition. Many recent visual SLAM and VO systems [2], [7], [8] solve place recognition using the DBoW2 bag of words library [9]. DBoW2 requires temporal consistency, matching three consecutive keyframes to the same area, before checking geometric consistency, boosting precision at the expense of recall. As a result, the system is too slow at closing loops and reusing previously mapped areas. We propose a novel place recognition algorithm, in which candidate keyframes are first checked for geometrical consistency, and then for local consistency with three covisible keyframes, that in most occasions are already in the map. This strategy increases recall and densifies data association improving map accuracy, at the expense of a slightly higher computational cost.

改进召回的位置识别系统。很多最近的视觉SLAM和VO系统用DBoW2 bag-of-words库来求解位置识别问题。DBoW2需要时间上的连贯性，对同样区域的三个连续的关键帧进行匹配，在检查几何一致性之前，以召回为代价提升精确度。结果是，系统在回环闭合和重用之前建图的区域时就很慢。我们提出了一种新的位置识别算法，其中首先检查候选关键帧的几何一致性，然后检查对三个互可见的关键帧的局部一致性，这在多数情况中都是已经在地图中的。这种策略提升了召回，增加了数据关联的密度，改进了地图准确率，代价是略微增加了计算量。

• ORB-SLAM Atlas, the first complete multi-map SLAM system able to handle visual and visual-inertial systems, in monocular and stereo configurations. The Atlas can represent a set of disconnected maps, and apply to them all the mapping operations smoothly: place recognition, camera relocalization, loop closure and accurate seamless map merging. This allows to automatically use and combine maps built at different times, performing incremental multi-session SLAM. A preliminary version of ORB-SLAM Atlas for visual sensors was presented in [10]. Here we add the new place recognition system, the visual-inertial multi-map system and its evaluation on public datasets.

ORB-SLAM Atlas，第一个完整的多地图SLAM系统，可以在单目和立体视觉的配置中处理视觉和视觉-惯性系统。Atlas可以表示不连续地图的集合，对它们平滑的使用所有建图操作：位置识别，相机重定位，回环闭合和准确无缝的地图合并。这可以自动的使用、合并在不同时间构建的地图，进行增量的多环节SLAM。[10]给出了视觉传感器的ORB-SLAM Atlas的初始版本。这里我们加入了新的位置识别系统，视觉-惯性多地图系统，以及在公开数据集上的评估。

• An abstract camera representation making the SLAM code agnostic of the camera model used, and allowing to add new models by providing their projection, unprojection and Jacobian functions. We provide the implementations of pin-hole [11] and fisheye [12] models.

一个抽象的相机表示，使SLAM代码不需要知道实用的相机模型，还可以加入新的模型，只要提供其投影函数，反投影函数和Jacobian函数。我们给出了小孔模型和鱼眼模型的实现。

All these novelties, together with a few code improvements make ORB-SLAM3 the new reference visual and visual-inertial open-source SLAM library, being as robust as the best systems available in the literature, and significantly more accurate, as shown by our experimental results in section VII. We also provide comparisons between monocular, stereo, monocular-inertial and stereo-inertial SLAM results that can be of interest for practitioners.

所有这些创新，与一些代码改进，使ORB-SLAM3称为视觉和视觉-惯性开源SLAM库的新的参考，与文献中可用的最好系统一样稳健，而且明显更加准确，我们在第7部分的试验结果进行了展示。我们还与单目，立体，单目-惯性和立体-惯性SLAM结果进行了比较。

## 2. Related Work

Table I presents a summary of the most representative visual and visual-inertial systems, showing the main techniques used for estimation and data association. The qualitative accuracy and robustness ratings included in the table are based on the results presented in section VII, and the comparison between PTAM, LSD-SLAM and ORB-SLAM reported in [2].

表1给出了最具代表性的视觉和视觉-惯性系统的总结，展示了用于估计和数据关联的主要技术。表中的定性的准确率和稳健性评分，是基于第7部分的结果，以及与PTAM，LSD-SLAM和ORB-SLAM的比较。

### 2.1. Visual SLAM

Monocular SLAM was first solved in MonoSLAM [13], [14], [52] using an Extended Kalman Filter (EKF) and Shi-Tomasi points that were tracked in subsequent images doing a guided search by correlation. Mid-term data association was significantly improved using techniques that guarantee that the feature matches used are consistent, achieving hand-held visual SLAM [53], [54].

单目SLAM问题首先在MonoSLAM中得到解决，实用了扩展Kalman滤波器(EKF)和Shi-Tomasi点，在后续的图像中用关联的导向搜索进行跟踪。确保使用的匹配的特征是一致的，中期数据关联得到了明显的改进，得到了手持的视觉SLAM。

In contrast, keyframe-based approaches estimate the map using only a few selected frames, discarding the information coming from intermediate frames. This allows to perform the more costly, but more accurate, BA optimization at keyframe rate. The most representative system was PTAM [16], that split camera tracking and mapping in two parallel threads. Keyframe-based techniques are more accurate than filtering for the same computational cost [55], becoming the gold standard in visual SLAM and VO. Large scale monocular SLAM was achieved in [56] using sliding-window BA, and in [57] using a double-window optimization and a covisibility graph.

比较起来，基于关键帧的方法只用几个选定的帧来估计地图，抛弃了中间帧的信息。这就允许以帧率的速度进行计算量更大但更精确的BA优化。最有代表性的系统是PTAM，将相机跟踪和建图分成了两个并行的线程。基于关键帧的技术比基于滤波的技术，在同样的计算量下，更加精确，是视觉SLAM和VO的金标准。[56]使用滑窗BA得到了大规模单目SLAM，[57]使用了双窗优化和互可见图。

Building on these ideas, ORB-SLAM [2], [3] uses ORB features, whose descriptor provides short-term and mid-term data association, builds a covisibility graph to limit the complexity of tracking and mapping, and performs loop closing and relocalization using the bag-of-words library DBoW2 [9], achieving long-term data association. To date is the only visual SLAM system integrating the three types of data association, which we believe is the key of its excellent accuracy. In this work we improve its robustness in pure visual SLAM with the new Atlas system that starts a new map when tracking is lost, and its accuracy in loopy scenarios with the new place recognition method with improved recall.

在这些思想的基础上，ORB-SLAM使用了ORB特征，其描述子给出了短期和中期数据关联，构建了一个互可见图来限制跟踪和建图的复杂度，使用bag-of-words库DBoW2来进行回环闭合和重定位，获得了长期数据关联。迄今为止，是包含三种类型的数据关联的唯一视觉SLAM系统，我们相信这是其优异准确率的关键。本文中，我们用新的Atlas系统改进了纯视觉SLAM的稳健性，当跟踪丢失时开启一个新图；用新的改进召回的位置识别方法，改进了在带有回环的场景中的准确率。

Direct methods do not extract features, but use directly the pixel intensities in the images, and estimate motion and structure by minimizing a photometric error. LSD-SLAM [20] was able to build large scale semi-dense maps using high gradient pixels. However, map estimation was reduced to pose-graph (PG) optimization, achieving lower accuracy than PTAM and ORB-SLAM [2]. The hybrid system SVO [23], [24] extracts FAST features, uses a direct method to track features and any pixel with nonzero intensity gradient from frame to frame, and optimizes camera trajectory and 3D structure using reprojection error. SVO is extremely efficient but, being a pure VO method, it only performs short-term data association, which limits its accuracy. Direct Sparse Odometry DSO [27] is able to compute accurate camera poses in situations where point detectors perform poorly, enhancing robustness in low textured areas or against blurred images. It introduces local photometric BA that simultaneously optimizes a window of seven recent keyframes and the inverse depth of the points. Extensions of this work include stereo [29], loop closing using features and DBoW2 [58] [59], and visual-inertial odometry [46]. Direct Sparse Mapping DSM [31] introduces the idea of map reusing in direct methods, showing the importance of mid-term data association. In all cases, the lack of integration of short, mid, and long-term data association results in lower accuracy than our proposal (see section VII).

直接方法不提取特征，但直接使用图像中的像素灰度，通过最小化光度误差来估计运动和结构。LSD-SLAM可以使用高梯度像素来构建大规模半密集的地图。但是，地图估计可以简化为姿态图优化问题，获得比PTAM和ORB-SLAM更低的准确率。混合系统SVO提取的是FAST特征，使用直接方法来逐帧跟踪特征，和所有非零灰度梯度的像素，使用重投影误差来优化相机轨迹和3D结构。SVO效率非常高，但是，作为一个纯VO方法，它只进行短期数据关联，这限制了它的准确率。直接稀疏里程计DSO[27]可以在点检测器表现很差的情况下，准确的计算相机姿态，在低纹理区域或在模糊的图像中也有很好的稳健性；它提出了局部光度BA，同时对7个最近的关键帧的窗口，和点的逆距离进行优化。这篇文章的拓展包括了立体相机[29]，使用特征和DBoW2的回环闭合，和视觉-惯性里程计。直接稀疏建图DSM [31]提出了直接方法中的地图重用的思想，表明了中期数据关联的重要性。在所有情况下，缺少短期、中期和长期数据关联的整合，比我们提出的方法都会得到更低的准确率（见第7部分）。

### 2.2. Visual-Inertial SLAM

The combination of visual and inertial sensors provide robustness to poor texture, motion blur and occlusions, and in the case of monocular systems, make scale observable.

视觉和惯性传感器的组合，可以对缺少纹理，运动模糊和遮挡的情况提供稳健性，在单目系统的情况下，使尺度具有观测性。

Research in tightly coupled approaches can be traced back to MSCKF [33] where the EKF quadratic cost in the number of features is avoided by feature marginalization. The initial system was perfected in [34] and extended to stereo in [35], [36]. The first tightly coupled visual odometry system based on keyframes and bundle adjustment was OKVIS [38], [39] that is also able to use monocular and stereo vision. While these systems rely on features, ROVIO [41], [42] feeds an EFK with photometric error using direct data association.

紧耦合方法的研究可以回溯到MSCKF[33]，通过特征marginalization，避免了特征数量的EKF二次代价。原始系统由[34]进行完善，[35,36]拓展到了立体相机。基于关键帧和BA的第一个紧耦合的视觉里程计系统是OKVIS[38,39]，可以使用单目和立体视觉的配置。这些系统都是依赖于特征的，但ROVIO [41,42]使用直接数据关联，用光度误差来送入EKF中。

ORB-SLAM-VI [4] presented for the first time a visual-inertial SLAM system able to reuse a map with short-term, mid-term and long-term data association, using them in an accurate local visual-inertial BA based on IMU preintegration [60], [61]. However, its IMU initialization technique was too slow, taking 15 seconds, which harmed robustness and accuracy. Faster initialization techniques were proposed in [62], [63], based on a closed-form solution to jointly retrieve scale, gravity, accelerometer bias and initial velocity, as well as visual features depth. Crucially, they ignore IMU noise properties, and minimize the 3D error of points in space, and not their reprojection errors, that is the gold standard in feature-based computer vision. Our previous work [64] shows that this results in large unpredictable errors.

ORB-SLAM-VI [4]第一次提出了可以用短期、中期、长期数据关联进行地图重用的视觉-惯性SLAM系统，并再一个准确的基于IMU预整合的局部视觉-惯性BA中进行使用。但是，其IMU初始化技术太慢了，耗时15秒，这有损于稳健性和准确率。[62,63]提出了更快的初始化技术，基于闭合形式解同时获取尺度、重力和加速度计偏差和初始速度，以及视觉特征深度。关键是，它们忽略了IMU噪声性质，最小化点在空间中的3D误差，而并不是它们的重投影误差，而这是基于特征的计算机视觉的金标准。我们之前的工作[64]表明，这会导致很大的无法预测的误差。

VINS-Mono [7] is a very accurate and robust monocular-inertial odometry system, with loop closing that uses DBoW2 and 4 DoF pose-graph optimization, and map-merging. Feature tracking is performed with Lucas-Kanade tracker, being slightly more robust than descriptor matching. In VINS-Fusion [44] it has been extended to stereo and stereo-inertial.

VINS-Mono [7]是非常精确、稳健的单目-惯性里程计系统，其回环闭合使用的是DBoW2和4 DoF姿态图优化，以及地图融合。特征跟踪是用Lucas-Kanade跟踪器进行的，比描述子跟踪要略微更加稳健。VINS-Fusion [44]将其拓展到了立体视觉和立体-惯性系统。

VI-DSO [46] extends DSO to visual-inertial odometry, proposing a bundle adjustment that combines inertial observations with the photometric error of selected high gradient pixels, what renders very good accuracy. As the information from high gradient pixels is successfully exploited, the robustness in scene regions with poor texture is also boosted. Their initialization method relies on visual-inertial BA and takes 20-30 seconds to converge within 1% scale error.

VI-DSO [46]将DSO拓展到视觉-惯性里程计，提出了一种BA，将惯性系统的观察与选定的高梯度像素的光度误差进行结合，给出了很好的准确率。由于高梯度像素的信息进行了成功利用，缺少纹理的场景区域的稳健性也得到了提高。其初始化方法依赖于视觉-惯性BA，需要20-30秒才能收敛到1%以内的尺度误差。

The recent BASALT [47] is a stereo-inertial odometry system that extracts non-linear factors from visual-inertial odometry to use them in BA, and closes loops matching ORB features, achieving very good to excellent accuracy. Kimera [8] is a novel outstanding metric-semantic mapping system, but its metric part consists in stereo-inertial odometry plus loop closing with DBoW2 and pose-graph optimization, achieving similar accuracy to VINS-Fusion.

最近的BASALT [47]是一个立体-惯性里程计系统，从视觉-惯性里程计中提取了非线性因素，在BA中进行利用，回环闭合利用的是ORB特征匹配，获得了非常好的准确率。Kimera [8]是一个新的非常好的度量-语义建图系统，但其度量部分是由立体-惯性里程计组成的，还有DBoW2的回环闭合，和姿态图优化，其准确率与VINS-Fusion类似。

In this work we build on ORB-SLAM-VI and extend it to stereo-inertial SLAM. We propose a novel fast initialization method based on Maximum-a-Posteriori (MAP) estimation that properly takes into account visual and inertial sensor uncertainties, and estimates the true scale with 5% error in 2 seconds, converging to 1% scale error in 15 seconds. All other systems discussed above are visual-inertial odometry methods, some of them extended with loop closing, and lack the capability of using mid-term data associations. We believe that this, together with our fast and precise initialization, is the key of the better accuracy consistently obtained by our system, even in sequences without loops.

本文中，我们在ORB-SLAM-VI的基础上，将其拓展到立体-惯性SLAM。我们提出了基于MAP估计的快速新型初始化方法，将视觉和惯性传感器的不确定性合理的考虑进去，在2秒内估计真正的尺度，误差5%，在15秒内收敛到1%的尺度误差。上面所讨论的所有其他系统都是视觉-惯性里程计方法，其中一些用回环闭合进行了拓展，缺少使用中期数据关联的能力。我们相信，这和我们的快速精确初始化一起，是我们系统持续得到更好的准确率的关键，即使在没有回环的序列中。

### 2.3. Multi-Map SLAM

The idea of adding robustness to tracking losses during exploration by means of map creation and fusion was first proposed in [65] within a filtering approach. One of the first keyframe-based multi-map systems was [66], but the map initialization was manual, and the system was not able to merge or relate the different sub-maps. Multi-map capability has been researched as a component of collaborative mapping systems, with several mapping agents and a central server that only receives information [67] or with bidirectional information flow as in C2TAM [68]. MOARSLAM [69] proposed a robust stateless client-server architecture for collaborative multi-device SLAM, but the main focus was the software architecture and did not report accuracy results.

通过地图创建和融合，对探索时跟踪丢失的情况增加稳健性，这个思想首先在[65]中提出，利用了滤波的方法。第一个基于关键帧的多梯度系统是[66]，但是地图初始化是手动的，系统不能对不同的子地图进行融合或关联。多地图的能力作为协作建图系统的组成部分，进行了研究，有几个建图智能体，还有一个中央服务器，只接收信息，或双向信息流，如C2TAM [68]。MOARSLAM [69]提出了一种稳健的无状态客户端-服务器架构，进行多设备协作SLAM，但其主要焦点是软件架构，并没有给出准确的结果。

More recently, CCM-SLAM [70], [71] proposes a distributed multi-map system for multiple drones with bidirectional information flow, built on top of ORB-SLAM. Their focus is on overcoming the challenges of limited bandwidth and distributed processing, while ours is on accuracy and robustness, achieving significantly better results on the EuRoC dataset. SLAMM [72] also proposes a multi-map extension of ORB-SLAM2, but keeps sub-maps as separated entities, while we perform seamless map merging, building a more accurate global map.

最近，CCM-SLAM [70,71] 为多个无人机提出了一种分布式多地图系统，有双向信息流，是在ORB-SLAM的基础上构建的。其关注的是，克服有限的带宽和分布式处理的挑战，而我们关注的则是准确率和稳健性，在EuRoC数据集上获得明显更好的结果。SLAMM [72]也提出了ORB-SLAM2的一个多地图拓展，但将子地图作为单独的实体，而我们则进行无缝的地图融合，构建了一个更精确的全局地图。

VINS-Mono [7] is a visual odometry system with loop closing and multi-map capabilities that rely on the place recognition library DBoW2 [9]. Our experiments show that ORB-SLAM3 is 2.6 times more accurate than VINS-Mono in monocular-inertial single-session operation on the EuRoc dataset, thanks to the ability to use mid-term data association. Our Atlas system also builds on DBoW2, but proposes a novel higher-recall place recognition technique, and performs more detailed and accurate map merging using local BA, increasing the advantage to 3.2 times better accuracy than VINS-Mono in multi-session operation on EuRoC.

VINS-Mono [7]是一个视觉里程计系统，带有回环闭合和多地图的能力，依赖于位置识别库DBoW2。我们的试验表明，ORB-SLAM3在EuRoC数据集上的单目-惯性单环节操作上，比VINS-Mono快了2.6倍，是由于我们使用中期数据关联的能力。我们的Atlas系统也是在DBoW2的基础上构建的，但提出了一种新的更高召回的位置识别技术，使用局部BA来进行更详细和准确的地图融合，在EuRoC的多环节运算上，比VINS-Mono的准确率提高到了3.2倍。

## 3. System Overview

ORB-SLAM3 is built on ORB-SLAM2 [3] and ORB-SLAM-VI [4]. It is a full multi-map and multi-session system able to work in pure visual or visual-inertial modes with monocular, stereo or RGB-D sensors, using pin-hole and fisheye camera models. Figure 1 shows the main system components, that are parallel to those of ORB-SLAM2 with some significant novelties, that are summarized next:

ORB-SLAM3是在ORB-SLAM2和ORB-SLAM-VI的基础上构建的。这是一个完全的多地图和多环节系统，可以在纯视觉或视觉-惯性模式运行，用单目，立体视觉或RGB-D传感器，使用针孔相机模型和鱼眼相机模型，都可以运行。图1展示了主要的系统组成部分，与ORB-SLAM2的类似，有明显的创新，总结如下：

• Atlas is a multi-map representation composed of a set of disconnected maps. There is an active map where the tracking thread localizes the incoming frames, and is continuously optimized and grown with new keyframes by the local mapping thread. We refer to the other maps in the Atlas as the non-active maps. The system builds a unique DBoW2 database of keyframes that is used for relocalization, loop closing and map merging.

Atlas是一个多地图表示，由于未连接的地图的集合组成。有一个活跃的地图，跟踪线程对输入的帧进行定位，随着新加入的关键帧，由局部建图线程对地图进行尺度优化并增长。我们将Atlas中的其他地图称为非活跃地图。系统构建了关键帧的DBoW2数据库，用于重定位，回环闭合和地图融合。

• Tracking thread processes sensor information and computes the pose of the current frame with respect to the active map in real-time, minimizing the reprojection error of the matched map features. It also decides whether the current frame becomes a keyframe. In visual-inertial mode, the body velocity and IMU biases are estimated by including the inertial residuals in the optimization. When tracking is lost, the tracking thread tries to relocalize the current frame in all the Atlas’ maps. If relocalized, tracking is resumed, switching the active map if needed. Otherwise, after a certain time, the active map is stored as non-active, and a new active map is initialized from scratch.

跟踪线程处理传感器信息，实时计算当前帧对活跃地图的姿态，最小化匹配地图特征的重投影误差。跟踪线程还决定当前帧是否会变成一个关键帧。在视觉-惯性模式，体速度和IMU偏差的估计，要将惯性残差纳入到优化过程中。当跟踪丢失时，跟踪线程在Atlas的所有地图中重新定位当前帧。如果重新定位成功，就恢复跟踪，如果需要的话，就切换活跃地图。否则，在一定时间之后，活跃地图就存储未非活跃的，从头初始化一个新的活跃地图。

• Local mapping thread adds keyframes and points to the active map, removes the redundant ones, and refines the map using visual or visual-inertial bundle adjustment, operating in a local window of keyframes close to the current frame. Additionally, in the inertial case, the IMU parameters are initialized and refined by the mapping thread using our novel MAP-estimation technique.

局部建图线程对活跃地图添加关键帧和地图点，移除冗余，使用视觉或视觉-惯性BA精炼地图，在当前帧附近的关键帧局部窗口上进行运算。另外，在惯性的情况下，IMU参数的初始化和精炼，是用我们新的MAP估计技术由建图线程完成。

• Loop and map merging thread detects common regions between the active map and the whole Atlas at keyframe rate. If the common area belongs to the active map, it performs loop correction; if it belongs to a different map, both maps are seamlessly merged into a single one, that becomes the active map. After a loop correction, a full BA is launched in an independent thread to further refine the map without affecting real-time performance.

回环和地图融合线程，以帧率的速度检测活跃地图和整个Atlas之间的共同区域。如果共同区域属于活跃地图，就进行回环修正；如果属于不同的地图，那么两个地图就无缝融合成一个地图，然后成为活跃地图。在回环修正后，在一个独立线程中启动完整BA，以在不影响实时性能的情况下，进一步精炼地图。

## 4. Camera Model

ORB-SLAM assumed in all system components a pin-hole camera model. Our goal is to abstract the camera model from the whole SLAM pipeline by extracting all properties and functions related to the camera model (projection and unprojection functions, Jacobian, etc.) into separate modules. This allows our system to use any camera model by providing the corresponding camera module. In ORB-SLAM3 library, apart from the pin-hole model, we provide the Kannala-Brandt [12] fisheye model.

ORB-SLAM在所有系统组成部分中，都假设针孔相机模型。我们的目标是，通过提取与相机模型相关的所有属性和函数（投影函数，反投影函数，Jacobian，等），成为单独的模块，将相机模型从整个SLAM过程中抽象出来。这使得我们的系统可以使用任意相机模型，只要给定对应的相机模块。在ORB-SLAM3库中，除了针孔模型，我们还给出了Kannala-Brandt[12]鱼眼模型。

As most popular computer vision algorithms assume a pin-hole camera model, many SLAM systems rectify either the whole image, or the feature coordinates, to work in an ideal planar retina. However, this approach is problematic for fisheye lenses, that can reach or surpass a field of view (FOV) of 180 degrees. Image rectification is not an option as objects in the periphery get enlarged and objects in the center loose resolution, hindering feature matching. Rectifying the feature coordinates requires using less than 180 degrees FOV and causes trouble to many computer vision algorithms that assume uniform reprojection error along the image, which is far from true in rectified fisheye images. This forces to crop-out the outer parts of the image, losing the advantages of large FOV: faster mapping of the environment and better robustness to occlusions. Next, we discuss how to overcome these difficulties.

由于多数流行的计算机视觉算法都假设针孔相机模型，很多SLAM系统要么校正整幅图像，要么是特征坐标点，以在理想的平面视网膜中进行工作。但是，这种方法对鱼眼镜头是有问题的，这种镜头可以达到或超过180度FOV。图像校正不能成为一种选项，因为在外围的目标被放大了，而中央的目标则失去了分辨率，阻碍了特征匹配。修正特征坐标点，则需要使用小于180度的FOV，很多计算机视觉算法假设沿着图像的重投影误差是均匀的，对这些算法会造成很大的麻烦，在修正的鱼眼图像中，远不是这种情况。这就被迫将图像的外围部分剪切出来，失去了大FOV的优势：对环境更快的建图，对遮挡有更好的稳健性。下面，我们讨论如何克服这些困难。

### 4.1. Relocalization

A robust SLAM system needs the capability of relocalizing the camera when tracking fails. ORB-SLAM solves the relocalization problem by setting a Perspective-n-Points solver based on the ePnP algorithm [73], which assumes a calibrated pin-hole camera along all its formulation. To follow up with our approach, we need a PnP algorithm that works independently of the camera model used. For that reason, we have adopted Maximum Likelihood Perspective-n-Point algorithm (MLPnP) [74] that is completely decoupled from the camera model as it uses projective rays as input. The camera model just needs to provide an unprojection function passing from pixels to projection rays, to be able to use relocalization.

一个稳健的SLAM系统，需要在跟踪丢失的时候，具有重定位相机的能力。ORB-SLAM通过设置一个基于ePnP算法的PnP求解器，来解决重定位问题，这假定在所有公式中都有一个校准过的针孔相机。为跟上我们的方法，我们需要一个PnP算法，与所使用的相机模型无关。为此，我们采用了最大似然PnP算法[74]，完全与相机模型解耦，因为它使用投影射线作为输入。相机模型只需要给出从像素到投影射线的反投影函数，以能够使用重定位。

### 4.2. Non-rectified Stereo SLAM

Most stereo SLAM systems assume that stereo frames are rectified, i.e. both images are transformed to pin-hole projections using the same focal length, with image planes co-planar, and are aligned with horizontal epipolar lines, such that a feature in one image can be easily matched by looking at the same row in the other image. However the assumption of rectified stereo images is very restrictive and, in many applications, is neither suitable nor feasible. For example, rectifying a divergent stereo pair, or a stereo fisheye camera would require severe image cropping, loosing the advantages of a large FOV.

多数立体SLAM系统都假设，立体视觉帧是校正过的，即，两幅图像都变换到了针孔投影，使用了相同的焦距，图像平面同平面，与水平对极线是对齐的，这样一幅图像中的一个特征可以很容易的匹配，只要查看另一幅图像同样的行。但是，假设立体图像是校正过的，这个限制是很大的，在很多应用中，既不合适，也不可行。比如，校正一个发散的立体图像对，或一个立体鱼眼相机，这需要严重的图像剪切，损失了大FOV的优势。

For that reason, our system does not rely on image rectification, considering the stereo rig as two monocular cameras having: 为此，我们的系统并不依赖于图像校正，将立体设备考虑为两个单目相机，有如下性质：

1) a constant relative SE(3) transformation between them, and 在其之间有不变的相对SE(3)变换

2) optionally, a common image region that observes the same portion of the scene. 可能有共同的图像区域，观察的是一个场景中的同样部分。

These constrains allow us to effectively estimate the scale of the map by introducing that information when triangulating new landmarks and in the bundle adjustment optimization. Following up with this idea, our SLAM pipeline estimates a 6 DoF rigid body pose, whose reference system can be located in one of the cameras or in the IMU sensor, and represents the cameras with respect to the rigid body pose.

这种约束使我们可以有效的估计地图的尺度，在对新的特征点进行三角定位和在BA优化时引入这种信息就可以了。按照这个思想，我们的SLAM流程估计了一个6 DoF刚体姿态，其参考系统可以定位到其中一个相机或在IMU传感器上，并对刚体姿态来表示相机。

If both cameras have an overlapping area in which we have stereo observations, we can triangulate true scale landmarks the first time they are seen. The rest of both images still has a lot of relevant information that is used as monocular information in the SLAM pipeline. Features first seen in these areas are triangulated from multiple views, as in the monocular case.

如果两个相机有重叠的区域，这样我们就有了立体视觉观测，我们可以对真正尺度的特征点，在第一次看到的时候进行三角定位。两幅图像的剩余部分仍然有大量相关信息，可在SLAM流程中用作单目信息。在这些区域中第一次看到的特征，从多个视角中进行三角定位，就像在单目的情况中一样。

## 5. Visual-Inertial SLAM

ORB-SLAM-VI [4] was the first true visual-inertial SLAM system capable of map reusing. However, it was limited to pin-hole monocular cameras, and its initialization was too slow, failing in some challenging scenarios. In this work, we build on ORB-SLAM-VI providing a fast and accurate IMU initialization technique, and an open-source SLAM library capable of monocular-inertial and stereo-inertial SLAM, with pin-hole and fisheye cameras.

ORB-SLAM-VI是第一个真正的能够进行地图重用的视觉-惯性SLAM系统。但是，局限于针孔单目相机，而且其初始化非常慢，在一些有挑战的场景中会失败。在本文中，我们以ORB-SLAM-VI为基础，给出了一个快速精确的IMU初始化技术，和一个开源SLAM库能够进行单目-惯性和立体-惯性SLAM，可以在针孔和鱼眼相机中适用。

### 5.1. Fundamentals

While in pure visual SLAM, the estimated state only includes the current camera pose, in visual-inertial SLAM, additional variables need to be computed. These are the body pose $T_i = [R_i, p_i] ∈ SE(3)$ and velocity $v_i$ in the world frame, and the gyroscope and accelerometer biases, $b^g_i$ and $b^a_i$, which are assumed to evolve according to a Brownian motion. This leads to the state vector:

在纯视觉SLAM中，估计的状态只包括当前相机的姿态，在视觉-惯性SLAM中，需要计算额外的变量。这是世界框架中的体姿态$T_i = [R_i, p_i] ∈ SE(3)$和速度$v_i$，陀螺仪和加速度计的偏差，$b^g_i$和$b^a_i$，一般假设是按照布朗运动进行演化。这带来了状态向量：

$$S_i = \{ T_i, v_i, b^g_i, b^a_i \}$$(1)

For visual-inertial SLAM, we preintegrate IMU measurements between consecutive visual frames, i and i+1, following the theory developed in [60], and formulated on manifolds in [61]. We obtain preintegrated rotation, velocity and position measurements, denoted as $∆R_{i,i+1}$, $∆v_{i,i+1}$ and $∆p_{i,i+1}$, as well a covariance matrix $Σ_{I_{i,i+1}}$ for the whole measurement vector. Given these preintegrated terms and states $S_i$ and $S_{i+1}$, we adopt the definition of inertial residual $r_{I_{i,i+1}}$ from [61]:

对于视觉-惯性SLAM，我们在连续视觉帧i和i+1之间预先整合IMU的测量，按照[60]中提出的理论，并在[61]的流形上给出公式。我们得到了预先整合的旋转，速度和位置测量，表示为$∆R_{i,i+1}$, $∆v_{i,i+1}$ 和 $∆p_{i,i+1}$，以及对于整个测量向量的一个协方差矩阵$Σ_{I_{i,i+1}}$。给定这些预先整合的项，和状态$S_i$和$S_{i+1}$，我们采用[61]中的惯性残差的定义$r_{I_{i,i+1}}$：

$$r_{I_{i,i+1}} = [r_{∆R_{i,i+1}}, r_{∆v_{i,i+1}}, r_{∆p_{i,i+1}}], \\ r_{∆R_{i,i+1}} = Log(∆R^T_{i,i+1} R^T_i R_{i+1}), \\ r_{∆v_{i,i+1}} = R^T_i (v_{i+1} − v_i − g∆t_{i,i+1}) − ∆v_{i,i+1}, \\ r_{∆p_{i,i+1}} = R^T_i (p_j − p_i − v_i ∆t_{i,i+1} − g∆t^2/2) − ∆p_{i,i+1}$$(2)

where Log : SO(3) → R^3 maps from the Lie group to the vector space. Together with inertial residuals, we also use reprojection errors r_ij between frame i and 3D point j at position x_j:

其中Log : SO(3) → R^3是从李群到向量空间的映射。与惯性残差一起，我们还用到了帧i和3D点j在位置x_j上的重投影误差r_ij：

$$r_{ij} = u_{ij} − Π(T_{CB} T^{−1}_i ⊕ x_j)$$(3)

where Π : R^3 → R^n is the projection function for the corresponding camera model, u_ij is the observation of point j at image i, having a covariance matrix Σ_ij, T_CB ∈ SE(3) stands for the rigid transformation from body-IMU to camera (left or right), known from calibration, and ⊕ is the transformation operation of SE(3) group over R^3 elements.

其中Π : R^3 → R^n是对应相机模型的投影函数，u_ij是点j在图像i中的观测值，有协方差矩阵Σ_ij，T_CB ∈ SE(3)表示从体IMU到相机（左或右）的刚性变换，从标定中可以得到，⊕是SE(3)群中的变换运算对R^3元素的应用。

Combining inertial and visual residual terms, visual-inertial SLAM can be posed as a keyframe-based minimization problem [39]. Given a set of k + 1 keyframes and its state $\bar S_k$ = {S_0 ... S_k}, and a set of l 3D points and its state X = {x_0 ... x_l−1}, the visual-inertial optimization problem can be stated as:

将惯性残差和视觉残差项结合到一起，视觉-惯性SLAM可以表示为一个基于关键帧的最小化问题[39]。给定k+1个关键帧的集合，和其状态$\bar S_k$ = {S_0 ... S_k}，以及l个3D点的集合，及其状态X = {x_0 ... x_l−1}，视觉-惯性优化问题可以表述为

$$min_{\bar S_k, X} (\sum_{i=1}^k ||r_{I_{i-1,i}}||^2_{Σ_{I_{i, i+1}^{-1}}} + \sum_{j=0}^{l-1} \sum_{i∈K^j} ρ_{Hub} (||r_{ij}||_{Σ_{ij}^{-1}}))$$(4)

where K^j is the set of keyframes observing 3D point j. This optimization may be outlined as the factor-graph shown in figure 2a. Note that for reprojection error we use a robust Huber kernel ρ_Hub to reduce the influence of spurious matchings, while for inertial residuals it is not needed, since miss-associations do not exist. This optimization needs to be adapted for efficiency during tracking and mapping, but more importantly, it requires good initial seeds to converge to accurate solutions.

其中K^j是观察3D点j的关键帧的集合。这个优化可以表示为图2a中的因子-图。注意，对于重投影误差，我们适用了一种稳健的Huber核ρ_Hub，来减少虚假匹配的影响，而惯性残差则不需要这个，因为误关联是不存在的。这个优化需要在跟踪和建图的过程中进行调整，以获得更高的效率，但更重要的是，需要很好的初始种子，以收敛到精确的解

### 5.2. IMU Initialization

The goal of this step is to obtain good initial values for the inertial variables: body velocities, gravity direction, and IMU biases. Some systems like VI-DSO [46] try to solve from scratch visual-inertial BA, sidestepping a specific initialization process, obtaining slow convergence for inertial parameters (up to 30 seconds).

这个步骤的目标，是对惯性变量得到很好的初始值：体速度，重力方向，和IMU偏差。一些系统，如VI-DSO[46]试图从头开始解决视觉-惯性BA，回避了特定的初始化过程，收敛到惯性参数很慢（达30s）。

In this work we propose a fast and accurate initialization method based on three key insights: 本文中，我们提出一种快速精确的初始化方法，基于三种关键的洞见：

• Pure monocular SLAM can provide very accurate initial maps [2], whose main problem is that scale is unknown. Solving first the vision-only problem will enhance IMU initialization. 纯单目SLAM可以给出非常准确的初始地图[2]，其主要问题是，尺度是未知的。首先求解这个只有视觉的问题，会有助于IMU的初始化。

• As shown in [56], scale converges much faster when it is explicitly represented as an optimization variable, instead of using the implicit representation of BA. 如[56]所示，尺度在显式的表示为一个优化变量时，会收敛的更快，而不是适用BA的隐式表示。

• Ignoring sensor uncertainties during IMU initialization produces large unpredictable errors [64]. 在IMU的初始化过程中，忽略传感器的不确定性，会产生很大的不可预测的误差。

So, taking properly into account sensor uncertainties, we state the IMU initialization as a MAP estimation problem, split in three steps:

所以，将传感器的不确定性合适的进行考虑，我们将IMU初始化问题表述为一个MAP估计问题，分成三个步骤：

1) Vision-only MAP Estimation: We initialize pure monocular SLAM [2] and run it during 2 seconds, inserting keyframes at 4Hz. After this period, we have an up-to-scale map composed of k = 10 camera poses and hundreds of points, that is optimized using visual-only BA (figure 2b). These poses are transformed to body reference, obtaining the trajectory $\bar T_{0:k} = [R, \bar p]_{0:k}$ where the bar denotes up-to-scale variables in the monocular case.

只有视觉的MAP估计：我们初始化纯单目SLAM [2]，运行2秒，以4Hz的频率插入关键帧。在这个阶段后，我们有了一个依赖于尺度的地图，由k=10个相机姿态，和数百个点组成，适用只有视觉的BA进行优化（图2b）。这些姿态变换到体参考，得到轨迹$\bar T_{0:k} = [R, \bar p]_{0:k}$，其中上横线表示在单目的情况中依赖于尺度的变量。

2) Inertial-only MAP Estimation: In this step we aim to obtain the optimal estimation of the inertial variables, in the sense of MAP estimation, using only $\bar T_{0:k}$ and inertial measurements between these keyframes. These inertial variables may be stacked in the inertial-only state vector:

只有惯性的MAP估计：在这个步骤中，我们的目标是得到惯性变量的MAP估计意义上的最佳估计，只使用$\bar T_{0:k}$和在这些关键帧之间的惯性测量。这些惯性变量可以在只有惯性的状态向量中堆叠：

$$Y_k = \{ s, R_{wg}, b, \bar v_{0:k} \}$$(5)

where $s ∈ R^+$ is the scale factor of the vision-only solution; $R_{wg} ∈ SO(3)$ is a rotation matrix, used to compute gravity vector g in the world reference as $g = R_{wg}g_I$, where $g_I = (0, 0, G)^T$ and G is the gravity magnitude; $b = (b_a, b_g) ∈ R^6$ are the accelerometer and gyroscope biases assumed to be constant during initialization; and $\bar v_{0:k} ∈ R^3$ is the up-to-scale body velocities from first to last keyframe, initially estimated from $\bar T_{0:k}$. At this point, we are only considering the set of inertial measurements $I_{0:k} = \{I_{0,1} . . . I_{k−1,k} \}$. Thus, we can state a MAP estimation problem, where the posterior distribution to be maximized is:

其中$s ∈ R^+$是尺度因子，用于只有视觉的解；$R_{wg} ∈ SO(3)$是一个旋转矩阵，用于计算世界参考坐标系中的梯度向量g，为$g = R_{wg}g_I$，其中$g_I = (0, 0, G)^T$，G是重力；$b = (b_a, b_g) ∈ R^6$是加速度计和陀螺仪的偏差，在初始化的过程中被认为是常数；$\bar v_{0:k} ∈ R^3$是根据尺度的体速度，从第一帧到最后一帧关键帧，从$\bar T_{0:k}$中得到初始估计。在这一点上，我们只考虑惯性测量$I_{0:k} = \{I_{0,1} . . . I_{k−1,k} \}$的集合。因此，我们可以表述一个MAP估计问题，其中要最大化的后验概率分布为

$$p(Y_k|I_{0:k}) ∝ p(I_{0:k}|Y_k)p(Y_k)$$(6)

where $p(I_{0:k}|Y_k)$ stands for likelihood and $p(Y_k)$ for prior. Considering independence of measurements, the inertial-only MAP estimation problem can be written as:

其中$p(I_{0:k}|Y_k)$表示似然，$p(Y_k)$表示先验。考虑测量的独立性，只有惯性的MAP估计问题可以写为：

$$Y^∗_k = argmax_{Y_k} (p(Y_k) \prod_{i=1}^k p(I_{i−1,i}|s, R_{wg}, b, \bar v_{i−1}, \bar v_i))$$(7)

Taking negative logarithm and assuming Gaussian error for IMU preintegration and prior distribution, this finally results in the optimization problem:

取-log，并假设IMU预整合和先验分布为高斯误差，这最后得到了优化问题：

$$Y^∗_k = argmax_{Y_k} (||b||^2_{Σ_b^{-1}} + \sum_{i=1}^k ||r_{I_{i-1,i}}||^2_{Σ^{-1}_{I_{i-1,i}}})$$(8)

This optimization, represented in figure 2c, differs from equation 4 in not including visual residuals, as the up-to-scale trajectory estimated by visual SLAM is taken as constant, and adding a prior residual that forces IMU biases to be close to zero. Covariance matrix Σ_b represents prior knowledge about the range of values IMU biases may take. Details for preintegration of IMU covariance $Σ_{I_{i−1,i}}$ can be found at [61].

这个优化如图2c表示，与式(4)不同的是，没有包含视觉残差，因为视觉SLAM估计的依赖于尺度的轨迹，被认为是常数，加入了一个先验残差，迫使IMU偏差接近于0。协方差矩阵Σ_b表示IMU偏差的范围的先验知识。IMU预整合的协方差矩阵$Σ_{I_{i−1,i}}$的细节可以在[61]中找到。

As we are optimizing in a manifold we need to define a retraction [61] to update $R_{wg}$ during the optimization. Since rotation around gravity direction does not suppose a change in gravity, this update is parameterized with two angles ($δα_g, δβ_g$):

因为我们在一个流形中进行优化，我们需要定义一个retraction [61]来在优化的过程中更新$R_{wg}$。由于围绕重力方向的旋转并不会改变重力，这个更新的参数应当包含2个角度($δα_g, δβ_g$)：

$$R_{wg}^{new} = R_{wg}^{old} Exp(δα_g, δβ_g, 0)$$(9)

being Exp(.) the exponential map from R^3 to SO(3). To guarantee that scale factor remains positive during optimization we define its update as: Exp(.)为R^3到SO(3)的指数图。为确保尺度因子在优化的过程中保持为正，我们定义其更新为：

$$s^{new} = s^{old} exp(δs)$$(10)

Once the inertial-only optimization is finished, the frame poses and velocities and the 3D map points are scaled with the estimated scale factor and rotated to align the z axis with the estimated gravity direction. Biases are updated and IMU preintegration is repeated, aiming to reduce future linearization errors.

一旦只有惯性的优化结束了，帧的姿态和速度和3D地图点就用估计的尺度因子进行缩放并旋转，以将z轴与估计的重力方向进行对齐。偏差进行了更新，IMU预整合进行了重复，目标是降低未来的线性化误差。

3) Visual-Inertial MAP Estimation: Once we have a good estimation for inertial and visual parameters, we can perform a joint visual-inertial optimization for further refining the solution. This optimization may be represented as figure 2a but having common biases for all keyframes and including the same prior information for biases than in the inertial-only step.

视觉-惯性MAP估计。一旦我们对惯性和视觉参数有了很好的估计，我们可以进行视觉-惯性联合优化，以进一步对解进行精炼。这个优化可以表示为图2a，但是对所有关键帧都有一样的偏差，与只有惯性的步骤相比，包含对偏差的同样先验信息。

Our exhaustive initialization experiments on the EuRoC dataset [6] show that this initialization is very efficient, achieving 5% scale error with trajectories of 2 seconds. To improve the initial estimation, visual-inertial BA is performed 5 and 15 seconds after initialization, converging to 1% scale error as shown in section VII. After these BAs, we say that the map is mature, meaning that scale, IMU parameters and gravity directions are already accurately estimated.

我们在EuRoC数据集上的全面初始化试验表明，这个初始化是非常高效的，用2秒的轨迹，尺度误差只有5%。为改进初始估计，在初始化后5秒和15秒进行视觉-惯性BA，收敛到1%的尺度误差，第7部分有展示。在这些BA后，我们说地图就成熟了，意思是尺度，IMU参数和重力方向就已经被准确的估计了。

Our initialization is much more accurate than joint initialization methods that solve a set of algebraic equations [62]–[64], and much faster than the initialization used in ORB-SLAM-VI [4] that needed 15 seconds to get the first scale estimation, or that used in VI-DSO [46], that starts with a huge scale error and requires 20-30 seconds to converge to 1% error. Comparisons between different initialization methods may be found at [6].

我们的初始化比求解代数方程组的联合初始化方法要准确的多，比ORB-SLAM-VI里需要15秒才能得到第一次尺度估计的初始化要快的多，以及VI-DSO里使用的，以巨大的尺度误差开始，需要20-30秒才能收敛到1%的误差。不同初始化方法的比较可以见[6]。

In some specific cases, when slow motion does not provide good observability of the inertial parameters, initialization may fail to converge to accurate solutions in just 15 seconds. To get robustness against this situation, we propose a novel scale refinement technique, based on a modified inertial-only optimization, where all inserted keyframes are included but scale and gravity direction are the only parameters to be estimated (figure 2d). Notice that in that case, the assumption of constant biases would not be correct. Instead, we use the values estimated from mapping, and we fix them. This optimization, which is very computationally efficient, is performed in the Local Mapping thread every ten seconds, until the map has more than 100 keyframes, or more than 75 seconds have passed since initialization.

在一些特定情况中，当缓慢的运动不能给出惯性参数很好的观测，初始化可能不能在15秒内收敛到精确的解。为对这种情况得到稳健性，我们提出了一种新的尺度精炼技术，基于一种修正的只有惯性的优化，其中包含所有插入的关键帧，但只有尺度和重力方向进行估计（图2d）。注意，在这种情况下，不变偏移的假设可能不是正确的。我们使用了从建图中估计的值，对其进行修正。这种优化计算量很小，在局部建图线程中每10秒运行一次，直到地图的关键帧数量超过100，或者自从初始化已经超过了75秒。

Finally, we have easily extended our monocular-inertial initialization to stereo-inertial by fixing the scale factor to one and taking it out from the inertial-only optimization variables, enhancing its convergence.

最后，我们很容易了将我们的单目-惯性初始化拓展到了立体-惯性系统，固定尺度因子为1，将其只有惯性的优化变量中取出，增强了其收敛能力。

### 5.3. Tracking and Mapping

For tracking and mapping we adopt the schemes proposed in [4]. Tracking solves a simplified visual-inertial optimization where only the states of the last two frames are optimized, while map points remain fixed.

对于跟踪和建图，我们采用[4]中的方案。跟踪求解的是一种简化的视觉-惯性优化，其中只要优化最后两针的状态，而地图点则保持固定。

For mapping, trying to solve the whole optimization from equation 4 would be intractable for large maps. We use as optimizable variables a sliding window of keyframes and their points, including also observations to these points from covisible keyframes but keeping their pose fixed.

对于建图，求解式4的完全优化对于大型地图可能是不可行的。我们将关键帧的滑窗及其点作为可优化的变量，包括从互可见帧中对这些点的观察，但保持其姿态固定。

### 5.4. Robustness to tracking loss

In pure visual SLAM or VO systems, temporal camera occlusion and fast motions result in losing track of visual elements, getting the system lost. ORB-SLAM pioneered the use of fast relocalization techniques based on bag-of-words place recognition, but they proved insufficient to solve difficult sequences in the EuRoC dataset [3]. Our visual-inertial system enters into visually lost state when less than 15 point maps are tracked, and achieves robustness in two stages:

在纯视觉SLAM或VO系统中，时域中的相机遮挡和快速运动会导致视觉元素的跟踪丢失，使系统丢失。ORB-SLAM率先使用快速重定位技术，基于bag-of-words位置识别，但在EuRoC数据集这样的困难序列中，求解效率很低。我们的视觉-惯性系统，在跟踪的地图点少于15个时，进入视觉丢失状态，然后用两个阶段获得稳健性：

• Short-term lost: the current body state is estimated from IMU readings, and map points are projected in the estimated camera pose and searched for matches within a large image window. The resulting matches are included in visual-inertial optimization. In most cases this allows to recover visual tracking. Otherwise, after 5 seconds, we pass to the next stage.

短期丢失：当前的体状态由IMU读数来估计，地图点投影到估计的相机姿态中，在大的图像窗口中搜索匹配点。得到的匹配在视觉-惯性优化中使用。在多数情况下，这会恢复视觉跟踪。否则，在5秒后，我们进入下一阶段。

• Long-term lost: A new visual-inertial map is initialized as explained above, and it becomes the active map. 长期丢失：初始化一个新的视觉-惯性地图，上面已经进行了解释，这个地图成为活跃地图。

If the system gets lost within 15 seconds after IMU initialization, the map is discarded. This prevents to accumulate inaccurate and meaningless maps. 如果系统在IMU初始化后15秒内丢失，地图就抛弃掉。这避免了不准确、无意义地图的累积。

## 6. Map Merging and Loop Closing

Short-term and mid-term data-associations between a frame and the active map are routinely found by the tracking and mapping threads by projecting map points into the estimated camera pose and searching for matches in an image window of just a few pixels. To achieve long-term data association for relocalization and loop detection, ORB-SLAM uses the DBoW2 bag-of-words place recognition system [9], [75]. This method has been also adopted by most recent VO and SLAM systems that implement loop closures (Table I).

跟踪和建图线程，将地图点投影到估计的相机姿态中，在只有几个像素的图像窗口中搜索匹配点，可以惯常的发现一帧图像和活跃地图的短期和中期的数据关联。为对重定位和回环检测得到长期数据关联，ORB-SLAM使用DBoW2 bag-of-words位置识别系统。最近实现了回环闭合的多数VO和SLAM系统，都采用了这个方法（表1）。

Unlike tracking, place recognition does not start from an initial guess for camera pose. Instead, DBoW2 builds a database of keyframes with their bag-of-words vectors, and given a query image is able to efficiently provide the most similar keyframes according to their bag-of-words. Using only the first candidate, raw DBoW2 queries achieve precision and recall in the order of 50-80% [9]. To avoid false positives that would corrupt the map, DBoW2 implements temporal and geometric consistency checks moving the working point to 100% precision and 30-40% recall [9], [75]. Crucially, the temporal consistency check delays place recognition at least during 3 keyframes. When trying to use it in our Atlas system, we found that this delay and the low recall resulted too often in duplicated areas in the same or in different maps.

与跟踪不一样的是，位置识别不是从相机姿态的初始估计开始。DBoW2构建了用关键帧的bag-of-words向量构建了其数据库，给定一幅查询图像，会根据其bag-of-words，高效的给出最类似的关键帧。只使用第一个候选，原始DBoW2查询获得的精度和召回在50%-80%的级别。为避免假阳性会损害地图，DBoW2实现了时域和几何的一致性检查，将工作点移到了100%的精度和30%-40%的召回。关键是，时域一致性检查将位置识别退后了至少3个关键帧。当在我们的Atlas系统中使用时，我们发现，这种延迟和低召回通常会导致在相同或不同地图中的重复区域。

In this work we propose a new place recognition algorithm with improved recall for long-term and multi-map data association. Whenever the mapping thread creates a new keyframe, place recognition is launched trying to detect matches with any of the keyframes already in the Atlas. If the matching keyframe found belongs to the active map, a loop closure is performed. Otherwise, it is a multi-map data association, then, the active and the matching maps are merged. As a second novelty in our approach, once the relative pose between the new keyframe and the matching map is estimated, we define a local window with the matching keyframe and its neighbours in the covisibility graph. In this window we intensively search for mid-term data associations, improving the accuracy of loop closing and map merging. These two novelties explain the better accuracy obtained by ORB-SLAM3 compared with ORB-SLAM2 in the EuRoC experiments. The details of the different operations are explained next.

本文中，我们提出了一种新的位置识别算法，对长期和多地图数据关联改进了召回。不论什么时候建图线程创建了一个新的关键帧，就启动位置识别，检测与已经在Atlas中的任意关键帧的匹配。如果找到的匹配关键帧属于活跃地图，就进行回环闭合。否则，这就是一个多地图数据关联，那么，活跃地图和匹配地图就进行融合。我们方法的第二个创新是，一旦估计出了新的关键帧和匹配地图之间的相对姿态，我们就用匹配的关键帧及其互可见图中的邻域定义一个局部窗口。在这个窗口中，我们广泛搜索中期数据关联，改进回环闭合和地图融合的准确率。ORB-SLAM3与ORB-SLAM2相比，在EuRoC数据集中得到了更好的准确率，这两个创新就可以解释。不同操作的细节在下面进行解释。

### 6.1. Place Recognition

To achieve higher recall, for every new active keyframe we query the DBoW2 database for several similar keyframes in the Atlas. To achieve 100 % precision, each of these candidates goes through several steps of geometric verification. The elementary operation of all the geometrical verification steps consists in checking whether there is an ORB keypoint inside an image window whose descriptor matches the ORB descriptor of a map point, using a threshold for the Hamming distance between them. If there are several candidates in the search window, to discard ambiguous matches, we check the distance ratio to the second-closest match [76]. The steps of our place recognition algorithm are:

为得到更高的召回，对每个新的活跃关键帧，我们查询DBoW2数据库，获得Atlas中的类似关键帧。为获得100%的精度，这些候选中的每个都经过几步几何验证。所有几何验证步骤的初级操作，都包括检查在图像窗口中，是否有一个ORB关键点，其描述子与一个地图点的ORB描述子匹配，使用的是其之间的Hamming距离的一个阈值。如果在搜索窗口中有几个候选，为抛弃疑义的匹配，我们检查距离率到第二接近的匹配[76]。我们的位置识别算法的步骤为：

1) DBoW2 candidate keyframes. We query the Atlas DBoW2 database with the active keyframe K_a to retrieve the three most similar keyframes, excluding keyframes covisible with K_a. We refer to each matching candidate for place recognition as K_m.

DBoW2候选关键帧。我们用活跃关键帧K_a，查询Atlas DBoW2数据库，得到三个最相似的关键帧，排除掉与K_a互可见的关键帧。我们称每个匹配的位置识别候选为K_m。

2) Local window. For each K_m we define a local window that includes K_m, its best covisible keyframes, and the map points observed by all of them. The DBoW2 direct index provides a set of putative matches between keypoints in K_a and in the local window keyframes. For each of these 2D-2D matches we have also available the 3D-3D match between their corresponding map points.

局部窗口。对每个K_m我们定义一个局部窗口，包括K_m，其最好的互可见帧，以及它们可以观察到的地图点。DBoW2的直接索引，给出了在K_a中的关键点，与局部窗口关键帧中的推定的匹配集合。对这些2D-2D匹配中的每一个，我们也有其对应的地图点之间的3D-3D匹配。

3) 3D aligning transformation. We compute using RANSAC the transformation T_am that better aligns the map points in K_m local window with those of K_a. In pure monocular, or in monocular-inertial when the map is still not mature, we compute T_am ∈ Sim(3), otherwise T_am ∈ SE(3). In both cases we use Horn algorithm [77] using a minimal set of three 3D-3D matches to find each hypothesis for T_am. The putative matches that, after transforming the map point in K_a by T_am, achieve a reprojection error in K_a below a threshold, give a positive vote to the hypothesis. The hypothesis with more votes is selected, provided the number is over a threshold.

3D对齐变换。我们使用RANSAC计算变换T_am，将K_m局部窗口中的地图点与K_a的地图点进行更好的对齐。在纯单目的情况下，或在单目-惯性的情况下，当地图仍然不成熟时，我们计算T_am ∈ Sim(3)，否则T_am ∈ SE(3)。在两种情况下，我们都使用Horn算法，使用三个3D-3D匹配的最小集合，来找到每个假设的T_am。用T_am对K_a中的地图点进行变换，获得在K_a中的重投影误差，低于一定阈值时，这样的推定匹配，对这个假设给出了一个正投票。如果投票数高于一定阈值，就选择有更多投票的假设。

4) Guided matching refinement. All the map points in the local window are transformed with T_am to find more matches with the keypoints in K_a. The search is also reversed, finding matches for K_a map points in all the keyframes of the local window. Using all the matchings found, T_am is refined by non-linear optimization, where the goal function is the bidirectional reprojection error, using Huber influence function to provide robustness to spurious matches. If the number of inliers after the optimization is over a threshold, a second iteration of guided matching and non-linear refinement is launched, using a smaller image search window.

引导的匹配改进。局部窗口中的所有地图点，用T_am进行变换，以找到与K_a中的关键点的更多匹配。搜索还反向进行，在局部窗口的所有关键帧中，找到K_a地图点的匹配。使用找到的匹配，T_am用非线性优化进行改进，目标函数是双向重投影误差，使用Huber影响函数来对错误匹配稳健。如果在优化后内点的数量大于一定阈值，就启动第二轮引导匹配和非线性优化，使用一个更小的图像搜索窗口。

5) Verification in three covisible keyframes. To avoid false positives, DBoW2 waited for place recognition to fire in three consecutive keyframes, delaying or missing place recognition. Our crucial insight is that, most of the time, the information required for verification is already in the map. To verify place recognition, we search in the active part of the map two keyframes covisible with K_a where the number of matches with points in the local window is over a threshold. If they are not found, the validation is further tried with the new incoming keyframes, without requiring the bag-of-words to fire again. The validation continues until three keyframes verify T_am, or two consecutive new keyframes fail to verify it.

在三个互可见关键帧中的验证。为避免假阳性，DBoW2等待位置识别在三个连续的关键帧中运行，这就推迟或错过了位置识别。我们关键的洞见是，在大多数时间中，验证需要的信息就已经在地图中了。为验证位置识别，我们在地图的活跃部分进行搜索，要两个关键帧与K_a是互可见的，在局部窗口中点的匹配的数量要大于一定阈值。如果没有找到，这个验证就进一步在新来的关键帧中进行尝试，不需要bag-of-words再次运行。这个验证持续下去，直到三个关键帧验证了T_am，或两个连续的新关键帧不能验证。

6) VI Gravity direction verification. In the visual-inertial case, if the active map is mature, we have estimated T_am ∈ SE(3). We further check whether the pitch and roll angles are below a threshold to definitively accept the place recognition hypothesis.

VI重力方向验证。在视觉-惯性的情况下，如果活跃地图成熟了，我们估计了T_am ∈ SE(3)。我们进一步检查，pitch和roll的角度是在一定阈值之下，以确切的接受位置识别的假设。

### 6.2. Visual Map Merging

When a successful place recognition produces multi-map data association between keyframe K_a in the active map M_a, and a matching keyframe K_m from a different map stored in the Atlas M_m, with an aligning transformation T_am, we launch a map merging operation. In the process, special care must be taken to ensure that the information in M_m can be promptly reused by the tracking thread to avoid map duplication. For this, we propose to bring the M_a map into M_m reference. As M_a may contain many elements and merging them might take a long time, merging is split in two steps. First, the merge is performed in a welding window defined by the neighbours of K_a and K_m in the covisibility graph, and in a second stage, the correction is propagated to the rest of the merged map by a pose-graph optimization. The detailed steps of the merging algorithm are:

当成功的位置识别产生了多地图关联，将活跃地图M_a中的关键帧K_a，和存储在Atlas中的一个不同的地图M_m中的关键帧K_m匹配上了，对齐变换为T_am，我们就发起地图融合操作。在这个过程中，必须注意确保，M_m中的信息可以由跟踪线性进行迅速的重用，以避免地图重复。为此，我们提出将M_a带到M_m的参考坐标系中。因为M_a可能会包含很多元素，合并耗时会较长，所以合并就分成两个步骤。第一，合并在一个结合窗口中进行，结合窗口由K_a和K_m在互可见图中的邻域定义，在第二阶段，修正传播到合并地图的剩下部分，进行姿态图优化。合并算法的详细步骤如下：

1) Welding window assembly. The welding window includes K_a and its covisible keyframes, K_m and its covisible keyframes, and all the map point observed by them. Before their inclusion in the welding window, the keyframes and map points belonging to M_a are transformed by T_ma to align them with respect to M_m.

结合窗口组合。结合窗口包含K_a及其互可见帧，K_m及其互可见帧，和它们观察到的所有地图点。在将其纳入到结合窗口中之前，属于M_a的关键帧和地图点由T_ma进行变换，以与M_m进行对齐。

2) Merging maps. Maps M_a and M_m are fused together to become the new active map. To remove duplicated points, matches are actively searched for M_a points in the M_m keyframes. For each match, the point from M_a is removed, and the point in M_m is kept accumulating all the observations of the removed point. The covisibility and essential graphs [2] are updated by the addition of edges connecting keyframes from M_m and M_a thanks to the new mid-term point associations found.

合并地图。地图M_a和M_m合并到一起，成为新的活跃地图。为移除重复的点，对M_a中的点，搜索到M_m的关键帧中的匹配。对每个匹配，在M_a中的点就移除掉，保留M_m中的点，累积移除的点的所有观察。互可见图和essential图要进行更新，加入连接M_m和M_a的关键帧的边，多亏了新发现的中期点关联。

3) Welding bundle adjustment. A local BA is performed optimizing all the keyframes from M_a and M_m in the welding window along with the map points which are observed by them (Fig. 3a). To fix gauge freedom, the keyframes of M_m not belonging to the welding window but observing any of the local map points are included in the BA with their poses fixed. Once the optimization finishes, all the keyframes included in the welding area can be used for camera tracking, achieving fast and accurate reuse of map M_m.

结合BA。对M_a和M_m在结合窗口中的所有关键帧，以及它们能观察到的所有地图点进行优化，即进行局部BA（图3a）。为固定测量的自由度，M_m中不属于结合窗口，但是观察到任意局部地图点的关键帧，也纳入到BA中，但保持其姿态固定。一旦优化结束，所有结合区域中的关键帧都可以用于相机跟踪，获得M_m的快速和精确的重用。

4) Essential-graph optimization. A pose-graph optimization is performed using the essential graph of the whole merged map, keeping fixed the keyframes in the welding area. This optimization propagates corrections from the welding window to the rest of the map.

Essential graph优化。使用整个合并的地图的essential graph，进行姿态图优化，保持结合区域中的关键帧固定。这个优化将修正从结合窗口传播到地图的剩余部分。

### 6.3. Visual-Inertial Map Merging

The visual-inertial merging algorithm follows similar steps than the pure visual case. Steps 1) and 3) are modified to better exploit the inertial information:

视觉-惯性合并算法与纯视觉情况下有着类似的步骤。步骤1)和3)要进行修正，以更好的利用惯性信息：

1) VI welding window assembly: If the active map is mature, we apply the available T_ma ∈ SE(3) to map M_a before its inclusion in the welding window. If the active map is not mature, we align M_a using the available T_ma ∈ Sim(3).

VI结合窗口组合：如果活跃地图是成熟的，我们对M_a在其纳入到结合窗口之前，使用可用的T_ma ∈ SE(3)。如果活跃地图还没成熟，我们就使用可用的T_ma ∈ Sim(3)来对齐M_a。

2) VI welding bundle adjustment: Poses, velocities and biases of keyframes K_a and K_m and their five last temporal keyframes are included as optimizable. These variables are related by IMU preintegration terms, as shown in Figure 3b. For M_m, the keyframe immediately before the local window is included but fixed, while for M_a the similar keyframe is included but its pose remains optimizable. All map points seen by the above mentioned keyframes are optimized, together with poses from K_m and K_a covisible keyframes. All keyframes and points are related by means of reprojection error.

VI结合BA：关键帧K_a和K_m，以及其最后5个时域上的关键帧，其姿态，速度和偏差都纳入为可优化的。这些变量由IMU预整合项进行关联，如图3b所示。对M_m，局部窗口前的最接近的关键帧也包括进来，但保持固定，而对于M_a，类似的关键帧也包括进来，但其姿态是可优化的。上述提到的所有关键帧的所有地图点，以及K_m和K_a互可见帧的姿态一起，都进行优化。所有关键帧和点由重投影误差来进行关联。

### 6.4. Loop Closing

Loop closing correction algorithm is analogous to map merging, but in a situation where both keyframes matched by place recognition belong to the active map. A welding window is assembled from the matched keyframes, and point duplicates are detected and fused creating new links in the covisibility and essential graphs. The next step is a pose-graph optimization to propagate the loop correction to the rest of the map. The final step is a global BA to find the MAP estimate after considering the loop closure mid-term and long-term matches. In the visual-inertial case, the global BA is only performed if the number of keyframes is below a threshold to avoid a huge computational cost.

回环闭合修正算法与地图融合类似，但位置识别匹配的两个关键帧都属于活跃地图。由匹配的关键帧组合出一个结合窗口，重复的点进行检测，并融合到一起，在互可见图和essential graph中创建新的连接。下一步史姿态图优化，将回环修正传播到剩余的地图部分。最后的步骤是全局BA，在考虑了回环闭合中期和长期匹配后的MAP估计。在视觉-惯性情况中，全局BA只在关键帧数量低于阈值时进行，以避免巨量的计算代价。

## 7. Experimental Results

The evaluation of the whole system is split in: 整个系统的评估分为：

• Single session experiments in EuRoC [79]: each of the 11 sequences is processed to produce a map, with the four sensor configurations: Monocular, Monocular-Inertial, Stereo and Stereo-Inertial.

在EuRoC中的单环节试验：11个序列中的每个都处理得到一个地图，有四种传感器配置，单目，单目-惯性，立体，和立体-惯性。

• Performance of monocular and stereo visual-inertial SLAM with fisheye cameras, in the challenging TUM VI Benchmark [80]. 带有鱼眼相机的单目和立体视觉-惯性SLAM的性能，在TUM VI基准测试中。

• Multi-session experiments in both datasets. 在两个数据集上的多环节试验。

As usual in the field, we measure accuracy with RMS ATE [81], aligning the estimated trajectory with ground-truth using a Sim(3) transformation in the pure monocular case, and a SE(3) transformation in the rest of sensor configurations. Scale error is computed using s from Sim(3) alignment, as |1 − s|. All experiments have been run on an Intel Core i7-7700 CPU, at 3.6GHz, with 32 GB memory, using only CPU.

我们用RMS ATE来测量准确性，将估计的轨迹与真值进行对齐，在纯单目的情况下，用Sim(3)变换，在其余传感器配置中，用SE(3)变换。尺度误差用Sim(3)对齐的s来进行计算，即|1 − s|。所有试验的运行，都是在Intel Core i7-7700 CPU上，3.6GHz，32G内存，只使用CPU。

### 7.1. Single-session SLAM on EuRoC

Table II compares the performance of ORB-SLAM3 using its four sensor configurations with the most relevant systems in the state-of-the-art. Our reported values are the median after 10 executions. As shown in the table, ORB-SLAM3 achieves in all sensor configurations more accurate result than the best systems available in the literature, in most cases by a wide margin.

表2比较了ORB-SLAM3在四种类传感器配置的情况下，与最相关的目前最好的系统的结果。我们给出的值是在10次运行结果后的中值。如表所示，ORB-SLAM3在所有传感器配置中，比最好的可用系统，都给出了更精确的结果，在很多情况下超出了很多。

In monocular and stereo configurations our system is more precise than ORB-SLAM2 due to the better place recognition algorithm that closes loops earlier and provides more mid-term matches. Interestingly, the next best results are obtained by DSM that also uses mid-term matches, even though it does not close loops.

在单目和立体的配置中，我们的系统比ORB-SLAM2更加精确，因为有更好的位置识别算法，会更早的闭合回环，给出更多的中期匹配。有趣的是，第二好的结果是DSM，也使用了中期匹配，但它并没有闭合回环。

In monocular-inertial configuration, ORB-SLAM3 is five to ten times more accurate than MCSKF, OKVIS and ROVIO, and more than doubles the accuracy of VI-DSO and VINS-Mono, showing again the advantages of mid-term and long-term data association. Compared with ORB-SLAM VI, our novel fast IMU initialization allows ORB-SLAM3 to calibrate the inertial sensor in a few seconds and use it from the very beginning, being able to complete all EuRoC sequences, and obtaining better accuracy.

在单目-惯性的配置中，ORB-SLAM3比MCSKF，OKVIS和ROVIO精确5-10倍，比VI-DSO和VINS-Mono的精确率高2倍以上，再次表明了中期和长期数据关联的优势。与ORB-SLAM VI相比，我们新的快速IMU初始化使ORB-SLAM3可以在几秒钟内校准惯性传感器，并在最开始的时候就使用，可以完成所有EuRoC序列，得到更好的准确率。

In stereo-inertial configuration, ORB-SLAM3 is three to four times more accurate than and Kimera and VINS-Fusion. It’s accuracy is only approached by the recent BASALT that, being a native stereo-inertial system, was not able to complete sequence V203, where some frames from one of the cameras are missing. Comparing our monocular-inertial and stereo-inertial systems, the latter performs better in most cases. Only for two Machine Hall (MH) sequences a lower accuracy is obtained. We hypothesize that greater depth scene for MH sequences may lead to less accurate stereo triangulation and hence a less precise scale.

在立体-惯性配置中，ORB-SLAM3比Kimera和VINS-Fusion要精确3-4倍。最近的BASALT接近其其准确度，因为是一个原生立体-惯性系统，但不能完成序列V203，其中一个相机的一些帧是缺失的。比较我们的单目-惯性和立体-惯性系统，后者在多数情况下表现更好。只有在两个MH序列中，得到了更低的准确率。我们推测，MH序列更大的深度场景，导致了立体三角测量没那么准确，因此精确程度没有那么高。

To summarize performance, we have presented the median of ten executions for each sensor configuration. For a robust system, the median represents accurately the behavior of the system. But a non-robust system will show high variance in its results. This can be analyzed using figure 4 that shows with colors the error obtained in each of the ten executions. Comparison with the figures for DSO, ROVIO and VI-DSO published in [46] confirms the superiority of our method.

为总结性能，对每种传感器配置，我们都给出了10次执行的中间结果。对于一个稳健的系统，中值准确的表示了系统行为。但不稳健的系统，其结果的方差会很高。这可以用图4进行分析，用颜色表明了10次执行中每次的误差。与[46]中发表的DSO，ROVIO和VI-DSO的图的比较，确认了我们方法的优越性。

In pure visual configurations, the multi-map system adds some robustness to fast motions by creating a new map when tracking is lost, that is merged later with the global map. This can be seen in sequences V103 monocular and V203 stereo that could not be solved by ORB-SLAM2 and are successfully solved by our system in most executions. As expected, stereo is more robust than monocular thanks to its faster feature initialization, with the additional advantage that the real scale is estimated.

在纯视觉的配置中，多地图系统对快速运动增加了一些稳健性，当跟踪丢失的时候，会创建一个新地图，后来会与全局地图进行合并。这可以在序列V103单目和V203立体中看到，ORB-SLAM2不能进行求解，但我们的系统在多数执行中都很成功的求解了。如同期待的一样，立体比单目更加稳健，因为特征初始化非常快速，而且其真实尺度也进行了估计，这也是一个优势。

However, the big leap in robustness is obtained by our novel visual-inertial SLAM system, both in monocular and stereo configurations. The stereo-inertial system has a very slight advantage over monocular-inertial, particularly in the most challenging V203 sequence.

但是，我们新的视觉-惯性SLAM系统才得到了稳健性的巨大提升，单目和立体的配置都是这种情况。立体-惯性系统比单目-惯性系统有些许优势，尤其是在最有挑战性的V203序列中。

We can conclude that inertial integration not only boosts accuracy, reducing the median ATE error compared to pure visual solutions, but it also endows the system with excellent robustness, having a much more stable performance.

我们可以得到结论，惯性系统的整合不仅提升了准确率，与纯视觉解决方案比，降低了中值ATE误差，而且还使系统有极好的稳健性，有更稳定的性能。

### 7.2. Visual-Inertial SLAM on TUM-VI Benchmark

The TUM-VI dataset [80] consists of 28 sequences in 6 different environments, recorded using a hand-held fisheye stereo-inertial rig. Ground-truth for the trajectory is only available at the beginning and at the end of the sequences, which for most of them represents a very small portion of the whole trajectory. Many sequences in the dataset do not contain loops. Even if the starting and ending point are in the same room, point of view directions are opposite and place recognition cannot detect any common region. Using this ground-truth for evaluation amounts to measuring the accumulated drift along the whole trajectory.

TUM-VI数据集包含6种不同环境的28个序列，使用手持式鱼眼立体-惯性设备进行的录制。轨迹的真值只在序列的开始和最后可用，它们的多数只表示了整个轨迹的一小部分。数据集中的很多序列，都不包含回环。即使开始点和结束点都在同样的房间中，观察点的方向是相反的，位置识别不能检测到任何共同区域。使用这种真值进行评估，等于估计沿着整个轨迹的累积漂移。

We extract 1500 ORB points per image in monocular-inertial setup, and 1000 points per image in stereo-inertial, after applying CLAHE equalization to address under and over exposure found in the dataset. For outdoors sequences, our system struggles with very far points coming from the cloudy sky, that is very visible in fisheye cameras. These points may have slow motion that can introduce drift in the camera pose. For preventing this, we discard points further than 20 meters from the current camera pose, only for outdoors sequences. A more sophisticated solution would be to use an image segmentation algorithm to detect and discard the sky.

在单目-视觉的设置中，我们对每幅图像提取了1500个ORB点，在立体-惯性的设置中，每幅图像提取1000个点，并在开始的时候先对数据集中发现的欠曝和过曝的情况进行了CLAHE均衡。对于室外序列，我们的系统难以处理有云蓝天上的很远的点，这在鱼眼相机中是可见的。这些点的运动很慢，会引入相机姿态的漂移。为避免这个，我们抛弃了相对于当前相机姿态远于20米的点，只对室外序列执行。一个更复杂的解决方案是，使用图像分隔算法来检测并丢弃天空。

The results obtained are compared with the most relevant systems in the literature in table III, that clearly shows the superiority of ORB-SLAM3 both in monocular-inertial and stereo-inertial. The closest systems are VINS-Mono and BASALT, that are essentially visual-inertial odometry systems with loop closures, and miss mid-term data associations.

得到的结果与文献中最相关的系统的比较，如表3所示，清楚的表明了ORB-SLAM3的优越性，在单目-惯性和立体-惯性的情况都是。最接近的系统是VINS-Mono和BASALT，它们都是带有回环闭合的VI里程计系统，但没有中期数据关联。

Analyzing more in detail the performance of our system, it gets lowest error in small and medium indoor environments, room and corridor sequences, with errors below 10 cm for most of them. In these trajectories, the system is continuously revisiting and reusing previously mapped regions, which is one of the main strengths of ORB-SLAM3. Also, tracked points are typically closer than 5 m, what makes easier to estimate inertial parameters, preventing them from diverging.

对系统性能进行更详细的分析，对小型和中型室内环境，房间和走廊的序列，得到了最低的误差，大部分误差低于10cm。在这些轨迹中，系统持续的重复访问并重用之前建图的区域，这是ORB-SLAM3系统的主要特点。而且，跟踪的点一般都在5m以内，这使估计惯性参数更加简单，防止发散。

In magistrale indoors sequences, that are up to 900 m long, most tracked points are relatively close, and ORB-SLAM3 obtains errors around 1 m except in one sequence that goes close to 5 m. In contrast, in some long outdoors sequences, the scarcity of close visual features may cause drift of the inertial parameters, notably scale and accelerometer bias, which leads to errors in the order of 10 to 70 meters. Even though, ORB-SLAM3 is the best performing system in the outdoor sequences.

在magistrale室内序列中，总计有900米长，多数跟踪的点是相对接近的，ORB-SLAM3得到了大约1m的误差，但在一个序列中误差接近5m。比较起来，在一些长的室外序列中，缺少接近的视觉特征，会导致惯性参数的漂移，可注意到的尺度和加速度计偏移，这会导致10到70m范围的误差。即使如此，ORB-SLAM3是室外序列表现最好的系统。

This dataset also contains three really challenging slides sequences, where the user descends though a dark tubular slide with almost total lack of visual features. In this situation, a pure visual system would be lost, but our visual-inertial system is able to process the whole sequence with competitive error, even if no loop-closures can be detected. Interestingly, VINS-Mono and BASALT, that track features using Lukas-Kanade, obtain in some of these sequences better accuracy than ORB-SLAM3, that matches ORB descriptors.

这个数据集还包含三个非常有挑战性的滑梯序列，其中用户通过一个暗的管状滑梯下降，里面几乎是完全黑暗的，缺少视觉特征。在这种情况下，纯视觉系统就会丢失，但我们的视觉-惯性系统可以处理整个序列，误差非常不错，虽然没有检测到任何回环闭合。有趣的是，VINS-Mono和BASALT，使用Lukas-Kanade来跟踪特征，在一些序列中可以比ORB-SLAM3得到更好的准确率，ORB-SLAM3采用的是ORB描述子进行匹配。

Finally, the room sequences can be representative of typical AR/VR applications, where the user moves with a hand-held or head-mounted device in a small environment. For these sequences ground-truth is available for the entire trajectory. Table III shows that ORB-SLAM3 is significantly more accurate that competing approaches. The results obtained using our four sensor configurations are compared in table IV. The better accuracy of pure monocular compared with stereo is only apparent: the monocular solution is up-to-scale and is aligned with ground-truth with 7 DoFs, while stereo provides the true scale, and is aligned with 6 DoFs. Using monocular- inertial, we further reduce the average RMS ATE error close to 1 cm, also obtaining the true scale. Finally, our stereo-inertial SLAM brings error below 1 cm, making it an excellent choice for AR/VR applications.

最后，room序列是典型的AR/VR应用的代表，其中用户用一个手持或头戴式的设备在一个小型环境中移动。对于这些序列，整个轨迹的真值都是可用的。表3表明，ORB-SLAM3与其他方法相比，明显更加精确。使用我们的四种传感器配置得到的结果，在表4中进行了比较。纯单目的准确率比立体相机更好，其原因是很明显的：单目的解是根据尺度的，与真值在7DoF上进行了对齐，而立体则给出了真的尺度，是用6DoFs进行对齐的。使用单目-惯性，我们进一步降低了平均RMS ATE误差，到了接近1cm，同时获得了真正的尺度。最后，我们的立体-惯性SLAM带来的误差小于1cm，使其成为AR/VR应用的优秀选择。

### 7.3. Multi-session SLAM

EuRoC dataset contains several sessions for each of its three environments: 5 in Machine Hall, 3 in Vicon1 and 3 in Vicon2. To test the multi-session performance of ORB-SLAM3, we process sequentially all the sessions corresponding to each environment. Each trajectory in the same environment has ground-truth with the same world reference, which allows to perform a single global alignment to compute ATE.

EuRoC数据集的3个环境中，每个都包含几个环节：Machine Hall中5个，Vicon1中3个，Vicon2中3个。为测试ORB-SLAM3的多环节性能，我们对每个环境中的所有环节进行顺序处理。同样环境中的每个轨迹都有真值，世界坐标系是一样的，使我们可以进行单个全局对齐，以计算ATE。

The first sequence in each room provides an initial map. Processing the following sequences starts with the creation of a new active map, that is quickly merged with the map of the previous sessions, and from that point on, ORB-SLAM3 profits from reusing the previous map.

在每个room中的第一个序列会给出一个初始地图。处理后面的序列，会创建一个新的活跃地图，但很快就会与前一个环节的地图合并到一起，从这一点开始，ORB-SLAM3就会从重用以前的地图中受益。

Table V reports the global multi-session RMS ATE for the four sensor configurations in the three rooms, comparing with the two only published multi-session results in EuRoC dataset: CCM-SLAM [71] that reports pure monocular results in MH01-MH03, and VINS-Mono [7] in the five Machine Hall sequences, using monocular-inertial. In both cases ORB-SLAM3 more than doubles the accuracy of competing methods. In the case of VINS-Mono, ORB-SLAM3 obtains 2.6 better accuracy in single-session, and the advantage goes up to 3.2 times in multi-session, showing the superiority of our map merging operations.

表5对三个rooms里四种传感器配置的情况，给出了全局的多环节RMS ATE，与仅有的两个发表的多环节结果在EuRoC数据集上进行了比较：CCM-SLAM在MH01-MH03上给出了纯单目的结果，VINS-Mono在5个Machine Hall序列上使用单目-惯性给出了结果。在两种情况中，ORB-SLAM3都比两种方法准确率高了至少2倍。在VINS-Mono中，ORB-SLAM3在单环节中得到了2.6倍更好的准确率，在多环节中，优势提升到了3.2倍，表明我们的地图合并操作的优势。

Comparing these multi-session performances with the single-session results reported in Table II the most notable difference is that multi-sessions monocular and stereo SLAM can robustly process the difficult sequences V103 and V203, thanks to the exploitation of the previous map.

将这些多环节的结果，与表2中的单环节结果进行比较，最值得注意的差异是，多环节单目和立体SLAM可以很稳健的处理困难序列V103和V203，这多亏了对之前的地图的利用。

We have also performed some multi-session experiments on the TUM-VI dataset. Figure 5 shows the result after processing several sequences inside the TUM building. In this case, the small room sequence provides loop closures that were missing in the longer sequences, bringing all errors to centimeter level. Although ground-truth is not available outside the room, comparing the figure with the figures published in [82] clearly shows our point: our multi-session SLAM system obtains far better accuracy that existing visual-inertial odometry systems. This is further exemplified in Figure 6. Although ORB-SLAM3 ranks higher in stereo inertial single-session processing of outdoors1, there is still a significant drift (≈ 60 m). In contrast, if outdoors1 is processed after magistrale2 in a multi-session manner, this drift is significantly reduced, and the final map is much more accurate.

我们还在TUM-VI数据集上进行了几个多环节试验。图5给出了TUM building内部的几个序列的处理的结果。在这种情况中，小型room序列给出了回环闭合，在更长的序列中是没有的，将所有的误差都带到了cm级别。虽然在room以外真值是没有，将图与[82]中的图进行比较，明显表明：我们的多环节SLAM系统，比现有的视觉-惯性里程计系统，得到了更好的准确率。这在图6中进一步展示。虽然ORB-SLAM3在立体-惯性单阶段outdoors1的处理排名更高，但仍然有明显的漂移(≈ 60 m)。比较之下，如果outdoor1在magistrale2之后，以多环节的方式处理，这个漂移就可以显著降低，最终的地图就会更加准确。

### 7.4. Computing Time

Table VI summarizes the running time of the main operations performed in the tracking and mapping threads, showing that our system is able to run in real time at 30-40 frames and at 3-6 keyframes per second. The inertial part takes negligible time during tracking and, in fact can render the system more efficient as the frame rate could be safely reduced. In the mapping thread, the higher number of variables per keyframe has been compensated with a smaller number of keyframes in the inertial local BA, achieving better accuracy, with similar running time. As the tracking and mapping threads work always in the active map, multi-mapping does not introduce significant overhead.

表6总结了在跟踪和建图线程中，进行的主要运算的运行时间，表明我们的系统可以以30-40fps的速度实时运行，每秒3-6关键帧。惯性部分在跟踪时耗时几乎可以忽略，而且可以使得系统更加有效率，因为帧率可以安全的降低。在建图线程中，在惯性局部BA中，每个关键帧中更高数量的变量，被较少数量的关键帧补偿了，获得了更高的准确率，运行时间类似。由于跟踪和建图线程一直都在活跃地图中运行，多地图并不会带来显著的开销。

Table VII summarizes the running time of the main steps for loop closing and map merging. The novel place recognition method only takes 10 ms per keyframe. Times for merging and loop closing remain below one second, running only a pose-graph optimization. For loop closing, performing a full bundle adjustment may increase times up to a few seconds, depending on the size of the involved maps. In any case, as both operations are executed in a separate thread (Fig. 1) they do not interfere with the real time performance of the rest of the system. The visual-inertial systems perform just two map merges to join three sequences, while visual systems perform some additional merges to recover from tracking losses. Thanks to their lower drift, visual-inertial systems also perform less loop closing operations compared with pure visual systems.

表7总结了回环闭合和地图融合的主要步骤运行时间。新的位置识别方法，每关键帧只耗时10ms。地图合并和回环闭合的时间保持在1秒一下，只运行一个姿态图优化。对于回环闭合，进行完整的BA会将时间增加到几秒，这依赖于涉及到的地图的大小。在任意情况下，因为两个操作都在独立的线程中运行（图1），它们并没有影响系统其余部分的实时性能。视觉-惯性系统只进行了两次地图合并，将三个序列连接了起来，而视觉系统进行了额外的合并，以从跟踪丢失中恢复回来。多亏了更低的漂移，视觉-惯性系统与纯视觉系统相比，进行的回环闭合操作也更少一些。

Although it would be interesting, we do not compare running time against other systems, since this would require a significant effort that is beyond the scope of this work.

我们没有与其他系统的运行时间进行对比，因为这需要大量其他的工作，不在本文工作范围之内。

## 8. Conclusions

Building on [2]–[4], we have presented ORB-SLAM3, the most complete open-source library for visual, visual-inertial and multi-session SLAM, with monocular, stereo, RGB-D, pin-hole and fisheye cameras. Our main contributions, apart from the integrated library itself, are the fast and accurate IMU initialization technique, and the multi-session map-merging functions, that rely on an new place recognition technique with improved recall.

在[2-4]的基础上，我们提出了ORB-SLAM3，视觉-视觉惯性和多环节SLAM最完整的开源库，支持单目，立体，RGB-D，针孔和鱼眼相机。我们的主要贡献，除了库以外，就是快速准确的IMU初始化技术，多环节地图合并函数，改进了召回的新位置识别技术。

Our experimental results show that ORB-SLAM3 is the first visual and visual-inertial system capable of effectively exploiting short-term, mid-term, long-term and multi-map data associations, reaching an accuracy level that is beyond the reach of existing systems. Our results also suggest that, regarding accuracy, the capability of using all these types of data association overpowers other choices such as using direct methods instead of features, or performing keyframe marginalization for local BA, instead of assuming an outer set of static keyframes as we do.

我们的试验结果表明，ORB-SLAM3能够有效的利用短期，中期，长期和多地图数据关联，准确率超过了现有的系统，是第一个这种的视觉和视觉-惯性系统。我们的结果还表明，对于准确率，使用所有类型的数据关联比其他的选择更重要，比如使用直接方法而不是特征，或对局部BA进行关键帧marginalization。

The main failure case of ORB-SLAM3 is low-texture environments. Direct methods are more robust to low-texture, but are limited to short-term [27] and mid-term [31] data association. On the other hand, matching feature descriptors successfully solves long-term and multi-map data association, but seems to be less robust for tracking than Lucas-Kanade, that uses photometric information. An interesting line of research could be developing photometric techniques adequate for the four data association problems. We are currently exploring this idea for map building from endoscope images inside the human body.

ORB-SLAM3的主要失败案例，是低纹理的环境。直接方法对低纹理环境更加稳健，但局限于短期和中期数据关联。另外，匹配特征描述子成功的解决了长期和多地图数据关联，但与使用了光度信息的Lucas-Kanade相比，对跟踪仍然不够稳健。一个有趣的研究线是，开发足以解决四种数据关联问题的光度方法。我们目前在探索这个思想，用于人体内内窥镜图像的地图构建。

About the four different sensor configurations, there is no question, stereo-inertial SLAM provides the most robust and accurate solution. Furthermore, the inertial sensor allows to estimate pose at IMU rate, which is orders of magnitude higher than frame rate, being a key feature for some use cases. For applications where a stereo camera is undesirable because of its higher bulk, cost, or processing requirements, you can use monocular-inertial without missing much in terms of robustness and accuracy. Only keep in mind that pure rotations during exploration would not allow to estimate depth.

关于四种不同的传感器配置，毫无疑问的是，立体-惯性SLAM给出了最稳健最精确的解。而且，惯性传感器使我们可以以IMU的速度来估计姿态，比帧率要高了几个数量级，是一些使用情况下的关键特点。对一些应用，立体相机可能因为体积大，价钱高，处理要求高，而不太理想，我们可以使用单目-惯性系统，稳健性和准确率也不会有多少损失。只要记住，在探索时的纯旋转，不能进行估计深度。

In applications with slow motions, or without roll and pitch rotations, such as a car in a flat area, IMU sensors can be difficult to initialize. In those cases, if possible, use stereo SLAM. Otherwise, recent advances on depth estimation from a single image with CNNs offer good promise for reliable and true-scale monocular SLAM [83], at least in the same type of environments where the CNN has been trained.

在运动速度很慢的应用中，或没有roll和pitch旋转的，比如一辆车在一个平坦的区域，IMU会很难初始化。在这些情况中，如果可能的话，就使用立体SLAM。否则，最近用CNNs在单幅图像上估计深度的进展会很好的应用，得到可靠的，真实尺度的单目SLAM[83]，这在一些类型的环境中，CNN进行训练过后，是可以应用的。