# Measuring the Objectness of Image Windows

Bogdan Alexe, Thomas Deselaers, Vittorio Ferrari, Computer Vision Laboratory, ETH Zurich

## 0. Abstract

We present a generic objectness measure, quantifying how likely it is for an image window to contain an object of any class. We explicitly train it to distinguish objects with a well-defined boundary in space, such as cows and telephones, from amorphous background elements, such as grass and road. The measure combines in a Bayesian framework several image cues measuring characteristics of objects, such as appearing different from their surroundings and having a closed boundary. These include an innovative cue to measure the closed boundary characteristic. In experiments on the challenging PASCAL VOC 07 dataset, we show this new cue to outperform a state-of-the-art saliency measure, and the combined objectness measure to perform better than any cue alone. We also compare to interest point operators, a HOG detector, and three recent works aiming at automatic object segmentation. Finally, we present two applications of objectness. In the first, we sample a small number of windows according to their objectness probability and give an algorithm to employ them as location priors for modern class-specific object detectors. As we show experimentally, this greatly reduces the number of windows evaluated by the expensive class-specific model. In the second application, we use objectness as a complementary score in addition to the class-specific model, which leads to fewer false positives. As shown in several recent papers, objectness can act as a valuable focus of attention mechanism in many other applications operating on image windows, including weakly supervised learning of object categories, unsupervised pixelwise segmentation, and object tracking in video. Computing objectness is very efficient and takes only about 4 sec. per image.

我们给出了一个通用目标度量，量化了一个图像窗口包含了任意类别的目标的可能性。我们显示的进行训练其区分空间中边缘明确目标，比如牛和电话，而背景元素模糊或不明确，如草和道路。在贝叶斯框架中，本度量与几种图像线索度量目标特征相结合，比如与周围看起来不同，有闭合的边缘。这包括一个新的线索，度量闭合的边缘特征。在很有挑战性的PASCAL VOC 07数据集的实验中，我们表明这个线索超过了目前最好的显著性度量，综合目标性度量比任何单独的线索表现都要好。我们还比较了感兴趣点算子，HOG检测器，和三种最近用于自动目标分割的方法。最后，我们给出目标度的两种应用。第一个，我们根据其目标度概率采样了少数几个窗口，给出一个算法来用之作为位置先验，用在现代类别目标检测器中。我们通过实验表明，这极大的降低了昂贵的类别模型评估的窗口数量。第二个，我们使用目标度在类别模型中作为一种补充度量，带来了更少的假阳性。最近几篇文章都表示，目标度可以作为一种有价值的注意力机制的焦点，在很多应用中在图像窗口中应用，包括目标类别的弱监督学习，无监督的像素分割，视频中的目标跟踪。计算目标度非常高效，每幅图像只需要4秒。

**Index Terms** — Objectness measure, object detection, object recognition

## 1. Introduction

In recent years, object class detection has become a major research area. Although a variety of approaches exist [4], [11], [35], most state-of-the-art detectors follow the sliding-window paradigm [11], [12], [18], [25], [33]. A classifier is first trained to distinguish windows containing instances of a given class from all other windows. The classifier is then used to score every window in a test image. Local maxima of the score localize instances of the class.

近几年，目标类别检测已经成为一个主要研究领域。虽然存在很多方法，多数目前最好的检测器都是滑窗的模式。首先训练一个分类器来区分包含给定类别实例的窗口与所有其他窗口。然后分类器给测试图像中的每个窗口进行打分。分数的局部极大值定位了这个类别的实例。

While object detectors are specialized for one object class, such as cars or swans, in this paper we define and train a measure of objectness generic over classes. It quantifies how likely it is for an image window to cover an object of any class. Objects are stand-alone things with a well-defined boundary and center, such as cows, cars, and telephones, as opposed to amorphous background stuff, such as sky, grass, and road (as in the things versus stuff distinction of [26]). Fig. 1 illustrates the desired behavior of an objectness measure. It should score highest windows fitting an object tight (green), score lower windows covering partly an object and partly the background (blue), and score lowest windows containing only stuff (red).

目标检测器是专用于一个目标类别的，比如车或天鹅，本文中我们定义并训练一个目标的度量，在所有类别中通用。它量化了一个图像窗口覆盖了任意类别的一个目标的可能性。目标是独立的，其边缘和中心是明确的，比如牛，车和电话，与一些模糊背景的事物相对应，比如天空，草地和道路（如[26]中things和stuff的区别）。图1给出了目标度量的理想性质，它应当在紧紧包含目标的窗口中分数最高（绿色），部分目标部分背景的框分数较低（蓝色），在只包含事物的窗口中分数最低（红色）。

In order to define the objectness measure, we argue that any object has at least one of three distinctive characteristics: 为定义目标度量，我们认为任意目标至少应当有下面三个特征中的一个：

- a well-defined closed boundary in space, 空间中明确的闭合边缘
- a different appearance from its surroundings [36], [38], 与周围外观不同
- sometimes it is unique within the image and stands out as salient [7], [21], [27], [28]. 在图像中有时候是唯一的，非常突出。

Many objects have several of these characteristics at the same time (Figs. 2, 3, 4, 5, and 6). In Sections 2.1, 2.2, 2.3, and 2.4 we explore several image cues designed to measure these characteristics. 很多目标同时拥有几种性质。在2.1，2.2.，2.3和2.4中，我们探索了几种图像线索，设计用于度量这些特征。

Although objects can appear at a variety of locations and sizes within an image, some windows are more likely to cover objects than others, even without analyzing the pixel patterns inside them. A very elongated window in an image corner is less probable a priori than a square window in the middle of it. We explore this idea in Section 2.5 and propose an objectness cue based on the location and size of a window.

虽然目标可以在图像很多位置出现，大小不同，一些窗口比其他窗口更可能覆盖目标，即使不分析其中的像素模式。在图像角落处的一个长长的窗口，比图像中间的一个方形窗口而言，更不太可能包含目标。我们在2.5中探索这个思想，基于窗口位置和大小提出一种目标度的线索。

This paper makes several contributions: 本文贡献如下：

1. We design an objectness measure and explicitly train it to distinguish windows containing an object from background windows. This measure combines in a Bayesian framework several cues based on the above characteristics (Sections 2 and 3). 设计了一种目标度度量，训练其用于区分包含目标和背景的窗口。这种度量在贝叶斯框架下结合了几种基于上述性质的cue。

2. On the task of detecting objects in the challenging PASCAL VOC 07 dataset [17], we demonstrate that the combined objectness measure performs better than any cue alone, and also outperforms traditional salient blob detectors [27], [28], interest point detectors [41], the Semantic Labeling technique of Gould et al. [22], and a HOG detector [11] trained to detect objects of arbitrary classes. 在PASCAL VOC 07数据集的目标检测任务上，我们证明了，综合目标度度量比其他任意cue单独都要好，也超过了传统的salient blob detectors [27], [28], interest point detectors [41], the Semantic Labeling technique of Gould et al. [22]，训练了一个HOG检测器来检测任意类别的目标。

3. We give an algorithm that employs objectness to greatly reduce the number of evaluated windows evaluated by modern class-specific object detectors [11], [18], [50] (Section 6). Differently from ESS [33], our method imposes no restriction on the class model used to score a window. 我们给出了一种利用目标度的算法，极大的降低了现代类别目标检测器所要评估的窗口数量。与ESS不同，我们的方法对用于给窗口打分的类别模型没有施加任意限制。

4. Analogously, we show how to use objectness for reducing the number of false-positives returned by class-specific object detectors (Section 7). 类似的，我们展示了怎样用目标度来降低假阳性。

In addition to the two applications above, objectness is valuable in others as well. Since the publication of a preliminary version of this work [3], objectness has already been used successfully to: 除了上面两个应用，目标度对其他应用中也很有价值。本文出版发表后，目标度已经成功应用于：

1. Facilitate learning new classes in a weakly supervised scenario [13], [30], [47], where the location of object instances is unknown. Objectness steers the localization algorithm toward objects rather than backgrounds, e.g., avoiding to select only grass regions in a set of sheep images. 在弱监督场景中促进学习新的类别，其中目标实例的位置是未知的。目标度使定位算法朝向目标，而不是背景，在很多羊的图像中如避免选择草的区域。

2. Analogously, to support weakly supervised pixelwise segmentation of object classes [2], [52] and unsupervised object discovery [34]. 类似的，支持了弱监督逐像素分割目标类别和无监督目标发现。

3. Learn spatial models of interactions between humans and objects [45]. 学习了人类与目标间相互作用的空间模型。

4. Content-aware image resizing [5], [48]. 对内容敏感的图像大小改变。

5. Object tracking in video, as an additional likelihood term preventing the tracker from drifting to the background (ongoing work by Van Gool’s group at ETH Zurich). 视频中的目标跟踪，作为一个额外的似然，防止跟踪器漂移到背景中。

The source code of the objectness measure is available at http://www.vision.ee.ethz.ch/~calvin. Computing objectness is very efficient and takes only about 4 sec. per image.

### 1.1 Related Work

This paper is related to several research strands, which differ in how they define “saliency.” 本文与几个研究线相关，与他们定义的显著性不太一样。

**Interest points**. Interest point detectors (IPs) [29], [41] respond to local textured image neighborhoods and are widely used for finding image correspondences [41] and recognizing specific objects [37]. IPs focus on individual points, while our approach is trained to respond to entire objects. Moreover, IPs are designed for repeatable detection in spite of changing imaging conditions, while our objectness measure is trained to distinguish objects from backgrounds. In Section 5, we experimentally evaluate IPs on our task.

**感兴趣点**。感兴趣点检测器IPs，对局部纹理的图像邻域进行响应，广泛用于寻找图像对应性，识别特定目标。IPs关注单独的点，而我们的方法训练用于对整个目标进行响应。而且，IPs设计用于可重复的检测，即使是改变成像条件的情况下，而我们的目标度度量训练用于区分目标和背景，在第5部分中，我们在任务中实验性的评估IPs。

**Class-specific saliency**. A few works [40], [42], [53] define as salient the visual characteristics that best distinguish a particular object class (e.g., cars) from others. This class-specific saliency is very different from the class-generic task we tackle here.

**特定类别的显著性**。几篇文章将显著性定义为视觉特征将特定目标类别与其他的最好的区分。这种特定类别的显著性与我们处理的通用类别的任务非常不一样。

**Generic saliency**. Since [28], numerous works [7], [21], [24], [27], [31] appeared to measure the saliency of pixels, as the degree of uniqueness of their neighborhood w.r.t. the entire image or the surrounding area [36], [38]. Salient pixels form blobs that “stand out” from the image. These works implement selective visual attention from a bottom-up perspective and are often inspired by studies of human eye movements [14], [15], [32].

**通用显著性**。自从[28]，很多工作都度量像素的显著性，作为其邻域与整个图像或周围区域的唯一性的程度。显著的像素形成blobs，在图像中突出出来。这些工作从自底向上的方式实现了选择性的视觉注意力，经常受到人眼运动的启发。

Liu et al. [36] detect a single dominant salient object in an image. They formulate the problem as image segmentation. The method combines local, regional, and global pixel-based saliency measurements in a CRF and derives a binary segmentation separating the object from the background. Valenti et al. [49] measure the saliency of individual pixels, and then returns the region with the highest sum of pixel-saliency as the single dominant object. Achanta et al. [1] find salient regions using a frequency-tuned approach. Their approach segments objects by adaptive thresholding of fully resolution saliency maps. The saliency maps are obtained by combining several band-pass filters that retain a wider range of spatial frequencies than [24], [27], [38]. Berg and Berg [6] find iconic images representative for an object class. Their approach returns images having a large, centered object which is clearly separated from the background. These approaches do not seem suitable for the PASCAL VOC 07 dataset [17] where many objects are present in an image and they are rarely dominant (Figs. 15 and 16).

Liu等[36]检测图像中单个主要的显著目标。他们将这个问题表述为图像分割问题。该方法将基于局部、区域和全局的像素显著性度量，结合到CRF中，推理出一个二值分割，将目标和背景区分开来。Valenti等[49]度量单个像素的显著度，然后将像素显著度之和最高的区域确定为单个主要的目标。Achanta等[1]使用频率调节的方法找到显著区域。其方法通过全分辨率的显著性图的自适应阈值来分割目标。显著性图通过结合几个带通滤波器得到。Berg等[6]找到对一个目标类别寻找标志性的图像代表。其方法返回的图像，有一个大的中心目标，与背景与明显的区分。这些方法似乎并不适合PASCAL VOC 07数据集，其中图像中存在很多目标，基本上都不占主要部分。

This paper is related to the above works, as we are looking for generic objects. We incorporate a state-of-the-art saliency detector [27] as one cue into our objectness measure. However, we also include other cues than “stand out” saliency and demonstrate that our combined measure performs better at finding objects (Section 5).

本文与上述工作相关，因为我们要寻找的是通用目标。我们采用了目前最好的显著性检测器，作为我们的目标度度量的一个cue。但是，我们还包含了其他cues，证明了我们的综合度量在寻找目标上比其他的要好。

Our work differs from the above also in other respects. The unit of analysis is not a pixel as possibly belonging to an object, but a window as possibly containing an entire object. This enables scoring all windows in an image and sampling any desired number of windows according to their scores. These can then directly be fed as useful location priors to object class learning and detection algorithms, rather than making hard decisions early on. We experimentally demonstrate this with applications to speeding up object detectors (Section 6) and to reducing their false-positive rates (Section 7).

我们的工作也在其他方面与上述不同。分析单位不是属于一个目标的一个像素，而是一个窗口，很可能包含一整个目标。这可以对图像中的所有窗口进行评分，根据其分数对任意数量的窗口进行取样。这些可以直接作为有用的位置先验，送入目标类别学习和检测算法中，而不用很早就做出很难的决策。我们实验证明了结论，并在应用中加速了目标检测，降低了其假阳率。

Analyzing windows also enables evaluating more complex measures of objectness. For example, in Section 2.4 we propose a new image cue and demonstrate it performs better than traditional saliency cues [27] at finding entire objects.

分析窗口也可以评估更复杂的目标度度量。比如，在2.4节中，我们提出一种新的图像cue，证明了其比传统的显著性cue在找到完整目标上表现要好。

Finally, we evaluate our method on a more varied and challenging dataset (PASCAL VOC 07), where most images contain many objects and they appear over a wide range of scales. We explicitly train our objectness measure to satisfy the strict PASCAL-overlap criterion, and evaluate its performance using it. This matters because it is the standard criterion for evaluating the intended clients of our objectness measure, i.e., object detection algorithms.

最后，我们在一个变化更多的挑战性数据集PASCAL VOC 07上评估了我们的方法，其中多数图像包含多个目标，尺度变化非常多。我们显式的训练了我们的目标度度量，以满足严苛的PASCAL重叠规则，使用其评估了我们的性能。这很重要，因为是评估我们的目标度量的标准规则，即目标检测算法。

**Object proposals**. Since the publication of the first version of this work [3], two closely related independent works appeared [9], [16]. They share with our work the overall idea of proposing a few hundred class-independent candidates likely to cover all objects in an image. While both works [9], [16] produce rough segmentations as object candidates, they are computationally very expensive and take 2-7 minutes per image (compared to a few seconds for our method). In Section 5, we compare the ability of our method to detect generic objects to [9], [16].

**目标建议**。自从本文第一版发表，出现了两个关系很紧密的工作。它们与我们的总体思想类似，提出几百个类别独立的候选窗口，很可能覆盖图像中的所有目标。其他两篇文章生成大致的分割作为目标候选，它们的计算量非常大，每幅图像耗时2-7分钟（我们的方法则是几秒钟）。在第5部分，我们将我们的方法检测通用目标的能力与[9,16]相比。

**Efficient sliding window object detection**. State-of-the-art object detection algorithms often rely on the sliding window paradigm [11], [18], [50], which scores a large number of windows in a test image using a classifier. To keep the computational cost manageable, researchers are careful in selecting classifiers that can be evaluated rapidly (e.g., linear SVMs [11], [20]).

**高效滑窗目标检测**。目前最好的目标检测算法通常依赖于滑窗的方式，用一个分类器，对测试图像中的大量窗口进行打分。为使计算代价可控，在选择分类器时候很小心，评估速度要快，如线性SVMs。

Recently, several techniques have been proposed to reduce the cost of sliding window algorithms. One approach is to reduce the cost of evaluating a complex classifier on a window. Excellent approximations to the histogram intersection kernel [39] and the $χ^2$ kernel [51] have been presented. Another approach is to reduce the number of times the complex classifier is evaluated. In [25], [50] this is achieved by first running a linear classifier over all windows and then evaluating a nonlinear kernel classifier only on a few highly scored windows. Lampert et al. [33] presented a branch-and-bound scheme that finds exactly the single highest scored window while minimizing the number of times the classifier is evaluated.

最近，已经提出了几种技术，降低了滑窗算法的代价。一种方法是降低在一个窗口上评估一个复杂分类器的代价。另一种方法是降低复杂分类器评估的次数。

In Section 6, we apply objectness to greatly reduce the number of windows evaluated by the classifier. Our method can be applied to any window classifier, whereas [33] is restricted to those for which the user can provide a good bound on the highest score in a contiguous set of windows. Our speedup is orthogonal to techniques to reduce the cost of evaluating one window, and could be combined with them [25], [39], [50], [51].

在第6部分中，我们使用目标度来极大的降低分类器评估的窗口数量。我们的方法可以应用于任意窗口分类器，而[33]则有所限制。我们的加速与降低一个窗口上评估代价的方法是不相关的，可以与之结合。

### 1.2 Plan of the Paper

Section 2 describes the cues composing our objectness measure. Sections 3 and 4 show how to learn the cue parameters and how to combine them in a Bayesian framework. In Section 5 we evaluate objectness on the very challenging PASCAL VOC 2007 dataset, and then show applications to aiding class-specific object detectors [11], [18], [50] in Sections 6 and 7.

第2部分描述我们的目标度度量的组成cue。第3和第4部分，展示了怎样学习cue参数，怎样将之在贝叶斯框架中结合到一起。在第5部分，我们在PASCAL VOC 07数据集中评估了目标度度量，然后在第6和第7部分展示了对特定类别目标检测有帮助的应用。

## 2. Objectness cues

As mentioned in Section 1, objects in an image are characterized by a closed boundary in 3D space, a different appearance from their immediate surrounding and sometimes by uniqueness. In Sections 2.1, 2.2, 2.3, and 2.4 we present four image cues to measure these characteristics for an image window. As a complementary strategy, we define a last cue based on the location and size of a window, without analyzing the actual image pixels (Section 2.5).

如第1部分所述，图像中的目标的特征是，在3D空间中有一个闭合的边缘，与其周围的背景外表不同，有时候还是唯一的。在2.1，2.2，2.3，2.4节中，我们提出4种图像cue，对一个图像窗口度量这些特征。作为一个互补的策略，我们基于窗口的位置和大小，定义了最后一个cue，而不分析其真实的图像像素（2.5节）。

### 2.1 Multiscale Saliency (MS)

Hou and Zhang [27] proposed a global saliency measure based on the spectral residual of the FFT which favors regions with an unique appearance within the entire image f. The saliency map I of an image f is obtained at each pixel p as

Hou和Zhang[27]基于FFT的谱残余提出了一种全局显著性度量，倾向于在整幅图像f中外观唯一的。图像f的显著性图I在每个像素处计算如下

$$I(p) = g(p)*F^{-1}[exp(R(f)+P(f))]^2$$(1)

where F is the FFT, R(f) and P(f) are the spectral residual and the phase spectrum [27] of the image f, and g is a Gaussian filter used for smoothing the output. Since this measure prefers objects at a certain scale, we extend it to multiple scales(Fig.2). Moreover, as [27] suggests, we process the color channels independently as separate images. For each scale s, we use [27] to obtain a saliency map $I^s_{MS}(p)$. Based on this, we define the saliency of a window w at scale s as

其中F是FFT，R(f)和P(f)是图像f的谱残余和相位谱，g是高斯滤波器，用于平滑输出。由于这个度量倾向于在一定尺度中的目标，我们将其拓展到多尺度中。而且，如[27]建议的，我们将色彩通道作为独立图像进行处理。对每个尺度s，我们使用[27]得到一个显著性图$I^s_{MS}(p)$。基于此，我们将一个窗口w在尺度s中的显著性为

$$MS(w, θ_{MS}^s(p)) = \sum_{p ∈ w | I^s_{MS}(p) ≥ θ_{MS}^s(p)} I^s_{MS}(p) × \frac {|p ∈ w | I^s_{MS}(p) ≥ θ_{MS}^s(p)|} {|w|}$$(2)

where the scale-specific thresholds $θ_{MS}^s$ are parameters to be learned (Section 3), and |⋅| indicates the number of pixels. Saliency is higher for windows with higher density of salient pixels (second factor), with a bias toward larger windows (first factor). Density alone would score highest windows comprising just a few very salient pixels. Instead, our measure is designed to score highest windows around entire blobs of salient pixels, which correspond better to whole objects (Fig. 2). The need for multiple scales is evident in Fig. 2 as the windows covering the two objects in the image (airplane, giraffe) score highest at different scales. This MS cue measures the uniqueness characteristic of objects.

其中特定尺度的阈值$θ_{MS}^s$是要学习的参数（第3部分），|⋅|为像素数量。对于显著像素密度更高（第二因素）的窗口来说，其显著性更高，并倾向于更大的窗口（第一因素）。单独考虑密度的话，对只有几个非常显著的像素的窗口来说，会分数最高。而我们的度量设计，要对显著像素整个blob附近的窗口分数最高，这更好的对应着整个目标（图2）。在图2中，多尺度的需求是明显的，因为覆盖图中两个目标（飞机，长颈鹿）的窗口，在两个不同的尺度上分数最高。这种多尺度cue度量的是目标性质的唯一性。

### 2.2 Color Contrast (CC)

CC is a measure of the dissimilarity of a window to its immediate surrounding area (Fig. 3). The surrounding Surr(w, θ_CC) of a window w is a rectangular ring obtained by enlarging the window by a factor θ_CC in all directions so that

CC是一个窗口与其周围紧邻区域的不相似性的度量（图3）。一个窗口w的紧邻Surr(w, θ_CC)，是一个矩形环，通过将窗口在各个方向增大一个系数θ_CC，这样

$$\frac {|Surr(w, θ_{CC})|} {|w|} = θ^2_{CC} - 1$$(3)

The CC between a window and its surrounding is computed as the Chi-square distance between their LAB histograms h: 一个窗口和其周围的CC，是通过其LAB直方图h的Chi平方距离计算的：

$$CC(w, θ_{CC}) = χ^2 (h(w), h(Surr(w, θ_{CC})))$$(4)

The ring size parameter θ_CC is learned in Section 3. 环形参数θ_CC在第3部分中学习。

CC is a useful cue because objects tend to have a different appearance (color distribution) than the background behind them. In Fig. 3a, windows on the grass score lower than windows half on a sheep and half on the grass. Windows fitting a sheep tightly score highest. This cue measures the different appearance characteristic of objects.

CC是一个有用的cue，因为目标通常与其后的背景外表不同（色彩分布不同）。在图3a中，在草上的窗口，比半在羊上半在草中的窗口，分数要低。包含羊更紧的窗口，分数最高。这个cue度量的是目标的不同外表特征。

CC is related to the center-surround histogram cue of [36]. However, [36] computes a center-surround histogram centered at a pixel, whereas CC scores a whole window as whether it contains an entire object. The latter seems a more appropriate level of analysis.

CC与中心周围直方图cue[36]相关。但是[36]计算的是一个像素为中心的中心周围直方图，而CC对整个窗口评分，不论其是否包含整个目标。后者似乎是一个更合适的分析目标。

### 2.3 Edge Density (ED)

ED measures the density of edges near the window borders. The inner ring Inn(w, θ_ED) of a window w is obtained by shrinking it by a factor θ_ED in all directions so that

ED度量的是窗口边缘附近的边缘密度。窗口w的内环Inn(w, θ_ED)是通过在各个方向缩减一个系数θ_ED得到的

$$\frac {|Inn(w, θ_{ED}|} {|w|} = \frac {1} {θ^2_{ED}}$$(5)

The ED of a window w is computed as the density of edgels in the inner ring: 窗口w的ED计算为内环中的边缘的密度：

$$ED(w, θ_{ED}) = \frac {\sum_{p∈Inn(w, θ_{ED})} I_{ED}(p)} {Len(Inn(w, θ_{ED}))}$$(6)

The binary edgemap I_ED(p)∈{0,1} is obtained using the Canny detector [8], and Len(⋅) measures the perimeter of the inner ring. Note how the expected number of boundary edgels grows proportionally to the perimeter, not the area, because edgels have constant thickness of 1 pixel. This normalization avoids a bias toward very small windows that would otherwise arise. The ring size parameter θ_ED is learned in Section 3.

二值边缘图I_ED(p)∈{0,1}通过使用Canny检测器得到，Len(⋅)度量的内环的周长。注意边缘的期望数量相对于周长来说按比例的增长，而不是相对于区域面积增长，因为边缘的宽度为固定的1个像素。这个归一化避免了非常小的窗口的倾向，否则值就会很大。环大小参数θ_ED在第3部分学习。

The ED cue captures the closed boundary characteristic of objects, as they tend to have many edgels in the inner ring (Fig. 4).

ED cue捕获的是目标闭合边缘的特性，因为在内环中通常会有很多边缘片段（图4）。

### 2.4 Superpixels Straddling (SS)

A different way of capturing the closed boundary characteristic of objects rests on using superpixels [19] as features. Superpixels segment an image into small regions of uniform color or texture. A key property of superpixels is to preserve object boundaries: All pixels in a superpixel belong to the same object [46] (ideally). Hence, an object is typically oversegmented into several superpixels, but none straddles its boundaries (Fig. 5). Based on this property, we propose here a cue to estimate whether a window covers an object.

捕获目标闭合边缘性质的一种不同方式是使用超像素作为特征。超像素将图像分割成小的区域，其中的色彩或纹理是统一的。超像素的一个关键性质是，保持了目标的边缘：在一个超像素中的所有像素，属于同一目标（理想情况下）。因此，一个目标一般是过分割成了几个超像素，但都不会横跨其边缘（图5）。基于这种性质，我们这里提出一个cue，来估计一个窗口是否覆盖了一个目标。

A superpixel s straddles a window w if it contains at least one pixel inside and at least one outside w. Most of the surface of an object window is covered by superpixels contained entirely inside it (w1 in Fig. 5c). Instead, most of the surface of a “bad” window is covered by superpixels straddling it (i.e., superpixels continuing outside the window, w2 in Fig. 5c). The SS cue measures for all superpixels s the degree by which they straddle w:

一个超像素s横跨了一个窗口，如果其包含了至少一个像素在w中和一个像素在w外。一个目标窗口的多数表面，是由完全在其内的超像素覆盖的（图5c中的w1）。而一个坏窗口的大部分表面，是由横跨其上的超像素覆盖的（即，延续到窗口外的超像素，图5c中的w2）。SS cue度量的是所有超像素横跨w的程度：

$$SS(w, θ_{SS}) = 1 - \sum_{s∈S(θ_{SS})} \frac {min(|s \backslash w|, |s∩w|)} {|w|}$$(7)

where S(θ_SS) is the set of superpixels obtained using [19] with a segmentation scale θ_SS (learned in Section 3). For each superpixel s, (7) computes its area |s∩w| inside w and its area |s\w| outside w. The minimum of the two is the degree by which s straddles w and is its contribution to the sum in (7).

其中S(θ_SS)是[19]计算得到的超像素的集合，使用的分割尺度θ_SS，在第3部分中学习得到。对于每个超像素s，(7)计算其在w中的区域|s∩w|，和在w外的区域|s\w|。这两者的最小值是s横跨w的程度，是对(7)的和的贡献。

Superpixels entirely inside or outside w contribute 0 to the sum. For a straddling superpixel s, the contribution is lower when it is contained either mostly inside w, as part of the object, or mostly outside w, as part of the background (Fig. 5c). Therefore, SS(w, θ_SS) is highest for windows w fitting tightly around an object, as desired.

全部在w内或w外的超像素，对此和的贡献为0。对于一个横跨的超像素s，当其几乎完全在w中，作为目标的一部分，或几乎在w外，作为背景的一部分（图5c）时，贡献更小。因此，SS(w, θ_SS)对于窗口w紧紧包裹住一个目标时值最大，就像期望的那样。

In Section 5, we show that SS outperforms all other cues we consider (including MS and its original form [27]).

在第5部分中，我们证明了SS超过了我们考虑的所有其他cues（包括MS及其原始形式）。

**Efficient computation of SS**. Naively computing the score SS(w, θ_SS) for window w in an image I requires O(|I|) operations. For each superpixel in the whole image we need to sum over all the pixels it contains to compute (7). Hence, the total cost of computing the score SS(w, θ_SS) for all windows w in a set W is O(|W||I|) operations.

**SS的高效计算**。对图像I中的窗口w直接计算分数SS(w, θ_SS)需要O(|I|)的运算复杂度。对整个图像中的每个超像素，我们需要对包含的所有像素求和，来计算(7)。因此，对集合W中的所有窗口w计算分数SS(w, θ_SS)是O(|W||I|)次运算。

We present here an efficient procedure to reduce this complexity. For each superpixel s we build a separate integral image I^s(x,y) giving the number of pixels of s in the rectangle (0, 0) -> (x, y). Then, using I^s(x, y), we can compute the number |s∩w| of pixels of s contained in any window w in constant time, independent of its area (Fig. 7). The area |s\w| outside w is also readily obtained as

我们这里给出降低这种复杂度的高效过程。对每个超像素s，我们构建一个单独的积分图像I^s(x,y)，给出矩形(0, 0) -> (x, y)中s的像素数量。然后使用I^s(x, y)，我们可以以常数时间计算s在任意窗口w中的像素数量|s∩w|，与其区域面积无关（图7）。w外的区域面积|s\w|可以通过下式很容易得到

$$|s\w| = |s| - |s∩w|$$(8)

Therefore, we can efficiently compute all elements of SS(w, θ_SS) using just four operations per superpixel, giving a total of 4S operations for a window (with S = |S(θ_SS)| the number of superpixels in the image).

因此，我们可以高效的计算SS(w, θ_SS)的所有元素，每个超像素只要4次运算，一个窗口只需要4S次运算（S = |S(θ_SS)|，是图像中超像素的数量）。

Computing all integral images I^s takes O(S|I|) operations and it is done only once (not for each window). Therefore, our efficient procedure takes only O(S(|W|+|I|)) operations, compared to O(|W||I|) for the naive algorithm. This is much faster because S ≪ |I|. In Section 5, we use this procedure for computing (7) for 100,000 windows in an image.

计算所有积分图像I^s，需要O(S|I|)次运算，而且只做一次（不是对每个窗口）。因此，我们的高效过程只需要O(S(|W|+|I|))次运算，与直接计算的O(|W||I|)次运算相比，快了很多，因为S ≪ |I|。在第5部分中，我们用这个过程计算(7)，图像中有100000个窗口。

### 2.5 Location and Size (LS)

Windows covering objects vary in size and location within an image. However, some windows are more likely to cover objects than others: An elongated window located at the top of the image is less probable a priori than a square window in the image center. Based on this intuition, we propose a new cue to assess how likely an image window is to cover an object, based only on its location and size, not on the pixels inside it.

在图像中，覆盖目标的窗口在大小和位置上变化很多。但是，一些窗口比其他的更可能覆盖一些目标：在图像上方的一个长条形窗口，比图像中央的一个方形窗口，包含目标的概率更小。基于这种直觉，我们提出一种新的cue，评估一个图像窗口覆盖一个目标的可能性，只基于其位置和大小，并不基于其中的像素。

We compute the probability of an image window to cover an object using kernel density estimation [44] in the 4D space W of all possible windows in an image. The space W is parameterized by the (x, y) coordinates of the center, the width, and the height of a window. We equip W with a probability measure pW defining how likely a window is to cover an object. This probability pW is computed by kernel density estimation on a large training set of N windows {w1; w2; . . . ; wN} covering objects which are points in W. As training images might have different sizes, we normalize the coordinate frame of all images to 100 × 100 pixels.

我们计算一个图像窗口中覆盖一个目标的概率，使用4D空间W中的核密度估计，W是一幅图像中所有可能的窗口。我们为W定义一个概率度量pW，定义了一个窗口可能覆盖一个目标的概率。这个概率pW的计算，是在一个N个窗口的大型训练集上通过核密度估计来进行的，{w1; w2; . . . ; wN}都覆盖了目标，是W中的一些点。由于训练图像可能大小不同，我们将所有图像的大小归一到100 × 100像素。

We perform kernel density estimation in the space W ⊂ [0, 100]^4 by placing a Gaussian with mean wi and covariance matrix θ_LS for each training window wi. The covariance matrix θ_LS is learned as explained in Section 3.3. The probability of a test window w ∈ W is computed as

我们在空间W ⊂ [0, 100]^4上进行核密度估计，在每个训练窗口wi上加上一个高斯函数，均值wi，协方差矩阵θ_LS。协方差矩阵θ_LS是学习得到的。测试窗口w ∈ W的概率计算如下：

$$p_W (w, θ_{LS}) = \frac {1}{Z} \sum_{i=1}^N \frac {1} {(2π)^2 |θ_{LS}|^{1/2}} e^{-(w-w_i)^T (θ_{LS})^{-1} (w-w_i)/2}$$(9)

where the normalization constant Z ensures that pW is a probability, i.e., $\sum_{w∈W} p_W(w) = 1$. Note that the query window w can be any point in W, not necessarily corresponding to a training window. The LS cue assesses the likelihood of a window w to cover an object as $LS(w,θ_{LS}) = p_W(w, θ_{LS})$.

其中归一化常数Z确保了pW是一个概率，即，$\sum_{w∈W} p_W(w) = 1$。注意query窗口w可以是W中的任意点，并不一定对应着训练窗口。LS cue评估一个窗口覆盖一个目标的概率为$LS(w,θ_{LS}) = p_W(w, θ_{LS})$。

**Efficient computation of LS**. In practice, we consider a discretized version of the 4D space W that contains only windows with integer coordinates in a normalized image of 100 × 100 pixels. We precompute (9) for all these windows, obtaining a lookup table. For a test window w we find its nearest neighbor w^nn in the discretized space W and read out its value from the lookup table. This procedure is very efficient as it takes very few operations for each test window compared to N operations in (9).

**LS的高效计算**。在实践中，我们考虑了一个4D空间W的离散版本，在100 × 100大小的归一化图像中，只包含了整数坐标的窗口。我们对这些窗口预先计算(9)，得到了一个查找表。对于一个测试窗口w，我们在离散空间W中找到其最近邻w^nn，从查找表中读出其值。这个过程是非常高效的，对每个测试窗口，只需要几个运算，而(9)则需要N个运算才可以。

### 2.6 Implementation Details

**MS**. For every scale s ∈ {16, 24, 32, 48, 64} and channel c we rescale the image to s × s and then compute MS(w, θ_MS) using one integral image [10] (indicating the sum over the
saliency of pixels in a rectangle). 对每个尺度s ∈ {16, 24, 32, 48, 64}和通道c，我们将图像改变到s × s大小，然后使用积分图像计算MS(w, θ_MS)（指示一个矩形中的显著像素的和）。

**CC**. We convert the image to the quantized LAB space 4 × 8 × 8 and then compute CC(w, θ_CC) using one integral image per quantized color. 我们将图像转换到量化的LAB空间4 × 8 × 8，然后计算CC(w, θ_CC)，每个量化的色彩使用一个积分图像。

**ED**. We rescale the image to 200 × 200 pixels and then compute ED(w, θ_ED) using one integral image (indicating the number of edgels in a rectangle). 我们将图像大小改变到200 × 200，然后使用一个积分图像计算ED(w, θ_ED)（指示在一个矩形中的边缘数量）。

**SS**. We obtain superpixels S(θ_SS) using the algorithm of [19] with segmentation scale θ_SS. We efficiently compute SS(w, θ_SS) using one integral image per superpixel, as
detailed in Section 2.4. 我们使用算法[19]在分割尺度θ_SS下，得到超像素S(θ_SS)。我们对每个超像素计算一个积分图像，用其高效的计算SS(w, θ_SS)，在2.4节中详述。

## 3. Learning Cue Parameters

We learn the parameters of the objectness cues from a training dataset T consisting of 1,183 images from the PASCAL VOC 07 train+val dataset [17]. In PASCAL VOC 07 each image is annotated with ground-truth bounding-boxes for all objects from 20 categories (boat, bicycle, horse, etc.). We use for training all images containing only objects of the six classes bird, car, cat, cow, dog, and sheep, for a total of 1,587 instances O (objects marked as difficult are ignored).

我们从训练集T中学习目标度cue的参数，训练集包含PASCAL VOC 07 train+val数据集中的1183幅图像。在PASCAL VOC 07中，每幅图像都用真值边界框对所有目标进行了标注，共包括20个类别。我们用作训练的图像，只包含6个类别，共计1587个实例（标记为难的目标都忽略了）。

The parameters to be learned are θ_CC, θ_ED, θ_SS, θ_LS, and θ^s_MS (for five scales s). The first three are learned in a unified manner (Section 3.1), whereas specialized methods are
given for θ_MS (Section 3.2) and θ_LS (Section 3.3).

学习的参数是θ_CC, θ_ED, θ_SS, θ_LS, 和 θ^s_MS（包含5个尺度）。前3个的学习方式是一样的（3.1节），θ_MS和θ_LS学习方法见3.2和3.3节。

### 3.1 Learning the Parameters of CC, ED, SS

### 3.2 Learning the Parameters of MS

### 3.3 Learning the Parameters of LS

## 4. Bayesian Cue Integration

Since the proposed cues are complementary, using several of them at the same time appears promising. MS gives only a rough indication of where an object is as it is designed to find blob-like things (Fig. 2). Instead, CC provides more accurate windows, but sometimes misses objects entirely (Fig. 3). ED provides many false positives on textured areas (Fig. 4). SS is very distinctive but depends on good superpixels, which are fragile for small objects (Fig. 6). LS provides a location-size prior without analyzing image pixels.

由于提出的cues是互补的，同时使用它们的几个，似乎是很有希望的。MS给出了一个目标的大致位置，因为其设计是用于找到类似blob的东西。CC则给出更准确的窗口，但有时候会完全漏掉目标。ED在纹理区域给出了很多假阳性。SS非常独特，但依赖于好的超像素，对于小目标，这是非常脆弱的。LS在不分析图像像素的情况下，给出了位置-大小的先验。

To combine n cues C ⊆ {MS, CC, ED, SS, LS} we train a Bayesian classifier to distinguish between positive and negative n-tuples of values (one per cue). For each training image, we sample 100,000 windows from the distribution given by the MS cue (thus biasing toward better locations), and then compute the other cues in C for them. Windows covering an annotated object are considered as positive examples W^obj, all others are considered as negative W^bg.

为结合n个cues C ⊆ {MS, CC, ED, SS, LS}，我们训练了一个贝叶斯分类器，来区分正样本和负样本的n元组（一个cue一个值）。对于每个训练图像，我们从MS cue的分布中采样10万个窗口（因此会倾向于更好的位置），然后对这些窗口计算C中的其他cues。覆盖了标注目标的窗口是正样本W^obj，其他的认为是负样本W^bg。

A natural way to combine the cues is to model them jointly. Unfortunately, integrating many cues C would require an enormous number of samples to estimate the joint likelihood p(cue_1, . . . , cue_n|obj), where cue_i ∈ C. Therefore, we choose a Naive Bayes approach. We have also tried a linear discriminant, but it performed worse in our experiments, probably because it combines cues in a too simplistic a manner (i.e., a weighted sum).

将cues结合到一起的一种很自然的方法是，对其同时建模。不幸的是，整合很多cues C需要大量样本来估计联合概率p(cue_1, . . . , cue_n|obj)，其中cue_i ∈ C。因此，我们选择了一个朴素Bayes方法。我们还尝试了一种线性判别式，但在我们的实验中效果很差，可能是因为其结合cues的方式太简单（即，加权和）。

In the Naive Bayes model, the cues are independent, so training consists of estimating the priors p(obj), p(bg), which we set by relative frequency, and the individual cue likelihoods p(cue|c), for cue ∈ C and c ∈ {obj, bg}, from the large sets of training windows W^obj, W^bg.

在朴素贝叶斯模型中，cues是独立的，所以训练就需要估计先验p(obj), p(bg)，我们通过相对频率进行设置，还要设置独立的cue似然p(cue|c)，对cue ∈ C和c ∈ {obj, bg}，从训练窗口W^obj, W^bg的大型集合中。

After training, when a test image is given, we can sample any desired number T of windows from MS and then compute the other cues for them (as done above for the training windows). The posterior probability of a test window w is

训练后，当测试图像给定时，我们可以从MS中采样任意数量T窗口，然后计算其他cues。测试窗口w的后验概率是

$$p(obj|C) = \frac {P(C|obj)p(obj)} {p(C)} = \frac {p(obj) ∏_{cue∈C} p(cue|obj)} {\sum_{c∈obj,bg} p(c) ∏_{cue∈C} p(cue|c)}$$(13)

The posterior given by (13) constitutes the final objectness score of w. The T test windows and their scores (13) form a distribution from which we can sample any desired final number F of windows. Note how (13) allows us to combine any subset C of cues, e.g., pairs of cues C = {MS, CC}, triplets C = {MS, CC, SS}, or all cues C = {MS, CC, ED, SS, LS}. Function (13) can combine any subset rapidly without recomputing the likelihoods.

(13)计算得到的后验就是w最终的目标度分数。T测试窗口和其分数(13)形成了一个分布，从中我们可以采样任意期望数量的F个窗口。注意(13)可以使我们结合cue集合C的任意子集，如，成对的cues C = {MS, CC}，三元组C = {MS, CC, SS}，或所有cues C = {MS, CC, ED, SS, LS}。函数(13)可以快速结合任意子集，不需要重新计算似然。

**Multinomial sampling**. This procedure samples independently windows according to their scores. T window scores form a multinomial distribution D. Naively sampling F windows from D requires T⋅F operations, so we use an efficient sampling procedure. From the T scores we build the cumulative sum score vector v. Note how the elements of v are sorted in ascending order and the last vector element v(T) is the sum of all scores. To sample a window we first generate a random number u uniformly distributed in [0, v(T)]. Then, we do a binary search in v to retrieve the interval [v_i-1, v_i] containing u. The chosen sample i has score v_i. Hence, sampling F windows only costs F⋅log2(T) operations.

**多项式采样**。这个过程根据其分数独立的采样窗口。T窗口分数形成了一个多样式分布D。从D中朴素采样F窗口，需要T⋅F个运算，所以我们使用了一个高效的采样过程。从T个分数中，我们构建了累加和分数向量v。注意v的元素是怎样以升序排列的，最后一个向量元素v(T)是所有分数的和。为采样一个窗口，我们首先生成一个随机数，在[0, v(T)]中均匀分布。然后，我们在v中进行一个二值搜索，以获得包含u的区间[v_i-1, v_i]。选择的样本i的分数为v_i。因此，采样F个窗口只需F⋅log2(T)次运算。

**NMS sampling**. This procedure samples windows according to their individual scores and the spatial overlap between windows. The goal is twofold: sample high scored windows and cover diverse image locations. This helps in detecting more objects. We start by sampling the single highest scored window. Then, we iteratively consider the next highest scored window and sample it if it does not overlap strongly with any higher scored window (i.e., intersection-over-union > 0:5). This is repeated until the desired F samples are obtained.

## 5. Experiments

## 8. Conclusions

We presented an objectness measure trained to distinguish object windows from background ones. It combines several image cues, including the innovative SS cue. On the task of detecting objects of new classes unseen during training, we have demonstrated that objectness outperforms traditional saliency [27], [28], interest point detectors, Semantic Labeling [22], and the HOG detector [11]. Moreover, we have demonstrated algorithms to employ objectness to greatly reduce the number of windows evaluated by class-specific detectors and to reduce their false-positive rates. Several recent works are increasingly demonstrating the value of objectness in other applications, such as learning object classes in weakly supervised scenarios [13], [30], [47], pixelwise segmentation of objects [2], [52], unsupervised object discovery [34], and learning humans-object interactions [45]. The source code of the objectness measure is available at http://www.vision.ee.ethz.ch/~calvin.

我们给出一个目标度度量，训练用于区分目标窗口和背景窗口。它结合了几种图像cues，包括新的SS cue。在检测新类别目标时（训练时未曾见过），我们证明了目标度超过了传统的显著性度量，感兴趣点检测器，语义标签和HOG检测器。而且，我们证明了采用目标度的算法极大的降低了检测器评估的窗口数量，降低了假阳性率。几个最近的工作都证明了目标度在其应用的价值，比如在弱监督的情况下学习其目标类别，逐像素的目标分割，无监督的目标发现，学习人类与目标的交互。代码已开源。