# Rapid Object Detection using a Boosted Cascade of Simple Features

Paul Viola, Michael Jones

Cambridge

## 0. Abstract

This paper describes a machine learning approach for visual object detection which is capable of processing images extremely rapidly and achieving high detection rates. This work is distinguished by three key contributions. The first is the introduction of a new image representation called the “Integral Image” which allows the features used by our detector to be computed very quickly. The second is a learning algorithm, based on AdaBoost, which selects a small number of critical visual features from a larger set and yields extremely efficient classifiers[6]. The third contribution is a method for combining increasingly more complex classifiers in a “cascade” which allows background regions of the image to be quickly discarded while spending more computation on promising object-like regions. The cascade can be viewed as an object specific focus-of-attention mechanism which unlike previous approaches provides statistical guarantees that discarded regions are unlikely to contain the object of interest. In the domain of face detection the system yields detection rates comparable to the best previous systems. Used in real-time applications, the detector runs at 15 frames per second without resorting to image differencing or skin color detection.

本文描述了一种视觉目标检测的机器学习方法，可以非常迅速的处理图像，得到很高的检测率。本文有三个关键的贡献，非常独特。第一是引入了一种新的图像表示，称为积分图像，我们的检测器可以使用这个特征计算的非常快速。第二是一个学习算法，基于AdaBoost，从一个大的集合中选择了少量关键视觉特征，得到了极其高效的分类器[6]。第三个贡献是以一种级联的方式将越来越复杂的分类器结合到一起，这可以使得图像背景区域可以迅速的丢弃掉，而在有希望的像目标的区域进行更多的计算。这个级联可以视为一种针对目标的聚焦注意力机制，之前的方法是给出统计的确保，丢弃掉不太可能的感兴趣目标。在人脸检测的领域，系统得到的检测率与之前最好的系统类似。在实时应用中可以应用，检测器的运行速度为15FPS，不需要进行图像差分或肤色检测。

## 1. Introduction

This paper brings together new algorithms and insights to construct a framework for robust and extremely rapid object detection. This framework is demonstrated on, and in part motivated by, the task of face detection. Toward this end we have constructed a frontal face detection system which achieves detection and false positive rates which are equivalent to the best published results [16, 12, 15, 11, 1]. This face detection system is most clearly distinguished from previous approaches in its ability to detect faces extremely rapidly. Operating on 384 by 288 pixel images, faces are detected at 15 frames per second on a conventional 700 MHz Intel Pentium III. In other face detection systems, auxiliary information, such as image differences in video sequences, or pixel color in color images, have been used to achieve high frame rates. Our system achieves high frame rates working only with the information present in a single grey scale image. These alternative sources of information can also be integrated with our system to achieve even higher frame rates.

本文为构建稳健、非常快速的目标检测框架，带来了新的算法和洞见。这个框架在人脸检测的任务上进行了展示，也是部分由这个任务推动的。为此，我们构建了一个正面人脸的检测系统，其检测率和假阳性率与目前最好的结果类似。这个人脸检测系统与之前的方法明显有区别，因为检测人脸的速度特别快。在384x288的图像上，人脸的检测速度是15 FPS，运行平台是传统的Intel Pentium III 700MHz电脑。在其他人脸检测系统，辅助信息，比如视频序列中的图像差，或彩色图像中的像素颜色，进行了使用以得到高帧率。我们的系统只用单幅灰度图像的信息，就获得了高帧率。其他信息源也可以整合进我们的系统，以获得更高的帧率。

There are three main contributions of our object detection framework. We will introduce each of these ideas briefly below and then describe them in detail in subsequent sections.

我们的目标检测框架有三个主要贡献。下面我们会简要介绍这些思想，然后在后续章节中详细描述。

The first contribution of this paper is a new image representation called an integral image that allows for very fast feature evaluation. Motivated in part by the work of Papageorgiou et al., our detection system does not work directly with image intensities [10]. Like these authors we use a set of features which are reminiscent of Haar Basis functions (though we will also use related filters which are more complex than Haar filters). In order to compute these features very rapidly at many scales we introduce the integral image representation for images. The integral image can be computed from an image using a few operations per pixel. Once computed, any one of these Harr-like features can be computed at any scale or location in constant time.

本文的第一个贡献，是一种的新的图像表示，称为积分图像，可以进行非常快速的特征评估。部分受到了Papageorgiou等人工作的启发，我们的检测系统没有直接用图像灰度进行计算。与这些作者类似，我们使用的特征集合与Haar基函数相关（虽然我们还使用了相关的滤波器，比Haar滤波器更加复杂）。为在几个尺度上非常快速的计算这些特征，我们引入了积分图像的表示。

The second contribution of this paper is a method for constructing a classifier by selecting a small number of important features using AdaBoost [6]. Within any image sub-window the total number of Harr-like features is very large, far larger than the number of pixels. In order to ensure fast classification, the learning process must exclude a large majority of the available features, and focus on a small set of critical features. Motivated by the work of Tieu and Viola, feature selection is achieved through a simple modification of the AdaBoost procedure: the weak learner is constrained so that each weak classifier returned can depend on only a single feature [2]. As a result each stage of the boosting process, which selects a new weak classifier, can be viewed as a feature selection process. AdaBoost provides an effective learning algorithm and strong bounds on generalization performance [13, 9, 10].

本文的第二种贡献，是构建一个分类器的方法，使用AdaBoost选择了少量重要的特征。在任何图像子窗口中，Haar类特征的总量都是非常大的，远比像素的数量要大。为确保快速分类，学习过程必须排除绝大部分可用的特征，聚焦在少数关键特征上。受到Tieu和Viola工作的启发，通过一个简单的修正AdaBoost过程就可以进行特征选择：弱学习器是受到约束的，这样返回的每个弱分类器都只依赖于单个特征[2]。结果是，在boosting过程的每个阶段，都选择了一个新的弱分类器，可以视为一个特征选择过程。AdaBoost给出了一个有效的学习算法，和在泛化性能上的很强的边界。

The third major contribution of this paper is a method for combining successively more complex classifiers in a cascade structure which dramatically increases the speed of the detector by focusing attention on promising regions of the image. The notion behind focus of attention approaches is that it is often possible to rapidly determine where in an image an object might occur [17, 8, 1]. More complex processing is reserved only for these promising regions. The key measure of such an approach is the “false negative” rate of the attentional process. It must be the case that all, or almost all, object instances are selected by the attentional filter.

本文第三个主要的贡献是，将更复杂的分类器，以级联结构结合起来，这极大的提高了检测器的速度，将注意力聚焦在图像中有希望的区域。注意力聚焦方法背后的概念是，通常快速确定在图像中哪里有目标，是可能的。更复杂的处理只给这些有希望的区域保留。这样一种方法的关键度量是，注意力过程的假阴性率。必须是这种情况，所有，或几乎所有的目标实例，是由注意力滤波器选择的。

We will describe a process for training an extremely simple and efficient classifier which can be used as a “supervised” focus of attention operator. The term supervised refers to the fact that the attentional operator is trained to detect examples of a particular class. In the domain of face detection it is possible to achieve fewer than 1% false negatives and 40% false positives using a classifier constructed from two Harr-like features. The effect of this filter is to reduce by over one half the number of locations where the final detector must be evaluated.

我们会描述一个过程，训练一个极其简单和有效的分类器，用作注意力算子的有监督聚焦。术语有监督是指，注意力算子是训练用于检测特定类别的样本。在人脸检测的领域，用2个Haar类的特征可以构建出一个分类器，假阴性率少于1%，假阳性率40%。这个滤波器的效果是，最终检测器必须计算的位置减少了一半。

Those sub-windows which are not rejected by the initial classifier are processed by a sequence of classifiers, each slightly more complex than the last. If any classifier rejects the sub-window, no further processing is performed. The structure of the cascaded detection process is essentially that of a degenerate decision tree, and as such is related to the work of Geman and colleagues [1, 4].

这些没有被初始分类器拒绝的子窗口，由一系列分类器进行处理，每个都比前一个略微复杂。如果任何分类器拒绝了子窗口，就不进行任何进一步的处理。级联检测过程的结构实质上就是一个退化的决策树，因此与Geman及其同事的工作相关。

An extremely fast face detector will have broad practical applications. These include user interfaces, image databases, and teleconferencing. In applications where rapid frame-rates are not necessary, our system will allow for significant additional post-processing and analysis. In addition our system can be implemented on a wide range of small low power devices, including hand-helds and embedded processors. In our lab we have implemented this face detector on the Compaq iPaq handheld and have achieved detection at two frames per second (this device has a low power 200 mips Strong Arm processor which lacks floating point hardware).

一个极其快速的人脸检测器会有广泛的实际应用。这包括用户界面，图像数据集，和远程会议。在快速帧率不是必须的应用中，我们的系统可以进行更多的后处理和分析。另外，我们的系统可以在很多小型低能耗设备上实现，包括手持设备和嵌入式处理器。在我们的实验室中，我们在Compaq iPaq手持设备上进行了实现，得到2 FPS的检测速度。

The remainder of the paper describes our contributions and a number of experimental results, including a detailed description of our experimental methodology. Discussion of closely related work takes place at the end of each section.

本文余下部分描述了我们的贡献，和几个试验结果，包括我们试验方法论的详细描述。在每节的最后，都讨论了相关的工作。

## 2. Features

Our object detection procedure classifies images based on the value of simple features. There are many motivations for using features rather than the pixels directly. The most common reason is that features can act to encode ad-hoc domain knowledge that is difficult to learn using a finite quantity of training data. For this system there is also a second critical motivation for features: the feature based system operates much faster than a pixel-based system.

我们的目标检测过程基于简单特征的值进行图像分类。使用特征，而不是直接使用像素，有很多动力。最常见的原因是，特征可以可以编码ad-hoc领域知识，使用有限数量的训练数据很难学习。对于这个系统，还有一个关键的使用特征的动力：基于特征的系统比基于像素的系统的运算速度要快很多。

The simple features used are reminiscent of Haar basis functions which have been used by Papageorgiou et al. [10]. More specifically, we use three kinds of features. The value of a two-rectangle feature is the difference between the sum of the pixels within two rectangular regions. The regions have the same size and shape and are horizontally or vertically adjacent (see Figure 1). A three-rectangle feature computes the sum within two outside rectangles subtracted from the sum in a center rectangle. Finally a four-rectangle feature computes the difference between diagonal pairs of rectangles.

使用的简单特征与Haar基函数相关，在Papageorgiou等[10]中进行了使用。更具体的，我们使用三种特征。双矩形特征的值，是在两个矩形区域中，像素和的差值。这些区域大小和形状都相同，是水平或竖直相邻的（见图1）。三矩形特征计算的是中间矩形的和，减去两边矩形的和。最后，四矩形特征计算的是，对角矩形对的差。

Given that the base resolution of the detector is 24x24, the exhaustive set of rectangle features is quite large, over 180,000. Note that unlike the Haar basis, the set of rectangle features is overcomplete.

检测器的基础分辨率是24x24，矩形特征的穷举集合数量是很大的，超过180000。注意，与Haar基不一样的是，矩形特征集合的是过完备的。

### 2.1. Integral Image

Rectangle features can be computed very rapidly using an intermediate representation for the image which we call the integral image. The integral image at location x, y contains the sum of the pixels above and to the left of x, y, inclusive:

矩形特征可以使用图像的中间表示进行快速计算，我们称之为积分图像。在位置x,y处的积分图像，包含x,y处往上往左的所有像素之和，包括：

$$ii(x,y) = \sum_{x'≤x, y'≤y} i(x',y')$$

where ii(x,y) is the integral image and i(x,y) is the original image. Using the following pair of recurrences: 其中ii(x,y)是积分图像，i(x,y)是原始图像。使用下面的循环对：

$$s(x,y) = s(x,y-1) + i(x,y)$$(1)

$$ii(x,y) = ii(x-1, y) + s(x,y)$$(2)

(where s(x,y) is the cumulative row sum, s(x,-1)=0, and ii(-1,y)=0) the integral image can be computed in one pass over the original image.

其中s(x,y)是行的累积和，s(x,-1)=0, ii(-1,y)=0。积分图像可以使用原始图像进行一次性计算。

Using the integral image any rectangular sum can be computed in four array references (see Figure 2). Clearly the difference between two rectangular sums can be computed in eight references. Since the two-rectangle features defined above involve adjacent rectangular sums they can be computed in six array references, eight in the case of the three-rectangle features, and nine for four-rectangle features.

使用积分图像可以用四个阵列参考计算任意矩形和（见图2）。很清楚，两个矩形和的差可以用8个参考来计算。由于上面定义的两矩形特征涉及到临近的矩形和，可以在6个阵列参考中计算，在三矩形特征时是8个，四矩形特征是9个。

### 2.2. Feature Discussion

Rectangle features are somewhat primitive when compared with alternatives such as steerable filters [5, 7]. Steerable filters, and their relatives, are excellent for the detailed analysis of boundaries, image compression, and texture analysis. In contrast rectangle features, while sensitive to the presence of edges, bars, and other simple image structure, are quite coarse. Unlike steerable filters the only orientations available are vertical, horizontal, and diagonal. The set of rectangle features do however provide a rich image representation which supports effective learning. In conjunction with the integral image, the efficiency of the rectangle feature set provides ample compensation for their limited flexibility.

当与一些其他的比较时，如steerable滤波器，矩形特征有些原始。Steerable滤波器及其相关衍生，对于边缘的详细分析，图像压缩和纹理分析，是非常优秀的。比较起来，矩形特征，对边缘、条状物和其他简单的图像结构的存在非常敏感，是非常粗糙的。与steerable滤波器不同，可用的方向只有垂直，水平和对角线。但是矩形特征的集合确实提供了丰富的图像表示，支持有效的学习。与积分图像一起，矩形特征集合的高效率，对其有限的灵活性提供了充足的补偿。

## 3. Learning Classification Functions

Given a feature set and a training set of positive and negative images, any number of machine learning approaches could be used to learn a classification function. In our system a variant of AdaBoost is used both to select a small set of features and train the classifier [6]. In its original form, the AdaBoost learning algorithm is used to boost the classification performance of a simple (sometimes called weak) learning algorithm. There are a number of formal guarantees provided by the AdaBoost learning procedure. Freund and Schapire proved that the training error of the strong classifier approaches zero exponentially in the number of rounds. More importantly a number of results were later proved about generalization performance [14]. The key insight is that generalization performance is related to the margin of the examples, and that AdaBoost achieves large margins rapidly.

给定特征集合，和包含正图像和负图像的训练集，可以使用很多机器学习方法来学习一个分类函数。在我们的系统中，使用了AdaBoost的一个变体，来选择特征的一个小集合，并来训练分类器。以其原始形式，AdaBoost学习算法用于提高简单学习算法（有时候称为弱学习算法）的分类性能。AdaBoost学习过程提供了几个正式的确保。Freund和Schapire证明了，强分类器方法的训练误差在几轮中就以指数速度趋向于0。更重要的是，后来还证明了关于泛化性能的几个结果[14]。关键的洞见是，泛化性能与样本的余地相关，AdaBoost迅速获得了很大的余地。

Recall that there are over 180,000 rectangle features associated with each image sub-window, a number far larger than the number of pixels. Even though each feature can be computed very efficiently, computing the complete set is prohibitively expensive. Our hypothesis, which is borne out by experiment, is that a very small number of these features can be combined to form an effective classifier. The main challenge is to find these features.

回忆一下，对每个图像子窗口，有超过180000个矩形特征，比像素的数量要多的多。虽然每个特征都可以很高效的计算，但是计算完成的集合仍然是非常的耗时。我们被试验证明的假设是，很少量的这些特征可以结合到一起形成一个有效的分类器。主要的挑战是找到这些特征。

In support of this goal, the weak learning algorithm is designed to select the single rectangle feature which best separates the positive and negative examples (this is similar to the approach of [2] in the domain of image database retrieval). For each feature, the weak learner determines the optimal threshold classification function, such that the minimum number of examples are misclassified. A weak classifier $h_j(x)$ thus consists of a feature $f_j$, a threshold $θ_j$ and a parity $p_j$ indicating the direction of the inequality sign:

为支持这个目标，设计了弱学习算法来选择单个的矩形特征，对正样本和负样本进行最好的区分（这与[2]中的方法在图像数据库检索的领域类似）。对于每个特征，弱学习器确定最佳分类函数阈值，这样误分类的样本数量达到最小。弱分类器$h_j(x)$因此包括了特征$f_j$，阈值$θ_j$和parity $p_j$，指示的是不等式符号的方向：

$$h_j(x) = \left\{ \begin{matrix} 1 && if p_j f_j(x)<p_j θ_j \\ 0 && otherwise \end{matrix} \right.$$

Here x is a 24x24 pixel sub-window of an image. See Table 1 for a summary of the boosting process. 这里x是一幅图像的24x24像素的子窗口。表1给出了boosting过程的总结。

In practice no single feature can perform the classification task with low error. Features which are selected in early rounds of the boosting process had error rates between 0.1 and 0.3. Features selected in later rounds, as the task becomes more difficult, yield error rates between 0.4 and 0.5.

在实践中，没有哪个单个特征可以以很低的误差进行分类任务。在早期boosting过程的几轮选择的特征，其错误率在0.1到0.3之间。在后期选择的特征，因为任务越来越难，得到的错误率在0.4到0.5之间。

### 3.1. Learning Discussion

Many general feature selection procedures have been proposed (see chapter 8 of [18] for a review). Our final application demanded a very aggressive approach which would discard the vast majority of features. For a similar recognition problem Papageorgiou et al. proposed a scheme for feature selection based on feature variance [10]. They demonstrated good results selecting 37 features out of a total 1734 features.

提出了很多通用的特征选择过程，见[18]的第8章的回顾。我们的最终应用需要一个非常激进的方法，可以抛弃大量特征。对于一个类似的识别问题，Papageorgiou等提出了一个基于特征变化的特征选择的方案。他们证明了在从1734个特征中选择37个特征，得到了很好的结果。

Roth et al. propose a feature selection process based on the Winnow exponential perceptron learning rule [11]. The Winnow learning process converges to a solution where many of these weights are zero. Nevertheless a very large number of features are retained (perhaps a few hundred or thousand).

Roth等提出一个特征选择的过程，基于Winnow指数感知机学习规则[11]。Winnow学习过程收敛到一个解，其中很多权重都是0。尽管如此，还是保留了大量的特征（可能有几百或几千）。

### 3.2. Learning Results

While details on the training and performance of the final system are presented in Section 5, several simple results merit discussion. Initial experiments demonstrated that a frontal face classifier constructed from 200 features yields a detection rate of 95% with a false positive rate of 1 in 14084. These results are compelling, but not sufficient for many real-world tasks. In terms of computation, this classifier is probably faster than any other published system, requiring 0.7 seconds to scan an 384 by 288 pixel image. Unfortunately, the most straightforward technique for improving detection performance, adding features to the classifier, directly increases computation time.

训练细节和最终系统的性能在第5部分给出，但几个简单的结果值得进行讨论。初始的试验表明，从200个特征构建的正面人脸分类器得到了95%的检测率，FP率为14084中有1个。这些结果是非常好的，但对很多真实世界任务仍然不足够。在计算上，这个分类器可能比任何其他发表的系统都要快，处理一幅384x288的图像，只需要0.7秒。不幸的是，多数直接改进检测性能的技术，对分类器增加特征，都会直接增加计算时间。

For the task of face detection, the initial rectangle features selected by AdaBoost are meaningful and easily interpreted. The first feature selected seems to focus on the property that the region of the eyes is often darker than the region of the nose and cheeks (see Figure 3). This feature is relatively large in comparison with the detection sub-window, and should be somewhat insensitive to size and location of the face. The second feature selected relies on the property that the eyes are darker than the bridge of the nose.

对于人脸检测的任务，AdaBoost选择的初始的矩形特征是有意义的，可以很容易的解释。第一个选择的特征似乎聚焦在下面这个性质，即眼部区域通常比鼻子和脸颊区域要更暗（见图3）。这个特征与检测子窗口比较，相对较大，应当对人脸的大小和位置相对不太敏感。第二个选择的特征是，眼镜部分要比鼻梁部分要暗的性质。

## 4. The Attentional Cascade

This section describes an algorithm for constructing a cascade of classifiers which achieves increased detection performance while radically reducing computation time. The key insight is that smaller, and therefore more efficient, boosted classifiers can be constructed which reject many of the negative sub-windows while detecting almost all positive instances (i.e. the threshold of a boosted classifier can be adjusted so that the false negative rate is close to zero). Simpler classifiers are used to reject the majority of subwindows before more complex classifiers are called upon to achieve low false positive rates.

本节描述了一个算法，构建分类器的级联，可以得到更高的检测性能，同时急剧的降低计算时间。关键的洞见是，更小因此更有效率的boosted分类器，可以通过那些拒绝很多负子窗口同时检测了几乎所有正样本的来构建（即，一个boosted分类器的阈值可以进行调整，这样假阴性率接近于0）。更简单的分类器用于拒绝多数的子窗口，然后更复杂的分类器被调用以获得低的假阳性率。

The overall form of the detection process is that of a degenerate decision tree, what we call a “cascade” (see Figure 4). A positive result from the first classifier triggers the evaluation of a second classifier which has also been adjusted to achieve very high detection rates. A positive result from the second classifier triggers a third classifier, and so on. A negative outcome at any point leads to the immediate rejection of the sub-window.

检测过程的总体形式是一个退化的决策树，我们称之为一个级联（见图4）。第一个分类器的一个正结果，会触发第二个分类器的评估，这也经过了调整，以获得非常高的检测率。第二个分类器的一个正结果会触发第三个分类器，等等。在任意点处的负结果，会导致子窗口被立刻拒绝。

Stages in the cascade are constructed by training classifiers using AdaBoost and then adjusting the threshold to minimize false negatives. Note that the default AdaBoost threshold is designed to yield a low error rate on the training data. In general a lower threshold yields higher detection rates and higher false positive rates.

级联中的阶段，是使用AdaBoost训练的分类器来构建的，然后调整阈值以最小化假阴性。注意，默认的AdaBoost阈值是设计用于在训练数据上得到一个低误差率。一般来说，更低的阈值会得到更高的检测率和更高的假阳性率。

For example an excellent first stage classifier can be constructed from a two-feature strong classifier by reducing the threshold to minimize false negatives. Measured against a validation training set, the threshold can be adjusted to detect 100% of the faces with a false positive rate of 40%. See Figure 3 for a description of the two features used in this classifier.

比如，一个优秀的第一阶段分类器，可以由一个双特征强分类器，通过降低阈值以最小化假阳性来构建得到。在一个验证训练集上进行度量，阈值经过调整，可以检测100%的人脸，假阳性率为40%。图3给出了在这个分类器中使用的两个特征的描述。

Computation of the two feature classifier amounts to about 60 microprocessor instructions. It seems hard to imagine that any simpler filter could achieve higher rejection rates. By comparison, scanning a simple image template, or a single layer perceptron, would require at least 20 times as many operations per sub-window.

双特征分类器的计算，需要大约60个微处理器指令。似乎很难想象，任何更简单的滤波器可以获得更高的拒绝率。比较起来，扫描一个简单的图像模板，或一个层的感知机，在每个子窗口中至少需要20倍的运算。

The structure of the cascade reflects the fact that within any single image an overwhelming majority of subwindows are negative. As such, the cascade attempts to reject as many negatives as possible at the earliest stage possible. While a positive instance will trigger the evaluation of every classifier in the cascade, this is an exceedingly rare event.

级联的结构反应了下面的事实，在任何一幅图像中，绝大部分子窗口都是负的。这样，级联就需要在最早的阶段拒绝尽量多的负窗口。一个正实例会触发级联中每个分类器的评估，但这是一个极其罕见的事件。

Much like a decision tree, subsequent classifiers are trained using those examples which pass through all the previous stages. As a result, the second classifier faces a more difficult task than the first. The examples which make it through the first stage are “harder” than typical examples. The more difficult examples faced by deeper classifiers push the entire receiver operating characteristic (ROC) curve downward. At a given detection rate, deeper classifiers have correspondingly higher false positive rates.

有一个决策树很像，后续的分类器的训练，使用的是通过了所有之前阶段的样本。结果是，第二个分类器比第一个分类器面临着更难的任务。通过了第一个阶段的样本，是比典型样本更难的。更深的分类器面临的这些更难的样本，会将整个ROC曲线向下推。在给定的检测率上，更深的分类器有对应的更高的假阳性率。

### 4.1. Training a Cascade of Classifiers

The cascade training process involves two types of tradeoffs. In most cases classifiers with more features will achieve higher detection rates and lower false positive rates. At the same time classifiers with more features require more time to compute. In principle one could define an optimization framework in which: i) the number of classifier stages, ii) the number of features in each stage, and iii) the threshold of each stage, are traded off in order to minimize the expected number of evaluated features. Unfortunately finding this optimum is a tremendously difficult problem.

级联的训练过程涉及到两种类型的折中。在多数情况下，有更多特征的分类器会得到更高的检测率和更低的假阳性率。同时，有更多特征的分类器需要更多的时间来计算。原则上，可以定义一个优化框架，其中：i)分类器阶段的数量，ii)在每个阶段中的特征数量，iii)每个阶段的阈值，都是折中的，以最小化期望数量的评估特征。不幸的是，找到这种最佳条件是极其困难的问题。

In practice a very simple framework is used to produce an effective classifier which is highly efficient. Each stage in the cascade reduces the false positive rate and decreases the detection rate. A target is selected for the minimum reduction in false positives and the maximum decrease in detection. Each stage is trained by adding features until the target detection and false positives rates are met (these rates are determined by testing the detector on a validation set). Stages are added until the overall target for false positive and detection rate is met.

实践中，使用了一种非常简单的框架来生成有效的分类器，效率也很高。级联的每个阶段都降低了假阳性率和检测率。选择了一个目标，作为假阳性降低的最小值，和检测率降低的最大值。每个阶段的训练是通过增加特征，直到达到目标检测率和假阳性率（这些率是通过在验证集上测试检测器得到的）。阶段数量不断增加，直到达到假阳性率和检测率的总体目标。

### 4.2. Detector Cascade Discussion

The complete face detection cascade has 38 stages with over 6000 features. Nevertheless the cascade structure results in fast average detection times. On a difficult dataset, containing 507 faces and 75 million sub-windows, faces are detected using an average of 10 feature evaluations per subwindow. In comparison, this system is about 15 times faster than an implementation of the detection system constructed by Rowley et al. [12].

完整的人脸检测级联，包含38个阶段，超过6000个特征。尽管如此，级联结构也会得到非常快的平均检测时间。在一个困难的数据集上，包含507张人脸，7500万个子窗口，对每个子窗口，人脸使用了平均10个特征就检测到了。比较之下，这个系统比Rowley等[12]构建的检测系统的实现快了大约15倍。

A notion similar to the cascade appears in the face detection system described by Rowley et al. in which two detection networks are used [12]. Rowley et al. used a faster yet less accurate network to prescreen the image in order to find candidate regions for a slower more accurate network. Though it is difficult to determine exactly, it appears that Rowley et al.’s two network face system is the fastest existing face detector.

与级联类似的概念在Rowley等的检测系统中也出现了，其中使用了2个检测网络[12]。Rowley等使用了一个更快但是不太精确的网络，来预先筛选图像，以找到候选区域，送入一个更慢却更精确的网络。虽然很难精确的确定，似乎Rowley等的双网络人脸系统是现有最快的人脸检测器。

The structure of the cascaded detection process is essentially that of a degenerate decision tree, and as such is related to the work of Amit and Geman [1]. Unlike techniques which use a fixed detector, Amit and Geman propose an alternative point of view where unusual co-occurrences of simple image features are used to trigger the evaluation of a more complex detection process. In this way the full detection process need not be evaluated at many of the potential image locations and scales. While this basic insight is very valuable, in their implementation it is necessary to first evaluate some feature detector at every location. These features are then grouped to find unusual co-occurrences. In practice, since the form of our detector and the features that it uses are extremely efficient, the amortized cost of evaluating our detector at every scale and location is much faster than finding and grouping edges throughout the image.

级联检测过程的结构，实际上就是一个退化的决策树，因此与Amit和Geman [1]的工作相关。Amit和Geman的方法与使用固定检测器的技术不一样，他们提出了另一种观点，其中不常见的简单图像特征的联合出现，用于触发一个更复杂的检测过程。这样，完整的检测过程不需要在很多潜在的图像位置和尺度进行计算。这种基本洞见是非常宝贵的，在其实现中，首先在每个位置都计算某个特征检测器，是很必要的。这些特征然后进行分组，以找到不常见的联合出现。在实践中，由于我们的检测器的形式和其使用的特征是非常高效的，在每个尺度和位置上计算我们的检测器的摊销代价，比在图像中找到和分组边缘要快的多。

In recent work Fleuret and Geman have presented a face detection technique which relies on a “chain” of tests in order to signify the presence of a face at a particular scale and location [4]. The image properties measured by Fleuret and Geman, disjunctions of fine scale edges, are quite different than rectangle features which are simple, exist at all scales, and are somewhat interpretable. The two approaches also differ radically in their learning philosophy. The motivation for Fleuret and Geman’s learning process is density estimation and density discrimination, while our detector is purely discriminative. Finally the false positive rate of Fleuret and Geman’s approach appears to be higher than that of previous approaches like Rowley et al. and this approach. Unfortunately the paper does not report quantitative results of this kind. The included example images each have between 2 and 10 false positives.

在最近的工作中，Fleuret和Geman提出了一种人脸检测技术，依赖于一种测试链，表明在特定的尺度和位置存在人脸[4]。由Fleuret和Geman测量的图像属性，精细尺度边缘的分裂，与矩形特征是非常不一样的，因为矩形特征简单，在所有尺度都存在，而且在一定程度上是可解释的。这两种方法在学习哲学上也非常不一样。Fleuret和Geman的学习过程的动机，是密度估计和密度区分，而我们的检测器则纯粹是区分性的。最后，Fleuret和Geman的方法的假阳性率，似乎比之前的方法要更高，比如Rowley等和这种方法。不幸的是，文章并没有给出这种的量化结果。这包括每个包含2-10个假阳性的样本图像。

## 5. Results

A 38 layer cascaded classifier was trained to detect frontal upright faces. To train the detector, a set of face and nonface training images were used. The face training set consisted of 4916 hand labeled faces scaled and aligned to a base resolution of 24 by 24 pixels. The faces were extracted from images downloaded during a random crawl of the world wide web. Some typical face examples are shown in Figure 5. The non-face subwindows used to train the detector come from 9544 images which were manually inspected and found to not contain any faces. There are about 350 million subwindows within these non-face images.

训练了一个38层的级联分类器，来检测正面竖直的人脸。为训练检测器，使用了人脸和非人脸的训练图像集合。人脸训练集包含4916幅手工标注的人脸，并进行了缩放和对齐，得到24x24像素的基础分辨率。图像是从网络上随机爬取的，并从中提取出人脸。一些典型的人脸样本如图5所示。用于训练检测器的非人脸子窗口，包含9544幅图像，经过人工检查，确保并不包含任何人脸。再这些非人脸图像中，大约有350 million个子窗口。

The number of features in the first five layers of the detector is 1, 10, 25, 25 and 50 features respectively. The remaining layers have increasingly more features. The total number of features in all layers is 6061.

在检测器前5层中，分别有1，10，25，25，50个特征。剩下的层的特征数量逐渐增加。所有层中的特征总计数量为6061。

Each classifier in the cascade was trained with the 4916 training faces (plus their vertical mirror images for a total of 9832 training faces) and 10,000 non-face sub-windows (also of size 24 by 24 pixels) using the Adaboost training procedure. For the initial one feature classifier, the nonface training examples were collected by selecting random sub-windows from a set of 9544 images which did not contain faces. The non-face examples used to train subsequent layers were obtained by scanning the partial cascade across the non-face images and collecting false positives. A maximum of 10000 such non-face sub-windows were collected for each layer.

在级联中的每个分类器，是用4916幅训练人脸图像（包含其竖直镜像图像，总计有9832幅训练人脸），和10000幅非人脸子窗口（大小也是24x24），使用AdaBoost训练过程训练得到的。对于初始的一个特征的分类器，非人脸训练样本是通过从9544幅不包含人脸的图像中选择随机子窗口得到的。用于训练后续层的非人脸样本，是通过查找非人脸图像中的部分级联，收集找到那些假阳性样本。对每个子窗口，最多收集了10000幅这种非人脸子窗口。

**Speed of the Final Detector**

The speed of the cascaded detector is directly related to the number of features evaluated per scanned sub-window. Evaluated on the MIT+CMU test set [12], an average of 10 features out of a total of 6061 are evaluated per sub-window. This is possible because a large majority of sub-windows are rejected by the first or second layer in the cascade. On a 700 Mhz Pentium III processor, the face detector can process a 384 by 288 pixel image in about .067 seconds (using a starting scale of 1.25 and a step size of 1.5 described below). This is roughly 15 times faster than the Rowley-Baluja-Kanade detector [12] and about 600 times faster than the Schneiderman-Kanade detector [15].

级联检测器的速度，与每个扫描子窗口中计算的特征数量直接相关。在MIT+CMU测试集[12]上进行的评估，在每个子窗口中，平均计算了6061个特征中的10个。因为大量子窗口在级联中的第一或第二层就被拒绝掉了，所以这是可能的。在一个700 MHz Pentium III处理器上，人脸检测器可以在.067秒内处理一幅384x288像素的图像（使用的初始尺度为1.25，步长为1.5，下面会进行叙述）。这比Rowley-Baluja-Kanade检测器快了大约15倍，比Schneiderman-Kanade检测器快了大约600倍。

**Image Processing**

All example sub-windows used for training were variance normalized to minimize the effect of different lighting conditions. Normalization is therefore necessary during detection as well. The variance of an image sub-window can be computed quickly using a pair of integral images. Recall that $σ^2 = m^2 - \sum_x^2/N$, where σ is the standard deviation, m is the mean, and x is the pixel value within the sub-window. The mean of a sub-window can be computed using the integral image. The sum of squared pixels is computed using an integral image of the image squared (i.e. two integral images are used in the scanning process). During scanning the effect of image normalization can be achieved by post-multiplying the feature values rather than pre-multiplying the pixels.

所有用于训练的样本子窗口，其方差都进行了归一化，以最小化不同光照条件的影响。因此在检测的时候，归一化也是必须的。一个图像子窗口的方差，可以用一对积分图像进行很快的计算。回忆一下，$σ^2 = m^2 - \sum_x^2/N$，其中σ是标准差，m是均值，x是子窗口中的像素值。一个子窗口的均值可以用积分图像来计算。像素平方的和，是用图像平方的积分图像计算的（即，在扫描过程中使用了两个积分图像）。在扫描过程中，图像归一化的效果可以通过特征值的后相乘得到，而不用预先相乘。

**Scanning the Detector**

The final detector is scanned across the image at multiple scales and locations. Scaling is achieved by scaling the detector itself, rather than scaling the image. This process makes sense because the features can be evaluated at any scale with the same cost. Good results were obtained using a set of scales a factor of 1.25 apart.

最终的检测器对图像的多个尺度、多个位置进行扫描。缩放是通过对检测器缩放得到的，而不是缩放图像。这个过程是有意义的，因为特征可以在任意尺度用相同的代价进行计算。使用的尺度集合系数差距为1.25，就得到了很好的结果。

The detector is also scanned across location. Subsequent locations are obtained by shifting the window some number of pixels Δ. This shifting process is affected by the scale of the detector: if the current scale is s the window is shifted by [sΔ], where [] is the rounding operation.

检测器还在多个位置进行了扫描。后续的位置是通过将图像窗口平移几个像素Δ得到的。这个平移过程受到检测器的尺度的影响：如果当前尺度是s，窗口就偏移[sΔ]，其中[]是四舍五入运算。

The choice of Δ affects both the speed of the detector as well as accuracy. The results we present are for Δ = 1.0. We can achieve a significant speedup by setting Δ = 1.5 with only a slight decrease in accuracy.

Δ的选择既影响了检测器的速度，也影响准确率。我们给出的结果是Δ = 1.0。设置Δ = 1.5，可以得到显著的加速，准确率只有略微下降。

**Integration of Multiple Detections**

Since the final detector is insensitive to small changes in translation and scale, multiple detections will usually occur around each face in a scanned image. The same is often true of some types of false positives. In practice it often makes sense to return one final detection per face. Toward this end it is useful to postprocess the detected sub-windows in order to combine overlapping detections into a single detection.

由于最终检测器对平移和尺度的较小变化是不敏感的，在一幅扫描的图像中，在每个人脸附近，会出现多个检测结果。对于一些类型的假阳性，也会出现类似的现象。在实践中，对每张人脸得到一个最终检测结果，才是有意义的。为此，对检测到的子窗口进行后处理，将重叠的检测结合成一个检测，这是有用的。

In these experiments detections are combined in a very simple fashion. The set of detections are first partitioned into disjoint subsets. Two detections are in the same subset if their bounding regions overlap. Each partition yields a single final detection. The corners of the final bounding region are the average of the corners of all detections in the set.

在试验中，检测结果的结合是非常简单的。检测结果集合首先分成不相交的子集。两个检测结果如果其区域重叠，就在相同的子集中。每个分割都会得到单个最终检测结果。最终边界区域的边角，是集合中所有检测的边角的平均。

**Experiments on a Real-World Test Set**

We tested our system on the MIT+CMU frontal face test set [12]. This set consists of 130 images with 507 labeled frontal faces. A ROC curve showing the performance of our detector on this test set is shown in Figure 6. To create the ROC curve the threshold of the final layer classifier is adjusted from -∞ to +∞. Adjusting the threshold to +∞ will yield a detection rate of 0.0 and a false positive rate of 0.0. Adjusting the threshold to -∞, however, increases both the detection rate and false positive rate, but only to a certain point. Neither rate can be higher than the rate of the detection cascade minus the final layer. In effect, a threshold of -∞ is equivalent to removing that layer. Further increasing the detection and false positive rates requires decreasing the threshold of the next classifier in the cascade. Thus, in order to construct a complete ROC curve, classifier layers are removed. We use the number of false positives as opposed to the rate of false positives for the x-axis of the ROC curve to facilitate comparison with other systems. To compute the false positive rate, simply divide by the total number of sub-windows scanned. In our experiments, the number of sub-windows scanned is 75,081,800.

我们在MIT+CMU正面人脸测试集[12]上测试我们的系统。这个集合包含130幅图像，有507个标注的正面人脸。在这个测试集上我们检测器性能的ROC曲线，如图6所示。为创建ROC曲线，最后一层分类器的阈值从-∞调整到+∞。调整阈值到+∞，会得到检测率为0.0，假阳性率也为0.0。将阈值调整到-∞，会同时提升检测率和假阳性率，但只会到一个特定点。两个率都不会高于检测级联减去最终层的检测率。实际上，阈值-∞等价于去掉那个层。进一步提升检测率和假阳性率，需要降低级联中下一个分类器的阈值。因此，为构建一个完整的ROC曲线，分类器层需要被去除。我们使用假阳性率作为ROC曲线的x轴，以促进与其他系统的比较。为计算假阳性率，将假阳性数量除以扫描的子窗口的总数量。在我们的试验中，扫描的子窗口的数量为75,081,800。

Unfortunately, most previous published results on face detection have only included a single operating regime (i.e. single point on the ROC curve). To make comparison with our detector easier we have listed our detection rate for the false positive rates reported by the other systems. Table 2 lists the detection rate for various numbers of false detections for our system as well as other published systems. For the Rowley-Baluja-Kanade results [12], a number of different versions of their detector were tested yielding a number of different results, they are all listed in under the same heading. For the Roth-Yang-Ahuja detector [11], they reported their result on the MIT+CMU test set minus 5 images containing line drawn faces removed.

不幸的是，多数之前发表的人脸检测结果，都只有ROC曲线上的一个点。为与我们的检测器进行更容易的比较，我们对其他系统的假阳性率列出了我们的检测率。表2给出了在几种假阳性数量的条件下，我们的系统和其他发表的系统的检测率。对于Rowley-Baluja-Kanade的结果[12]，其检测器的几个不同版本进行了测试，得到了几个不同的结果，它们都列在了一起。对于Roth-Yang-Ahuja检测器[11]，他们给出的结果是在MIT+CMU测试集去掉5幅线条画的脸。

Figure 7 shows the output of our face detector on some test images from the MIT+CMU test set. 图7展示了在MIT+CMU测试上，我们的检测器在一些测试图像上的输出。

**A simple voting scheme to further improve results**

In table 2 we also show results from running three detectors (the 38 layer one described above plus two similarly trained detectors) and outputting the majority vote of the three detectors. This improves the detection rate as well as eliminating more false positives. The improvement would be greater if the detectors were more independent. The correlation of their errors results in a modest improvement over the best single detector.

在表2中，我们还展示了运行三个检测器（38层检测器，和两个类似训练得到的检测器）并输出三个检测器的主要投票的结果。这改进了检测率，并消除了更多的假阳性率。如果检测器更加独立，这个改进会更大。其误差的关联，会比单个最好的检测器有些许改进。

## 6. Conclusions

We have presented an approach for object detection which minimizes computation time while achieving high detection accuracy. The approach was used to construct a face detection system which is approximately 15 faster than any previous approach.

我们提出了一种目标检测的方法，计算时间非常短，检测率也非常高。利用这种方法构建了一个人脸检测系统，比之前的方法快了大约15倍。

This paper brings together new algorithms, representations, and insights which are quite generic and may well have broader application in computer vision and image processing.

本文将新算法、表示和洞见结合到了一起，非常通用，在计算机视觉和图像处理中会有更广泛的应用。

Finally this paper presents a set of detailed experiments on a difficult face detection dataset which has been widely studied. This dataset includes faces under a very wide range of conditions including: illumination, scale, pose, and camera variation. Experiments on such a large and complex dataset are difficult and time consuming. Nevertheless systems which work under these conditions are unlikely to be brittle or limited to a single set of conditions. More importantly conclusions drawn from this dataset are unlikely to be experimental artifacts.

最后，本文在一个困难的人脸检测数据集上给出了详细的试验。这个数据集包含了在很多情况下的人脸，包括：光照，尺度，姿态和相机变化。在这样一个大型复杂数据集上的试验，是非常困难和耗时的。尽管如此，在这些条件下工作的系统，不太可能是脆弱的，或局限于单个条件集合。更重要的是，从这个数据集上得出的结论，不太可能是试验的假象。