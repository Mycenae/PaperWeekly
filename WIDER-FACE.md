# WIDER FACE: A Face Detection Benchmark (2015)

Shuo Yang et al. University of Hong Kong

## Abstract 摘要

Face detection is one of the most studied topics in the computer vision community. Much of the progresses have been made by the availability of face detection benchmark datasets. We show that there is a gap between current face detection performance and the real world requirements. To facilitate future face detection research, we introduce the WIDER FACE dataset, which is 10 times larger than existing datasets. The dataset contains rich annotations, including occlusions, poses, event categories, and face bounding boxes. Faces in the proposed dataset are extremely challenging due to large variations in scale, pose and occlusion, as shown in Fig. 1. Furthermore, we show that WIDER FACE dataset is an effective training source for face detection. We benchmark several representative detection systems, providing an overview of state-of-the-art performance and propose a solution to deal with large scale variation. Finally, we discuss common failure cases that worth to be further investigated.

人脸检测是计算机视觉中研究最多的领域。大多数研究进展要归功于人脸检测基准测试数据集的建立。我们发现，现在的人脸检测性能与真实世界的需要有一定的差距。为方便将来的人脸检测研究，我们提出了WIDER FACE数据集，比现有的数据集大了10倍以上。这个数据集有丰富的标注，包括遮挡、姿态、事件类别和人脸边界框。提出的数据集中的人脸非常有挑战性，因为在尺度、姿态和遮挡上的变化都很大，如图1所示。而且，我们说明，WIDER FACE数据集对人脸检测训练非常高效。我们基准测试了几种有代表性的检测系统，给出了目前最好性能的概览，提出了处理大的尺度变化的方法。最后，我们讨论了常用的失败案例，将来值得进一步研究。

Figure 1. We propose a WIDER FACE dataset for face detection, which has a high degree of variability in scale, pose, occlusion, expression, appearance and illumination. We show example images (cropped) and annotations. The annotated face bounding box is denoted in green color. The WIDER FACE dataset consists of 393, 703 labeled face bounding boxes in 32, 203 images (Best view in color).

图1. 我们提出了WIDER FACE人脸检测数据集，其尺度、姿态、遮挡、表情、外貌和光照等情况变化很大。我们给出了例子图像（剪切过）和标注。标注的人脸边界框表示为绿色。WIDER FACE数据集包含393,703标注的人脸边界框，共32,203幅图像。

## 1. Introduction 引言

Face detection is a critical step to all facial analysis algorithms, including face alignment, face recognition, face verification, and face parsing. Given an arbitrary image, the goal of face detection is to determine whether or not there are any faces in the image and, if present, return the image location and extent of each face [27]. While this appears as an effortless task for human, it is a very difficult task for computers. The challenges associated with face detection can be attributed to variations in pose, scale, facial expression, occlusion, and lighting condition, as shown in Fig. 1. Face detection has made significant progress after the seminal work by Viola and Jones [22]. Modern face detectors can easily detect near frontal faces and are widely used in real world applications, such as digital camera and electronic photo album. Recent research [3, 15, 18, 25, 28] in this area focuses on the unconstrained scenario, where a number of intricate factors such as extreme pose, exaggerated expressions, and large portion of occlusion can lead to large visual variations in face appearance.

人脸检测是所有面部分析算法的关键步骤，包括人脸对齐、人脸识别、人脸验证和人脸解析。给定任意一幅图像，人脸检测的目标是确定图像中是否有人脸存在，如果存在，返回每张人脸的图像位置和范围[27]。虽然这对于人类来说是一项非常简单的任务，但对于计算机来说非常困难。人脸检测相关的挑战主要是由于姿态、尺度、面部表情、遮挡和光照条件的变化，如图1所示。人脸检测自从最初Viola和Jones[22]的工作后有了显著的进展。现代人脸检测器可以轻松检测到距离较近的正面人脸，在真实世界应用中得到了广泛应用，如数字相机和电子相册。这个领域近年来的研究[3,15,18,25,28]聚焦在不受限的场景，其中一些复杂的因素，如极端的姿态、夸张的表情和很大比例的遮挡，可以导致人脸的外观变化较大。

Publicly available benchmarks such as FDDB [12], AFW [30], PASCAL FACE [24], have contributed to spurring interest and progress in face detection research. However, as algorithm performance improves, more challenging datasets are needed to trigger progress and to inspire novel ideas. Current face detection datasets typically contain a few thousand faces, with limited variations in pose, scale, facial expression, occlusion, and background clutters, making it difficult to assess for real world performance. As we will demonstrate, the limitations of datasets have partially contributed to the failure of some algorithms in coping with heavy occlusion, small scale, and atypical pose.

公开可用的基准测试，如FDDB[12]，AFW[30]，PASCAL FACE[24]，在过去刺激了人脸检测研究的兴趣和发展。但是，随着算法性能改进，需要更有挑战性的数据集来触发进展和启发新想法。目前的人脸检测数据集一般包含数千张人脸，在姿态、尺度、表情、遮挡和复杂的背景变化有限，这就很难估计在真实世界中应用的性能。我们会证明，数据集的限制是一些算法难以应对严重的遮挡、小尺度和非典型姿态的一部分原因。

In this work, we make three contributions. (1) We introduce a large-scale face detection dataset called WIDER FACE. It consists of 32, 203 images with 393, 703 labeled faces, which is 10 times larger than the current largest face detection dataset [13]. The faces vary largely in appearance, pose, and scale, as shown in Fig. 1. In order to quantify different types of errors, we annotate multiple attributes: occlusion, pose, and event categories, which allows in depth analysis of existing algorithms. (2) We show an example of using WIDER FACE through proposing a multi-scale two-stage cascade framework, which uses divide and conquer strategy to deal with large scale variations. Within this framework, a set of convolutional networks with various size of input are trained to deal with faces with a specific range of scale. (3) We benchmark four representative algorithms [18, 22, 25, 28], either obtained directly from the original authors or reimplemented using open-source codes. We evaluate these algorithms on different settings and analyze conditions in which existing methods fail.

在本文中，我们作出三个贡献。(1)我们提出了大规模人脸检测数据集WIDER FACE，包括32203幅图像，有393703个标记的人脸，比现有最大的人脸检测数据集[13]大上10倍。数据集中的人脸在外表、姿态和尺度上变化很大，如图1所示。为量化不同类型的错误，我们标注了多种属性：遮挡、姿态和事件类别，这就可以深度分析现有的算法。(2)我们展示了使用WIDER FACE的一个例子，提出了一种多尺度两阶段级联框架，使用分而治之的策略来处理大的尺度变化。在这个框架下，多个不同输入大小的卷积神经网络训练用来处理特定尺度范围的人脸。(3)我们基准测试了四种有代表性的算法[18,22,25,28]，要么是原作者的实现，要么是从开源代码得到的重新实现。我们在不同设置下评估了这些算法，分析了现有方法失败的情况。

## 2. Related Work 相关的工作

**Brief review of recent face detection methods**: Face detection has been studied for decades in the computer vision literature. Modern face detection algorithms can be categorized into four categories: cascade based methods [3, 11, 16, 17, 22], part based methods [20, 24, 30], channel feature based methods [2, 25], and neural network based methods [7, 15, 28]. Here we highlight a few notable studies. A detailed survey can be found in [27, 29]. The seminal work by Viola and Jones [22] introduces integral image to compute Haar-like features in constant time. These features are then used to learn AdaBoost classifier with cascade structure for face detection. Various later studies follow a similar pipeline. Among those variants, SURF cascade [16] achieves competitive performance. Chen et al. [3] learns face detection and alignment jointly in the same cascade framework and obtains promising detection performance.

**近年来人脸检测算法的简要回归**：人脸检测已经在计算机视觉文献中研究了几十年了。现代人脸检测算法可以分类四类：基于级联的方法[3,11,16,17,22]，基于部位的方法[20,24,30]，基于通道特征的方法[2,25]和基于神经网络的方法[7,15,28]。这里我们强调了几个著名的研究。详细的调查可以参考[27,29]。Viola和Jones的开创性工作[22]提出了积分图像来计算类Haar特征。这些特征然后用于学习级联结构的AdaBoost分类器进行人脸检测。后续的各种研究都遵循了类似的流程。在这些变体中，级联SURF[16]取得了非常好的性能。Chen等[3]在同样的级联框架下学习了人脸检测和人脸对齐算法，得到了很有希望的检测性能。

One of the well-known part based methods is deformable part models (DPM) [8]. Deformable part models define face as a collection of parts and model the connections of parts through Latent Support Vector Machine. The part based methods are more robust to occlusion compared with cascade-based methods. A recent study [18] demonstrates state-of-the art performance with just a vanilla DPM, achieving better results than more sophisticated DPM variants [24, 30]. Aggregated channel feature (ACF) is first proposed by Dollar et al. [4] to solve pedestrian detection. Later on, Yang et al. [25] applied this idea on face detection. In particular, features such as gradient histogram, integral histogram, and color channels are combined and used to learn boosting classifier with cascade structure. Recent studies [15, 28] show that face detection can be further improved by using deep learning, leveraging the high capacity of deep convolutional networks. We anticipate that the new WIDER FACE data can benefit deep convolutional network that typically requires large amount of data for training.

一个非常有名的基于部位的方法是可变部位模型(DPM)[8]。可变部位模型将人脸定义为部位的集合，用Latent Support Vector Machine对部位间的连接进行建模。基于部位的方法与基于级联的方法相比，对遮挡的稳健性更好。最近的一项研究[18]表明，一种普通DPM的目前最好的性能，比更复杂的DPM变体[24,30]性能更好。通道特征聚集(ACF)首先由Dollar等[4]提出，用于解决行人检测问题。后来，Yang等[25]将这种思想用于人脸检测。尤其是，像梯度直方图、积分直方图和色彩通道这样的特征综合到一起，用于学习级联结构的boosting分类器。最近的研究[15,28]表明，人脸检测可以用深度学习进一步改进，利用深度卷积网络的高性能。我们期待新的WIDER FACE数据集可以使深度卷积网络受益，这通常都需要很多数据进行训练。

**Existing datasets**: We summarize some of the well-known face detection datasets in Table 1. AFW [30], FDDB [12], and PASCAL FACE [24] datasets are most widely used in face detection. AFW dataset is built using Flickr images. It has 205 images with 473 labeled faces. For each face, annotations include a rectangular bounding box, 6 landmarks and the pose angles. FDDB dataset contains the annotations for 5, 171 faces in a set of 2, 845 images. PASCAL FACE consists of 851 images and 1, 341 annotated faces. Recently, IJB-A [13] is proposed for face detection and face recognition. IJB-A contains 24, 327 images and 49, 759 faces. MALF is the first face detection dataset that supports fine-grained evaluation. MALF [26] consists of 5, 250 images and 11, 931 faces. The FDDB dataset has helped driving recent advances in face detection. However, it is collected from the Yahoo! news website which biases toward celebrity faces. The AFW and PASCAL FACE datasets contain only a few hundred images and has limited variations in face appearance and background clutters. The IJBA dataset has large quantity of labeled data; however, occlusion and pose are not annotated. The MALF dataset labels fine-grained face attributes such as occlusion, pose and expression. The number of images and faces are relatively small. Due to the limited variations in existing datasets, the performance of recent face detection algorithms saturates on current face detection benchmarks. For instance, on AFW, the best performance is 97.2% AP; on FDDB, the highest recall is 91.74%; on PASCAL FACE, the best result is 92.11% AP. The best few algorithms have only marginal difference.

**现有的数据集**：我们在表1中总结了一些有名的人脸检测数据集。AFW[30], FDDB[12]和PASCAL FACE[24]数据集在人脸检测中得到了广泛使用。AFW数据集是用Flickr构建的，包含205幅图像，473个标注的人脸。对于每个人脸，标注包括一个矩形边界框，6个特征点和姿态角度。FDDB数据集包括2845幅图像，标注了5171个人脸。PASCAL FACE包含851幅图像，1341个标注的人脸。最近，[13]提出了IJB-A数据集进行人脸检测和人脸识别。IJB-A包含24327幅图像，标注了49759个人脸。MALF是第一个支持细粒度评估的人脸检测数据集。MALF[26]包括5250幅图像，11931个人脸。FDDB数据集曾经帮助推动了人脸检测在近年来的发展。但是，它是从Yahoo! news网站上收集的，里面名人的人脸非常多。AFW和PASCAL FACE数据集只包括了几百幅图像，人脸外表和背景复杂度上的变化很有限。IJBA数据集中标注的数据很多，但是没有标注遮挡情况和姿态。MALF数据集标注了细粒度人脸属性，如遮挡、姿态和表情，但图像和人脸数量相对较少。由于现有数据集中人脸的变化有限，近来的人脸检测算法的性能在现有的人脸检测基准测试中逐渐饱和。比如，在LFW中，最好的性能是97.2% AP；在FDDB中，最高的召回率为91.74%；在PASCAL FACE，最好的结果是92.11% AP。最好的几种算法间只有很小的差异。

## 3. WIDER FACE Dataset

### 3.1. Overview 概览

To our knowledge, WIDER FACE dataset is currently the largest face detection dataset, of which images are selected from the publicly available WIDER dataset [23]. We choose 32, 203 images and label 393, 703 faces with a high degree of variability in scale, pose and occlusion as depicted in Fig. 1. WIDER FACE dataset is organized based on 60 event classes. For each event class, we randomly select 40%/10%/50% data as training, validation and testing sets. Here, we specify two training/testing scenarios:

据我们所知，WIDER FACE数据集是目前最大的人脸检测数据集，图像是从公开可用的WIDER数据集中选择的[23]。我们选择了32303幅图像，标注了393703个人脸，在尺度、姿态和遮挡等方面人脸变化都很大，如图1所示。WIDER FACE数据集基于60类时间组织。对于每个事件类别，我们随机的选择了40%/10%/50%的数据作为训练、验证和测试集。这里，我们指定两种训练/测试方案：

- Scenario-Ext: A face detector is trained using any external data, and tested on the WIDER FACE test partition. 人脸检测器是由外部数据训练得到，然后在WIDER FACE测试部分进行测试。

- Scenario-Int: A face detector is trained using WIDER FACE training/validation partitions, and tested on WIDER FACE test partition. 人脸检测器在WIDER FACE训练/验证集上训练，在WIDER FACE测试集上测试。

We adopt the same evaluation metric employed in the PASCAL VOC dataset [6]. Similar to MALF [26] and Caltech [5] datasets, we do not release bounding box ground truth for the test images. Users are required to submit final prediction files, which we shall proceed to evaluate.

我们采用与PASCAL VOC数据集[6]相同的评估准则。与MALF[26]和Caltech[5]数据集类似，我们没有放出测试图像的真值边界框。用户需要提交其最终预测文件，然后我们来进行评估。

### 3.2. Data Collection 数据采集

**Collection methodology**. WIDER FACE dataset is a subset of the WIDER dataset [23]. The images in WIDER were collected in the following three steps: 1) Event categories were defined and chosen following the Large Scale Ontology for Multimedia (LSCOM) [19], which provides around 1, 000 concepts relevant to video event analysis. 2) Images are retrieved using search engines like Google and Bing. For each category, 1, 000-3, 000 images were collected. 3) The data were cleaned by manually examining all the images and filtering out images without human face. Then, similar images in each event category were removed to ensure large diversity in face appearance. A total of 32, 203 images are eventually included in the WIDER FACE dataset.

**收集方法**。WIDER FACE数据集是WIDER数据集[23]的子集。WIDER中的图像是通过以下三个步骤收集来的：1)根据Large Scale Concept Ontology for Multimedia(LSCOM)，定义并选择事件类别，LSCOM提供了大约1000种与视频事件分析相关的概念。2)用搜索引擎如Google和Bing检索得到图像，对每个种类，收集1000-3000幅图像。3)数据清洗是通过手工检查所有图像，并滤除没有人脸的图像。然后，每个事件类别中类似的图像被去除掉，来确保人脸外表的多样性。WIDER FACE数据集中包含了共计32203幅图像。

**Annotation policy**. We label the bounding boxes for all the recognizable faces in the WIDER FACE dataset. The bounding box is required to tightly contain the forehead, chin, and cheek, as shown in Fig. 2. If a face is occluded, we still label it with a bounding box but with an estimation on the scale of occlusion. Similar to the PASCAL VOC dataset [6], we assign an ’Ignore’ flag to the face which is very difficult to be recognized due to low resolution and small scale (10 pixels or less). After annotating the face bounding boxes, we further annotate the following attributes: pose (typical, atypical) and occlusion level (partial, heavy). Each annotation is labeled by one annotator and cross-checked by two different people.

**标注策略**。我们对WIDER FACE数据集中所有可辨识的人脸标注边界框。边界框需要紧密包围着额头、下巴、脸颊，如图2所示。如果脸部被遮挡了，我们仍然用边界框进行标注，但会估计遮挡的程度。与PASCAL VOC数据集[6]类似，我们对于非常难以辨认的人脸会指定一个“忽略”标志位，这可能是由于分辨率低和尺度小（小于10个像素）。在标注了人脸边界框之后，我们进一步标注下列属性：姿态（典型，非典型），遮挡水平（部分，重度）。每个标注都由一个标注者标注，然后由另外两个人来交叉检查。

Figure 2. Examples of annotation in WIDER FACE dataset (Best view in color).

### 3.3. Properties of WIDER FACE 数据集的性质

WIDER FACE dataset is challenging due to large variations in scale, occlusion, pose, and background clutters. These factors are essential to establishing the requirements for a real world system. To quantify these properties, we use generic object proposal approaches [1, 21, 31], which are specially designed to discover potential objects in an image (face can be treated as an object). Through measuring the number of proposals vs. their detection rate of faces, we can have a preliminary assessment on the difficulty of a dataset and potential detection performance. In the following assessments, we adopt EdgeBox [31] as object proposal, which has good performance in both accuracy and efficiency as evaluated in [10].

WIDER FACE数据集非常具有挑战性，因为人脸的尺度、遮挡、姿态和背景复杂度都变化非常大。这些因素非常重要，可以确保满足真实世界系统的要求。为量化这些性质，我们使用通用目标候选方法[1,21,31]，这些算法设计是用来发现图像中的潜在目标（人脸可以看作是一种目标）。通过衡量候选的数量，对比其人脸检测率，我们可以对数据集的难度和潜在的检测性能有一个初步的评价。在下面的评价中，我们使用EdgeBox[31]作为目标候选，根据[10]的评估，这个算法在准确度和运算速度上都表现很好。

**Overall**. Fig. 3(a) shows that WIDER FACE has much lower detection rate compared with other face detection datasets. The results suggest that WIDER FACE is a more challenging face detection benchmark compared to existing datasets. Following the principles in KITTI [9] and MALF [26] datasets, we define three levels of difficulty: ’Easy’, ’Medium’, ’Hard’ based on the detection rate of EdgeBox [31], as shown in the Fig. 3(a). The average recall rates for these three levels are 92%, 76%, and 34%, respectively, with 8, 000 proposal per image.

**整体性质**。图3(a)表明，与其他人脸检测数据集相比，WIDER FACE的检测率非常低。这个结果说明，与现有的数据集相比，WIDER FACE是一个更有挑战性的人脸检测基准测试。遵循KITTI[9]和MALF[26]数据集的原则，基于EdgeBox[31]的检测率，我们定义了三个层次的难度：“容易”，“中等”，“困难”，如图3(a)所示。这三个层次的平均召回率分别为92%，76%和34%，每幅图像8000个候选。

**Scale**. We group the faces by their image size (height in pixels) into three scales: small (between 10-50 pixels), medium (between 50-300 pixels), large (over 300 pixels). We make this division by considering the detection rate of generic object proposal and human performance. As can be observed from Fig 3(b), the large and medium scales achieve high detection rate (more than 90%) with 8, 000 proposals per image. For the small scale, the detection rates consistently stay below 30% even we increase the proposal number to 10, 000.

**尺度性质**。我们依据人脸像素大小（高度的像素数）将人脸分成三种尺度：小（10-50个像素），中（50-300个像素），大（超过300个像素）。通过考虑通用目标候选和人的检测率，作出这种区分。从图3(b)可以观察到，在每幅图像8000个候选的情况下，大型和中型尺度的可以取得很高的检测率（超过90%）。在小尺度上，检测率一直保持在30%以下，即使我们将候选数量增加到了10000以上。

**Occlusion**. Occlusion is an important factor for evaluating the face detection performance. Similar to a recent study [26], we treat occlusion as an attribute and assign faces into three categories: no occlusion, partial occlusion, and heavy occlusion. Specifically, we ask annotator to measure the fraction of occlusion region for each face. A face is defined as ‘partially occluded’ if 1%-30% of the total face area is occluded. A face with occluded area over 30% is labeled as ‘heavily occluded’. Fig. 2 shows some examples of partial/heavy occlusions. Fig. 3(c) shows that the detection rate decreases as occlusion level increases. The detection rates of faces with partial or heavy occlusions are below 50% with 8, 000 proposals.

**遮挡**。遮挡是评估人脸检测性能的一个重要的因素。与最近的一项研究[26]类似，我们将遮挡作为一种属性，将人脸分为三类：无遮挡，部分遮挡，重度遮挡。具体的，我们要求标注者衡量每幅人脸遮挡区域的部分。如果1%-30%的人脸区域被遮挡，就定义为“部分遮挡”；如果遮挡超过30%，就标注为“重度遮挡”。图2给出了几个部分/重度遮挡的例子。图3(c)给出了随着遮挡水平增加，检测率会降低。部分遮挡或重度遮挡情况下的人脸检测率，即使8000候选，也低于50%。

**Pose**. Similar to occlusion, we define two pose deformation levels, namely typical and atypical. Fig. 2 shows some faces of typical and atypical pose. Face is annotated as atypical under two conditions: either the roll or pitch degree is larger than 30-degree; or the yaw is larger than 90-degree. Fig. 3(d) suggests that faces with atypical poses are much harder to be detected.

**姿态**。与遮挡类似，我们定义两个姿态变形水平，即典型与非典型。图2展示了一些典型与非典型姿态的例子。人脸在两种情况下会被标注成非典型姿态：要么roll或pitch度数大于30；或者yaw大于90度。图3(d)表明，非典型姿态的人脸更难于检测。

Figure 3. The detection rate with different number of proposals. The proposals are generated by using Edgebox [31]. Y-axis denotes for detection rate. X-axis denotes for average number of proposals per image. Lower detection rate implies higher difficulty. We show histograms of detection rate over the number of proposal for different settings (a) Different face detection datasets. (b) Face scale level. (c) Occlusion level. (d) Pose level.

图3. 不同数量候选下的检测率。候选是使用EdgeBox[31]生成的。Y轴表示检测率。X轴表示每幅图像的平均数量候选。低检测率表明更高的难度。我们给出了不同设置时不同数量候选下检测率直方图，(a)不同的人脸检测数据集，(b)人脸尺度水平，(c)遮挡水平，(d)姿态水平。

**Event**. Different events are typically associated with different scenes. WIDER FACE contains 60 event categories covering a large number of scenes in the real world, as shown in Fig. 1 and Fig. 2. To evaluate the influence of event to face detection, we characterize each event with three factors: scale, occlusion, and pose. For each factor we compute the detection rate for the specific event class and then rank the detection rate in an ascending order. Based on the rank, events are divided into three partitions: easy (41-60 classes), medium (21-40 classes) and hard (1-20 classes). We show the partitions based on scale in Fig. 4. Partitions based on occlusion and pose are included in the supplementary material.

**事件**。不同的事件通常对应不同的场景。WIDER FACE包含60个事件类别，覆盖了真实世界的大量场景，如图1和图2所示。为评估事件对人脸检测的影响，我们用三个因素描述每个事件：尺度，遮挡和姿态。对每个因素，我们都计算特定事件类别的检测率，然后将检测率按降序排列。基于排序，事件分成三种类别：简单（41-60类），中等（21-40类）和困难（1-20类）。我们在图4中给出了基于尺度的分类。基于遮挡和姿态的分类在附加材料中给出。

Figure 4. Histogram of detection rate for different event categories. Event categories are ranked in an ascending order based on the detection rate when the number of proposal is fixed at 10, 000. Top 1 − 20, 21 − 40, 41 − 60 event categories are denoted in blue, red, and green, respectively. Example images for specific event classes are shown. Y-axis denotes for detection rate. X-axis denotes for event class name.

图4. 不同事件类别的检测率直方图。事件类别基于检测率以升序排列，候选数量固定在10000个。1-20，21-40，41-60的事件类别分别表示为蓝色、红色和绿色。特定事件类别的样本图像如图所示。Y轴表示检测率。X轴表示事件类别名称。

**Effective training source**. As shown in the Table 1, existing datasets such as FDDB, AFW, and PASCAL FACE do not provide training data. Face detection algorithms tested on these datasets are frequently trained with ALFW [14], which is designed for face landmark localization. However, there are two problems. First, ALFW omits annotations of many faces with a small scale, low resolution, and heavy occlusion. Second, the background in ALFW dataset is relatively clean. As a result, many face detection approaches resort to generate negative samples from other datasets such as PASCAL VOC dataset. In contrast, all recognizable faces are labeled in the WIDER FACE dataset. Because of its event-driven nature, WIDER FACE dataset has a large number of scenes with diverse background, making it possible as a good training source with both positive and negative samples. We demonstrate the effectiveness of WIDER FACE as a training source in Sec. 5.2.

**高效的训练源**。如表1所示，现有的数据集，如FDDB，AFW和PASCAL FACE没有给出训练数据。在这些数据集上测试的人脸检测算法，很多都是在ALFW[14]上训练的，而这个数据集是为人脸特征点定位设计的。但是，这有两个问题。首先，ALFW忽略了很多尺度很小、低分辨率和遮挡严重的人脸。第二，ALFW数据集的背景相对干净。结果是，很多人脸检测算法从其他数据集如PASCAL VOC数据集中寻找生成负样本的方法。而在WIDER FACE数据集中，所有可识别的人脸都进行了标注。由于数据集是事件驱动的，WIDER FACE数据集场景众多，背景复杂，正样本和负样本都很充足，可以作为非常好的训练源。我们在5.2节中展示了其作为训练源的有效性。

## 4. Multi-scale Detection Cascade 多尺度检测级联

We wish to establish a solid baseline for WIDER FACE dataset. As we have shown in Table 1, WIDER FACE contains faces with a large range of scales. Fig. 3(b) further shows that faces with a height between 10-50 pixels only achieve a proposal detection rate of below 30%. In order to deal with the high degree of variability in scale, we propose a multi-scale two-stage cascade framework and employ a divide and conquer strategy. Specifically, we train a set of face detectors, each of which only deals with faces in a relatively small range of scales. Each face detector consists of two stages. The first stage generates multi-scale proposals from a fully-convolutional network. The second stage is a multi-task convolutional network that generates face and non-face prediction of the candidate windows obtained from the first stage, and simultaneously predicts for face location. The pipeline is shown in Fig. 5. The two main steps are explained as follow.

我们希望为WIDER FACE数据集建立一个坚实的基准。如表1所示，WIDER FACE包含的人脸尺度众多。图3(b)进一步指出，人脸高度在10-50个像素时，候选检测率低于30%。为处理尺度的高度变化性，我们提出一种多尺度两阶段级联框架，采用了分而治之的策略。具体来说，我们训练了一系列人脸检测器，每个都只处理尺度范围相对很小的人脸。每个人脸检测器包含两个阶段。第一阶段用一个全卷积网络生成多尺度候选，第二阶段是一个多任务卷积网络，从第一阶段得到的候选窗口中，生成人脸与非人脸预测，同时预测人脸位置。流程如图5所示。两个主要的步骤下面进行解释。

Figure 5. The pipeline of the proposed multi-scale cascade CNN.

Input image X -> Multiscale proposal networks(10-30 pixels, 30-120, 120-240, 340-480) -> Response maps -> Proposals -> Multiscale detection networks(30×30, 120×120, 240×240, 480×480) -> Detection results -> Final results

**Multi-scale proposal**. In this step, we joint train a set of fully convolutional networks for face classification and scale classification. We first group faces into four categories by their image size, as shown in the Table 2 (each row in the table represents a category). For each group, we further divide it into three subclasses. Each network is trained with image patches with the size of their upper bound scale. For example, Network 1 and Network 2 are trained with 30×30 and 120×120 image patches, respectively. We align a face at the center of an image patch as positive sample and assign a scale class label based on the predefined scale subclasses in each group. For negative samples, we randomly cropped patches from the training images. The patches should have an intersection-over-union (IoU) of smaller than 0.5 with any of the positive samples. We assign a value −1 as the scale class for negative samples, which will have no contribution to the gradient during training.

**多尺度候选**。在这个步骤中，我们训练一系列全卷积网络，同时进行人脸分类和尺度分类：我们首先将人脸分组成4个类别，根据其图像大小，如表2所示（表中每行代表一个类别）。对于每个分组，我们进一步将其分成三个子类。每个网络的训练都是用其尺度上限的图像块进行。比如，网络1和网络2，是分别用30×30和120×120的图像块进行处理的。我们将人脸对齐到图像块中央，作为正样本，基于每组中预定义的尺度子类别指定一个尺度类别标签。对于负样本，我们从训练图像中随机剪切图像块，这些图像块与任何正样本的IoU要小于0.5。对于负样本的尺度类别，我们指定为-1，其在训练过程中对于梯度没有贡献。

Table 2. Summary of face scale for multi-scale proposal networks.

Scale | Class 1 | Class 2 | Class 3
--- | --- | --- | ---
Network 1 | 10-15 | 15-20 | 20-30
Network 2 | 30-50 | 50-80 | 80-120
Network 3 | 120-160 | 160-200 | 200-240
Network 4 | 240-320 | 320-400 | 400-480

We take Network 2 as an example. Let $\{x_i\}_{i=1}^N$ be a set of image patches where $∀x_i ∈ R^{120×120}$. Similarly, let $\{y_i^f\}_{i=1}^N$ be the set of face class labels and $\{y_i^s\}_{i=1}^N$ be the set of scale class label, where $∀y_i^f ∈ R^{1×1}$ and $∀y_i^s ∈ R^{1×3}$. Learning is formulated as a multi-variate classification problem by minimizing the cross-entropy loss. $L=\sum_{i=1}^N y_i log p(y_i = 1|x_i)+(1-y_i)log(1-p(y_i = 1 | x_i))$, where $p(y_i|x_i)$ is modeled as a sigmoid funtion, indicating the probability of the presence of a face. This loss function can be optimized by the stochastic gradient descent with back-propagation.

我们以网络2为例子。令$\{x_i\}_{i=1}^N$为图像块集合，其中$∀x_i ∈ R^{120×120}$。类似的，令$\{y_i^f\}_{i=1}^N$为人脸类别标签集，$\{y_i^s\}_{i=1}^N$为尺度类别集，其中$∀y_i^f ∈ R^{1×1}$，$∀y_i^s ∈ R^{1×3}$。学习就是一个多变量分类问题，需要最小化交叉熵损失函数，$L=\sum_{i=1}^N y_i log p(y_i = 1|x_i)+(1-y_i)log(1-p(y_i = 1 | x_i))$，其中$p(y_i|x_i)$是一个sigmoid函数，表示人脸存在的概率。这个损失函数可以通过随机梯度下降的反向传播进行优化。

**Face detection**. The prediction of proposed windows from the previous stage is refined in this stage. For each scale category, we refine these proposals by joint training face classification and bounding box regression using the same CNN structure in the previous stage with the same input size. For face classification, a proposed window is assigned with a positive label if the IoU between it and the ground truth bounding box is larger than 0.5; otherwise it is negative. For bounding box regression, each proposal is predicted a position of its nearest ground truth bounding box. If the proposed window is a false positive, the CNN outputs a vector of [−1, −1, −1, −1]. We adopt the Euclidean loss and cross-entropy loss for bounding box regression and face classification, respectively. More details of face detection can be found in the supplementary material.

**人脸检测**。前一阶段得到的候选窗口的预测在这个阶段得到提炼。对于每个尺度类别，我们提炼这些候选，同时训练人脸分类并进行边界框回归，使用与前一阶段相同的CNN结构，同样的输入大小。对于人脸分类，候选窗口如果与真值框的IoU大于0.5，就指定正标签；否则就是负标签。对于边界框回归，每个候选都预测其最近的真值边界框的位置。如果候选窗口是false positive，CNN输出向量[−1, −1, −1, −1]。边界框回归和人脸分类分别采用欧几里得损失和交叉熵损失。人脸检测更多的细节见附加材料。

## 5. Experimental Results 试验结果

### 5.1. Benchmarks

As we discussed in Sec. 2, face detection algorithms can be broadly grouped into four representative categories. For each class, we pick one algorithm as a baseline method. We select VJ [22], ACF [25], DPM [18], and Faceness [28] as baselines. The VJ [22], DPM [18], and Faceness [28] detectors are either obtained from the authors or from open source library (OpenCV). The ACF [25] detector is reimplemented using the open source code. We adopt the Scenario-Ext here (see Sec. 3.1), that is, these detectors were trained by using external datasets and are used ‘as is’ without re-training them on WIDER FACE. We employ PASCAL VOC [6] evaluation metric for the evaluation. Following previous work [18], we conduct linear transformation for each method to fit the annotation of WIDER FACE.

我们在第2部分讨论过，人脸检测算法可以大致分成四个代表性的类别。对于每个类别，我们选择一种算法作为基准方法。我们的选择是VJ[22]，ACF[25]，DPM[18]和Faceness[28]作为基准。VJ[22], DPM[18]和Faceness[28]检测器的实现是从作者得到的，或者从OpenCV中开源的。ACF[25]检测器是从开源代码中重新实现的。我们这里采用Scenario-Ext（见3.1节），即，这些检测器使用外部数据集训练，不在WIDER FACE上进行重新训练。我们采用PASCAL VOC[6]的评估标准进行评估。遵循之前工作[18]的做法，我们对每种方法进行线性变换以适应WIDER FACE的标注。

**Overall**. In this experiment, we employ the evaluation setting mentioned in Sec. 3.3. The results are shown in Fig. 6 (a.1)-(a.3). Faceness [28] outperforms other methods on three subsets, with DPM [18] and ACF [25] as marginal second and third. For the easy set, the average precision (AP) of most methods are over 60%, but none of them surpasses 75%. The performance drops 10% for all methods on the medium set. The hard set is even more challenging. The performance quickly decreases, with a AP below 30% for all methods. To trace the reasons of failure, we examine performance on varying subsets of the data.

**总体结果**。在这个试验中，我们采用3.3节中提到的评估设置。结果如图6(a.1)-(a.3)所示。Faceness[28]在三个子集中都超过了其他方法，DPM[18]和ACF[25]略差一点排在第二和第三。对于“容易”子集，多数方法的AP超过了60%，但没有超过75%的。在“中等”集合上所有方法都降了10%。“困难”子集更有挑战性，性能迅速下降，对所有方法AP都低于30%。为追踪失败的原因，我们检查了在各种子集上数据的性能。

**Scale**. As described in Sec. 3.3, we group faces according to the image height: small (10-50 pixels), medium (50-300 pixels), and large (300 or more pixels) scales. Fig. 6 (b.1)- (b.3) show the results for each scale on un-occluded faces only. For the large scale, DPM and Faceness obtain over 80% AP. At the medium scale, Faceness achieves the best relative result but the absolute performance is only 70% AP. The results of small scale are abysmal: none of the algorithms is able to achieve more than 12% AP. This shows that current face detectors are incapable to deal with faces of small scale.

**尺度**。如3.3节所述，我们根据图像高度对人脸进行分组：小尺度（10-50像素），中等尺度（50-300像素），大尺度（大于300像素）。图6(b.1)-(b.3)给出了每个尺度下未遮挡人脸的检测结果。对于大尺度下，DPM和Faceness的检测率超过80%。在中等尺度下，Faceness取得了相对最好的结果，但性能只有70% AP。小尺度下的结果很遭的：没有一个算法能超过12% AP。这说明，现有的人脸检测器没法处理小尺度的人脸。

**Occlusion**. Occlusion handling is a key performance metric for any face detectors. In Fig. 6 (c.1)-(c.3), we show the impact of occlusion on detecting faces with a height of at least 30 pixels. As mentioned in Sec. 3.3, we classify faces into three categories: un-occluded, partially occluded (1%-30% area occluded) and heavily occluded (over 30% area occluded). With partial occlusion, the performance drops significantly. The maximum AP is only 26.5% achieved by Faceness. The performance further decreases in the heavy occlusion setting. The best performance of baseline methods drops to 14.4%. It is worth noting that Faceness and DPM, which are part based models, already perform relatively better than other methods on occlusion handling.

**遮挡**。遮挡处理对于任何人脸检测器来说都是一个关键的性能度量。在图6(c.1)-(c.3)中，我们给出了遮挡对于检测人脸的影响，这里的人脸高度至少30个像素。如3.3节所述，我们将人脸分成三类：未遮挡的，部分遮挡的（1%-30%区域遮挡）和遮挡严重的（超过了30%）。在部分遮挡的情况下，性能就会显著下降。Faceness得到的最高AP大约是26.5%。在严重遮挡的情况下，性能进一步下降。基准方法的最高性能下降到14.4%。值得注意的是，Faceness和DPM这两个基于部位的模型，在处理遮挡的情形时已经比其他方法表现相对更好一些。

**Pose**. As discussed in Sec. 3.3, we assign a face pose as atypical if either the roll or pitch degree is larger than 30-degree; or the yaw is larger than 90-degree. Otherwise a face pose is classified as typical. We show results in Fig. 6 (d.1)-(d.2). Faces which are un-occluded and with a scale larger than 30 pixels are used in this experiment. The performance clearly degrades for atypical pose. The best performance is achieved by Faceness, with a recall below 20%. The results suggest that current face detectors are only capable of dealing with faces with out-of-plane rotation and a small range of in-plane rotation.

**姿态**。如3.3节所述，如果roll或pitch度数大于30，或者yaw度数大于90度，我们就认为这个人脸是非典型的。否则这个人脸姿态就是典型的。我们在图6(d.1)-(d.2)中给出了结果。本试验中使用的是未遮挡的人脸，尺度大于30像素。在非典型姿态的情况下，性能明显下降。Faceness取得了最好的性能，但召回率低于20%。结果显示，目前的人脸检测器只能处理异面旋转和很小范围内的平面内旋转。

**Summary**. Among the four baseline methods, Faceness tends to outperform the other methods. VJ performs poorly on all settings. DPM gains good performance on medium/large scale and occlusion. ACF outperforms DPM on small scale, no occlusion and typical pose settings. However, the overall performance is poor on WIDER FACE, suggesting a large room of improvement.

**总结**。在这四种基准方法中，Faceness通常都比其他方法好。VJ在所有设置中都表现较差。DPM在中等尺度和大尺度上和遮挡情况下表现更好一些。ACF在小尺度、没有遮挡和典型姿态下超过了DPM。但是，在WIDER FACE中的整体表现都较差，说明可以改进的空间仍然很大。

Figure 6. Precision and recall curves of different subsets of WIDER FACES: (a.1)-(a.3) Overall Easy/Medium/Hard subsets. (b.1)-(b.3) Small/Medium/Large scale subsets. (c.1)-(c.3) None/Partial/Heavy occlusion subsets. (d.1)-(d.2) Typical/Atypical pose subsets.

### 5.2. WIDER FACE as an Effective Training Source

In this experiment, we demonstrate the effectiveness of WIDER FACE dataset as a training source. We adopt Scenario-Int here (see Sec. 3.1). We train ACF and Faceness on WIDER FACE to conduct this experiment. These two algorithms have shown relatively good performance on WIDER FACE previous benchmarks see (Sec. 5.1). Faces with a scale larger than 30 pixels in the training set are used to retrain both methods. We train the ACF detector using the same training parameters as the baseline ACF. The negative samples are generated from the training images. For the Faceness detector, we first employ models shared by the authors to generate face proposals from the WIDER FACE training set. After that, we train the classifier with the same procedure described in [28]. We test these models (denoted as ACF-WIDER and Faceness-WIDER) on WIDER FACE testing set and FDDB dataset.

在这个试验中，我们展示WIDER FACE数据集作为训练源的有效性。我们这里采用Scenario-Int（见3.1节）。我们在WIDER FACE上训练ACF和Faceness以进行这个-试验。这两个算法在WIDER FACE之前的基准测试（见5.1节）中表现相对较好。训练集中大于30像素的人脸用来重新训练这两个方法。我们使用相同的训练参数来训练ACF检测器，作为基准ACF。负样本从训练图像中生成，对于Faceness检测器，我们首先采用作者分享的模型，来从WIDER FACE训练集生成人脸候选。之后，我们采用与[28]一样的步骤来训练分类器。我们在WIDER FACE和FDDB数据集中测试这些模型（表示为ACF-WIEDR和Faceness-WIDER）。

**WIDER FACE**. As shown in Fig. 7, the retrained models perform consistently better than the baseline models. The average AP improvement of retrained ACF detector is 5.4% in comparison to baseline ACF detector. For the Faceness, the retrained Faceness model obtain 4.2% improvement on WIDER hard test set.

**WIDER FACE**。如图7所示，我们重新训练的模型比基准模型一直都要好。重新训练的ACF，比基准ACF检测器，在平均AP的增长达到了5.4%。对于Faceness，重新训练的Faceness模型，在WIDER困难测试集上，得到了4.2%的改进。

Figure 7. WIDER FACE as an effective training source. ACF-WIDER and Faceness-WIDER are retrained with WIDER FACE, while ACF and Faceness are the original models. (a)-(c) Precision and recall curves on WIDER Easy/Medium/Hard subsets. (d) ROC curve on FDDB dataset.

**FDDB**. We further evaluate the retrained models on FDDB dataset. Similar to WIDER FACE dataset, the retrained models achieve improvement in comparison to the baseline methods. The retrained ACF detector achieves a recall rate of 87.48%, outperforms the baseline ACF by a considerable margin of 1.4%. The retrained Faceness detector obtains a high recall rate of 91.78%. The recall rate improvement of the retrained Faceness detector is 0.8% in comparison to the baseline Faceness detector. It worth noting that the retrained Faceness detector performs much better than the baseline Faceness detector when the number of false positive is less than 300.

**FDDB**。我们在FDDB数据集上进一步评估重新训练的模型。与在WIDER FACE数据集上类似，与基准方法相比，重新训练的模型在性能上得到了改进。重新训练的ACF检测器的召回率为87.48%，超过了基准ACF 1.4%。重新训练的Faceness检测器得到了很高的召回率91.78%。与基准Faceness检测器相比，重新训练的Faceness检测器在召回率的改进上达到了0.8%。值得注意的是，在false positve小于300时，重新训练的Faceness检测器比基准Faceness检测器的表现要好很多。

**Event**. We evaluate the baseline methods on each event class individually and report the results in Table 3. Faces with a height larger than 30 pixels are used in this experiment. We compare the accuracy of Faceness and ACF models retrained on WIDER FACE training set with the baseline Faceness and ACF. With the help of WIDER FACE dataset, accuracies on 56 out of 60 event categories have been improved. It is interesting to observe that the accuracy obtained highly correlates with the difficulty levels specified in Sec. 3.3 (also refer to Fig. 4). For example, the best performance on ”Festival” which is assigned as a hard class is no more than 46% AP.

**事件**。我们在每个事件类别中都单独评估了基准方法，结果如表3所示。这个试验中所用到的是高度大于30像素的人脸。我们比较了在WIDER FACE上重新训练的Faceness和ACF模型，和基准Faceness和ACF的准确度。在WIDER FACE数据集的帮助下，60个事件类别中的56个都得到了改进。很有趣的是，我们观察到，得到的准确率与3.3节中指定的难度等级高度相关（参考图4）。比如，在“节日”这个类别上的最好表现不超过46%，这是一个困难级别的分类。

Table 3. Comparison of per class AP. To save space, we only show abbreviations of category names here. The event category is organized based on the rank sequence in Fig. 4 (from hard to easy events based on scale measure). We compare the accuracy of Faceness and ACF models retrained on WIDER FACE training set with the baseline Faceness and ACF. With the help of WIDER FACE dataset, accuracies on 56 out of 60 categories have been improved. The re-trained Faceness model wins 30 out of 60 classes, followed by the ACF model with 26 classes. Faceness wins 1 medium class and 3 easy classes.

### 5.3. Evaluation of Multi-scale Detection Cascade

In this experiment we evaluate the effectiveness of the proposed multi-scale cascade algorithm. Apart from the ACF-WIDER and Faceness-WIDER models (Sec. 5.2), we establish a baseline based on a ”Two-stage CNN”. This model differs to our multi-scale cascade model in the way it handles multiple face scales. Instead of having multiple networks targeted for different scales, the two-stage CNN adopts a more typical approach. Specifically, its first stage consists only a single network to perform face classification. During testing, an image pyramid that encompasses different scales of a test image is fed to the first stage to generate multi-scale face proposals. The second stage is similar to our multi-scale cascade model – it performs further refinement on proposals by simultaneous face classification and bounding box regression.

在这个试验中，我们评估了提出的多尺度级联算法的有效性。除了ACF-WIDER和Faceness-WIDER模型(5.2节)，我们还建立了基于两阶段CNN的基准。这个模型与我们的多尺度级联模型不同，它可以处理多个尺度的人脸。两阶段CNN采用了一种更典型的方法，而没有使用多个网络处理不同的尺度。特别的，其第一阶段只包括一个单个网络来进行人脸分类。在测试时，一个图像金字塔输入到第一阶段CNN中，生成多尺度人脸候选。第二阶段与我们的多尺度级联模型类似，通过同时进行人脸分类和边界框回归，对候选进行提炼。

We evaluate the multi-scale cascade CNN and baseline methods on WIDER Easy/Medium/Hard subsets. As shown in Fig. 8, the multi-scale cascade CNN obtains 8.5% AP improvement on the WIDER Hard subset compared to the retrained Faceness, suggesting its superior capability in handling faces with different scales. In particular, having multiple networks specialized on different scale range is shown effective in comparison to using a single network to handle multiple scales. In other words, it is difficult for a single network to handle large appearance variations caused by scale. For the WIDER Medium subset, the multi-scale cascade CNN outperforms other baseline methods with a considerable margin. All models perform comparably on the WIDER Easy subset.

我们在WIDER FACE容易/中等/困难子集上评估多尺度级联CNN和基准方法。如图8所示，多尺度级联CNN在WIDER FACE困难子集上，与重新训练的Faceness相比，得到了8.5% AP的改进，说明在处理多尺度人脸上，其能力非常好。特别是，有多个网络处理不同的尺度范围，与使用单个网络来处理多个尺度上相比，非常有效。换句话说，使用单个网络处理尺度造成的较大外观变化，非常困难。对于WIDER FACE中等难度子集，多尺度级联CNN超过了其他基准方法非常多。在WIDER FACE容易子集上，各种方法表现的比较类似。

## 6. Conclusion 结论

We have proposed a large, richly annotated WIDER FACE dataset for training and evaluating face detection algorithms. We benchmark four representative face detection methods. Even considering an easy subset (typically with faces of over 50 pixels height), existing state-of-the-art algorithms reach only around 70% AP, as shown in Fig. 8. With this new dataset, we wish to encourage the community to focusing on some inherent challenges of face detection – small scale, occlusion, and extreme poses. These factors are ubiquitous in many real world applications. For instance, faces captured by surveillance cameras in public spaces or events are typically small, occluded, and atypical poses. These faces are arguably the most interesting yet crucial to detect for further investigation.

我们提出了一个大型、标注丰富的WIDER FACE数据集，可训练和评估人脸检测算法。我们对四种有代表性的人脸检测方法进行了基准测试。即使是在容易的子集上（典型的是超过50个像素高度的人脸），现有的目前最好的算法也只得到了70% AP的结果，如图8所示。有了这个新的数据集，我们希望能鼓励研究团体关注人脸检测中的一些内在挑战，如小尺度，遮挡和极端姿态。这些因素在很多真实世界应用中都普遍存在。比如，监控摄像头在公共场所或事件中捕捉到的人脸，通常都很小，而且是遮挡的，非典型的姿态。这些人脸非常有趣，对于检测非常关键，值得进一步研究。