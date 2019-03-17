# Scene Text Detection and Recognition: The Deep Learning Era 场景文字检测与识别：深度学习时代

Shangbang Long, Xin He, Cong Yao 北京大学/旷视(Face++)

## Abstract 摘要

With the rise and development of deep learning, computer vision has been tremendously transformed and reshaped. As an important research area in computer vision, scene text detection and recognition has been inevitable influenced by this wave of revolution, consequentially entering the era of deep learning. In recent years, the community has witnessed substantial advancements in mindset, methodology and performance. This survey is aimed at summarizing and analyzing the major changes and significant progresses of scene text detection and recognition in the deep learning era. Through this article, we devote to: (1) introduce new insights and ideas; (2) highlight recent techniques and benchmarks; (3) look ahead into future trends. Specifically, we will emphasize the dramatic differences brought by deep learning and the grand challenges still remained. We expect that this review paper would serve as a reference book for researchers in this field. Related resources are also collected and compiled in our Github repository: https://github.com/Jyouhou/SceneTextPapers.

随着深度学习的崛起与发展，计算机视觉已经得到了极大的改变和重塑。作为计算机视觉中的一个重要研究领域，场景文字检测与识别也不可避免的被这股革命的浪潮所影响，结果进入了深度学习时代。近年来，学术团体见证了思维模式、方法论和性能上的大量实质性进展。这个调查目的是总结并分析场景文字检测和识别在深度学习时代的主要改变和显著进展。我们在本文中主要关注：(1)介绍新的思想和想法；(2)强调最近的技术和基准测试；(3)展望未来发展的趋势。特别的，我们会强调深度学习和重要的挑战赛带来的激动人心的改变。我们期待本综述文章能成为这个领域的研究者的参考书目。相关的资源收集编纂在我们的Github repo中：https://github.com/Jyouhou/SceneTextPapers。

**Index Terms** - Scene Text, Detection, Recognition, Deep Learning, Survey 索引词 - 场景文字，检测，识别，深度学习，调查

## 1 Introduction 引言

Undoubtedly, text is among the most brilliant and influential creations of humankind. As the written form of human languages, text makes it feasible to reliably and effectively spread or acquire information across time and space. In this sense, text constitutes the cornerstone of human civilization.

毫无疑问，文字是人类最美好最有影响力的发明之一。作为人类语言的书写形式，文字使得跨越时间与空间进行可靠有效的传播或获得信息成为可能。在这种意义下，文字构成了人类文明的基石。

On the one hand, text, as a vital tool for communication and collaboration, has been playing a more important role than ever in modern society; on the other hand, the rich, precise high level semantics embodied in text could be beneficial for understanding the world around us. For example, text information can be used in a wide range of real-world applications, such as image search [116], [134], instant translation [23], [102], robots navigation [21], [79], [80], [117], and industrial automation [16], [39], [47]. Therefore, automatic text reading from natural environments (schematic diagram is depicted in Fig. 1), a.k.a. scene text detection and recognition [172] or PhotoOCR [8], has become an increasing popular and important research topic in computer vision.

一方面，文字作为沟通和协作的关键工具，在现代社会已经成为了一个越来越重要的角色；另一方面，文字中所包含的丰富、精确的高层语义对于理解我们周围的世界是有好处的。比如，文字信息可以在广泛的真实世界应用中得到应用，比如图像搜索[116,134]，即时翻译[23,102]，机器人导航[21,79,80,117]和工业自动化[16,39,47]。所以，自然环境中的自动文字阅读（原理图如图1所示），即场景文字检测和识别[172]或PhotoOCR[8]，已经成为计算机视觉中越来越流行和重要的研究课题。

Fig. 1: Schematic diagram of scene text detection and recognition. The image sample is from Total-Text [15]. 图1. 场景文字检测和识别的原理图

However, despite years of research, a series of grand challenges may still be encountered when detecting and recognizing text in the wild. The difficulties mainly stem from three aspects [172]: 但是，尽管有多年的研究，在自然环境下检测与识别时仍然会遇到一系列重要挑战。困难主要源于三个方面[172]：

- **Diversity and Variability of Text in Natural Scenes**. Distinctive from scripts in documents, text in natural scene exhibit much higher diversity and variability. For example, instances of scene text can be in different languages, colors, fonts, sizes, orientations and shapes. Moreover, the aspect ratios and layouts of scene text may vary significantly. All these variations pose challenges for detection and recognition algorithms designed for text in natural scenes.

- **自然环境中文字的多样性和多变性**。与文件中的书写体不同，自然环境中的文字展示出了更高的多样性和多变性。比如，场景文字的案例可以是不同的语言、颜色、字体、大小、方向和形状。而且，场景文字的纵横比和布局可以变化很大。所有这些变化给自然场景文字的检测和识别算法带来了挑战。

- **Complexity and Interference of Backgrounds**. Backgrounds of natural scenes are virtually unpredictable. There might be patterns extremely similar with text (e.g., tree leaves, traffic signs, bricks, windows, and stockades), or occlusions caused by foreign objects, which may potentially lead to confusions and mistakes.

- **背景的复杂性和干扰**。自然场景的背景实际上是不可预测的。可能存在与文本非常相似的模式，（如树叶、交通灯、砖块、窗户和栅栏），或不相干物体的遮挡，这可能带来混淆和误解。

- **Imperfect Imaging Conditions**. In uncontrolled circumstances, the quality of text images and videos could not be guaranteed. That is, in poor imaging conditions, text instances may be with low resolution and severe distortion due to inappropriate shooting distance or angle, or blurred because of out of focus or shaking, or noised on account of low light level, or corrupted by highlights or shadows.

- **有缺陷的成像条件**。在不受控的情况下，文字图像和视频的质量不能够得到保证。也就是说，在劣质的成像条件下，文字的分辨率可能很低，由于不合适的拍摄距离或角度，可能存在严重的失真，或由于没有聚焦或晃动导致模糊，或由于低光照水平导致含噪，或由于高光、阴影导致降质。

These difficulties run through the years before deep learning showed its potential in computer vision as well as in other fields. As deep learning came to prominence after AlexNet [68] won the ILSVRC2012 [115] contest, researchers turn to deep neural networks for automatic feature learning and start with more in-depth studies. The community are now working on ever more challenging targets. The progresses made in recent years can be summarized as follows:

这些困难存在了很多年，然后深度学习在计算机视觉以及其他领域展现出了其潜能。随着深度学习在AlexNet[68]赢得ILSVRC2012竞赛[115]之后崛起，研究者转向深度神经网络以自动进行特征学习，开始更加深入的研究。学术团体正在研究更加有挑战的目标。近年来的进展可以总结如下：

- **Incorporation of Deep Learning**. Nearly all recent methods are built upon deep learning models. Most importantly, deep learning frees researchers from the exhausting work of repeatedly designing and testing hand-crafted features, which gives rise to a blossom of works that push the envelope further. To be specific, the use of deep learning substantially simplifies the overall pipeline. Besides, these algorithms provide significant improvements over previous ones on standard benchmarks. Gradient-based training routines also facilitate to end-to-end trainable methods.

- **利用深度学习**。几乎近年来所有的方法都是基于深度学习模型构建的。最重要的是，深度学习将研究者从重复设计并测试手工制作特征的繁复工作中解放出来，这使得大量工作可以推进最前沿算法。具体来说，使用深度学习极大的简化了整体的流程。另外，这些算法极大了改进了之前在标准测试基准上的算法的性能。基于梯度的训练方法也促进了端到端的可训练方法。

- **Target-Oriented Algorithms and Datasets**. Researchers are now turning to more specific aspects and targets. Against difficulties in real-world scenarios, newly published datasets are collected with unique and representative characteristics. For example, there are datasets that feature long text, blurred text, and curved text respectively. Driven by these datasets, almost all algorithms published in recent years are designed to tackle specific challenges. For instance, some are proposed to detect oriented text, while others aim at blurred and unfocused scene images. These ideas are also combined to make more general-purpose methods.

- **面向目标的算法和数据集**。研究者现在正转向更具体的方面和目标。针对真实世界场景的困难，新发布的数据集收集的数据都有唯一的、有代表性的特征。比如，有一些数据集的特征分别是长文字、模糊文字和弯曲的文字。受到这些数据集推动，几乎所有近年来发表的算法都是设计用来处理特定的挑战。比如，一些是提出来检测有方向的文字，其他的目标则是模糊未对焦的场景图像。这些想法经常还会结合在一起来得到更一般性的方法。

- **Advances in Auxiliary Technologies**. Apart from new datasets and models devoted to the main task, auxiliary technologies that do not solve the task directly also find their places in this field, such as synthetic data and bootstrapping.

- **辅助技术的进展**。除了专注于主要任务的新数据集和模型，还有很多辅助技术，没有直接解决这些任务，但也有一席之地，比如合成数据和bootstrapping。

In this survey, we present an overview of recent development in scene text detection and recognition with focus on the deep learning era. We review methods from different perspectives, and list the up-to-date datasets. We also analyze the status quo and predict future research trends.

在这个调查中，我们给出了场景文字检测和识别在近年来的发展概述，聚焦在深度学习技术时代。我们回顾了各种不同的方法，并列出了最新的数据集。我们还分析了现状并预测了未来研究的趋势。

There have been already several excellent review papers [136], [154], [160], [172], which also comb and analyze works related text detection and recognition. However, these papers are published before deep learning came to prominence in this field. Therefore, they mainly focus on more traditional and feature-based methods. We refer readers to these paper as well for a more comprehensive view and knowledge of the history. This article will mainly concentrate on text information extraction from still images, rather than videos. For scene text detection and recognition in videos, please also refer to [60], [160].

已经有了几篇精彩的综述文章[136,154,160,172]，都梳理并分析了与文字检测和识别的相关工作。但是，这些文章都是在深度学习主导这个领域之前发表的。所以，它们主要聚焦在更传统的基于特征的方法。我们也向读者推荐这些文章，以更综合的理解研究的历史。本文主要聚焦在从图像中进行文本信息提取，而不是视频。对于视频中的场景文字检测与识别，请参考[60,160]。

The remaining parts of this paper are arranged as follows. In Section 2, we briefly review the methods before the deep learning era. In Section 3, we list and summarize algorithms based on deep learning in a hierarchical order. In Section 4, we take a look at the datasets and evaluation protocols. Finally, we present potential applications and our own opinions on the current status and future trends.

本文的剩下部分组织如下。在第2部分，我们简要回顾了深度学习时代之前的方法。在第3部分，我们以层次化的顺序列出并总结了基于深度学习的算法。在第4部分，我们研究了数据集和评估协议。最后，我们给出了潜在的应用和对目前状态以及未来趋势的看法。

## 2. Methods Before The Deep Learning Era 深度学习时代之前的方法

#### 2.1 Overview 概览

In this section, we take a brief glance retrospectively at algorithms before the deep learning era. More detailed and comprehensive coverage of these works can be found in [136], [154], [160], [172]. For text detection and recognition, the attention has been the design of features. In this period of time, most text detection methods either adopt Connected Components Analysis (CCA) [24], [52], [58], [98], [135], [156], [159] or Sliding Window (SW) based classification [17], [70], [142], [144]. CCA based methods first extract candidate components through a variety of ways (e.g., color clustering or extreme region extraction), and then filter out non-text components using manually designed rules or classifiers automatically trained on hand-crafted features (see Fig.2). In sliding window classification methods, windows of varying sizes slide over the input image, where each window is classified as text segments/regions or not. Those classified as positive are further grouped into text regions with morphological operations [70], Conditional Random Field (CRF) [142] and other alternative graph based methods [17], [144].

本节中，我们简要回顾一下深度学习时代之前的算法。这些工作更详细的描述可以见[136,154,160,172]。对于文字检测和识别，主要关注的是特征的设计。在这段时间里，大多数文字检测方法要么采用了连接部件分析法(Connected Components Analysis, CCA)[24,52,58,98,135,156,159]，要么是基于分类的滑窗法(Sliding Window, SW)[17,70,142,144]。基于CCA的方法首先通过多种方法提取候选部件（比如颜色聚类或极端区域提取），然后使用手工设计的规则或从训练得到的分类器（在手工设计的特征上训练的）自动的滤除掉非文字的部件（见图2）。在滑窗分类方法中，不同大小的窗口滑过输入图像，其中每个窗口都分类成文本片段/区域或不是文本。分类为正的进一步使用形态学[70]、条件随机场(Conditional Random Field, CRF)[142]和其他基于图的方法分组成文本区域。

For text recognition, one branch adopted the feature-based methods. Shi et al. [126] and Yao et al. [153] proposed character segments based recognition algorithms. Rodriguez et al. [109], [110] and Gordo et al. [35] and Almazan et al. [3] utilized label embedding to directly perform matching between strings and images. Stoke [10] and character key-points [104] are also detected as features for classification. Another discomposed the recognition process into a series of sub-problems. Various methods have been proposed to tackle these sub-problems, which includes text binarization [71], [93], [139], [167], text line segmentation [155], character segmentation [101], [114], [127], single character recognition [12], [120] and word correction [62], [94], [138], [145], [165].

对于文字识别，一个分支采用了基于特征的方法。Shi等[126]和Yao等[153]提出基于特征字符片段的算法。Rodriguez等[109,110]和Gordo等[35]和Almazan等[3]利用了标签嵌套来直接匹配字符串和图像。Stoke[10]和字符关键点[104]也检测为特征以用于分类。将识别过程也可以分解为一系列子问题，提出了各种方法以处理这些子问题，包括字符二值化[71,93,139,167]，文本线片段[155]，字符片段[101,114,127]，单字符识别[12,120]和文字更正[62,94,138,145,165]。

There have been efforts devoted to integrated (i.e. end-to-end as we call it today) systems as well [97], [142]. In Wang et al. [142], characters are considered as a special case in object detection and detected by a nearest neighbor classifier trained on HOG features [19] and then grouped into words through a Pictorial Structure (PS) based model [26]. Neumann and Matas [97] proposed a decision delay approach by keeping multiple segmentations of each character until the last stage when the context of each character is known. They detected character segmentations using extremal regions and decoded recognition results through a dynamic programming algorithm.

也有一些工作致力于整合系统（即，现在称的端到端系统）[97,142]。在Wang等[142]，字符被看作是通用目标识别的一种特例，由一个在HOG特征[19]上训练的最近邻分类器检测，然后用一个基于Pictorial Structure(PS)的模型分组成文字。Neumann和Matas[97]提出一种决策延迟方法，首先保持每个字符的多个片段，直到最后阶段，每个字符的上下文都成为已知。他们使用极端区域检测字符片段，并将识别结果用一种动态规划算法进行解码。

In summary, text detection and recognition methods before the deep learning era mainly extract low-level or mid-level hand crafted image features, which entails demanding and repetitive pre-processing and post-processing steps. Constrained by the limited representation ability of hand crafted features and the complexity of pipelines, those methods can hardly handle intricate circumstances, e.g. blurred images in the ICDAR2015 dataset [63].

总结一下，深度学习时代之前的文本检测与识别方法，主要提取底层或中层手工设计的图像特征，这需要很多重复的预处理和后处理步骤。受到手工设计特征有限的表示能力和流程的复杂性限制，这些方法很难处理错综复杂的情况，如ICDAR2015数据集中的模糊图像[63]。

Fig. 2: Illustration of traditional methods with hand-crafted features: (1) Maximally Stable Extremal Regions (MSER) [98], assuming chromatic consistency within each character; (2) Stroke Width Transform (SWT) [24], assuming consistent stroke width within each character.

图2：手工设计特征的传统方法的描述：(1)最大稳定极端区域(MSER)[98]，假设每个字符中存在色彩连续性；(2)笔画宽度变换(SWT)[24]，假设每个字符中的笔画宽度连续。

## 3. Methodology in the Deep Learning Era 深度学习时代的方法

As implied by the title of this section, we would like to address recent advances as changes in methodology instead of merely new methods. Our conclusion is grounded in the observations as explained in the following paragraph.

就像本节标题所说的，我们要描述的是近年来的进展，不仅仅只是新的方法，而且是方法上的变化。我们的结论以观察为基础，在下面的小节中解释。

Methods in the recent years are characterized by the following two distinctions: (1) Most methods utilizes deep-learning based models; (2) Most researchers are approaching the problem from a diversity of perspectives. Methods driven by deep-learning enjoy the advantage that automatic feature learning can save us from designing and testing the large amount potential hand-crafted features. At the same time, researchers from different viewpoints are enriching and promoting the community into more in-depth work, aiming at different targets, e.g. faster and simpler pipeline [168], text of varying aspect ratios [121], and synthetic data [38]. As we can also see further in this section, the incorporation of deep learning has totally changed the way researchers approach the task, and has enlarged the scope of research by far. This is the most significant change compared to the former epoch.

近年来的方法特征在于下面的两个差异：(1)多数方法使用了基于深度学习的模型；(2)多数研究者从各种不同的角度来看待这个问题。以深度学习为驱动的方法，有自动学习特征的优势，使得我们不用设计并测试大量潜在的手工设计的特征。同时，不同视角的研究者使得学术团体更加深入的研究，对准不同的目标，如更快和更简单的流程[168]，不同纵横比的文字[121]，合成数据[38]。我们在本节中还可以看到，深度学习的使用完全的改变了研究者处理这个任务的方式，迄今扩大了研究的视野。这是与之前的时代相比最显著的改变。

In a nutshell, recent years have witness a blossoming expansion of research into subdivisible trends. We summarize these changes and trends in Fig.3, and we would follow this diagram in our survey.

简言之，近年来目睹了研究的爆发。我们在图3中总结了这些变化和趋势，我们在这个调查中详细描述这个图表。

In this section, we would classify existing methods into a hierarchical taxonomy, and introduce in a top-down style. First, we divide them into four kinds of systems: (1) text detection that detects and localizes the existence of text in natural image; (2) recognition system that transcribes and converts the content of the detected text region into linguistic symbols; (3) end-to-end system that performs both text detection and recognition in one single pipeline; (4) auxiliary methods that aim to support the main task of text detection and recognition, e.g. synthetic data generation, and deblurring of image. Under each system, we review recent methods from different perspectives.

在本节中，我们将现有的方法进行层次性的分类，引入了一种自上而下的分类形式。首先，我们将其分类为四种系统：(1)文本检测系统，检测并定位自然图像中文本的存在；(2)识别系统，将检测到的文本区域转变成语言符号；(3)端到端的系统，可以同时进行检测和识别；(4)辅助方法，目标是支持文本检测和识别的主要任务，如合成数据生成，图像去模糊。在每个系统下，我们从不同的角度回顾近年来的方法。

Fig. 3: Overview of recent progress and dominant trends

### 3.1 Detection 检测

There are three main trends in the field of text detection, and we would introduce them in the following sub-sections one by one. They are: (1) pipeline simplification; (2) changes in prediction units; (3) specified targets.

在文字检测领域有三种主要的趋势，我们将在下面的小节中一个一个进行介绍。分别是：(1)流程简化；(2)预测单元的变化；(3)指定目标。

#### 3.1.1 Pipeline Simplification 流程简化

One of the important trends is the simplification of the pipeline, as shown in Fig.4. Most methods before the era of deep-learning, and some early methods that use deep-learning, have multi-step pipelines. More recent methods have simplified and much shorter pipelines, which is a key to reduce error propagation and simplify the training process. More recently, separately trained two-staged methods are surpassed by jointly trained ones. The main components of these methods are end-to-end differentiable modules, which is an outstanding characteristic.

一个重要的趋势是流程的简化，如图4所示。深度学习时代之前的多数方法，以及使用深度学习较为前期的一些方法，其处理流程通常是多阶段的。近年来的方法的流程得到的简化，非常简短，这对于降低错误传播非常关键，并简化了训练过程。再近一些的时候，两阶段独立训练的方法被同时训练的方法超过。这些方法的主要部件是端到端的可微分模块，这是一个显著的特征。

Fig. 4: Typical pipelines of scene text detection and recognition. (a) [55] and (b) [152] are representative multi-step methods. (c) and (d) are simplified pipeline. (c) [168] only contains detection branch, and therefore is used together with a separate recognition model. (d) [45], [81] jointly train a detection model and recognition model.

图4. 场景文字检测和识别的典型流程。(a)[55]和(b)[152]是多阶段方法的代表。(c)和(d)为简化的流程。(c)[168]只包含检测分支，因此与一个单独的识别模型一起使用。(d)[45,81]同时训练一个检测模型和识别模型。

**Multi-step methods**: Early deep-learning based methods [152], [166], [41] cast the task of text detection into a multi-step process. In [152], a convolutional neural network is used to predict whether each pixel in the input image (1) belongs to a character, (2) is inside the text region, and (3) the text orientation around the pixel. Connected positive responses are considered as a detection of character or text region. For characters belonging to the same text region, Delaunay triangulation [61] is applied, after which graph partition based on the predicted orientation attribute groups characters into text lines.

**多阶段方法**：早期基于深度学习的方法[152,166,41]将文字检测作为多阶段过程。在[152]中，用卷积神经网络来预测输入图像中的每个像素是(1)属于一个字符，(2)在文字区域内，以及(3)像素周围文字的方向。连通的正响应被认为是检测到了字符或文字区域。对于属于同样文字区域的字符来说，应用Delaunay triangulation[61]算法，在这之后，基于预测的方向属性的图分割算法将字符分组成文字行。

Similarly, [166] first predicts a dense map indicating which pixels are within text line regions. For each text line region, MSER [99] is applied to extract character candidates. Character candidates reveal information of the scale and orientation of the underlying text line. Finally, minimum bounding box is extracted as the final text line candidate.

类似的，[166]首先预测一个密集图，指示了哪些像素是在文字行区域中。对于每个文字行区域，应用MSER[99]来提取候选字符。候选字符会显示潜在的文字行的尺度和方向。最后，提取出最小边界框作为最终的文字行候选。

In [41], the detection process also consists of several steps. First, text blocks are extracted. Then the model crops and only focuses on the extracted text block to extract text center line(TCL), which is defined to be a shrunk version of the original text line. Each text line represents the existence of one text instance. The extracted TCL map is then split into several TCLs. Each split TCL is then concatenated to the original image. A semantic segmentation model then classifies each pixel into ones that belong to the same text instance as the given TCL, and ones that do not.

在[41]中，检测过程也包括了几个步骤。首先，提取出文字块。然后模型剪切出提取的文字块，并只聚焦在这里，来提取出文字中心线(TCL)，其定义是原始文字行的缩小版。每个文字行代表一个文字实例的存在。提取出的TCL图然后切分成几个TCLs。每个TCL切片都与原始图像拼接起来。一个语义分割模型然后会将每个像素分类成属于与给定的TCL同一个文字实例的，或者不是。

**Simplified pipeline**: More recent methods [44], [59], [73], [82], [121], [163], [90], [111], [74], [119] follow a 2-step pipeline, consisting of an end-to-end trainable neural network model and a post-processing step that is usually much simpler than previous ones. These methods mainly draw inspiration from techniques in general object detection [27], [30], [31], [42], [76], [107], and benefit from the highly integrated neural network modules that can predict text instances directly. There are mainly two branches: (1) Anchor-based methods [44], [73], [82], [121] that predict the existence of text and regress the location offset only at pre-defined grid points of the input image; (2) Region proposal methods [59], [74], [90], [111], [119], [163] that predict and regress on the basis of extracted image region.

**简化的流程**：更近年来的方法[44,59,73,82,121,163,90,111,74,119]都采用了两步骤流程，包括一个端到端的可训练的神经网络模型，以及一个后处理步骤，通常都比前面要简单很多。这些方法主要从通用目标识别[27,30,31,42,76,107]中汲取灵感，高度集成的神经网络模块可以直接预测文字实例，这可以从中受益非常多。主要有两个分支：(1)基于锚窗的方法[44,73,82,121]，预测文字的存在，并在输入图像中预定义的网格点上对位置偏移进行回归；(2)区域候选法[59,74,90,111,119,163]在提取出的图像区域的基础上进行预测和回归。

Since the original targets of most of these works are not merely the simplification of pipeline, we only introduce some representative methods here. Other works will be introduced in the following parts.

由于大多数这些工作的原始目标都不仅仅是流程的简化，我们这里只介绍一些有代表性的方法。其他工作在随后的部分进行介绍。

**Anchor-based methods** draw inspiration from SSD [76], a general object detection network. As shown in Fig.5 (b), a representative work, TextBoxes [73], adapts SSD network specially to fit the varying orientations and aspect-ratios of text line. Specifically, at each anchor point, default boxes are replaced by default quadrilaterals, which can capture the text line tighter and reduce noise.

**基于锚框的方法**从SSD[76]中得到灵感，SSD是一个通用目标检测网络。如图5(b)所示，一个代表性的工作，TextBoxes[73]，修改了SSD网络以适应文字行变化的方向和纵横比。特别的，在每个锚点，默认框由默认四边形替换，这可以更紧凑的捕捉到文字行并降低噪声。

A variant of the standard anchor-based default box prediction method is EAST [168]. In the standard SSD network, there are several feature maps of different sizes, on which default boxes of different receptive fields are detected. In EAST, all feature maps are integrated together by gradual upsampling, or U-Net [113] structure to be specific. The size of the final feature map is 1/4 of the original input image, with c-channels. Under the assumption that each pixel only belongs to one text line, each pixel on the final feature map, i.e. the 1 × 1 × c feature tensor, is used to regress the rectangular or quadrilateral bounding box of the underlying text line. Specifically, the existence of text, i.e. text/non-text, and geometries, e.g. orientation and size for rectangles, and vertexes coordinates for quadrilaterals, are predicted. EAST makes a difference to the field of text detection with its highly simplified pipeline and the efficiency. Since EAST is most famous for its speed, we would re-introduce EAST in later parts, with emphasis on its efficiency.

标准的基于锚框的默认框的预测方法的一个变体为EAST[168]。在标准SSD网络中，有数个不同大小的特征图，在上面可以检测不同感受野大小的默认框。在EAST中，所有的特征图通过逐渐的上采样整合到一起，具体来说，是形成U-Net[113]的结构。最终特征图的大小是原始输入图像的1/4，有c个通道。假设每个像素只会属于一个文字行，最终特征图的每个像素，即1×1×c特征张量，可以用于回归得到潜在的文字行的的矩形或四边形边界框。具体来说，预测的是文字的存在，即文字/非文字，几何情况，即矩形的方向和大小，和四边形的顶点坐标。EAST检测流程及其简单，效率非常高。由于EAST以其速度著名，我们会在后面的部分重新介绍EAST，强调其运行效率。

**Region proposal methods** usually follow the standard object detection framework of R-CNN [30], [31], [107], where a simple and fast pre-processing method is applied, extracting a set of region proposals that could contain text lines. A neural network then classifies it as text/non-text and corrects the localization by regressing the boundary offsets. However, adaptations are necessary.

**候选区域方法**通常使用的是标准目标识别R-CNN的框架[30,31,107]，使用一个简单快速的预处理方法来提取一系列候选区域，其中可能包含文字行。一个神经网络将其分类成文字/非文字的情况，并通过回归边界偏移来修正位置。但是，需要对网络进行修改。

Rotation Region Proposal Networks [90] follows and adapts the standard Faster RCNN framework. To fit into text of arbitrary orientations, rotating region proposals are generated instead of the standard axis-aligned rectangles.

旋转候选区域网络(Rotation Region Proposal Networks)[90]修改了标准的Faster RCNN框架。为适应任意方向的文字，生成旋转候选区域，以替代标准的与坐标轴对齐的矩形。

Similarly, R2CNN [59] modifies the standard region proposal based object detection methods. To adapt to the varying aspects ratios, three Region of Interests Poolings of different sizes are used, and concatenated for further prediction and regression. In FEN [119], adaptively weighted poolings are applied to integrated different pooling sizes. The final prediction is made by leveraging the textness score for poolings of 4 different sizes.

类似的，R2CNN[59]修改了标准的基于区域候选的目标检测方法。为适应变化的纵横比，使用了三个不同大小的RoI Poolings并拼接在一起，以进行进一步的预测和回归。在FEN[119]中，使用了自适应权值的pooling以整合不同的pooling大小。衡量4种不同大小的pooling的textness分数，以得到最终的预测。

Fig. 5: High level illustration of existing anchor/roi-pooling based methods: (a) Similar to YOLO [105], predicting at each anchor positions. Representative methods include rotating default boxes [82]. (b) Variants of SSD [76], including Textboxes [73], predicting at feature maps of different sizes. (c) Direct regression of bounding boxes [168], also predicting at each anchor position. (d) Region Proposal based methods, including rotating Region of Interests (RoI) [90] and RoI of varying aspect ratios [59].

#### 3.1.2 Different Prediction Units 不同的预测单元

A main distinction between text detection and general object detection is that, text are homogeneous as a whole and show locality, while general object detection are not. By homogeneity and locality, we refer to the property that any part of a text instance is still text. Human do not have to see the whole text instance to know it belongs to some text.

文字检测和通用目标检测的主要区别在于，文字整体上是由同类型的东西组成的，并显示出局部性，而一般性的目标检测则不是。同质性和局部性的意思是，文字实例的一部分仍然是文字。人们不需要看到文字实例的整体来确认这是属于文字类别的。

Such a property lays a cornerstone for a new branch of text detection methods that only predict sub-text components and then assemble them into a text instance.

这样一个性质是一类新的文字检测方法分支的基础，这些算法只检测亚文字组件，然后将其组合成一个文字实例。

In this part, we take the perspective of the granularity of text detection. There are two main level of prediction granularity, text instance level and sub-text level.

在这部分中，我们以文字检测的粒度为视角。有两种主要的预测粒度水平，文字实例级和亚文字级。

In **text instance level** methods [18], [46], [59], [73], [74], [82], [90], [119], [163], [168], detection of text follows the standard routine of general object detection, where a region-proposal network and a refinement network are combined to make predictions. The region-proposal network produces initial and coarse guess for the localization of possible text instance, and then a refinement part discriminates the proposals as text/non-text and also correct the localization of the text.

在**文字实例级**的方法中[18,46,59,73,74,82,90,119,163,168]，文字的检测遵循通用目标检测的标准流程，即一个区域候选网络和一个提炼网络结合在一起进行预测。区域候选网络生成可能的文本实例位置的初始的粗糙猜测，然后提炼部分将候选区分成文字/非文字，同时也纠正文本的位置。

Contrarily, **sub-text level** detection methods [89], [20], [41], [148], [152], [44], [40], [121], [166], [133], [140], [171] only predicts parts that are combined to make a text instance. Such sub-text mainly includes pixel-level and components-level.

相反的，**亚文字级**的检测方法[89,20,41,148,152,44,40,121,166,133,140,171]只预测部件，然后组合形成文字实例。这种亚文字级检测主要包括像素级检测和component-level检测。

In **pixel-level** methods [20], [41], [44], [148], [152], [166], an end-to-end fully convolutional neural network learns to generate a dense prediction map indicating whether each pixel in the original image belongs to any text instances or not. Post-processing methods then groups pixels together depending on which pixels belong to the same text instance. Since text can appear in clusters which makes predicted pixels connected to each other, the core of pixel-level methods is to separate text instances from each other. PixelLink [20] learns to predict whether two adjacent pixels belong to the same text instance by adding link prediction to each pixel. Border learning method [148] casts each pixels into three categories: text, border, and background, assuming that border can well separate text instances. In Holistic [152], pixel-prediction maps include both text-block level and character center levels. Since the centers of characters do not overlap, the separation is done easily.

在**像素级**的方法[20,41,44,148,152,166]中，端到端的全卷积神经网络学习生成密集预测图，表明原始图像每个像素是否属于文字实例。后处理方法依据哪些像素属于同一个文字实例，将这些像素分组。由于文字可以成群出现，这使得预测的像素互相连通，像素级的方法的核心就是将文字实例区分开来。PixelLink[20]对每个像素增加连接预测，来学习预测两个相邻的像素是否属于相同的文字实例。边界学习方法[148]将所有像素分为三类：文字，边界和背景，假设边界可以很好的分离文字实例。在Holistic[152]，预测像素的图包括文字块级和字符中心水平。因为字符的中心不会重叠，所以可以很容易的分离。

In this part we only intend to introduce the concept of prediction units. We would go back to details regarding the separation of text instances in the section of Specific Targets.

在本部分中，我们只介绍预测单元的概念，我们会在“特定目标”部分详细介绍文字实例分割的情况。

**Components-level** methods [40], [89], [121], [133], [140], [171] usually predicts at a medium granularity. Component refer to a local region of text instance, sometimes containing one or more characters.

**Components-level**方法[49,89,121,133,140,171]通常预测的是中等粒度的对象。Component指的是文字实例的局部区域，有时候包含一个或多个字符。

As shown in Fig.6 (a), SegLink [121] modified the original framework of SSD [76]. Instead of default boxes that represent whole objects, default boxes used in SegLink have only one aspect ratio and predict whether the covered region belongs to any text instances or not. The region is called text segment. Besides, links between default boxes are predicted, indicating whether the linked segments belong to the same text instance.

如图6(a)所示，SegLink[121]修改了SSD[76]的原始框架。原始框架中默认框是代表整个目标，在SegLink中使用的默认框只有一种纵横比，预测覆盖的区域是否属于任何文字实例。区域称为文字片段。另外，预测默认框之间的连接，表明连接的片段是否属于同一文字实例。

Corner localization methods [89] proposes to detect the corners of each text instance. Since each text instance only has 4 corners, the prediction results and their relative position can indicate which corners should be grouped into the same text instance.

角点定位方法[89]提出检测每个文字实例的的角点。由于每个文字实例只有4个角点，预测结果及其相对位置可以表明哪些角点可以成为相同文字实例的组中。

SegLink [121] and Corner localization [89] are proposed specially for long and multi-oriented text. We only introduce the idea here and discuss more details in the section of Specific Targets, regarding how they are realized.

SegLink[121]和Corner localization[89]的提出是专门为了长文字和多角度的文字。我们在这里介绍其思想，在“特定目标”一节中更详细的讨论是如何实现的。

In a clustering based method [140], pixels are clustered according to their color consistency and edge information. The fused image segments are called superpixel. These superpixels are further used to extract characters and predict text instance.

在基于聚类的方法[140]中，像素聚类的依据是其颜色连续性及其边缘信息。合并的图像片段称为超像素(superpixel)。这些超像素用来进一步提取特征并预测实例。

Another branch of component-level method is Connectionist Text Proposal Network (CTPN) [133], [147], [171]. CTPN models inherit the idea of anchoring and recurrent neural network for sequence labeling. They usually consist of a CNN-based image classification network, e.g. VGG, and stack an RNN on top of it. Each position in the final feature map represents features in the region specified by the corresponding anchor. Assuming that text appear horizontally, each row of features are fed into an RNN and labeled as text/non-text. Geometries are also predicted.

Component-level方法的另一个分支是Connectionist Text Proposal Network(CTPN)[133,147,171]。CTPN模型继承了锚的思想，和用于序列标注的RNN的思想。他们通常包括一个基于CNN的图像分类网络，如VGG，并在其上堆叠一个RNN。最终特征图的每个位置都代表了对应的锚在区域中的特征。假设文字在水平方向出现，每一行特征都输入RNN并标注为文字/非文字。同时还预测几何形状。

Fig. 6: Illustration of representative bottom-up methods: (a) SegLink [121]: with SSD as base network, predict word segments at each anchor position, and connections between adjacent anchors. (b) PixelLink [20]: predict for each pixel, text/non-text classification and whether it belongs to the same text as adjacent pixels or not/ (c) Corner Localization [89]: predict the four corners of each text and group those belonging to the same text instances. (d) TextSnake [85]: predict text/non-text and local geometries, which are used to reconstruct text instance.

#### 3.1.3 Specific Targets 特定目标

Another characteristic of current text detection system is that, most of them are designed for special purposes, attempting to approach unique difficulties in detecting scene text. We broadly classify them into the following aspects.

目前的文字检测系统的另一个特征是，多数是设计用于特别目的的，尝试解决检测场景文字的独特困难。我们大致将其归类于下述的几个方面。

**3.1.3.1 Long Text**: Unlike general object detection, text usually come in varying aspect ratios. They have much larger height-width ratio, and thus general object detection framework would fail. Several methods have been proposed [59], [89], [121], specially designed to detect long text. R2CNN [59] gives an intuitive solution, where ROI pooling with different sizes are used. Following the framework of Faster R-CNN [107], three ROI-poolings with varying pooling sizes, 7 × 7 , 3 × 11 , and 11 × 3, are performed for each box generated by region-proposal network, and the pooled features are concatenated for textness score.

**3.1.3.1 长文本**：不像通用目标识别，文字的纵横比通常变化不一。其高度宽度比非常大，所以通用目标检测框架通常会检测不成功。对于长文本检测，特别提出设计了几种方法[59,89,121]。R2CNN[59]给出了一种直觉性的解决方案，其中使用了不同大小的ROI pooling。遵循Faster R-CNN[107]的框架，使用了三种大小的ROI pooling，7 × 7, 3 × 11和11 × 3，来处理区域候选网络生成的每个框，pooled得到的特征拼接起来得到textness分数。

Another branch learns to detect local sub-text components which are independent from the whole text [20], [89], [121]. SegLink [121] proposes to detect components, i.e. square areas that are text, and how these components are linked to each other. PixelLink [20] predicts which pixels belong to any text and whether adjacent pixels belong to the same text instances. Corner localization [89] detects text corners. All these methods learn to detect local components and then group them together to make final detections.

另一个分支学习检测局部亚文本部件，这是独立于整体本文的[20,89,121]。SegLink[121]提出来检测组件，即是文本的方形区域，以及这些组件是怎样相互连接的。PixelLink[20]预测哪些像素是属于任何文本的，以及邻近像素是否属于同样的文本实例。Corner localization[89]检测文本角点。所有这些方法学习检测局部组件，然后将组件分组，形成最终预测。

**3.1.3.2 Multi-Oriented Text**: Another distinction from general text detection is that text detection is rotation-sensitive and skewed text are common in real-world, while using traditional axis-aligned prediction boxes would incorporate noisy background that would affect the performance of the following text recognition module. Several methods have been proposed to adapt to it [59], [73], [74], [82], [90], [121], [168], [141].

**3.1.3.2 多方向文本**：另一个与通用文本检测的区别是，文本检测是对旋转很敏感的，歪斜的文本在真实世界非常常见，使用传统的与轴对齐的预测框会使用含噪的背景，这会影响后续的文本识别模块的性能。为应对这种问题，提出了几种方法[59,73,74,82,90,121,168,141]。

Extending from general anchor-based methods, rotating default boxes [73], [82] are used, with predicted rotation offset. Similarly, rotating region proposals [90] are generated with 6 different orientations. Regression-based methods [59], [121], [168] predict the rotation and positions of vertexes, which are insensitive to orientation. Further, in Liao et al. [74], rotating filters [169] are incorporated to model orientation-invariance explicitly. The peripheral weights of 3 × 3 filters rotate around the center weight, to capture features that are sensitive to rotation.

从通用的基于锚框的方法出发，[73,82]使用了旋转的默认框，以及预测的旋转偏移。类似的，[90]生成了6个不同方向的旋转候选区域。基于回归的方法[59,121,168]预测了顶点的旋转和位置，这些都是对方向不敏感的。进一步的，Liao等人[74]使用了旋转滤波器[169]来对方向不变性进行显式建模。3×3滤波器的外围权重围绕着中心权重进行旋转，以捕捉对旋转敏感的特征。

While the aforementioned methods may entail additional post-processing, Wang et al. [141] proposes to use a parametrized Instance Transformation Network (ITN) that learns to predict appropriate affine transformation to perform on the last feature layer extracted by the base network, to rectify oriented text instances. Their method, with ITN, can be trained end-to-end.

前面提到的方法会涉及到额外的后处理，Wang等[141]提出使用参数化的实例变换网络(Instance Transformation Network, ITN)学习预测适当的仿射变换在基础网络上最后的特征层上执行，以校正有方向的文字实例。他们的带有ITN的方法可以进行端到端的训练。

**3.1.3.3 Text of Irregular Shapes**: Apart from varying aspect ratios, another distinction is that text can have a diversity of shapes, e.g. curved text. Curved text poses a new challenge, since regular rectangular bounding box would incorporate a large proportion of background and even other text instances, making it difficult for recognition.

**3.1.3.3 不规则形状的文本**：除了多种纵横比，另一种区别，文字可以有很多不同的形状，如弯曲的文字。弯曲的文字提出了一种新的挑战，因为规则的矩形边界框会包括很大比例的背景，甚至其他文字实例，使其更难识别。

Extending from quadrilateral bounding box, it’s natural to use bounding ’boxes’ with more that 4 vertexes. Bounding polygons [163] with as many as 14 vertexes are proposed, followed by a bi-lstm [48] layer to refine the coordinates of the predicted vertexes. In their framework, however, axis-aligned rectangles are extracted as intermediate results in the first step, and the location bounding polygons are predicted upon them.

从四边形边界框扩展，很自然的，可以使用多于4个顶点的边界“框”。[163]提出了最多14个顶点的边界多边形，随后跟着一个bi-lstm[48]层来提炼预测的顶点的坐标。但在他们的框架中，第一步会提取出与坐标轴对齐的矩形作为中间结果，在这个基础上再预测边界多边形的位置。

Similarly, Lyu et al. [88] modifies the Mask R-CNN [42] framework, so that for each region of interest—in the form of axis-aligned rectangles—character masks are predicted solely for each type of alphabets. These predicted characters are then aligned together to form a polygon as the detection results. Notably, they propose their method as an end-to-end system. We would refer to it again in the following part.

类似的，Lyu等[88]修正了Mask R-CNN[42]框架，这样对于每个感兴趣区域(RoI，形式是与坐标轴对齐的矩形)，对每种类型的字符都预测了字符掩膜。这些预测的字符然后一起对齐，形成一个多边形作为检测结果。要注意的是，他们提出的系统的形式是端到端的。我们会在接下来的小节中再引用其工作。

Viewing the problem from a different perspective, Long et al. [85] argues that text can be represented as a series of sliding round disks along the text center line (TCL), which accord with the running direction of the text instance, as shown in Fig.7. With the novel representation, they present a new model, TextSnake, as shown in Fig.6 (d), that learns to predict local attributes, including TCL/non-TCL, text-region/non-text-region, radius, and orientation. The intersection of TCL pixels and text region pixels gives the final prediction of pixel-level TCL. Local geometries are then used to extract the TCL in the form of ordered point list, as demonstrated in Fig.6 (d). With TCL and radius, the text line is reconstructed. It achieves state-of-the-art performance on several curved text dataset as well as more widely used ones, e.g. ICDAR2015 [63] and MSRA-TD500 [135]. Notably, Long et al. proposes a cross-validation test across different datasets, where models are only fine-tuned on datasets with straight text instances, and tested on the curved datasets. In all existing curved datasets, TextSnake achieves improvements by up to 20% over other baselines in F1-Score.

从另一个不同的角度看这个问题，Long等[85]提出文字可以表示为沿着文字中心线(TCL)的一系列滑动圆盘，TCL也就是沿着文字实例的运行方向，如图7所示。他们使用这种新颖的表示提出了一种新模型，TextSnakes，可以学习预测局部属性，包括TCL/non-TCL，文本区域/非文本区域，半径和方向，如图6(d)所示。TCL像素与文字区域像素的交集给出像素级TCL的最终预测。然后使用局部几何来提取TCL，形成有序点列表的形式，如图6(d)所示。使用TCL和半径就可以重建文字线。这在几个弯曲文字数据集上都取得了目前最好的成绩，也包括使用非常广泛的一些，如ICDAR2015[63]和MSRA-TD500[135]。值得注意的是，Long等提出了一种不同数据集之间的交叉验证测试，其中模型只要在数据集上的直接文字实例上进行精调，并在弯曲数据集上进行测试。在所有现有的弯曲数据集中，TextSnake比其他方法在F1-Score上都取得了20%的改进。

Fig. 7: (a)-(c): Representing text as horizontal rectangles, oriented rectangles, and quadrilaterals. (d): The sliding-disk reprensentation proposed in TextSnake [85].

**3.1.3.4 Speedup**: Current text detection methods place more emphasis on speed and efficiency, which is necessary for application in mobile devices. 目前的文本检测方法更侧重于速度和效率，这对于在移动设备上的应用来说是必须的。

The first work to gain significant speedup is EAST [168], which makes several modifications to previous framework. Instead of VGG [129], EAST uses PVANet [67] as its basenetwork, which strikes a good balance between efficiency and accuracy in the ImageNet competition. Besides, it simplifies the whole pipeline into a prediction network and a non-maximum suppression step. The prediction network is a U-shaped [113] fully convolutional network that maps an input image $I ∈ R^{H,W,C}$ to a feature map $F ∈ R^{H/4,W/4,K}$, where each position $f = F_{i,j,:} ∈ R^{1,1,K}$ is the feature vector that describes the predicted text instance. That is, the location of the vertexes or edges, the orientation, and the offsets of the center, for the text instance corresponding to that feature position (i, j). Feature vectors that corresponds to the same text instance are merged with the non-maximum suppression. It achieves state-of-the-art speed with FPS of 16.8 as well as leading performance on most datasets.

第一个得到显著提速的工作是EAST[168]，其对之前的框架作出了几个改进。EAST没有使用VGG[129]，而使用了PVANet[67]作为其基础网络，PVANet在ImageNet竞赛中得到了很好的效率和准确率的折中。另外，还将整个流程简化为一个预测网络和一个非最大抑制的步骤。预测网络是U形[113]全卷积网络，将输入图像$I ∈ R^{H,W,C}$映射到特征图$F ∈ R^{H/4,W/4,K}$，特征图中的每个位置$f = F_{i,j,:} ∈ R^{1,1,K}$都是特征向量，描述的是预测的文字实例。即，在特征位置(i,j)上对应的文字实例的方向，中间点的偏移，顶点或边的位置。同一文字实例对应的特征向量通过非最大抑制(NMS)合并到一起。EAST在多数数据集上取得了目前最快的速度16.8FPS，性能也是领先的。

**3.1.3.5 Easy Instance Segmentation**: As mentioned above, recent years have witnessed methods with dense predictions, i.e. pixel level predictions [20], [41], [103], [148]. These methods generate a prediction map classifying each pixel as text or non-text. However, as text may come near each other, pixels of different text instances may be adjacent in the prediction map. Therefore, separating pixels become important.

**3.1.3.5 简单的实例分割**：如前所述，近年来看到了很多密集预测的方法，即，像素级预测的方法[20,41,103,148]。这些方法生成了一个预测图，将每个像素归类于文字或非文字。但是，由于文字之间可能非常接近，不同文字实例的像素在预测图中也可能邻接在一起。所以，分离像素变得非常重要。

Pixel-level text center line is proposed [41], since the center lines are far from each other. In [41], a prediction map indicating text lines is predicted. These text lines can be easily separated as they are not adjacent. To produce prediction for text instance, a binary map of text center line of a text instance is attached to the original input image and fed into a classification network. A saliency mask is generated indicating the detected text. However, this method involves several steps. The text-line generation step and the final prediction step can not be trained end-to-end, and error propagates.

像素级的文字中心线于是提了出来[141]，因为中心线彼此间比较远。在[41]中，预测了指代文字线的预测图。这些文字线可以很容易的区分开来，因为他们不是邻接的。为生成文字实例的预测，文字实例的文字中心线的二值图与原始图像一起送入分类网络。生成一个显著性掩膜，指代检测到的文字。但是，这种方法需要多个步骤。生成文字线的步骤和最后的预测步骤不能进行端到端的训练，所以误差容易传播。

Another way to separate different text instances is to use the concept of border learning [103], [148], [149], where each pixel is classified into one of the three classes: text, non-text, and text border. The text border then separates text pixels that belong to different instances. Similarly, in the work of Xue et al. [149], text are considered to be enclosed by 4 segments, i.e. a pair of long-side borders (abdomen and back) and a pair of short-side borders (head and tail). The method of Xue et al. is also the first to use DenseNet [51] as their basenet, which provides a consistant 2 − 4% performance boost in F1-score over that with ResNet [43] on all datasets that it’s evaluated on.

另一种分离不同的文字实例的是使用边界学习的概念[103,148,149]，其中所有像素分为三类：文字，非文字和文字边界。文字边界用于分离属于不同实例的文字像素。类似的是Xue等的工作[149]，认为文字被4条线段所包围，即，长边的一对（上下）和短边的一对（头尾）。Xue等的方法也是第一个使用DenseNet[51]作为基础网络的方法，这比使用ResNet[43]的工作在所有数据集上进行评估时，都有2-4%的性能提升。

Following the linking idea of SegLink, PixelLink [20] learns to link pixels belonging to the same text instance. Text pixels are classified into groups for different instances efficiently via disjoint set algorithm. Treating the task in the same way, Liu et al. [84] proposes a method for predicting the composition of adjacent pixels with Markov Clustering [137], instead of neural networks. The Markov Clustering algorithm is applied to the saliency map of the input image, which is generated by neural networks and indicates whether each pixel belongs to any text instances or not. Then, the clustering results give the segmented text instances.

PixelLink[20]学习了SegLink连接的思想，学习连接属于同一文字实例的像素。文字像素通过disjoint set算法高效的分成不同实例的组。Liu等[84]也是同样进行这样的任务，提出了一种方法使用Markov聚类预测邻接像素的组成，而没有使用神经网络。Markov聚类算法处理的是输入图像的显著性图，这是由神经网络生成的，指代的是每个像素是否属于哪个文字实例。然后，聚类结果给出了分离过的文字实例。

**3.1.3.6 Retrieving Designated Text**: Different from the classical setting of scene text detection, sometimes we want to retrieve a certain text instance given the description. Rong et al. [112] a multi-encoder framework to retrieve text as designated. Specifically, text is retrieved as required by a natural language query. The multi-encoder framework includes a Dense Text Localization Network (DTLN) and a Context Reasoning Text Retrieval (CRTR). DTLN uses an LSTM to decode the features in a FCN network into a sequence of text instance. CRTR encodes the query and the features of scene text image to rank the candidate text regions generated by DTLN. As much as we are concerned, this is the first work that retrieves text according to a query.

**3.1.3.6 检索指定文字**：有时候我们希望检索一些特定的文字实例，这与场景文字检测的经典设置不同。Rong等[112]提出了一个多编码器框架来检索指定文本。特别的，文字的检索是否自然语言查询发起的。多编码器的框架包括了一个密集文字定位网络(Dense Text Localization Network, DTLN)和一个上下文推理文字检索(Context Reasoning Text Retrival, CRTR)。DTLN使用一个LSTM来将一个FCN网络中的特征解码成一个文字实例序列。CRTR对查询和场景文字图像的特征进行了编码，以对DTLN生成的候选文字区域进行评级。据我们所知，这是第一个根据查询检索文字的工作。

**3.1.3.7 Against Complex Background**: Attention mechanism is introduced to silence the complex background [44]. The stem network is similar to that of the standard SSD framework predicting word boxes, except that it applies inception blocks on its cascading feature maps, obtaining what’s called Aggregated Inception Feature (AIF). An additional text attention module is added, which is again based on inception blocks. The attention is applied on all AIF, reducing the noisy background.

**3.1.3.7 复杂背景情况的处理**：[44]引入了注意力机制来处理复杂的背景。主体网络与标准SSD框架类似，预测文字框架，但是在其级联的特征图上应用了Inception模块，得到了一种特征称为Aggregated Inception Feature(AIF)。增加了一个额外的文字注意力模块，这也是基于Inception模块的。在所有的AIF上都应用了注意力，减少了含噪背景的影响。

### 3.2 Recognition 识别

In this section, we introduce methods that tackle the text recognition problem. Input of these methods are cropped text instance images which contain one word or one text line.

在本节中，我们介绍处理文字识别问题的方法。这些方法的输入是剪切的文字实例图像，包含一个单词或一个文本行。

In traditional text recognition methods [8], [127], the task is divided into 3 steps, including image pre-processing, character segmentation and character recognition. Character segmentation is considered the most challenging part due to the complex background and irregular arrangement of scene text, and largely constrained the performance of the whole recognition system. Two major techniques are adopted to avoid segmentation of characters, namely Connectionist Temporal Classification [36] and Attention mechanism. We introduce recognition methods in the literature based on which technique they employ, while other novel work will also be presented. Mainstream frameworks are illustrated in Fig.8.

在传统的文字识别方法[8,127]中，任务分成了三个步骤，包括图像预处理，字符分割和字符识别。字符分割一般认为是最有挑战性的部分，因为背景复杂，以及场景文字的不规则排布，是限制整个识别系统性能的主要部分。有两种技术可以避免分割字符，即Connectionist Temporal Classification[36]和注意力机制。我们根据文献采用的技术介绍其中的识别方法，同时也给出其他新颖的工作。主流的框架如图8所示。

Fig. 8: Frameworks of text recognition models. The basic methodology is to first resize the cropped image to a fixed height, then extract features and feed them to an RNN that produce a character prediction for each column. As the number of columns of the features is not necessarily equal to the length of the word, the CTC technique [36] is proposed as a post-processing stage. (a) RNN stacked with CNN [122]; (b) Sequence prediction with FCN [28]; (c) Attention-based models [14], [29], [69], [83], [123], [150], allowing decoding text of varying lengths; (d) Cheng et al. [13] proposed to apply supervision to the attention module; (e) To improve the misalignment problem in previous methods with fixed-length decoding with attention, Edit Probability [6] is proposed to reorder the predicted sequential distribution.

图8. 文字识别模型的框架。其基本方法是首先将剪切的图像改变大小，固定其高度，然后提取特征并送入一个RNN，每一列都生成一个特征预测。由于特征的列的数量比一定与单词的长度相等，所以CTC技术[36]用来后处理。(a)RNN上堆叠了CNN[122]；(b)用FCN进行序列预测[28]；(c)基于注意力机制的模型[14,29,69,83,123,150]，可以解码变长的文本；(d)Cheng等[13]提出对注意力模块进行监督；(e)为改进之前方法在定长文本注意力解码的错位问题，[8]提出了Edit Probability来对预测的序列分布进行重新排序。

#### 3.2.1 CTC-based Methods 基于CTC的方法

CTC computes the conditional probability P(L|Y) , where $Y = y_1 , ..., y_T$ represent the per-frame prediction of RNN and L is the label sequence, so that the network can be trained using only sequence level label as supervision. The first application of CTC in the OCR domain can be traced to the handwriting recognition system of Graves et al. [37]. Now this technique is widely adopted in scene text recognition [130], [78], [122], [28], [157].

CTC计算条件概率P(L|Y)，其中$Y = y_1 , ..., y_T$代表RNN的每帧预测，L是标签序列，所以网络可以只使用序列水平标签作为监督来训练。CTC在OCR领域的第一个应用可以追溯到Graves等[37]的手写体识别系统。现在这种技术在场景文本识别中得到了广泛的应用[130,78,122,28,157]。

Shi et al. [122] proposes a model that stacks CNN with RNN to recognize scene text images. As illustrated in Fig.8 (a), CRNN consists of three parts: (1) convolutional layers, which extract a feature sequence from the input image; (2) recurrent layers, which predict a label distribution for each frame; (3) transcription layer (CTC layer), which translates the per-frame predictions into the final label sequence.

Shi等[122]提出一个模型，将CNN堆叠在RNN上，以识别场景文本图像。如图8(a)所示，CRNN包含三部分：(1)卷积层，从输入图像中提取特征序列；(2)循环层，从每一帧中预测一个标签分布；(3)转录层(CTC层)，将每一帧的预测转化为最终的标签序列。

Instead of RNN, Gao et al. [28] adopt the stacked convolutional layers to effectively capture the contextual dependencies of the input sequence, which is characterized by lower computational complexity and easier parallel computation. Overall difference with other frameworks are illustrated in Fig.8 (b).

Gao等[28]没有使用RNN，而是使用了堆积的卷积层来有效的捕捉输入序列的上下文依赖，其计算复杂度较低，更容易进行并行计算。与其他框架的整体区别如图8(b)所示。

Yin et al. [157] also avoids using RNN in their model, they simultaneously detects and recognizes characters by sliding the text line image with character models, which are learned end-to-end on text line images labeled with text transcripts.

Yin等[157]也没有在其模型中使用RNN，他们使用滑动的文字线图像与特征模型同时检测并识别字符，可以使用带有文本标注的文本线图像进行端到端的学习。

#### 3.2.2 Attention-based methods 基于注意力机制的方法

The attention mechanism was first presented in [5] to improve the performance of neural machine translation systems, and flourished in many machine learning application domains including Scene text recognition [13], [14], [29], [69], [83], [123], [150].

注意力机制首次提出是在[5]中，为改进神经网络机器翻译系统的性能，然后在很多机器学习系统中得到了应用，包括场景文字识别[13,14,29,69,83,123,150]。

Lee et al. [69] presented a recursive recurrent neural networks with attention modeling (R2AM) for lexicon-free scene text recognition. the model first passes input images through recursive convolutional layers to extract encoded image features I, and then decodes them to output characters by recurrent neural networks with implicitly learned character-level language statistics. Attention-based mechanism performs soft feature selection for better image feature usage.

Lee等[69]提出了一种带有注意力机制模型(R2AM)递归循环神经网络，进行不用词典的场景文字识别。模型首先将输入图像送入递归卷集层，提取出图像特征编码I，然后用训练神经网络解码输出特征，其中显式的包含了学习得到的特征级别的语言统计信息。基于注意力的机制进行软性特征选择，以更好的利用图像特征。

Cheng et al. [13] observed the attention drift problem in existing attention-based methods and proposed an Focus Attention Network (FAN) to attenuate it. As shown in Fig.8 (d), the main idea is to add localization supervision to the attention module, while the alignment between image features and target label sequence are usually automatically learned in previous work.

Chang等[13]观察到在现有的基于注意力的方法中存在注意漂移的问题，提出了一种聚焦注意力网络(FAN)以减弱这种问题。如图8(d)所示，主要思想是向注意力模块中加入了定位监督，而图像特征与目标标注序列的对齐问题，通常在之前的工作中就自动学习到了。

In [6], Bai et al. proposed an edit probability (EP) metric to handle the misalignment between the ground truth string and the attention’s output sequence of probability distribution, as shown in Fig.8 (e). Unlike aforementioned attention-based methods, which usually employ a framewise maximal likelihood loss, EP tries to estimate the probability of generating a string from the output sequence of probability distribution conditioned on the input image, while considering the possible occurrences of missing or superfluous characters.

在[6]中，Bai等人提出了一种编辑概率(edit probability, EP)度量标准，来处理真值字符串和注意力的概率分布输出序列间的不对齐问题，如图8(e)所示。之前提到的基于注意力的方法通常使用逐帧的最大似然损失，EP与之不同，尝试来估计从给定输入图像的概率分布的输出序列中生成字符串的概率，同时还要考虑丢失或多余的字符的可能情况。

In [83], Liu et al. proposed an efficient attention-based encoder-decoder model, in which the encoder part is trained under binary constraints. Their recognition system achieves state-of-the-art accuracy while consumes much less computation costs than aforementioned methods.

在[83]中，Liu等提出一种有效的基于注意力的编码器-解码器模型，其中编码器部分是在二值约束下训练的。他们的识别系统取得了目前最好的准确率，而计算量与之前提到的方法比少了很多。

Among those attention-based methods, some work made efforts to accurately recognize irregular (perspectively distorted or curved) text. Shi et al. [123], [124] proposed a text recognition system which combined a Spatial Transformer Network (STN) [56] and an attention-based Sequence Recognition Network. The STN predict a Thin-Plate-Spline transformations which rectify the input irregular text image into a more canonical form.

在这些基于注意力的方法中，一些工作尝试去准确的识别不规则的文本（如变形的或弯曲的）。Shi等[123,124]提出了一种文本识别系统，将空域变换器网络(Spatial Transformer Network, STN)[56]与一种基于注意力的序列识别网络(Sequence Recognition Network, SRN)结合在一起。STN预测了一种Thin-Plate-Spline变换，将输入的不规则文本变换成一种更加规范的形式。

Yang et al. [150] introduced an auxiliary dense character detection task to encourage the learning of visual representations that are favorable to the text patterns.And they adopted an alignment loss to regularize the estimated attention at each time-step. Further, they use a coordinate map as a second input to enforce spatial-awareness.

Yang等[150]提出了一种辅助的密集字符检测任务，来鼓励学习倾向于这种文本模式的视觉表示。他们采用了一种对齐损失来在每个时间步骤中规范估计到的注意力。进一步的，他们使用了坐标图作为另一种输入来增强空间感知力。

In [14], Cheng et al. argue that encoding a text image as a 1-D sequence of features as implemented in most methods is not sufficient. They encode an input image to four feature sequences of four directions:horizontal, reversed horizontal, vertical and reversed vertical. And a weighting mechanism is designed to combine the four feature sequences.

在[14]中，Cheng等提出，大多数方法将文本编码为1D序列特征，这是不够用的。他们将输入图像编码为四个方向的四个特征序列：水平，逆水平，竖直，和逆竖直，并设计了一种加权机制来综合这四个特征序列。

Liu et al. [77] presented a hierarchical attention mechanism (HAM) which consists of a recurrent RoI-Warp layer and a character-level attention layer. They adopt a local transformation to model the distortion of individual characters, resulting in an improved efficiency, and can handle different types of distortion that are hard to be modeled by a single global transformation.

Liu等[77]提出了一种层次化的注意力机制(Hierachical Attention Mechanism, HAM)，包括了一个循环的RoI变形层，一个特征级的注意力层。他们采用了一种局部变换，来对单个字符的变形进行建模，得到了改进的效率，并可以处理不同类型的变形，有的很难通过单个全局变换进行建模。

#### 3.2.3 Other Efforts 其他努力

Jaderberg et al. [53], [54] perform word recognition on the whole image holistically. They train a deep classification model solely on data produced by a synthetic text generation engine, and achieve state-of-the-art performance on some benchmarks containing English words only. But application of this method is quite limited as it cannot be applied to recognize long sequences such as phone numbers.

Jaderberg等[53,54]在整个图像上进行文字识别，他们训练了一个深度分类模型，所用的数据是从合成文本生成引擎得到的，并在一些只包含英文文字的基准测试中得到了目前最好的效果。但这种方法的应用非常有限，因为不能识别长的文本序列，如电话号码。

#### 3.3 End-to-End System 端到端的系统

In the past, text detection and recognition are usually cast as two independent sub-problems that are combined together to perform text retrieval from images. Recently, many end-to-end text detection and recognition systems (also known as text spotting systems) have been proposed, profiting a lot from the idea of designing differentiable computation graphs. Efforts to build such systems have gained considerable momentum as a new trend.

在过去，文本检测和识别通常作为两个独立的子问题对待，两者结合起来从图像中检索文本。最近，提出了很多端到端的文本检测和识别系统（也称为text spotting系统），从设计可微计算图的思想中获益良多。构建这种系统的努力非常多，成为了一种新的趋势。

While earlier work [142], [144] first detect single characters in the input image, recent systems usually detect and recognize text in word level or line level. Some of these systems first generate text proposals using a text detection model and then recognize them with another text recognition model [38], [55], [73]. Jaderberg et al. [55] use a combination of Edge Box proposals [173] and a trained aggregate channel features detector [22] to generate candidate word bounding boxes. Proposal boxes are filtered and rectified before being sent into their recognition model proposed in [54]. In [73], Liao et al. combined an SSD [76] based text detector and CRNN [122] to spot text in images. Lyu et al. [88] proposes a modification of Mask R-CNN that is adapted to produce shape-free recognition of scene text, as shown in Fig.9 (c). For each region of interest, character maps are produced, indicating the existence and location of a single character. A post-processing that links these character together gives the final results.

早期的工作[142,144]首先从输入图像中检测单个字符，最近的系统通常检测识别文字或一段文字。一些系统首先使用一个文本检测模型来生成文本候选，然后使用另一个文字识别模型[38,55,73]来进行识别。Jaderberg等[55]将Edge Box候选[173]和一个训练好的ACF检测器[22]结合起来生成文字候选框。候选框经过过滤和整形，送入识别模型[54]。在[73]中，Liao等将一个基于SSD[76]的文本检测器和CRNN[122]结合起来检测图像中的文本。Lyu等[88]修改了Mask R-CNN，可以生成与形状无关的场景文字识别结果，如图9(c)所示。对每个感兴趣区域(RoI)，都生成特征图，表明单个字符的存在和位置。后处理过程将这些字符连接起来，给出最后结果。

One major drawbacks of the two-step methods is that the propagation of error between the text detection models and the text recognition models will lead to less satisfactory performance. Recently, more end-to-end trainable networks are proposed to tackle the this problem [7], [11], [45], [72], [81].

两步方法的一个主要缺点是，文本检测模型和文本识别模型的错误的传播，导致性能不尽如人意。最近，提出了更多的端到端的可训练网络来解决这个问题[7,11,45,72,81].

Bartz et al. [7] presented an solution which utilize a STN [56] to circularly attend to each word in the input image, and then recognize them separately. The united network is trained in a weakly-supervised manner that no word bounding box labels are used. Li et al. [72] substitute the object classification module in Faster-RCNN [107] with an encoder-decoder based text recognition model and make up their text spotting system. Lui et al. [81], Busta et al. [11] and He et al. [45] developed a unified text detection and recognition systems with a very similar overall architecture which consist of a detection branch and a recognition branch. Liu et al. [81] and Busta et al. [11] adopt EAST [168] and YOLOv2 [106] as their detection branch respectively, and have a similar text recognition branch in which text proposals are mapped into fixed height tensor by bilinear sampling and then transcribe in to strings by a CTC-based recognition module. He et al. [45] also adopted EAST [168] to generate text proposals, and they introduced character spatial information as explicit supervision in the attention-based recognition branch.

Bartz等[7]提出了一种解决方案，使用了一个STN[56]来循环的处理输入图像中的每个单词，然后分别识别它们。联合网络进行弱监督训练，没有使用任何文字边界框。Li等[72]将Faster R-CNN[107]中的目标分类模块替换成一个基于编码器-解码器的文本识别模型，构成了他们的text spotting系统。Lui等[81]，Busta等[11]和He等[45]都提出了文本检测和识别的联合系统，其总体框架都很类似，包括一个检测分支和识别分支。Liu等[81]和Busta等[11]分别采用了EAST[168]和YOLOv2[106]作为其检测分支，文本识别分支都很类似，都是将文本候选通过双线性采样映射成固定高度的张量，然后通过一个基于CTC的识别模块转录成字符串。He等[45]也采用了EAST[168]来生成文本候选，然后引入了特征的空间信息作为基于注意力的识别分支的显式分支。

Fig. 9: Illustration of mainstream end-to-end scene text detection and recognition framework. The basic idea is to concatenate the two branch. (a): In SEE [7], the detection results are represented as grid matrices. Image regions are cropped and transformed before being fed into the recognition branch. (b): In contrast to (a), some methods crop from the feature maps and feed them to the recognition branch [11], [45], [72], [81]. (c): While frameworks (a) and (b) utilize CTC-based and attention-based recognition branch, it’s also possible to retrieve each character as generic objects and compose the text [88].

### 3.4 Auxiliary Techniques 辅助技术

Recent advances are not limited to detection and recognition models that aim to solve the tasks directly. We should also give credit to auxiliary techniques that have played an important role. In this part, we briefly introduce several promising trends: synthetic data, bootstrapping, text deblurring, incorporating context information, and adversarial training.

除了检测和识别模型，还有一些不是直接解决这些问题的最新进展。这包括一些辅助技术，也起到了重要的作用。在本部分中，我们简要的介绍了几个有希望的趋势：合成数据，bootstrapping，文本去模糊，利用上下文信息，和对抗训练。

#### 3.4.1 Synthetic Data 合成数据

Most deep learning models are data-thirsty. Their performance is guaranteed only when enough data are available. Therefore, artificial data generation has been a popular research topic, e.g. Generative Adversarial Nets (GAN) [34]. In the field of text detection and recognition, this problem is more urgent since most human-labeled datasets are small, usually containing around merely 1K − 2K data instances. Fortunately, there have been work [38], [54], [164] that can generate data instances of relatively high quality, and they have been widely used for pre-training models for better performance.

多数深度学习模型对非常依赖数据。只有数据充足，算法性能才能得到保证。因此，生成人工数据就成了一个受欢迎的研究课题，如，生成式对抗网络(Generative Adversarial Nets, GAN)[34]。在文本检测和识别的研究领域，这个问题更加紧迫，因为大多数人类标注的数据集都很小，通常只包括1K-2K数据实例。幸运的是，已经有一些工作[38,54,164]可以生成相对质量较好的数据实例了，并且已经广泛的用于预训练模型，以得到更好的性能。

Jaderberg et at. [54] first proposes synthetic data for text recognition. Their method blends text with randomly cropped natural image from human-labeled datasets after rending of font, border/shadow, color, and distortion. The results show that training merely on these synthetic data can achieve state-of-the-art performance and that synthetic data can act as augmentative data sources for all datasets.

Jaderberg等[54]首先提出用于文本识别的数据合成。他们的方法将文本与随机剪切的自然图像混合到一起，图像是人类标注的数据集中的，文本经过字体、边缘/阴影、色彩和变形的渲染。结果表明，只在这些合成数据上进行训练，就可以得到目前最好的性能，并且合成数据可以作为所有数据集的数据扩充的源。

SynthText [38] first proposes to embed text in natural scene images for training of text detection, while most previous work only print text on a cropped region and these synthetic data are only for text recognition. Printing text on the whole natural images poses new challenges, as it needs to maintain semantic coherence. To produce more realistic data, SynthText makes use of depth prediction [75] and semantic segmentation [4]. Semantic segmentation groups pixels together as semantic clusters, and each text instance is printed on one semantic surface, not overlapping multiple ones. Dense depth map is further used to determine the orientation and distortion of the text instance. Model trained only on SynthText achieves state-of-the-art on many text detection datasets. It’s later used in other works [121], [168] as well for initial pre-training.

SynthText[38]首先提出将文本嵌入自然场景图像中进行文字检测训练，而多数之前的工作只是将文本打印到图像的剪切区域中，这些合成数据只用于文本识别。将文本打印到全自然图像中给出了新的挑战，因为需要保持语义的一致性。为生成更多的实际数据，SynthText使用深度预测[75]和语义分割[4]。语义分割将像素按语义聚类分组，每个文本实例只打印在一个语义表面上，而不覆盖多个语义表面。进而将密集深度图用于确定文本实例的方向和变形。只在SynthText上训练的模型在很多文本检测数据集上得到了目前最好的结果。后来在其他工作中[121,168]也用于初始预训练。

Further, Zhan et al. [164] equips text synthesis with other deep learning techniques to produce more realistic samples. They introduce selective semantic segmentation so that word instances would only appear on sensible objects, e.g. a desk or wall in stead of someone’s face. Text rendering in their work is adapted to the image so that they fit into the artistic styles and do not stand out awkwardly.

进一步的，Zhan等[164]将文本合成用于其他深度学习技术中，以生成更真实的样本。他们提取选择性的语义分割，这样单词实例只出现在有意义的目标上，如一个桌子或墙上，而不是某人的脸上。他们工作中的文本渲染适应于对应的图像，这样在艺术风格上很匹配，不会非常突兀。

#### 3.4.2 Bootstrapping

Bootstrapping, or Weakly and semi supervision, is also important in text detection and recognition [50], [111], [132]. It’s mainly used in word [111] or character [50], [132] level annotations.

Bootstrapping，即弱监督或半监督，在文本检测和识别中也很重要[50,111,132]。主要使用于单词[111]或字母[50,132]级的标注。

**Bootstrapping for word-box**. Rong et al. [111] proposes to combine an FCN-based text detection network with Maximally Stable Extremal Region (MSER) features to generate new training instances annotated on box-level. First, they train an FCN, which predicts the probability of each pixel belonging to text. Then, MSER features are extracted from regions where the text confidence is high. Using single linkage criterion (SLC) based algorithms [32], [128], final prediction is made.

**单词框的bootstrapping**。Rong等[111]提出，将一种基于FCN的文本检测网络，和最大稳定极端区域(Maximally Stable Extremal Region, MSER)特征，结合到一起，以产生新的训练样本，标注用边界框。首先，他们训练了一个FCN，预测了每个像素属于文字的概率。然后，从文本概率高的区域提取出MSER特征。使用基于单连接原则(single linkage criterion, SLC)的算法[32,128]，得到了最终的预测。

**Bootstrapping for character-box**. Character level annotations are more accurate and better. However, most existing datasets do not provide character-level annotating. Since character is smaller and close to each other, character-level annotation is more costly and inconvenient. There have been some work on semi-supervised character detection [50], [132]. The basic idea is to initialize a character-detector, and applies rules or threshold to pick the most reliable predicted candidates. These reliable candidates are then used as additional supervision source to refine the character-detector. Both of them aim to augment existing datasets with character level annotations. They only differ in details.

**字母框的bootstrapping**。字母级的标注更加准确好用。但是，多数现有的数据集都没有提供字母级的标注。因为字母更小，互相之间距离非常接近，字母级的标注耗费更大，而且非常不方便。在半监督字母检测中有一些工作[50,132]。其基本思想是初始化一个字母检测器，然后应用一些规则或阈值，来挑出最可靠的预测候选。这些可靠的候选然后用作额外的监督源，来优化字母检测器。这两个工作的目标都是用字母级的标注扩充现有的数据集，只在一些细节上有不同之处。

WordSup [50] first initializes the character detector by training 5K warm-up iterations on synthetic dataset, as shown in Fig.10 (b). For each image, WordSup generates character candidates, which are then filtered with wordboxes. For characters in each word box, the following score is computed to select the most possible character list:

WordSup[50]首先通过在合成数据集上的5k热身迭代训练来初始化字母检测器，如图10(b)所示。对于每幅图像，WordSup生成候选字母，然后用wordboxes进行过滤。对于每个wordbox中的文字，计算下面的分数以选择最可能的字母列表：

$$s = w · s_1 + (1 − w) · s_2 = = w · \frac {area(B_{chars})}{area(B_{word})} + (1-w) · (1-\frac{λ_2}{λ_1})$$(1)

where $B_{chars}$ is the union of the selected character boxes; $B_{word}$ is the enclosing word bounding box; $λ_1$ and $λ_2$ are the first and second largest eigenvalues of a covariance matrix C, computed by the coordinates of the centers of the selected character boxes; w is a weight scalar. Intuitively, the first term measures how complete the selected characters can cover the word boxes, while the second term measures whether the selected characters are located on a straight line, which is a main characteristic for word instances in most datasets.

其中$B_{chars}$是选择的字符框的并集，$B_{word}$是包围单词的边界框，$λ_1$和$λ_2$是协方差矩阵C的最大和第二大的特征值，C是由选择的字符框的中心坐标计算得到的，w是一个标量权重。直觉上来说，第一项是选择的字符框能覆盖单词框的比例，第二项是选择的字母是不是处在一条直线上，这在大多数数据集中都是单词实例的一个主要特征。

WeText [132] starts with a small datasets annotated on character level. It follows two paradigms of bootstrapping: semi-supervised learning and weakly-supervised learning. In the semi-supervised setting, detected character candidates are filtered with a high thresholding value. In the weakly-supervised setting, ground-truth word boxes are used to mask out false positives outside. New instances detected in either way is added to the initial small datasets and re-train the model.

WeText[132]开始于标注在字符级的小数据集上。进行了两种bootstrapping的方案：半监督学习和弱监督学习。在半监督学习的设置中，检测到的字符候选用高阈值过滤。在弱监督学习的设置中，使用真值单词框来将false positive掩膜到外面。任何方法检测到的新实例都加入初始的小数据集，然后重新训练模型。

#### 3.4.3 Text Deblurring 文本去模糊

By nature, text detection and recognition are more sensitive to blurring than general object detection. Some methods [49], [66] have been proposed for text deblurring.

很自然的，文本检测和识别比通用目标检测对模糊更加敏感。提出了一些方法[49,66]进行文本去模糊。

Hradis et al. [49] proposes an FCN-based deblurring method. The core FCN maps the input image which is blurred and generates a deblurred image. They collect a dataset of well-taken images of documents, and process them with kernels designed to mimic hand-shake and defocus.

Hradis等[49]提出了一种基于FCN的去模糊方法。FCN核心将模糊的输入图像映射并生成一个去模糊的图像。他们收集了拍摄很清晰的文档数据集，然后进行处理，模拟晃动和未聚焦的效果。

Khare et al. [66] proposes a quite different framework. Given a blurred image, g , it aims to alternatively optimize the original image f and kernel k by minimizing the following energy value:

Khare等[66]提出了一种非常不同的框架。给定一个模糊的图像g，其目标是通过最小化下面的能量函数，来交替优化原始图像f和卷积核k：

$$E = \int (k(x,y)*f(x,y)-g(x,y))^2 dxdy + λ \int w R(k(x,y))dxdy$$(2)

where λ is the regularization weight, with operator R as the Gaussian weighted (w) L1 norm. The optimization is done by alternatively optimizing over the kernel k and the original image f.

其中λ是正则化权重，算子R是加权高斯L1范数。优化过程就是对卷积核k和原始图像f交替优化的过程。

#### 3.4.4 Context Information 上下文信息

Another way to make more accurate predictions is to take into account the context information. Intuitively, we know that text only appear on a certain surfaces, e.g. billboards, books, and etc.. Text are less likely to appear on the face of a human or an animal. Following this idea, Zhu et al. [170] proposes to incorporate the semantic segmentation result as part of the input. The additional feature filters out false positives where the patterns look like text.

进行更精准的预测的另一种方法，是加入上下文信息。直觉上来讲，我们知道文本只会在一些特定平面上才出现，如，广告牌，书本等等。文本不太可能在人脸或动物的脸上出现。按照这个思想，Zhu等[170]提出利用语义分割结果作为部分输入。额外的特征滤除掉了false positives，那些模式看起来比较像文本。

#### 3.4.5 Adversarial Attack 对抗性攻击

Text detection and recognition has a broad range of application. In some scenarios, the security of the applied algorithms becomes a key factor, e.g. autonomous vehicles and identity verification. Yuan et al. [162] proposes the first adversarial attack algorithm for text recognition. They propose a white-box attack algorithm that induces a trained model to generate a desired wrong output. Specifically, they aim to optimize a joint target of: (1) D(x, x') for minimizing the alteration applied to the original image; (2) L($x_{targeted}$) for the loss function with regard to the probability of the targeted output. They adapt the automated weighting method proposed by Kendall et al. [65] to find the optimum weight of the two targets. Their method realizes a success rate over 99.9% with 3 − 6× speedup compared to other state-of-the-art attack methods. Most importantly, their method showed a way to carry out sequential attack.

文本检测和识别有广泛的应用。在一些场景中，算法应用的安全性成为了一个关键因素，如，自动驾驶和身份验证。Yuan等[162]提出了第一种文本识别的对抗攻击算法。他们提出了一种白盒攻击算法，诱导训练的模型来生成一种期望的错误结果。特别的，他们的目标是优化一种联合目标函数：(1)D(x,x')最小化应用在原图上的变更；(2)L($x_{targeted}$)是对目标输出概率的损失函数。他们将Kendall等[65]提出的自动加权方法进行修改，以发现两个目标函数的最佳加权。他们的方法实现了超过了99.9%的成功率，但与目前最好的攻击方法相比，速度快了3-6倍。最重要的是，他们的方法展现了一种序列攻击的途径。

## 4 Benchmark Datasets and Evaluation Protocols 基准测试数据集和评估方案

As cutting edge algorithms achieved better on previous datasets, researchers were able to tackle more challenging aspects of the problems. New datasetes aimed at different real-world challenges have been and are being crafted, benefiting the development of detection and recognition methods further.

由于最尖端的技术在之前的数据集上会取得更好的结果，研究者可以处理更有挑战性的问题。新的数据集建立的目标是应对不同的真实世界挑战，以使检测和识别方法进一步发展。

In this section, we list and briefly introduce the existing datasets and the corresponding evaluation protocols. We also identify current state-of-the-art performance on the widely used datasets when applicable.

在本节中，我们列出现有的数据集并简单进行介绍，以及相应的评估方案。我们还给出在目前广泛使用的数据集上目前最好的性能。

### 4.1 Benchmark Datasets 基准测试数据集

We collect existing datasets and summarize their features in Tab.1. We also select some representative image samples from some of the datasets, which are demonstrated in Fig.11. Links to these datasets are also collected in our Github repository mentioned in abstract, for readers’ convenience.

我们收集现有的数据集，并在表1中总结了其特征。我们还从一些数据集中选出了一些代表性图像样本，如图11所示。这些数据集的连接也收集并放在了我们的Github repo中，以方便读者。

TABLE 1: Existing datasets: * indicates datasets that are the most widely used across recent publications. Newly published ones representing real-world challenges are marked in bold. EN stands for English and CN stands for Chinese.

Dataset(Year) | Image Num (train/test) | Text Num (train/test) | Orientation | Language | Characteristics | Detection Task | Recognition Task
--- | --- | --- | --- | --- | --- | --- | ---
ICDAR03(2003) | 258/251 | 1110/1156 | Horizontal | EN | - | y | y
**ICDAR13 Scene Text(2013)*** | 229/233 | 848/1095 | Horizontal | EN | - | y | y
**ICDAR15 Incidental Text(2015)*** | 1000/500 | -/- | Multi-Oriented | EN | Blur/Small | y | y
**ICDAR RCTW(2017)** | 8034/4229 | -/- | Multi-Oriented | CN | - | y | y
**Total-Text(2017)** | 1255/300 | -/- | Curved | EN,CN | Polygon label | y | y
SVT(2010) | 100/250 | 257/647 | Horizontal | EN | - | y | y
**CUTE(2014)*** | -/80 | -/- | Curved | EN | - | y | y
CTW(2017) | 25k/6k | 812k/205k | Multi-Oriented | CN | fine-grained annotation | y | y
CASIA-10K(2018) | 7k/3k | -/- | Multi-Oriented | CN | - | y | y
**MSRA-TD500(2012)*** | 300/200 | 1068/651 | Multi-Oriented | EN,CN | Long text | y | n
**HUST-TR400(2014)** | 400/- | -/- | Multi-Oriented | EN,CN | Long text | y | n
**ICDAR17MLT(2017)** | 9000/9000 | -/- | Multi-Oriented | 9 languages | - | y | n
**CTW1500(2017)** | 1000/500 | -/- | Curved | EN | | y | n
**IIIT 5K-Word(2012)*** | 2000/3000 | 2000/3000 | Horizontal | - | - | n | y
**SVTP(2013)** | -/639 | -/639 | Multi-Oriented | EN | Perspective text | n | y
SVHN(2010) | 73257/26032 | 73257/26032 | Horizontal | - | House number digits | n | y

Fig. 11: Selected samples from Chars74K, SVT-P, IIIT5K, MSRA-TD500, ICDAR2013, ICDAR2015, ICDAR2017 MLT, ICDAR2017 RCTW, and Total-Text.

#### 4.1.1 Datasets with both detection and recognition tasks 检测与识别任务都有的数据集

- The ICDAR 2003&2005 and 2011&2013

Held in 2003 , the ICDAR 2003 Robust Reading Competition [87] is the first such benchmark dataset that’s ever released for scene text detection and recognitio.Among the 509 images, 258 are used for training and 251 for testing. The dataset is also used in ICDAR 2005 Text Locating Competition [86]. ICDAR 2015 also includes a digit recognition track.

2003年举办了ICDAR2003稳健阅读竞赛[87]，这是场景文本检测和识别基准测试数据集的第一个，包括509幅图像，258幅用于训练，251幅用于测试。这个数据集也用于ICDAR2005文本定位竞赛[86]。ICDAR2015也包括一个数字识别的赛道。

In the ICDAR 2011 and 2013 Robust Reading Competitions, previous datasets are modified and extended, which make the new ICDAR 2011 [118] and 2013 [64] datasets. Problems in previous datasets are corrected, e.g. imprecise bounding boxes. State-of-the-art results are shown in Tab.2 for detection and Tab.8 for recognition.

在ICDAR2011和2013稳健阅读竞赛中，之前的数据集得到了修改和拓展，这就形成了ICDAR2011[118]和ICDAR2013[64]数据集。之前数据集中的问题得到了修正，如，不精确的边界框。目前最好的检测结果如图2所示，识别结果如图8所示。

TABLE 2: Detection performance on ICDAR2013. ∗ means multi-scale, † stands for the base net of the model is not VGG16. The performance is based on DetEval.

Method | Precision | Recall | F-measures | FPS
--- | --- | --- | --- | ---
Zhang et al. [166] | 88 | 78 | 83 | -
SynthText [38] | 92.0 | 75.5 | 83.0 | -
Holistic [152] | 88.88 | 80.22 | 84.33 | -
PixelLink [20] | 86.4 | 83.6 | 84.5 | -
CTPN [133] | 93 | 83 | 88 | 7.1 
He et al. ∗ [41] | 93 | 79 | 85 | -
SegLink [121] | 87.7 | 83.0 | 85.3 | 20.6
He et al. ∗ † [46] | 92 | 80 | 86 | 1.1
TextBox++ [73] | 89 | 83 | 86 | 1.37
EAST [168] | 92.64 | 82.67 | 87.37 | -
SSTD [44] | 89 | 86 | 88 | 7.69
Lyu et al. [89] | 93.3 | 79.4 | 85.8 | 10.4
Liu et al. [84] | 88.2 | 87.2 | 87.7 | -
He et al. ∗ [45] | 88 | 87 | 88 | -
Xue et al. † [149] | 91.5 | 87.1 | 89.2 | -
WordSup ∗ [50] | 93.34 | 87.53 | 90.34 | -
Lyu et al. ∗ [88] | 94.1 | 88.1 | 91.0 | 4.6
FEN [119] | 93.7 | 90.0 | 92.3 | 1.11

- ICDAR 2015

In real world application, images containing text may be too small, blurred, or occluded. To represent such a challenge, ICDAR2015 is proposed as the Challenge 4 of the 2015 Robust Reading Competition [63] for incidental scene text detection. Scene text images in this dataset are taken by Google Glasses without taking care of the image quality. A large proportion of images are very small, blurred, and multi-oriented. There are 1000 images for training and 500 images for testing. The text instances from this dataset are labeled as word level quadrangles. State-of-the-art results are shown in Tab.3 for detection and Tab.8 for recognition.

在真实世界应用中，图像中包含的文字可能太小，模糊或遮挡。为代表这些挑战，ICDAR2015提出了2015稳健阅读竞赛[63]的挑战4，即场景附带文本检测。这个数据集中的场景文本图像是由谷歌眼镜拍摄的，并没有照顾到图像质量。大部分图像中的文字都非常小、模糊且包含多个方向。有1000幅图像进行训练，500幅图像进行测试。这个数据集中的文本实例标注为单词级的四边形。目前最好的检测结果如表3所示，识别结果如表8所示。

TABLE 3: Detection performance on ICDAR2015. ∗ means multi-scale, † stands for the base net of the model is not VGG16.

Method | Precision | Recall | F-measure | FPS
--- | --- | --- | --- | ---
Zhang et al. [166] | 71 | 43.0 | 54 | -
CTPN [133] | 74 | 52 | 61 | 7.1
Holistic [152] | 72.26 | 58.69 | 64.77 | -
He et al. ∗ [41] | 76 | 54 | 63 | -
SegLink [121] | 73.1 | 76.8 | 75.0 | -
SSTD [44] | 80 | 73 | 77 | -
EAST [168] |83.57 | 73.47 | 78.20 | 13.2
He et al. ∗ † [46] | 82 | 80 | 76 | -
R2CNN [59] | 85.62 | 79.68 | 82.54 | 0.44
Liu et al. [84] | 72 | 80 | 76 | -
WordSup ∗ [50] | 79.33 | 77.03 | 78.16 | -
Wang et al. † [141] | 85.7 | 74.1 | 79.5 | -
Lyu et al. [89] | 94.1 | 70.7 | 80.7 | 3.6
TextSnake [85] | 84.9 | 80.4 | 82.6 | 1.1
He et al. ∗ [45] | 84 | 83 | 83 | -
Lyu et al. ∗ [88] | 85.8 | 81.2 | 83.4 | 4.8
PixelLink [20] | 85.5 | 82.0 | 83.7 | 3.0

- ICDAR 2017 RCTW

In ICDAR2017 Competition on Reading Chinese Text in the Wild [125], Shi et al. propose a new dataset, called CTW-12K, which mainly consists of Chinese. It is comprised of 12, 263 images in total, among which 8, 034 are for training and 4, 229 are for testing. Text instances are annotated with parallelograms. It’s the first large scale Chinese dataset, and was also the largest published one by then.

在ICDAR2017的阅读自然环境下的中文文本竞赛[125]上，Shi等提出了一个新的数据集，称为CTW-12K，主要包括中文文本，共计包括12263幅图像，8034幅进行训练，4339幅进行测试。文本实例用平行四边形标注。这是第一个大型中文文本数据集，那时也是最大的一个。

- CTW

The Chinese Text in the Wild (CTW) dataset proposed by Yuan et al. [161] is the largest annotated dataset to date. It has 32, 285 high resolution street view image of Chinese text, with 1, 018, 402 character instances in total. All images are annotated at the character level, including its underlying character type, bounding box, and 6 other attributes. These attributes indicate whether its background is complex, whether it’s raised, whether it’s hand-written or printed, whether it’s occluded, whether it’s distorted, whether it uses word-art. The dataset is split into a training set of 25, 887 images with 812, 872 characters, a recognition test set of 3, 269 images with 103, 519 characters, and a detection test set of 3, 129 images with 102, 011 characters.

Yuan等[161]提出的自然环境下的中文文本(CTW)数据集，是迄今为止最大的标注数据集，包括32285幅高分辨率街景中文文本图像，共计1018402个字符实例。所有图像都是字符级标注，包括其潜在的字符类型，边界框和6个其他的属性。这些属性表明其背景是否复杂，是不是浮雕文字，是手写还是打印的，是否被遮挡，是否有形变，是否使用了艺术字体。数据集分割成25887幅图像的训练集，812872个字符，识别测试集包括3269幅图像，103519个字符，检测测试集包括3129幅图像102011个字符。

- Total-Text

Unlike most previous datasets which only include text that are in straight lines, Total-Text consists of 1555 images with more than 3 different text orientations: Horizontal, Multi-Oriented, and Curved. Text instances in Total-Text are annotated with both quadrilateral boxes and polygon boxes of a variable number of vertexes. State-of-the-art results for Total-Text are shown in Tab.4 for detection and recognition.

之前大多数数据集包含的文本都是直线排列的，Total-Text则包含了三种不同文本方向的1555幅图像：水平排列，多方向排列和曲线排列。Total-Text中的文本实例用四边形框和不同顶点数量的多边形框。Total-Text上目前最好的检测和识别结果如表4所示。

TABLE 4: Performance on Total-Text.

Method | Detection P | Detection R | Detection F | Word Spotting None | Word Spotting Full
--- | --- | --- | --- | --- | --- | ---
DeconvNet[100] | 33 | 40 | 36 | - | -
Lyu et al.*[88] | 69.0 | 55.0 | 61.3 | 52.9 | 71.8
TextSnake[85] | 82.7 | 74.5 | 78.4 | - | -

-  SVT

The Street View Text (SVT) dataset [142], [143] is a collection of street view images. SVT has 350 images. It only has word-level annotations. 街景文本数据集[142,143](Street View Text, SVT)是街景图像的集合，包含350幅图像，只有单词级的标注。

- CUTE80 (CUTE)

CUTE is proposed in [108]. The dataset focuses on curved text. It contains 80 high-resolution images taken in natural scenes. No lexicon is provided.  CUTE是在[108]中提出，聚焦在弯曲文字上，包含80幅高分辨率自然场景中的图像，没有提供词汇表。

#### 4.1.2 Datasets with only detection task 只有检测任务的数据集

- MSRA-TD 500 and HUST-TR 400

The MSRA Text Detection 500 Dataset (MSRA-TD500) [135] is a benchmark dataset featuring long and multi-oriented text. Text instances in MSRA-TD500 have much larger aspect ratios than other datasets. Later, an additional set of images, called HUST-TR400 [151], are collected in the same way as MSRA-TD500, usually used as additional training data for MSRA-TD500.

MSRA文本检测500数据集(MSRA-TD5000)[135]是一个基准检测数据集，其特点是很长而且多方向的文本。MSRA-TD500中的文本实例比其他数据集的宽高比更大。后来，另一个图像集，称为HUST-TR400[151]，与MSRA-TD500收集的方式一样，通常用作MSRA-TD500的额外训练数据。

- ICDAR2017 MLT

The dataset of ICDAR2017 MLT Challenge [95] contains 18K images with scripts of 9 languages, 2K for each. It features the largest number of languages up till now. 包含18K幅图像，9种语言，每种2K幅图像，是迄今位置包含语言最多的数据集。

- CASIA-10K

CASIA-10K is a newly published Chinese scene text dataset. This dataset contains 10000 images under various scenarios, with 7000 for training and 3000 testing. As Chinese characters are not segmented by spaces, line-level annotations are provided. 这是一个新发布的中文场景文本数据集，包含10000幅图像，各种场景，7000幅训练，3000幅测试。由于中文字符不是靠空格分割的，所以标注了线段级标注。

- SCUT-CTW1500 (CTW1500)

CTW1500 is another dataset which features curved text. It consists of 1000 training images and 500 test images. Annotations in CTW1500 are polygons with 14 vertexes. Performances on CTW1500 are shown in Tab.5 for detection. CTW1500是另一个以曲线文本为特征的数据集，包含1000幅训练图像，500幅测试图像，其中的标注是14个顶点的多边形，表5给出了检测性能的表现。

TABLE 5: Detection performance on CTW1500.

Method | Precision | Recall | F-measure
--- | --- | --- | ---
SegLink [121] | 42.3 | 40.0 | 40.8
EAST [168] | 78.7 | 49.1 | 60.4
DMPNet [82] | 69.9 | 56.0 | 62.2
CTD+TLOC [163] | 77.4 | 69.8 | 73.4
TextSnake [85] | 67.9 | 85.3 | 75.6

#### 4.1.3 Datasets with only recognition task 只有识别任务的数据集

- IIIT 5K-Word

IIIT 5K-Word [94] is the largest dataset, containing both digital and natural scene images. Its variance in font, color, size and other noises makes it the most challenging one to date. There are 5000 images in total, 2000 for training and 3000 for testing. 这是最大的一个数据集，包含数字和自然场景图像，在字体、颜色、大小和其他噪声上变化都很大，所以成为迄今为止最有挑战性的数据集。共有5000幅图像，2000幅训练图像，3000幅测试图像。

- SVT-Perspective (SVTP)

SVTP is proposed in [104] for evaluating the performance of recognizing perspective text. Images in SVTP are picked from the side-view images in Google Street View. Many of them are heavily distorted by the non-frontal view angle. The dataset consists of 639 cropped images for testing, each with a 50-word lexicon inherited from the SVT dataset. [104]里提出SVTP的目的是评估识别透视文本的性能。SVTP种的图像是从谷歌街景图像侧视角图像种挑选出来的，其中很多由于非正面视角的原因严重变形，数据集包括639幅剪切图像进行测试用，每幅图像包括50个单词的字典，这是从SVT数据集中继承下来的。

- SVHN

The street view house numbers (SVHN) dataset [96] contains more than 600000 digits of house numbers in natural scenes. The images are collected from Google View images. This dataset is usually used in digit recognition. 街景门牌号数据集(SVHN)[96]包含自然场景中多余60000个门牌号数字，图像从谷歌街景图像中收集的，这个数据集通常用于数字识别。

### 4.2 Evaluation Protocol 评估方案

In this part, we briefly summarize the evaluation protocols for text detection and recognition. 这个部分中，我们简要的总结一下文本检测和识别的评估方案。

As metrics for performance comparison of different algorithms, we usually refer to their precision, recall and F1-score. To compute these performance indicators, the list of predicted text instances should be matched to the ground truth labels in the first place. Precision, denoted as P, is calculated as the proportion of predicted text instances that can be matched to ground truth labels. Recall, denoted as R , is the proportion of ground truth labels that have correspondents in the predicted list. F1-score is a then computed by  $F_1 = \frac{2∗P∗R}/{P+R}$, taking both precision and recall into account. Note that the matching between the predicted instances and ground truth ones comes first.

作为不同算法性能比较的衡量标准，我们通常参考其精确度，召回率和F1-分数。为计算这些性能指标，预测的文本实例列表首先需要与真值标签匹配。精确度，表示为P，就是计算预测的文本实例可以匹配到真值标签的比例。召回率，表示为R，就是真值标签有预测列表中有对应匹配的比例。F1分数计算公式为$F_1 = \frac{2∗P∗R}/{P+R}$，是精确度和召回率的综合评判。注意，预测实例与真值的匹配首先进行。

#### 4.2.1 Text Detection 文本检测

There are mainly two different protocols for text detection, the IOU based PASCAL Eval and overlap based DetEval. They differ in the criterion of matching predicted text instances and ground truth ones. In the following part, we use these notations: $S_{GT}$ is the area of the ground truth bounding box, $S_P$ is the area of the predicted bounding box, $S_I$ is the area of the intersection of the predicted and ground truth bounding box, $S_U$ is the area of the union.

主要有两种文本检测的评估方法，基于IOU的PASCAL评估方法，和基于重叠的DetEval。它们之间的区别在于，匹配预测文本实例和真值的标准不同。在后面的部分，我们使用下面的表示符号：$S_{GT}$是真值边界框的区域，$S_P$是预测边界框的区域，$S_I$是预测边界框和真值边界框的交集，$S_U$是其并集。

- PASCAL [25]: The basic idea is that, if the intersection-over-union value, i.e. $S_I/S_U$, is larger than a designated threshold, the predicted and ground truth box are matched together. 其基本思想是，如果其交并比值，即$S_I/S_U$，比指定的阈值要大，那么预测边界框和真值边界框就匹配到一起了。

- DetEval: DetEval imposes constraints on both precision, i.e. $S_I/S_P$ and recall, i.e. $S_I/S_{GT}$. Only when both are larger than their respective thresholds, are they matched together. DetEval在精确度和召回率上都进行了限制，即$S_I/S_P$和$S_I/S_{GT}$。只有当两者分别大于其阈值，才酸匹配成功。

Most datasets follow either of the two evaluation protocols, but with small modifications. We only discusses those that are different from the two protocols mentioned above. 多数数据集遵循上述两种评估方案，但有很小的修改。我们只讨论那些与这两种方案不同的情况。

**4.2.1.1 ICDAR2003/2005**: The match score m is calculated in a way similar to IOU. It’s defined as the ratio of the area of intersection over that of the minimum bounding rectangular bounding box containing both. 匹配分数m的计算方法与IOU类似，其定义为交集与包含两者的最小矩形的比值。

**4.2.1.2 ICDAR2011/13**: One major drawback of the evaluation protocol of ICDAR2003/2005 is that it only considers one-to-one match. It does not consider one-to-many, many-to-many, and many-to-one matchings, which underestimates the actual performance. Therefore, ICDAR2011/2013 follows the method proposed by Wolf et al. [146]. The match score function, $m_D$ and $m_G$, gives different score for each types of matching: ICDAR2003/2005的评估方案的一个主要缺陷是，只考虑了一对一的匹配，没有考虑到一对多，多对多，和多对一的匹配，这低估了实际的性能。所以，ICDAR2011/2013按照Wolf等[146]提出的方法，采用匹配分数函数，$m_D$ and $m_G$，对不同的匹配类型给出不同的分数：

$$score = 1, one-to-one match$$(3)
$$score = 0, if no match$$(3)
$$score = f_{sc}(k), if many matches$$(3)

$f_{sc} (k)$ is a function for punishment of many-matches, controlling the amount of splitting or merging. $f_{sc} (k)$是惩罚多匹配情况的函数，控制分裂或合并的数量。

**4.2.1.3 MSRA-TD500**: Yao et al. [135] proposes a new evaluation protocol for rotated bounding box, where both the predicted and ground truth bounding box are revolved horizontal around its center. They are matched only when the standard IOU score is higher than the threshold and the rotation of the original bounding boxes are less a pre-defined value (in practice pi/4). Yao等[135]对旋转的边界框提出了一种新的评估方案，其中预测和真值边界框都沿着其中心旋转为水平，只有当其标准IOU值高于阈值，且原始边界框的旋转小于预定义的值（实际中是pi/4）。

#### 4.2.2 Text Recognition and End-to-End System 文本识别和端到端的系统

Text recognition is another task where a cropped image is given which contains exactly one text instance, and we need to extract the text content from the image in a form that a computer program can understand directly, e.g. string type in C++ or str type in Python. There is not need for matching in this task. The predicted text string is compared to the ground truth directly. The performance evaluation is in either character-level recognition rate (i.e. how many characters are recognized) or word level (whether the predicted word is 100% correct). ICDAR also introduces an edit-distance based performance evaluation. Note that in end-to-end evaluation, matching is first performed in a similar way to that of text detection. State-of-the-art recognition performance on the most widely used datasets are summarized in Tab.8.

文本识别任务中，给定的是图像剪切块，其中只包含一个文本实例，我们需要从图像中提取文本内容，并形成计算机程序可以直接理解的形式，即，C++或Python中的字符串。这个任务中不需要匹配，预测的文本字符串于真值直接进行比较。性能评估是字符级的识别率（即，识别了多少字符），或单词级的识别率（预测的单词是否是100%正确）。ICDAR还引入了一种基于edit-distance的性能评估。注意在端到端的评估中，首先进行的匹配，这与文本检测的方式类似。在使用最多的数据集中，目前最好的识别性能总结在表8中。

TABLE 8: State-of-the-art recognition performance across a number of datasets. “50”, “1k”, “Full” are lexicons. “0” means no lexicon. “90k” and “ST” are the Synth90k and the SynthText datasets, respectively. “ST + ” means including character-level annotations. “Private” means private training data.

The evaluation for end-to-end system is a combination of both detection and recognition. Given output to be evaluated, i.e. text location and recognized content, predicted text instances are first matched with ground truth instances, followed by comparison of the text content.

端到端系统的评估是检测性能与识别性能的综合。给定输出进行评估时，即，文本位置和识别出的内容，预测的文本实例首先于真值实例进行匹配，然后与文本内容进行对比。

The most widely used datasets for end-to-end systems are ICDAR2013 [64] and ICDAR2015 [63]. The evaluation over these two datasets are carried out under two different settings [1], the Word Spotting setting and the End-to-End setting. Under Word Spotting, the performance evaluation only focuses on the text instances from the scene image that appear in a predesignated vocabulary, while other text instances are ignored. On the contrary, all text instances that appear in the scene image are included under End-to-End. Three different vocabulary lists are provided for candidate transcriptions. They include Strongly Contextualised, Weakly Contextualised, and Generic. The three kinds of lists are summarized in Tab.7. Note that under End-to-End, these vocabulary can still serve as reference. State-of-the-art performances are summarized in Tab.9.

端到端系统使用最多的数据集为ICDAR2013[64]和ICDAR[63]。这两个数据集上的评估在两种不同的设置下进行[1]，单词定位(Word Spotting)设置和端到端的设置。在单词定位中，性能评估只关注的是，场景图像中的文本实例中，出现的预先指定词汇，其他文本实例则被忽略。相反的是，在端到端系统中，场景图像中出现的所有文本实例都被包含在内。提供了三种不同的词汇列表来进行候选转录，包括强语境化的，弱语境化的，和一般性的。这三种列表总结在表7中。注意在端到端的系统中，这些词汇表仍然可以用于参考。目前最好的性能总结于表9中。

TABLE 7: Characteristics of the three vocabulary lists used in ICDAR 2013/2015. S stands for Strongly Contextualised, W for Weakly Contextualised, and G for Generic

Vocab List | Description
--- | ---
S | a per-image list of 100 words/ all words in the image + seletected distractors
W | all words in the entire test set
G | a 90k -word generic vocabulary

TABLE 9: State-of-the-art performance of End-to-End and Word Spotting tasks on ICDAR2015 and ICDAR2013. multi-scale, † stands for the base net of the model is not VGG16.

## 5 Application 应用

The detection and recognition of text—the visual and physical carrier of human civilization—allow the connection between vision and the understanding of its content further. Apart from the applications we have mentioned at the beginning of this paper, there have been numerous specific application scenarios across various industries and in our daily lives. In this part, we list and analyze the most outstanding ones that have, or are to have, significant impact, improving our productivity and life quality.

文本的检测和识别-人类文明的视觉载体和物理载体-使视觉和其内容理解进一步联系起来。除了本文开始时提到的应用，还有非常多的特定应用场景，在各种工业场景和日常生活中。在这一部分，我们列出并分析那些最杰出的几个，已经或可能对于提高生产力和生活质量有显著影响的那些。

**Automatic Data Entry**. Apart from an electronic archive of existing documents, OCR can also improve our productivity in the form of automatic data entry. Some industries involve time-consuming data type-in, e.g. express orders written by customers in the delivery industry, and handwritten information sheets in the financial and insurance industries. Applying OCR techniques can accelerate the data entry process as well as protect customer privacy. Some companies have already been using this technologies, e.g. SF-Express. Another potential application is note taking, such as NEBO, a note-taking software on tablets like iPad that can perform instant transcription as user writes down notes.

**自动数据记录**。除了现有文档的电子存档，OCR还可以通过自动数据录入来提高生产力。一些工业应用中的数据录入非常耗时，如物流行业中客户写的快递订单，金融和保险行业中的手写信息单。使用OCR技术可以加速数据录入过程，也可以保护客户隐私。一些公司已经正在使用这种技术，如顺风快递。另一个潜在的应用是做笔记，如NEBO，一种平板上的笔记软件，可以在用户写下笔记时立刻进行转录。

**Identity Authentication**. Automatic identity authentication is yet another field where OCR can give a full play to. In fields such as Internet finance and Customs, users/passengers are required to provide identification (ID) information, such as identity card and passport. Automatic recognition and analysis of the provided documents would require OCR that reads and extracts the textual content, and can automate and greatly accelerate such processes. There are companies that have already started working on identification based on face and ID card, e.g. Megvii(Face++).

**身份验证**。自动身份验证是OCR可以完全应用的另一个领域。在互联网金融和海关领域，用户/乘客需要提供身份信息(ID)，如身份证和护照。自动识别和分析这些提供的文档需要OCR技术来阅读并提取其中的文本内容，而且OCR可以自动化并极大的加速这些过程。已经有企业开始进行基于人脸和ID卡的身份验证工作，如Megvii(Face++)。

**Augmented Computer Vision**. As text is an essential element for the understanding of scene, OCR can assist computer vision in many ways. In the scenario of autonomous vehicle, text-embedded panels carry important information, e.g. geo-location, current traffic condition, navigation, and etc.. There have been several works on text detection and recognition for autonomous vehicle [91], [92]. The largest dataset so far, CTW [161], also places extra emphasis on traffic signs. Another example is instant translation, where OCR is combined with a translation model. This can be extremely helpful and time-saving as people travel or consult documents written in foreign languages. Google’s Translate application can perform such instant translation. A similar application is instant text-to-speech equipped with OCR, which can help those with visual disability and those who are illiterate [2].

**增强计算机视觉**。由于文本是理解场景的关键元素，OCR可以很多方式协助计算机视觉。在自动驾驶的场景中，嵌入了文本的面板带有重要的信息，如地理位置信息，目前的交通状况，导航信息，等等。已经有几项自动驾驶领域的文本检测和识别工作[91,92]。迄今位置最大的数据集，CTW[161]，也额外重点强调交通指示。另一个例子是即时翻译，OCR与翻译模型结合起来使用。这在人们旅行，或参阅外语文档时可以非常有用处，节省人们的时间。谷歌翻译的应用可以进行这样的即时翻译。类似的应用是即时文本到语音的应用，这可以帮助那些有视觉障碍的，和那些不认识字的人[2]。

**Intelligent Content Analysis**. OCR also allows the industries to perform more intelligent analysis, mainly for platforms like video-sharing websites and e-commerce. Text can be extracted from images and subtitles as well as real-time commentary subtitles (a kind of floating comments added by users, e.g. those in Bilibili and Niconico). On the one hand, such extracted text can be used in automatic content tagging and recommendation system. They can also be used to perform user sentiment analysis, e.g. which part of the video attracts the users most. On the other hand, website administrator can impose supervision and filtration for inappropriate and illegal content, such as terrorist advocacy.

**智能内容分析**。OCR也使得业界可以进行更智能的分析，主要面向视频分享的网站和电子商务这样的平台。可以从图像、字幕以及实时弹幕中提取文本。一方面，这样提取的文本可用于自动内容标注和推荐系统，也可以用于分析用户情感，如视频的哪一部分最吸引用户。另一方面，网站管理员可以监督和过滤不合适的和非法的内容，如支持恐怖主义的内容。

## 6 Conclusion and Discussion 结论和讨论

### 6.1 Status Quo 现状

The past several years have witnessed the significant development of algorithms for text detection and recognition. As deep learning rose, the methodology of research has changed from searching for patterns and features, to architecture designs that takes up challenges one by one. We’ve seen and recognize how deep learning has resulted in great progress in terms of the performance of the benchmark datasets. Following a number of newly-designed datasets, algorithms aimed at different targets have attracted attention, e.g. for blurred images and irregular text. Apart from efforts towards a general solution to all sorts of images, these algorithms can be trained and adapted to more specific scenarios, e.g. bankcard, ID card, and driver’s license. Some companies have been providing such scenario-specific APIs, including Baidu Inc., Tencent Inc. and Megvii Inc.. Recent development of fast and efficient methods [107], [168] has also allowed the deployment of large-scale systems [9]. Companies including Google Inc. and Amazon Inc. are also providing text extraction APIs.

过去几年见证了文本检测和识别算法的显著进展。随着深度学习的崛起，研究方法开始从搜索模式和特征，变为架构设计，一个一个的接受挑战。我们看到了深度学习是怎样取得了极大的成功的，表现在基准测试数据集的性能上。随着几个新设计的数据集的提出，不同目标的算法得到了关注，如，模糊图像和不规则文本。除了对各种图像的一般解决方案，这些算法可以经过训练适应更特殊的场景，如，银行卡，ID卡和驾照。一些公司开始提供这些特定场景的APIs，包括百度、腾讯和旷视。最近快速高效方法的进展[107,168]也使得部署大规模系统成为可能[9]。包括谷歌和亚马逊这样的企业也提供文本提取的APIs。

Despite the success so far, algorithms for text detection and recognition are still confronted with several challenges. While human have barely no difficulties localizing and recognizing text, current algorithms are not designed and trained effortlessly. They have not yet reached human-level performance. Besides, most datasets are monolingual. We have no idea how these models would perform on other languages. What exacerbates it is that, the evaluation metrics we use today may be far from perfect. Under PASCAL evaluation, a detection result which only covers slightly more than half of the text instance would be judged as successful as it passes the IoU threshold of 0.5. Under DetEval, one can manually enlarge the detected area to meet the requirement of pixel recall, as DetEval requires a high pixel recall (0.8) but rather low pixel precision (0.4). Both cases would be judged as failure from oracle’s viewpoint, as the former can not retrieve the whole text, while the later encloses too much background. A new and more appropriate evaluation protocol is needed.

尽管目前取得了一定的成功，文本检测和识别算法还是遇到了几个挑战。虽然人类在定位和识别文本上几乎毫无困难，但现在的算法设计和训练还是很费力的，还没有达到人类水平的性能。另外，多数数据集都是单语言的。我们不知道这些模型在其他语言上会表现如何。我们使用的评估标准远未达到完美的标准，这更加剧了这种状况。在PASCAL评估标准下，检测结果如果覆盖了文本实例超过一半，就会被认为是成功的，因为超过了IOU阈值0.5。在DetEval下，可以手工增大检测到的区域，满足像素召回的需求，因为DetEval需要很高的像素召回率(0.8)，但较低的精准率(0.4)。两种情况在oracle视角下都是失败的，因为前者不能检索整个文本，而后者包括了太多的背景。需要一种新的更合适的评估方案。

Moreover, few works except for TextSnake [85] have considered the problem of generalization ability across datasets. Generalization ability is important as we aim to some application scenarios would require the adaptability to changing environments. For example, instant translation and OCR in autonomous vehicles should be able to perform stably under different situations: zoomed-in images with large text instances, far and small words, blurred words, different languages and shapes. However, these scenarios are only represented by different datasets individually. We would expect a more diverse dataset.

而且，除了TextSnake[85]，很少有工作考虑数据集间的泛化能力。泛化能力非常重要，因为我们一些应用场景会需要转换环境的适应性。比如，即时翻译和自动驾驶中的OCR应当在不同情况下表现稳定：放大的图像中的很大的文字实例，很小的单词，模糊的单词，不同的语言和形状。但是，这些场景只是在不同数据集中单独得到了很好的体现。我们期望会有一个更多样化的数据集。

Though synthetic data (such as SynthText [38]) has been widely adopted in recent scene text detection and recognition algorithms. The diversity and realistic degree are actually quite limited. To develop scene text detection and recognition models with higher accuracy and generalization ability, it is worthy of exploration to build more powerful engines for text image synthesis.

虽然合成数据（比如SynthText[38]）已经在最近的场景文本检测和识别算法中得到了广泛的应用，但多样性和与实际使用接近的程度还很有限。为开发更高准确率和泛化能力的场景文本检测和识别模型，值得探索构建更强力的文本图像合成的引擎。

Another shortcoming of deep learning based methods for scene text detection and recognition lies in their efficiency. Most of the current state-of-the-art systems are not able run in real-time when deployed on computers without GPUs or mobile devices. However, to make text information extraction techniques and services anytime anywhere, current systems should be significantly speed up while maintaining high accuracy.

基于深度学习的场景文字检测和识别的另一个缺陷是其计算效率。多数目前最好的系统，部署在没有GPU的计算机或移动设备上时，都不能实时运行。但是，为了使文本信息提取技术和服务随时随地都可用，现在的系统需要在保持准确度的前提下，显著的提升速度。

### 6.2 Future Trends 未来趋势

History is a mirror for the future. What we lack today tells us about what we can expect tomorrow. 历史是未来的镜子，我们现在所缺少的是我们将来可以期待的。

**Diversity among Datasets: More Powerful Model** Text detection and recognition is different from generic object detection in the sense that, it’s faced with unique challenges. We expect that new datasets aimed at new challenges, as we have seen so far [15], [63], [163], would draw attention to these aspects and solve more real world problems.

**数据集的多样性：更强大的模型**。文本检测与识别与通用目标检测不同，面临着独特的挑战。我们期待着新的数据集会面向新的挑战，就像我们在[15,63,163]中看到的，会将注意力吸引到这些方面，解决这个世界中的真实问题。

**Diversity inside Datasets: More Robust Model** Despite the success we’ve seen so far, current methods are only evaluated on single datasets after being trained on them separately. Tests of authentic generalization are needed, where a single trained model is evaluated on a more diverse held-out set, e.g. a combination of current datasets. Naturally, a new dataset representing several challenges would also provide extra momentum for this field. Evaluation of cross dataset generalization ability is also preferable, where the model is trained only on one dataset and then tested of another, as done in recent work in curved text [85].

**数据集内部的多样性：更稳健的模型**。尽管我们看到了一些成功，但现在的方法还只是在单个数据集上训练然后评估的。真实的泛化测试非常需要，即将训练的模型在更多样化的保留集上进行评估，如，现有数据集的组合。很自然的，代表几种挑战的新数据集也可以提供这个领域的新的动力。跨数据集泛化能力的评估非常需要，即在一个数据集上训练，在其他数据集上进行测试，就像最近的曲线文本工作[85]做的一样。

**Suitable Evaluation Metrics: a Fairer Play** As discussed above, an evaluation metric that fits the task more appropriately would be better. Current evaluation metrics (DetEval and PASCAL-Eval) are inherited from the more generic task of object detection, where detection results are all represented in rectangular bounding boxes. However, in text detection and recognition, the shapes and orientations matter. Tighter and noiseless bounding region would also be more friendly to recognizers. Neglecting some parts in object detection may be acceptable as it remains semantically the same, but it would be disastrous for the final text recognition results as some characters may be missing, resulting in different words.

**合适的评估标准：公平的比赛**。如上所述，需要更适合这项任务的评估标准。现有的评估标准(DetEval和PASCAL-Eval)都是从更通用的目标检测任务中继承下来的，其中检测结果都用矩形边界框进行表示。但是，在文本检测和识别中，形状和方向都很重要。更紧密和没有噪声的边界框区域会对识别者更友好。忽略目标检测中的某些部分可能可以接受，因为语义上都是一样的，但是会对最终的文本识别造成灾难性的后果，因为一些字符可能缺失，形成不同的单词。

**Towards Stable Performance: as Needed in Security** As we have seen work that breaks sequence modeling methods [162] and attacks that interfere with image classification models [131], we should pay more attention to potential security risks, especially, when applied in security services e.g. identity check.

**稳定的性能：安全所需**。我们看到过破解序列建模的方法[162]，用图像分类模型攻击其接口，我们应当更加注意潜在的安全风险，尤其是在安全领域的应用中，如，身份验证。