# DeepLesion: automated mining of large-scale lesion annotations and universal lesion detection with deep learning

Ke Yan et. al. NIH

## 0. Abstract

Extracting, harvesting, and building large-scale annotated radiological image datasets is a greatly important yet challenging problem. Meanwhile, vast amounts of clinical annotations have been collected and stored in hospitals’ picture archiving and communication systems (PACS). These types of annotations, also known as bookmarks in PACS, are usually marked by radiologists during their daily workflow to highlight significant image findings that may serve as reference for later studies. We propose to mine and harvest these abundant retrospective medical data to build a large-scale lesion image dataset. Our process is scalable and requires minimum manual annotation effort. We mine bookmarks in our institute to develop DeepLesion, a dataset with 32,735 lesions in 32,120 CT slices from 10,594 studies of 4,427 unique patients. There are a variety of lesion types in this dataset, such as lung nodules, liver tumors, enlarged lymph nodes, and so on. It has the potential to be used in various medical image applications. Using DeepLesion, we train a universal lesion detector that can find all types of lesions with one unified framework. In this challenging task, the proposed lesion detector achieves a sensitivity of 81.1% with five false positives per image.

提取，收获和构建大规模标注的放射图像数据集，是非常重要但是有挑战的问题。同时，在医院的PACS系统中已经收集并存储了大量的临床标注。这些类型的标注，也称为PACS中的书签，通常是由放射科医生在其日常工作流程中标注的，以强调显著的图像发现，可能为后续研究作为参考。我们提出对这些大量的回顾性的医学数据进行挖掘和收获，以构建一个大型病变图像数据集。我们的过程是可放大的，需要很少的人工标注努力。我们挖掘了我们研究所的书签，开发了DeepLesion，这个数据集在32120个CT slices中有32735个病变，是4427个唯一的病人的10594个研究。在这个数据集中有很多病变类型，比如肺结节，肝癌，淋巴结肿大，等。在各种医学图像应用中，都有使用的可能。使用DeepLesion，我们训练了一个通用的病变检测器，采用统一的框架即可发现所有类型的病变。在这个有挑战性的任务中，提出的病变检测器取得了81.1%的敏感度，每幅图像有5个假阳性。

**Keywords:** medical image dataset; lesion detection; convolutional neural network; deep learning; picture archiving and communication system; bookmark.

## 1. Introduction

Computer-aided detection/diagnosis (CADe/CADx) has been a highly prosperous and successful research field in medical image processing. Recent advances have attracted much interest to the application of deep learning approaches. Convolutional neural network (CNN) based deep learning algorithms perform significantly better than conventional statistical learning approaches combined with handcrafted image features. However, these performance gains are often achieved at the cost of requiring tremendous amounts of labeled training data. Unlike general computer vision tasks, medical image analysis currently lacks a large-scale annotated image dataset (comparable to ImageNet and MS COCO), which is mainly because the conventional methods for collecting image labels via Google search + crowd-sourcing from average users cannot be applied in the medical image domain, as medical image annotation requires extensive clinical expertise.

计算机辅助检测/诊断(CADe/CADx)是医学图像处理中一个非常活跃和成功的研究领域。最近的进展已经吸引了很多兴趣研究深度学习方法的应用。基于CNN的深度学习算法表现比传统的统计学习方法与手工设计的图像特征结合的算法要好很多。但是，这些性能提升通常是在需要大量标注的训练数据的代价下获得的。与通用的计算机视觉任务不同，医学图像分析目前缺少一个大规模标注图像数据集（与ImageNet和MS COCO可比较），这主要是因为传统的收集图像标签的方法是通过Google搜索+从普通用户众包，这不能应用到医学图像领域，因为医学图像标注需要大量的临床专业知识。

Detection and characterization of lesions are important topics in CADe/CADx. Existing detection/characterization algorithms generally target one particular lesion type, such as skin lesions, lung nodules, liver lesions, sclerotic lesions, and colonic polyps. While some common types of lesions receive much attention, vast infrequent types are ignored by most CADe programs. Besides, studying one lesion type at a time differs from the method radiologists routinely apply to read medical images and compile radiological reports. In practice, multiple findings can be observed and are often correlated. For instance, metastases can spread to regional lymph nodes or other body parts. By obtaining and maintaining a holistic picture of relevant clinical findings, a radiologist will be able to make a more accurate diagnosis. However, it remains challenging to develop a universal or multicategory CADe framework, capable of detecting multiple lesion types in a seamless fashion, partially due to the lack of a multicategory lesion dataset. Such a framework is crucial to building an automatic radiological diagnosis and reasoning system.

病变的检测和表征在CADe/CADx中是很重要的话题。现有的检测/表征算法一般以特定病症类型为目标，比如皮肤病症，肺结节，肝病，硬化病，结肠息肉。一些特定类型的病症会得到很多关注，但大量不常见的类型都被多数CADe程序忽视了。另外，一次研究一种病症类型，与放射科医生日常研读医学图像并编纂放射科报告的方法不同。在实践中，可以观察到多个发现，并通常是相关的。比如，转移瘤可以扩散到区域淋巴结或其他身体部位。放射科医生通过得到并维护相关临床发现的整体图像，可以进行更精确的诊断。但是，要开发一种统一的或多类别的CADe框架，可以以一种无缝的方式检测多种病变类型，这是非常有挑战性的，部分是因为缺少多类别病变数据集。这样一个框架对构建一种自动放射诊断和推理系统，是很关键的。

In this paper, we attempt to address these challenges. First, we introduce a paradigm to harvest lesion annotations from bookmarks in a picture archiving and communication system (PACS) with minimum manual effort. Bookmarks are metadata marked by radiologists during their daily work to highlight target image findings. Using this paradigm, we collected a large-scale dataset of lesions from multiple categories (Fig. 1). Our dataset, named DeepLesion, is composed of 32,735 lesions in 32,120 bookmarked CT slices from 10,594 studies of 4427 unique patients. Different from existing datasets, it contains a variety of lesions including lung nodules, liver lesions, enlarged lymph nodes, kidney lesions, bone lesions, and so on. DeepLesion is publicly released and may be downloaded from Ref. 11.

本文中，我们试图去解决这个挑战。首先，我们提出了一种从一个PACS系统的书签中收获病变标注的方式，需要的人工努力是很少的。书签是放射科医生在其日常工作中标注的元数据，用于强调目标图像的发现。使用这种方法，我们收集了一个大型病变数据集，包含多个类型（见图1）。我们的数据集，名为DeepLesion，是由32735个病变组成，是4427个不同病人的10594的研究的32120个带有书签的CT slices。与现有的数据集不同，这个数据集包含了很多病变类型，包括肺结节，肝癌，淋巴结肿大，肾病变，骨病变等。DeepLesion已经公开，可以从Ref. 11中下载。

Using this dataset, we develop an automatic lesion detection algorithm to find all types of lesions with one unified framework. Our algorithm is based on a regional convolutional neural network (faster RCNN). It achieves a sensitivity of 77.31% with three false positives (FPs) per image and 81.1% with five FPs. Note that the clinical bookmarks are not complete annotations of all significant lesions on a radiology image. Radiologists typically only annotate lesions of focus to facilitate follow-up studies of lesion matching and growth tracking. There are often several other lesions left without annotation. We empirically find that a large portion of the so-called FPs is actually true lesions, as demonstrated later. To harvest and distinguish those clinician unannotated lesions from “true” FPs will be an important future work.

使用这个数据集，我们提出了一种自动病变检测算法，使用一个统一的框架来找到所有类型的病变。我们的算法是基于一种区域CNN(Faster RCNN)。算法获得了77.31%的敏感度，每幅图像有3个假阳性，在81.1%的敏感度下有5个假阳性。注意，临床标签并不是一幅放射图像中所有显著病症的完全标注。放射科医生一般只标注关注的病变，以促进后续研究的病症匹配和增长追踪。通常还会有其他几个病变并没有标注。我们通过经验发现，很大一部分这些所谓的假阳性，实际上是真的病变，后面会证明。为从真的假阳性中收获并区分这些临床上未标注的病变，是未来的一个重要工作。

## 2 Materials and Methods

In this section, we will first introduce bookmarks as radiology annotation tools. Then, we will describe the setup procedure and data statistics of the DeepLesion dataset. The proposed universal lesion detector will be presented afterward.

本节中，我们首先提出书签作为放射标注工具。然后，我们会描述设置的过程，和DeepLesion数据集的数据统计结果。提出的统一病变检测器会随后给出。

### 2.1 Bookmarks

Radiologists routinely annotate and measure hundreds of clinically meaningful findings in medical images, which have been collected for two decades in our institute’s PACS. Figure 2 shows a sample of a bookmarked image. Many of the bookmarks are either tumors or lymph nodes measured according to the response evaluation criteria in solid tumors (RECIST) guidelines. According to RECIST, assessment of the change in tumor burden is an important feature of the clinical evaluation of cancer therapeutics. Therefore, bookmarks usually indicate critical lesion findings. It will be extremely useful if we can collect them into a dataset and develop CADe/CADx algorithms to detect and characterize them.

放射科医生会在医学图像中例行标注并衡量数百个临床上有意义的发现，在我们机构的PACS中已经收集了超过了二十年。图2给出了带有书签的图像的例子。很多书签是肿瘤或淋巴结，根据RECIST准则进行的衡量。根据RECIST，肿瘤负荷变化的评估，是肿瘤治疗临床评估的重要特征。因此，标签通常指出了关键的病症发现。如果我们可以将其收集到一起，成为一个数据集，以开发CADe/CADx算法，对其进行检测和表征，会非常有用。

To get an overview of the bookmarks, we analyze them by year, image modality, and annotation tool. From Fig. 3, we can see that the number of studies with bookmarks increases each year with a boost in 2015. This indicates that bookmarks are becoming more and more popular as radiologists discover that it is a helpful tool. By collecting these bookmarks every year, we can easily obtain a large-scale lesion dataset. The image modalities of the bookmarks are shown in Fig. 4. CT images make up the largest percentage, followed by MR and nuclear medicine.

为对标签进行概览，我们通过年份，图像模态，和标注工具对其进行分析。从图3中，我们可以看到带有标签的studies的数量逐渐递增，在2015年有一个突破。这说明书签正变得越来越流行，因为放射科医生发现，这是一个很有用的工具。通过逐年收集这些书签，我们可以很容易的得到一个大规模病变数据集。标签的图像模态如图4所示。CT图像占了最大的比重，然后是MR和核医学图像。

Radiologists can use various annotation tools to annotate the bookmarks, including arrows, lines, ellipses, bidimensional RECIST diameters, segmentations, and text. We downloaded all the bookmarks in CT studies and counted the usage of the tools (Fig. 5). RECIST diameters were applied most frequently. Each RECIST-diameter bookmark consists of two lines: one measuring the longest diameter of the lesion and the second measuring its longest perpendicular diameter in the plane of measurement. Examples can be found in Fig. 2. The RECIST-diameter bookmarks can tell us the exact location and size of a lesion. A line bookmark contains only one length measurement, which may be the longest or shortest diameter of a lesion, or even a measurement of a nonlesion. For line, ellipse, text, or arrow bookmarks, while we can infer the approximate location of a lesion, the exact location and/or size is not available.

放射科医生可以使用各种标注工具来对书签进行标注，包括箭头，线段，椭圆，双维度RECIST半径，分割和文字。我们下载了CT studies中所有的书签，对工具的使用进行了计数（图5）。RECIST半径应用的最为频繁。每个RECIST半径的书签包括两条线：一个是病变的最长半径，第二个是度量平面垂直方向上的最长半径。例子如图2所示。RECIST半径的书签可以告诉我们，一个病变的确切位置和大小。一个线书签只包含一个长度度量，可能是病变最长的或最短的半径，甚至是一个非病变的度量。对于线段，椭圆，文本或箭头书签，我们可以推断一个病变的大致位置，但确切位置和/或大小是不可用的。

### 2.2 DeepLesion Dataset

Because bookmarks can be viewed as annotations of critical lesions, we collected them to build a lesion dataset for CADe/CADx algorithms. This research has been approved by our Institutional Research Board. Without loss of generality, currently, we only focus on CT bookmarks, which are the most abundant. As for the annotation tools, now, we only consider RECIST diameters. Until January 2017, we have collected 33,418 bookmarks of this type. After filtering some noisy bookmarks (detailed in Sec. 2.2.1), we obtained the DeepLesion dataset with 32,120 axial slices from 10,594 CT studies of 4427 unique patients. There are one to three bookmarks in each slice, for a total of 32,735 bookmarks. The dataset will be introduced in detail from the following aspects: setup procedure, data statistics, advantages, limitations, and potential applications.

因为书签可以视为关键病变的标注，我们收集以构建病变数据集，用于开发CADe/CADx算法。这项研究已经由我们的机构研究委员会批准。不失一般性，目前，我们只关注CT书签，这是最充分的数据。至于标注工具，现在我们只考虑RECIST半径。到2017年1月，我们收集了33418个这种类型的书签。在过滤掉一些含噪的书签后（详见2.2.1），我们得到的DeepLesion数据集有32120个轴向的slices，是4427个病人的10594次studies。在每个slice中，有1到3个书签，总计有32735个书签。下面从以下方面详细介绍数据集：设置过程，数据统计，优势，局限性和可能的应用。

#### 2.2.1 Setup procedure

First, we acquired the accession numbers of the CT studies with bookmarks by querying the PACS (Carestream Vue V12.1.6.0117). Then, the bookmarks were downloaded according to them using a Perl script provided by the PACS manufacturer. We selected only the RECIST-diameter ones, which are represented by four vertices. Most of them were annotated on the axial plane. We filtered the nonaxial ones, and then converted the vertices to image coordinates. The conversion was done by first subtracting the “ImagePositionPatient” (extracted from the DICOM file) from each vertex and then dividing the coordinates of each vertex with the pixel spacing.

首先，我们通过查询PACS，获得了带有书签的CT studies的检索号。然后，使用一个PACS厂商提供的Perl脚本根据检索号进行书签下载。我们只选择了哪些有RECIST半径的，由四个顶点来表示。多数是在一个轴向面上标注的。我们滤除掉了那些非轴向的，然后将顶点转换到图像坐标系中。这个转换过程，首先是从每个顶点上减去从DICOM文件中提取出的ImagePositionPatient，然后将每个顶点的坐标除以像素间距。

The CT volumes that contain these bookmarks were also downloaded. We used MATLAB to convert each image slice from DICOM files to 16-bit portable network graphics (PNG) files for lossless compression and anonymization. Real patient IDs, accession numbers, and series numbers were replaced by self-defined indices of patient, study, and series (starting from 1) for anonymization. We named each volume with the format “{patient index}_{study index}_{series index}.” Note that one patient often underwent multiple CT examinations (studies) for different purposes or follow-up. Each study contains multiple volumes (series) that are scanned at the same time point but differ in image filters, contrast phases, etc. Every series is a three-dimensional (3-D) volume composed of tens to hundreds of axial image slices. Metadata, such as pixel spacing, slice interval, intensity window, and patient gender and age, were also recorded. The slice intervals were computed by differentiating the “ImagePositionPatient” (extracted from DICOM) of neighboring slices. We made sure that the slice indices increased from head to feet.

也下载了包含这些书签的CT。我们使用MATLAB来转换每个图像slice，从DICOM文件到16位的PNG文件，以进行无损压缩和匿名化。真实的患者ID，索引号和series号都替换为自定义的病人、study和series索引（从1开始），以进行匿名化。我们对每个volume命名为{patient index}_{study index}_{series index}的格式。注意一个病人通常会经历多次CT检查(studies)，每次是不同的目的或随访。每个study包含多个volume(series)，是在同一时间点扫描的，但图像滤波器、对比度相位等会不同。每个series是一个3D volume，由数十到上百个轴向图像slices组成。元数据，如像素间距，层厚，灰度窗和病人的性别和年龄，也进行了记录。通过相邻slices的ImagePositionPatient相减，可以得到层厚。我们确保，slice索引是从头到脚递增的。

To facilitate applications such as computer-aided lesion detection, we converted the RECIST diameters into bounding-boxes. Denote the four vertices as (x11;y11), (x12;y12), (x21;y21), and (x22;y22). The z coordinates are omitted since the vertices are on the same axial plane. A bounding box (left, top, right, and bottom) was computed to enclose the lesion measurement with 5-pixel padding in each direction, i.e., (xmin−5;ymin−5;xmax+5;ymax+5), where xmin = min(x11; x12;x21;x22), xmax = max(x11;x12;x21;x22), and similarly for ymin and ymax. The 5-pixel padding was applied to cover the lesion’s full spatial extent.

为促进计算机辅助病变检测这样的应用，我们将RECIST半径转换到边界框。将四个顶点表示为(x11;y11), (x12;y12), (x21;y21)和(x22;y22)。忽略了z坐标，因为顶点是在同样的轴向平面上的。计算一个边界框(left, top, right, bottom)，以包括病变的度量，并在每个方向上都有5个像素的补充，即(xmin−5;ymin−5;xmax+5;ymax+5)，其中xmin = min(x11; x12;x21;x22)，xmax = max(x11;x12;x21;x22)，ymin和ymax是类似的。5个像素的补充是为了覆盖病变的完整空间范围。

There are a limited number of incorrect bookmarks. For example, some bookmarks are outside the body, which is possibly caused by annotation error by the user. To remove these label noises, we computed the area and width-height-ratio of each bounding-box, as well as the mean and standard deviation of the pixels inside the box. Boxes that are too small/large/flat/dark or small in intensity range were manually checked. Another minor issue is duplicate annotations. A small number of lesions were bookmarked more than once possibly by different radiologists. We merged bounding-boxes that have more than 60% overlap by averaging their coordinates.

有一定数量的不正确的书签。比如，一些书签是在身体外面的，这可能是由于用户标注错误造成的。为去除这些标签噪声，我们计算了每个边界框的面积和宽高比，以及框内像素的均值和标准差值。太小/太大/太平坦/太暗的框，或灰度范围太小的，都经过人工检查。另一个小问题是重复的标注。少量病变是由多个放射科医生添加了多次书签的。我们把重叠超过60%的边界框都进行了合并，对其坐标值进行了平均。

#### 2.2.2 Data statistics 数据统计

The slice intervals of the CT studies in the dataset range between 0.25 and 22.5 mm. About 48.3% of them are 1 mm and 48.9% are 5 mm. The pixel spacings range between 0.18 and 0.98 mm∕pixel with a median of 0.82 mm∕pixel. Most of the images are 512 × 512 and 0.12% of them are 768 × 768 or 1024 × 1024. Figure 6 displays the distribution of the sizes of the bounding-boxes. The median values of the width and height are 22.9 and 22.3 mm, respectively. The diameter range of the lesions is 0.42 to 342.5 mm for long diameter and 0.21 to 212.4 mm for short diameter.

数据集中CT studies的层厚范围在0.25mm到22.5mm之间。48.3%的是1mm，48.9%的是5mm。像素间距范围是在0.18到0.98mm/像素，中位值为0.82mm/像素。大多数图像大小为512 × 512，0.12%的图像为768 × 768或1024 × 1024。图6展示了大小和边界框的分布。宽度和高度的中位值分别为22.9和22.3 mm。病变的半径范围，长轴为0.42到342.5mm，短轴为0.21到212.4mm。

To explore the lesion types in DeepLesion, we randomly selected 9816 lesions and manually labeled them into eight types: lung (2426), abdomen (2166), mediastinum (1638), liver (1318), pelvis (869), soft tissue (677), kidney (490), and bone (232). These are coarse-scale attributes of the lesions. The mediastinum type mainly consists of lymph nodes in the chest. Abdomen lesions are miscellaneous ones that are not in liver or kidney. The soft tissue type contains lesions in the muscle, skin, and fat. Examples of the lesions in the eight types can be found in Fig. 1, where a subset of the lesions is drawn on a scatter map to show their types and relative body coordinates. The map is similar to a frontal view of the human body. To obtain the approximate z-coordinate of each lesion, we adopted the unsupervised body part regressor to predict the slice score of each image slice. From Fig. 1, we can find that the dataset is clinically diversified.

为探索DeepLesion中的病变类型，我们随机选取了9816个病变，手工为其添加标签，共有8个类别：肺(2426)，腹(2166)，纵隔(1638)，肝(1318)，盆(869)，软组织(677)，肾(490)和骨(232)。这些是病变的粗糙尺度的属性。纵隔类型的主要包括包括胸部淋巴结。腹部病变是不在肝脏或肾脏中各种类型的病变。软组织类型的包含在肌肉、皮肤和脂肪中的病变。8种类型的病变的类型，在图1中有示例，其中一些病变的子集画成了散点图，以展示其类型和相对身体坐标系。这个图与人身体的前视图很像。为获得每个病变的近似z坐标，我们采用了无监督的身体部位回归器，来预测每个slice的分数。从图1中，我们可以发现，这个数据集在临床上是非常多样化的。

### 2.3 Universal Lesion Detection

In this section, we will introduce our universal lesion detector in detail. It is trained on DeepLesion, thus can detect all types of lesions that radiologists are interested in measuring with one unified framework. The algorithm is adapted from the faster RCNN method. Its flowchart is illustrated in Fig. 7.

本节中，我们会详细介绍我们的统一病变检测器。这是在DeepLesion上训练的，因此可以使用一个统一的框架，检测放射科医生感兴趣的所有类型的病变。算法是从Faster RCNN修改得到的。其流程图如图7所示。

#### 2.3.1 Image preprocessing

The 12-bit CT intensity range was rescaled to floating-point numbers in [0,255] using a single windowing (−1024 to 3071 HU) that covers the intensity ranges of lung, soft tissue, and bone. Every image slice was resized to 512×512. To encode 3-D information, we used three axial slices to compose a three-channel image and input it to the network. The slices were the center slice that contains the bookmark and its neighboring slices interpolated at 2-mm slice intervals. No data augmentation was used since our dataset is large enough to train a deep neural network.

12-bit的CT灰度范围使用了单个窗(−1024 to 3071 HU)，覆盖了肺、软组织和骨骼，缩放到了[0,255]的浮点数。每幅图像slice大小统一到512×512。为包含3D信息，我们使用三个轴向slices以组成三通道图像，输入到网络中。这些slices的中间slice包含书签信息，其相邻的slices是以2mm层厚插值得到的。没有使用数据扩增，因为我们的数据集足够大，可以训练一个CNN。

#### 2.3.2 Network architecture

The VGG-16 model was adopted as the backbone of the network. We also compared deeper architectures including ResNet-50 and DenseNet-121 and the shallower AlexNet on the validation set and observed that VGG-16 had the highest accuracy. As shown in Fig. 7, an input image was first processed by the convolutional blocks in VGG-16 (Conv1–Conv5) to produce feature maps. We removed the last two pooling layers (pool4 and pool5) to enhance the resolution of the feature map and to increase the sampling ratio of positive samples (candidate regions that contain lesions), since lesions are often small and sparse in an image.

采用VGG-16作为网络的骨干。我们还在验证集上比较了更深的架构，包括ResNet-50和DenseNet-121和更浅的AlexNet，发现VGG-16有最高的准确率。如图7所示，输入图像首先由VGG16的卷积层(Conv1-Conv5)处理，以产生特征图。我们去掉了最后两个pooling层(pool4和pool5)，以增强特征图的分辨率，增加正样本的采样率（即包含病变的候选区域），因为在一幅图像中，病变通常很小而且很稀疏。

Next, a region proposal network parsed the feature maps and proposes candidate lesion regions. It estimated the probability of “lesion/nonlesion” on a fixed set of anchors on each position of the feature maps. At the same time, the location and size of each anchor were fine-tuned via bounding box regression. After investigating the sizes of the bounding-boxes in DeepLesion, we used five anchor scales (16, 24, 32, 48, and 96) and three anchor ratios (1:2, 1:1, and 2:1) in this paper.

下一步，有一个区域建议网络对特征图进行解析，提出候选病变区域。网络在特征图的每个位置上，在一个固定的锚框集上估计了病变/非病变的概率。同时，每个锚框的位置和大小都通过边界框回归进行了精调。在研究了DeepLesion中的边界框的大小之后，我们在本文中使用了5个锚框的大小(16, 24, 32, 48, and 96)，和3个锚框的比率(1:2, 1:1, 2:1)。

Afterward, the lesion proposals and the feature maps were sent to a region of interest (RoI) pooling layer, which resampled the feature maps inside each proposal to a fixed size (7 × 7 in this paper). These feature maps were then fed into two convolutional layers, Conv6 and Conv7. Here, we replaced the original 4096D fully-connected (FC) layers in VGG-16 so that the model size was cut to 1/4 while the accuracy was comparable. Conv6 consisted of 512 3 × 3 filters with zero padding and stride 1. Conv7 consisted of 512 5×5 filters with zero padding and stride 1. Rectified linear units were inserted after the two convolutional layers. The 512D feature vector after Conv7 then underwent two FC layers to predict the confidence scores for each lesion proposal and ran another bounding box regression for further fine-tuning. Nonmaximum suppression (NMS) was then applied to the fine-tuned boxes to generate the final predictions. The intersection-over-union (IoU) thresholds for NMS were 0.7 and 0.3 in training and testing, respectively.

然后，病变的建议和特征图送入一个ROI池化层，将每个建议中的特征图重采样到一个固定的大小（本文中是7 × 7）。这些特征图然后送入两个卷积层，Conv6和Conv7。这里，我们将VGG16中原来的4096维全连接层替换掉，这样模型大小可以变为1/4，同时准确率还接近。Conv6由512个3 × 3滤波器组成，有补零，步长为1。Conv7由512个5×5的滤波器，有补零，步长为1。在这两个卷积层后，有两个ReLU层。Conv7中的512维的特征向量，然后经历2个全连接层，预测每个病变候选的信心分数，经过另一个边界框回归用于进一步的精调。然后用NMS来精调框，以生成最终的预测。在训练和测试中，NMS使用的IoU阈值分别为0.7和0.3。

#### 2.3.3 Implementation details

The proposed algorithm was implemented using MXNet. The weights in Conv1 to Conv5 were initialized with the ImageNet pretrained VGG-16 model, whereas all the other layers were randomly initialized. During training, we fixed the weights in Conv1 and Conv2. The two classification and two regression losses were jointly optimized. This end-to-end training strategy is more efficient than the four-step strategy in the original faster RCNN implementation. Each mini-batch had eight images. The number of region proposals per image for training was 32. We adopted the stochastic gradient descent optimizer and set the base learning rate to 0.002, and then reduced it by a factor of 10 after six epochs. The network converged within eight epochs.

提出的算法使用MXNet实现。Conv1到Conv5的权重使用ImageNet预训练的VGG-16模型初始化，而所有其他层都是随机初始化的。在训练中，我们固定Conv1和Conv2的权重。两个分类损失和两个回归损失进行了同时优化。这种端到端的训练策略，比原始Faster RCNN实现中的四步策略更加高效。每个mini-batch有8幅图像。每幅图像中用于训练的区域候选的数量为32。我们采用SGD优化器，设基础学习速率为0.002，然后在6轮训练后将其除以10。网络在8轮训练内就收敛了。

## 3 Results

To evaluate the proposed algorithm, we divided DeepLesion into training (70%), validation (15%), and test (15%) sets by randomly splitting the dataset at the patient level. The proposed algorithm only took 34 ms to process a test image on a Titan X Pascal GPU. Here, we report the free receiver operating characteristic (FROC) curves on the test set in Fig. 8. The sensitivity reaches 81.1% when there are five FPs on average on each image. In addition, the performance steadily improves as more training samples are used. As a result, the accuracy is expected to be better as we harvest more data in the future.

为评估提出的算法，我们将DeepLesion分成训练集(70%)，验证集(15%)和测试集(15%)，在病人的级别随机分割数据集。提出的算法在Titan X Pascal GPU上处理一幅图像耗时只有34ms。这里，我们在图8中给出测试集上的FROC曲线。在敏感度达到81.1%时，在每幅图像中平均有5个假阳性。另外，当使用更多的训练样本时，性能会持续改进。所以当我们在将来继续利用更多数据时，结果准确率会更好。

The FROC curves of different lesion types are shown in Fig. 9. Note that our network does not predict the type of each detected lesion, so the x-axis in Fig. 9 is the average number of FPs of all lesion types per image. Thus, the curves could not be directly compared with the literature. Instead, they reflect the relative performance of different types and sizes. From Fig. 9, we can find that liver, lung, kidney, and mediastinum lesions are among the easiest ones to detect. This is probably because their intensity and appearance is relatively distinctive from the background. It is more difficult to detect abdominal and pelvic lesions, where normal and abnormal structures including bowel and mesentery clutter the image and may have similar appearances (Figs. 18–21). Soft tissue and bone lesions have fewer training samples and small contrast with normal structures, thus have the lowest sensitivity.

图9中给出了不同病变类型的FROC曲线。注意，我们的网络并不预测检测到的病变的类型，所以图9中的x轴是每幅图像中所有病变类型的平均假阳性数量。因此，这些曲线不能直接与其他文献进行比较。它们反应的是不同类型和大小的相对性能。从图9中，我们可以发现，肝、肺、肾和纵隔病变是最容易检测的。这可能是因为其灰度和外观与背景相比相对最明显。腹部和盆部的病变更难检测到，在这些部位中，正常的器官和非正常的结构在图像中混杂在一起，包括肠管和隔膜，这些都有类似的外观（图18-21）。软组织和骨病变的训练样本较少一些，与正常组织间的对比度很小，因此敏感度更低。

The FROC curves of different lesion sizes are shown in Fig. 10. The size is computed by averaging the long and short diameters. In Fig. 10, it is not surprising that small lesions (<10 mm) are harder to detect. It is also easy to find very large (≥50 mm) lesions. However, when lesion size is between 10 and 50 mm, the sensitivity is not proportional with lesion size, which is possibly because detection accuracy can be affected by multiple factors, such as lesion size, lesion type, number of training samples, etc. The algorithm performs the best when the lesion size is 15 to 20 mm.

不同病变大小的FROC曲线如图10所示。尺寸的计算是将长轴和短轴进行平均，在图10中，较小的病变(<10mm)更难检测，这并不令人意外。很大的病变(≥50 mm)很容易发现。但是，当病变大小在10mm和50mm之间时，敏感度与病变大小并不成正比，这可能是因为检测准确率会受多个因素影响，如病变大小，病变类型，训练样本数量，等。当病变大小在15mm至20mm之间时，算法性能最好。

The detection accuracy also depends on the selected IoU threshold. From Fig. 11, we can find that the sensitivity decreases if the threshold is set higher.

检测准确率还依赖于选择的IoU阈值。从图11中，我们还可以发现，如果阈值设置的较高，那么敏感度就会下降。

Some qualitative results are randomly chosen from the test set and are shown in Figs. 12–21. The figure shows examples of true positives, FPs, and false negatives (FNs).

从测试集中随机选择了一些定性的结果，如图12-21所示。图中给出了真阳性，假阳性和假隐性的例子。

## 4 Discussion

### 4.1 DeepLesion Dataset

#### 4.1.1 Advantages

Compared to most other lesion medical image datasets that consist of only certain types of lesions, one major feature of our DeepLesion database is that it contains all kinds of critical radiology findings, ranging from widely studied lung nodules, liver lesions, and so on, to less common ones, such as bone and soft tissue lesions. Thus, it allows researchers to:

其他医学图像数据集大多数包含的是特定类型的病变，与之相比，DeepLesion数据集的一个主要特征是，包含了所有类型的放射学发现，包括研究的很多的肺结节，肝病变等等，到研究的较少的，如骨病变和软组织病变等。因此，这使研究者可以：

- Develop a universal lesion detector. The detector can help radiologists find all types of lesions within one unified computing framework. It may open the possibility to serve as an initial screening tool and send its detection results to other specialist systems trained on certain types of lesions.

开发一个统一病变检测器。检测器可以帮助放射科医生用一种统一的计算框架找到所有病变类型。这可能会是一种初始筛查工具，将其检测结果送到其他在特定类型病变上训练得到的专家系统中。

- Mine and study the relationship between different types of lesions. In DeepLesion, multiple findings are often marked in one study. Researchers are able to analyze their relationship to make discoveries and improve CADe/CADx accuracy, which is not possible with other datasets.

挖掘并研究不同类型的病变之间的关系。在DeepLesion中，在一个study中通常会标注多个发现。研究者们可以分析其关系，得出发现，改进CADe/CADx的准确率，在其他数据集上，这是不可能的。

Another advantage of DeepLesion is its large size and small annotation effort. ImageNet is an important dataset in computer vision, which are composed of millions of images from thousands of classes. In contrast, most publicly available medical image datasets have tens or hundreds of cases, and datasets with more than 5000 well-annotated cases are rare. DeepLesion is a large-scale dataset with over 32K annotated lesions from over 10K studies. It is still growing every year, see Fig. 3. In the future, we can further extend it to other image modalities, such as MR, and combine data from multiple hospitals. Most importantly, these annotations can be harvested with minimum manual effort. We hope the dataset will benefit the medical imaging area just as ImageNet benefitted the computer vision area.

DeepLesion的另一个优势是，其规模很大，但标注成本很低。ImageNet是计算机视觉中一个重要的数据集，由数千类的数百万幅图像构成。比较起来，多数公开可用的医学图像数据集只有数十个或数百个病例，超过5000个良好标注病例的数据集非常稀少。DeepLesion是一个大规模数据集，包含超过10K个studies，标注的病变数量超过32K。其规模仍在逐年递增，如图3所示。在将来，我们会进一步拓展到其他图像模态，如MR，将不同医院的数据结合起来。最重要的是，这些标注用很少的努力就可以得到。我们期望这个数据集会使得医学图像领域受益，就像ImageNet使得计算机视觉领域受益一样。

#### 4.1.2 Potential applications

- Lesion detection: This is the direct application of DeepLesion. Lesion detection is a key part of diagnosis and is one of the most labor-intensive tasks for radiologists. An automated lesion detection algorithm is highly useful because it can help human experts to improve the detection accuracy and decrease the reading time.

病变检测：这是DeepLesion的直接应用。病变检测是诊断的关键部分，对于放射科医生是工作量很大的工作。一种自动病变检测算法会非常有用，因为可以帮助人类专家改进检测准确率，降低读片时间。

- Lesion classification: Although the type of each lesion was not annotated along with the bookmarks, we can extract the lesion types from radiology reports coupled with each study. Nowadays, radiologists often put hyperlinks in reports to link bookmarks with lesion descriptions. Consequently, we can use natural language processing algorithms to automatically extract lesion types and other information cues.

病变分类：虽然每个病变的类型并没有与标签一起标注，但我们可以从与每个study一起的放射报告中提取出病变类型。现在，放射科医生通常会在报告中放置超链接，将书签与病变描述连接起来。结果是，我们可以使用NLP算法来自动提取病变类型以及其他信息线索。

- Lesion segmentation: With the RECIST diameters and bounding-boxes provided in the dataset, weakly supervised segmentation algorithms can be developed to automatically segment or measure lesions. One can also select lesions of interest and manually annotate them for training and testing. During the annotation process, active learning may be employed to alleviate human burden.

病变分割：使用数据集中提供的RECIST半径和边界框，可以开发弱监督分割算法以自动分割或度量病变。也可以选择感兴趣的病变，手工标注进行训练和测试。在标注过程中，可以使用主动学习来缓解人类的负担。

- Lesion retrieval: Considering its diversity, DeepLesion is a good data source for the study of content-based or text-based lesion retrival algorithms. The goal is to find the most relevant lesions given a query text or image.

病变检索：考虑其多样性，DeepLesion是研究基于内容的或基于文本的病变检索算法的很好的数据源。其目标是给定一个查询的文本或图像，发现最相关的病变。

- Lesion growth analysis: In the dataset, lesions (e.g., tumors and lymph nodes) are often measured multiple times for follow-up study. With these sequential data, one may be able to analyze or predict the change of lesions based on their appearance and other relative information.

病变增长分析：在数据集中，病变（如，肿瘤和淋巴结）通常测量了很多次，以进行后续的研究。用这些顺序数据，可以基于其外观和其他相关的信息，分析或预测病变的改变。

#### 4.1.3 Limitations

Since DeepLesion was mined from PACS, it has a few limitations: 由于DeepLesion是从PACS系统中收集的，所有有一些局限：

- Lack of complete labels: DeepLesion contains only two-dimensional diameter measurements and bounding-boxes of lesions. It has no lesion segmentations, 3-D bounding-boxes, or fine-grained lesion types. We are now working on extracting lesion types from radiology reports.

缺少完整的标签：DeepLesion只包含病变的二维半径测量和边界框。并没有病变的分割，3D的边界框，或细粒度的病变类型。我们现在正在进行从放射报告中提取病变类型。

- Missing annotations: Radiologists typically mark only representative lesions in each study. Therefore, some lesions remain unannotated. The unannotated lesions may harm or misrepresent the performance of the trained lesion detector because the negative samples (nonlesions) are not purely true. To solve this problem, one can leverage machine learning strategies, such as learning with noisy labels. It is also feasible to select negative samples from another dataset of healthy subjects. Furthermore, to more accurately evaluate the trained detector, it is better to have a fully labeled test set with all lesions annotated. The newly annotated lesions should also be similar to those already in DeepLesion, so lesions that do not exist in DeepLesion should not be annotated.

缺失的标注：放射科医生一般只在每个study中标注典型的病变。因此，一些病变是没有标注的。未标注的病变可能会对训练好的病变检测器造成损害或错误表示，因为负样本（非病变）并不完全是对的。为解决这个问题，可以利用机器学习的策略，比如使用含噪的标签进行学习。从另一个健康目标的数据集中选择负样本也是可能的。而且，为更精确的评估训练好的检测器，最好有一个完整标注的测试集，所有的病变都进行标注。新标注的病变应当与那些与DeepLesion中已经存在的类似，所以在DeepLesion中不存在的病变不会被标注。

- Noise in lesion annotations: According to manual examination, although most bookmarks represent abnormal findings or lesions, a small proportion of the bookmarks is actually measurement of normal structures, such as lymph nodes of normal size. We can design algorithms to either filter them (e.g., by using extracted lesion types from reports) or ignore them (e.g., by using machine learning models that are robust to noise).

病变标注的噪声：根据手工检查，虽然多数书签都表示异常发现或病变，一小部分书签实际上是正常组织的度量，比如正常大小的淋巴结。我们可以设计算法将其滤除（如，使用从报告中提取的病变类型）或忽略之（如，使用对噪声稳健的机器学习模型）。

### 4.2 Universal Lesion Detection

Because radiologists typically mark only representative lesions in each study, there are missing annotations in the test set. Therefore, the actual FP rates should be lower. We would argue that the current result is still a nonperfect but reasonable surrogate of the actual accuracy. From the qualitative detection results in Figs. 12–21, we can find that the universal lesion detector is able to detect various types of lesions in the test set of DeepLesion, including the annotated ones (ground-truth) as well as some unannotated ones, although a few FPs and FNs still present.

因为放射科医生一般只在每个study中标注典型的病变，因此在测试集中有缺失的标注。因此，实际的假阳性率应当是更低的。我们会争论说，目前的结果仍然不是完美的，只是实际准确率的一个合理的代理。从定性的检测结果中（图12-21），我们可以发现，统一病变检测器可以在DeepLesion的测试集中检测到各种类型的病变，包括标注了的（真值），以及一些未标注的，虽然仍然存在几个假阳性和假阴性。

- Lung, mediastinum, and liver lesions can be detected more accurately, as their intensity and appearance patterns are relatively distinctive from the background. 肺，纵隔和肝病变可以更精确的检测到，因为其灰度和外观模式与背景相对更容易区分；

- Lung scarring is not always detected, which is possibly because it is not commonly measured by radiologists, thus DeepLesion contains very few training samples. 肺部伤疤不会一直检测到，这可能是因为，放射科医生不太常测量，因此DeepLesion包含的这样的训练样本很少；

- Unenlarged lymph nodes are sometimes detected as FNs. This is probably because the design of faster RCNN (e.g., the RoI pooling layer) allows it to be robust to small scale changes. We can amend this issue by training a special lymph node detector and a lesion size regressor. 非肿大淋巴结有时候会检测为假阴性。这可能是因为Faster RCNN的设计（如RoI池化层）允许对小尺度的变化更稳健。我们通过训练一个特殊的淋巴结检测器和一个病变尺寸回归器就可以修复这个问题。

- There are more FPs and FNs in the abdominal and pelvic area, as normal and abnormal structures bowel and mesentery clutter inside the image and may have similar appearances (Figs. 18–21). This problem may be mitigated by applying ensemble of models and enhancing the model with 3-D context. 在腹部和盆部有更多的假阳性和假阴性，因为正常结构和非正常结构，肠管和隔膜都杂聚在图像中，有很类似的外观（图18-21）。这个问题通过模型集成和用3D上下文增强模型可能弥补一下。

It is not proper to directly compare our results with others' since most existing work can only detect one type of lesion. However, we can use them as references. Roth et al. proposed CNNs with random view aggregation to detect sclerotic bone lesions, lymph nodes, and colonic polyps. Their detection results are 70%, 77%, and 75% at three FPs per patient for the three types of lesions, respectively. Ben-Cohen et al. applied fully convolutional network and sparsity-based dictionary learning for liver lesion detection in CT. Their result is 94.6% at 2.9 FPs per case. Multilevel contextual 3-D CNNs were used to detect lung nodules with a sensitivity of 87.9 at two FPs per scan. The main reason that our result (77.31% at three FPs per image) is still inferior than those in Refs. 7–9 is that our task is considerably harder, which tries to detect all kinds of lesions including lung nodules, liver lesions, bone lesions, lymph nodes, and so on. Besides, our dataset is much larger (32,735 lesions with about 25% lung lesions and 13% liver ones, versus 123 liver lesions and 1186 lung nodules) with lesion sizes ranging widely from 0.21 to 342.5 mm. Furthermore, we did not use a fully annotated dataset of a specific lesion to train a sophisticated detection model such as those in Refs. 7–9. Improving the detection accuracy is one of our future works.

将我们的结果与其他的直接进行比较是不合适的，因为多数已有的工作只能检测一种类型的病变。但是，我们可以将其用作参考。Roth等提出随机视角聚积的CNNs来检测骨硬化病变，淋巴结和结肠息肉。其检测结果是，在每个病人3个假阳性的情况下，对这些三种类型的病变，分别为70%，77%和75%。Ben-Cohen等使用全卷积网络和基于稀疏的字典学习在CT中检测肝病变。其结果是在每个病例中2.9个假阳性的情况下，为94.6%。多级上下文3D-CNN用于检测肺结节，每个scan 2个假阳性的情况下，达到87.9%。我们的结果（每幅图像3个假阳性的情况下77.31%）比这些参考7-9要低的主要原因是，我们的任务要更难，要检测所有类型的病变，包括肺结节，肝病变，骨病变，淋巴结等等。除此以外，我们的数据集更大（32735个病变，25%是肺结节，13%是肝病变），病变大小范围非常大，从0.21mm到342.5mm。而且，我们并没有使用特定病变的全标注数据集，来训练一个复杂的检测模型，就像7-9中的那些。改进检测准确率，是我们未来工作的一种。

## 5 Conclusion

In this paper, we introduced a paradigm to collect lesion annotations and build large-scale lesion datasets with minimal manual effort. We made use of bookmarks in PACS, which are annotations marked by radiologists during their routine work to highlight significant clinical image findings that would serve as references for longitudinal studies. After analyzing their characteristics, we harvested and sorted them to create DeepLesion, a dataset with over 32K lesion bounding-boxes and measurements. DeepLesion is composed of a variety of lesions and has many potential applications. As a direct application, we developed a universal lesion detector that can find all types of lesions with one unified framework. Qualitative and quantitative results proved its effectiveness.

本文中，我们提出了一个范式，收集病变标注，构建大规模病变数据集，需要的人工努力非常少。我们使用PACS中的书签，是由放射科医生在其日常工作中的标注，强调了明显的临床影像发现，可以作为长期研究的参考。在分析其特征后，我们对其进行了挖掘和排序，创建了DeepLesion数据集，包含超过32K个病变边界框和测量。DeepLesion是由很多病变组成的，有很多潜在的应用。作为一个直接应用，我们开发了一个统一病变检测器，使用一种统一的框架来找到所有类型的病变。定性结果和定量结果都证明了其有效性。

In the future, we will keep on improving the DeepLesion dataset by collecting more data and extracting lesion types from radiology reports. We also plan to improve the universal lesion detector by leveraging 3-D and lesion type information.

未来，我们会持续改进DeepLesion数据集，收集更多的数据，从放射报告中提取病变类型。我们还计划利用3D和病变类型信息，改进统一病变检测器。