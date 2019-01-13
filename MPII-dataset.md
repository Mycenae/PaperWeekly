# 2D Human Pose Estimation: New Benchmark and State of the Art Analysis

Mykhaylo Andriluka et al. Max Planck Institute

## Abstract 摘要

Human pose estimation has made significant progress during the last years. However current datasets are limited in their coverage of the overall pose estimation challenges. Still these serve as the common sources to evaluate, train and compare different models on. In this paper we introduce a novel benchmark “MPII Human Pose” that makes a significant advance in terms of diversity and difficulty, a contribution that we feel is required for future developments in human body models. This comprehensive dataset was collected using an established taxonomy of over 800 human activities [1]. The collected images cover a wider variety of human activities than previous datasets including various recreational, occupational and householding activities, and capture people from a wider range of viewpoints. We provide a rich set of labels including positions of body joints, full 3D torso and head orientation, occlusion labels for joints and body parts, and activity labels. For each image we provide adjacent video frames to facilitate the use of motion information. Given these rich annotations we perform a detailed analysis of the leading human pose estimation approaches gaining insights for the success and failures of these methods.

人体姿态估计在过去的几年里取得了显著的进展。但是目前的数据集覆盖面受限，一些姿态估计挑战无法进行。但这些数据集仍然是不同模型训练、评估、比较的常见来源。本文中，我们提出一个新的基准测试"MPII人体姿态"数据集，在多样性和难度上都有显著提升，我们认为未来人体姿态模型的进步需要这样的数据集。这个综合数据集的建立使用了已有的人体活动的分类法，包括了800类活动[1]。收集的图像比之前的数据集覆盖了更广泛的人类活动，包括各种娱乐活动、职业活动和家庭活动，并且图像拥有更多的角度。我们给出了丰富的标注，包括人体关节的位置，完整的3D躯干和头部方向，关节和身体部位的遮挡标注，和活动标注。对于每幅图像我们给出了临近的视频帧，以利用运动信息。有了这些丰富的标注，我们对现在最好的姿态估计方法进行了分析比较，得到了这些方法成功和失败的原因。

## 1. Introduction 引言

Recent pose estimation methods employ complex appearance models [2, 9, 15] and rely on learning algorithms to estimate model parameters from the training data. The performance of these approaches crucially depends on the availability of annotated training images that are representative for the appearance of people clothing, strong articulation, partial (self-)occlusions and truncation at image borders. Although there exists training sets for special scenarios such as sport scenes [12, 13] and upright people [17, 2], these benchmarks are still limited in their scope and variability of represented activities. Sport scene datasets typically include highly articulated poses, but are limited with respect to variability of appearance since people are typically wearing tight sports outfits. In turn, datasets such as “FashionPose” [2] and “Armlets” [9] aim to collect images of people wearing a variety of different clothing types, and include occlusions and truncation but are dominated by people in simple upright standing poses.

近年来的姿态估计方法采用了复杂的appearance模型[2,9,15]，并需要学习算法从训练数据中估计出模型参数。这些方法的性能严重依赖于标注的训练数据，图像数据中包括人们穿衣的样子，关节连接的样式，部分（自我）遮挡以及图像边缘的切断。虽然有特殊场景的训练集，如运动场景[12,13]和直立人[17,2]，这些基准测试仍然受限于其视野和反应的活动的多样性。运动场景数据集一般包括关节变化多样的姿态，但外貌的多样性没有保证，因为一般都穿着紧身运动服。而像FashionPose[2]和Armlets[9]这样的数据集，收集的是各种着装的人，并包括遮挡和截断，但是里面的人主要是简单的直立站立姿势。

To the best of our knowledge no attempt has been made to establish a more representative benchmark aiming to cover a wide pallet of challenges for human pose estimation. We believe that this hinders further development on this topic and propose a new benchmark “MPII Human Pose”. Our benchmark significantly advances state of the art in terms of appearance variability and complexity, and includes more than 40,000 images of people. We used YouTube as a data source and collected images and image sequences using queries based on the descriptions of more than 800 activities. This results in a diverse set of images covering not only different activities, but indoor and outdoor scenes, a variety of imaging conditions, as well as both amateur and professional recordings (c.f. Fig. 1). This allows us to study existing body pose estimation techniques and identify their individual failure modes.

据我们所知，目前还没有人要建立一个更有代表性的基准测试，能为各种人体姿态估计挑战提供一个广阔的平台。我们相信这阻碍了这个课题的发展，并提出了新的MPII人体姿态基准测试。我们的基准测试显著提升了样貌多样性和复杂度，包括超过4万幅人类图像。我们使用Youtube作为数据源，进行图像和图像序列的收集，并使用了超过800种活动作为检索词。这得到了很大的数据集，包括的不只是不同的活动，而且包括室内和室外场景，各种图像条件，以及业余和专业录像（如图1）。这使我们可以研究现有的人体姿态估计技术，发现他们失败的模式。

Figure 1. Randomly chosen images from each of 20 activity categories of the proposed “MPII Human Pose” dataset. Image captions indicate activity category (1st row) and activity (2nd row). To view the full dataset visit human-pose.mpi-inf.mpg.de.

**Related work**. The commonly used publicly available datasets for evaluation of 2D human pose estimation are summarized in Tab. 1 according to the year of the corresponding publication. Both full body and upper body datasets are included.

**相关工作**。表1是经常使用的进行2D姿态估计评估的公开数据集，以及其对应的论文，包括了全身和上身数据集。

Table 1. Overview of the publicly available datasets for articulated human pose estimation. For each dataset we report the number of annotated people in training and test sets and the type of images the set include. The numbers indicate the number of unique annotated people without mirroring.

datasets | type | #training | #test | img. type
--- | --- | --- | --- | ---
Parse [16] | full body | 100 | 205 | diverse
LSP [12] | full body | 1,000 | 1,000 | sports (8 types)
PASCAL Person Layout [6] | full body | 850 | 849 | everyday
Sport [21] | full body | 649 | 650 | sports
UIUC people [21] | full body | 346 | 247 | sports (2 types)
LSP extended [13] | full body | 10,000 | - | sports (3 types)
FashionPose [2] | full body | 6,530 | 775 | fashion blogs
J-HMDB [11] | full body | 31,838 | - | diverse (21 act.)
Buffy Stickmen [8] | upper body | 472 | 276 | TV show (Buffy)
ETHZ PASCAL Stickmen [3] | upper body | - | 549 | PASCAL VOC
Human Obj. Int. (HOI) [23] | upper body | 180 | 120 | sports (6 types)
We Are Family [5] | upper body | 350 imgs. | 175 imgs. | group photos
Video Pose 2 [18] | upper body | 766 | 519 | TV show (Friends)
FLIC [17] | upper body | 6,543 | 1,016 | feature movies
Sync. Activities [4] | upper body | - | 357 imgs. | dance / aerobics
Armlets [9] | upper body | 9,593 | 2,996 | PASCAL VOC/Flickr
MPII Human Pose (this paper)| - | 28,821 | 11,701 | diverse (491 act.)

Existing benchmarks cover aspects of the human pose estimation task such as sport scenes [12, 21], frontal-facing people [8, 3, 17], people interacting with objects [23], pose estimation in group photos [5] and pose estimation of people performing synchronized activities [4].

现有的基准测试包括了几种人体姿态估计任务，比如运动场景[12,21]，正面人体[8,3,17]，与物体互动的人[23]，group photos中的姿态估计，和动作一致的人的姿态估计[4]。

Earlier datasets such as “Parse” [16] and “Buffy” [8] are still commonly found in evaluations [22, 15]. However the small training sets included in these datasets make them unsuitable for training models with complex appearance representations and multiple components [13, 17, 2], which have been shown to perform best.

更早的数据集在测试时也经常使用，如Parse[16]和Buffy[8]。但这些数据集中的训练集都很小，不适合训练复杂的模型[13,17,2]，而这些都是最好的模型。

Some efforts have been made to collect larger sets of images. For example [13] extends the LSP dataset to 10,000 images of people performing gymnastics, athletics and parkour. [2] proposes a large “FashionPose” dataset collected from fashion blogs. This dataset aims to cover a wide variety in people clothing. The LSP and FashionPose datasets are complementary and focus on two different challenges for human pose estimation: pose variability and variability of people appearance. However since they are collected with a specific focus in mind, these datasets do not cover real-life challenges such as truncation, occlusions by scene objects and variability of imaging conditions.

有一些收集更大的图像数据集的工作。如[13]拓展了LSP数据集到1万幅人类图像，进行体操活动、田径运动和跑酷活动。[2]提出了一个很大的FashionPose数据集，是从时尚博客里收集的，其目标是覆盖人们衣着的多样性。LSP和FashionPose数据集是互补的，专注于两种不同的人类姿态估计的挑战：姿态的多样性和人们外表的多样性。但是由于它们收集的特殊目标，这些数据集没有覆盖到真实生活中的挑战，如截断、场景目标的遮挡和图像条件的变化。

The works of [6] and [9] propose a challenging dataset building on the PASCAL VOC image collection. Results reported in [9] indicate that the best performing approaches for pose estimation of people in the presence of occlusion and complex appearance are under-performing on sport-oriented datasets such as LSP [12] and vice versa. There are qualitative differences between methods that work well for LSP and “Armlets” datasets. On LSP the best performing methods are typically based on flexible part-based models that are well suited for capturing pose variability. In contrary on the “Armlets” dataset the best performing approach [9] uses a set of rigid detectors for groups of parts, that are more robust to the variability in appearance.

[6,9]的工作提出了基于PASCAL VOC图像集的数据集。[9]中的结果说明，在遮挡和复杂外表情况下进行姿态估计，其中表现最好的模型，在运动类的数据集如LSP[12]中表现不好，反之亦然。在LSP和Armlets数据集上表现好的模型有质的区别。在LSP上表现最好的模型，一般是基于部位可变的模型，可以很好的捕捉到姿态的多样性。相反，在Armlets数据集中表现最好的方法[9]使用了rigid检测器集进行身体部位集检测，这类方法对于外表多样性更稳健。

Our dataset is complementary to the J-HMDB dataset [11] and provides more images and a wider coverage of activities (491 in our dataset vs. 21 in J-HMDB), whereas J-HMDB provides densely annotated image sequences and larger number of videos for each activity. Our dataset also addresses a different set of challenges compared to the datasets such as “HumanEva” [19] and “Human3.6M” [10] that include images and 3D poses of people but are captured in the controlled indoor environments, whereas our dataset includes real-world images but provides 2D poses only.

我们的数据集与J-HMDB数据集[11]是互补的，图像数量更多，包含的活动范围更广（我们的有491种，J-HMDB只有21种），J-HMDB提供了密集标注的图像序列和每种活动的大量视频。HumanEva[19]和Human3.6M[10]这样的数据集包含人们的3D姿态，但是在受控的室内环境收集，与之相比，我们的数据集处理的则是不太一样的挑战，包括的是真实世界的图像，但只有2D姿态。

## 2. Dataset 数据集

In this paper we introduce a large dataset of images that covers a wide variety of human poses and clothing types and includes people interacting with various objects and environments. The key rationale behind our data collection strategy is that we want to represent both common and rare human poses that might be missed when simply collecting more images without aiming for good coverage. To this end, we use a two-level hierarchy of human activities proposed in [1] to guide the collection process. This hierarchy was developed for the assignment of standardized energy levels during physical activity surveys and includes 823 activities in total of 21 different activity categories. The activities at the first level of the hierarchy correspond to thematically related groups of activities such as “Home Activities”, “Lawn and Garden” or “Sports”. The activities at the second level then correspond to individual activities such as “Washing windows”, “Picking fruit” or “Rock climbing”. Note that using the activity hierarchy for collection has an additional advantage that all images have an associated activity label. As a result one can assess and analyze any performance measure also on subsets of activities or activity categories.

本文中我们提出了一种大型图像数据集，包含了广泛的人体姿态、衣着类型，以及与各种不同目标和环境互动的人。我们收集数据的关键原因是，我们想代表常见和不常见的人体姿态，如果只收集更多的图片，而不是为了覆盖更多的情况，很可能会错过一些。为了这个目的，我们采用了[1]中提出的2级人类活动层次分类法来指导我们的收集过程。这个层次分类是为了体育活动调查中指定标准化的能量级别而制定的，包括21种不同类别的823种活动。第一层次的活动对应着主题相关的活动，如“家庭活动”，“草坪和花园”或者“运动”。第二个层次的活动就对应着个体活动，如“擦窗户”，“摘水果”或“攀岩”。注意使用活动层级进行收集有额外的优势，即所有的图像都有一个相关的运动标签。所以可以在任何活动子集或活动类别子集上分析性能度量。

Due to the coverage of the hierarchy the images in our dataset are representative of the diversity of human poses, overcoming one of the main limitations of previous collections. In Fig. 2 we visualize this diversity by comparing upper body annotations of the “Armlets” dataset Fig. 2(b) and our proposed dataset (c). Note that although “Armlets” contain about 13,500 images, the annotations resemble a person with arms down along the torso (distribution of red, cyan, green, and blue sticks).

由于这个层次的覆盖性很好，所以我们数据集种的图像代表了人类姿态的多样性，解决了之前数据集的一个主要问题。在图2中，我们将这种多样性进行了可视化，比较了上身标注的Armlets数据集（图2b）和我们的数据集（图2c）。注意虽然Armlets包括了大约13500幅图像，但标注很像一个人胳膊沿着躯干垂下来。

Figure 2. Visualization of upper body pose variability. From left to right we show, (a) color coding of the body parts (b) annotations of the “Armlets” dataset [9], and (c) annotations of this dataset.

We collect images from YouTube using queries based on the activity descriptions. Using YouTube allows us to access a rich collection of videos originating from various sources, including amateur and professional recordings and capturing a variety of public events and performances. In Fig. 2 (c) we show the distribution of upper body poses on our dataset. Note the variability in the location of hands and the absence of distinctive peaks for the upper and lower arms that are present in the case of the “Armlets” dataset.

我们从Youtube收集图像时，使用了基于活动描述的检索。使用Youtube使我们可以获得巨大的视频集合，来源众多，包括业余拍摄和职业拍摄的，覆盖了很多公共事件和表演。图2c中，我们给出了我们数据集中的上身姿态分布，注意手部位置的多样性，以及胳膊上部和下部没有明显的峰值分布，这在Armlets数据集中很明显。

**Data collection**. As a first step of the data collection we manually query YouTube using descriptions of activities from [1]. We select up to 10 videos for each activity filtering out videos of low quality and those that do not include people. This resulted in 3,913 videos spanning 491 different activities. Note that we merged a number of the original 823 activities due to high similarity between them, such as cycling at different speeds. In the second step we manually pick several frames with people from each video. As the focus of our benchmark is pose estimation we do not include video frames in which people are severely truncated or in which pose is not recognizable due to poor image quality or small scale. We aim to select frames that either depict different people present in the video or the same person in a substantially different pose. In addition we restrict the selected frames to be at least 5 seconds apart. This step resulted to a total of 24,920 extracted frames from all collected videos. Next, we annotate all people present in the collected images, but ignore dense people crowds in which significant number of people are almost fully occluded. Following this procedure we collect images of 40,522 people. We allocate roughly tree quaters of the collected images for training and use the rest for testing. Images from the same video are either all in the training or all in the test set. This results in a training/test set split of 28,821 to 11,701.

**数据收集**。数据收集的第一步，我们使用[1]中的活动描述来手动检索Youtube。我们筛选掉低质量的视频和不包含人的视频，对每个活动选择了最多10个视频。结果得到了491种不同活动的3913个视频。由于原始823种活动中很多相似度很高，所以合并了一些，比如以不同的速度骑车。第二步，我们从每个视频中手工挑选出一些包含人的帧。由于我们的基准测试关注的是姿态估计，所以没有选择那些人物图像被严重截断的，或由于视频质量很差或尺度太小导致姿态难以分辨的。我们的目标是选择那些包含了不同的人的帧，或同样的人但姿势完全不同的帧。另外，我们限制选定的帧之间至少有5秒的间隔。这一步骤之后，从所有收集到的视频中，提取出了共计24920帧图像。下一步，我们对所有收集到的图像中的人进行标注，但忽略了那些非常密集的几乎完全互相遮挡的人群。按照这个过程，我们收集到的图像中有40522个人。我们将大约3/4的图像用于训练，剩余的用作测试。同一视频中的图像要么都在训练集中，要么都在测试集中。得到的训练集/测试集之比为28821/11701。

**Data annotation**. We provide rich annotations for the collected images, an example can be seen in Fig. 3. Annotated are the body joints, 3D viewpoint of the head and torso, and position of the eyes and nose. Additionally for all body joints and parts visibility is annotated. Following [13, 9] we annotate joints in a “person centric” way, meaning that the left/right joints refer to the left/right limbs of the person. At test time this requires pose estimation with both a correct localization of the limbs of a person along with the correct match to the left/right limb. The annotations are performed by in-house workers and via Amazon Mechanical Turk (AMT). In our annotation process we build and extend the annotation tools described in [14]. Similarly to [13, 20] we found that effective use of AMT requires careful selection of qualified workforce. We pre-select AMT workers based on a qualification task, and then maintain data quality by manually inspecting the annotated data.

**数据标注**。我们对收集的图像进行了丰富的标注，如图3中的例子所示。标注的数据有，身体关节点，头部和躯干的3D视角，眼睛和鼻子的位置。另外对于所有的身体节点和部位，都标注了可见性。与[13,9]一样，我们以“人为中心”的方式标注关节点，意思是左/右关节点是指人的左/右肢体。在测试时，这需要姿态估计不仅要准确定位肢体的位置，还要正确的匹配左/右肢体。标注是通过Amazon Mechanical Turk(AMT)分包进行的。在标注过程中，我们构建并扩展了[14]中的标注工具。与[13,20]类似，我们发现有效的利用AMT需要仔细选择高质量的工作人手。我们基于资格鉴定工作来预选择AMT工人，然后通过人工检查标注的数据来保持数据质量。

Figure 3. Example of the provided annotations. Annotated are (a) positions and visibility of the main body joints, locations of the eyes and nose and the head bounding box (occluded joints are shown in red), (b) occlusion of the main body parts (occluded parts are shown with filled rectangles), and (c) 3D viewpoints of the head and torso. On the illustration the viewpoint is shown using a simplified body model, the front face of the model is shown in red.

**Experimental protocol and evaluation metrics**. We define the baseline evaluation protocol on our dataset following the current practices in the literature [13, 9, 15]. We assume that at test time the rough location and scale of a person are known, and we exclude the cases with multiple people in close proximity to each other from the evaluation. We feel that these simplifications are necessary for the rapid adoption of the dataset as the majority of the current approaches does not address multiple people pose estimation and does not search over people positions and scales.

**试验协议和评估标准**。我们依据现有的实践[13,9,15]定义了数据集上的基准评估协议。我们假设在测试时，人的大致位置和尺度是已知的，并排除了多人互相非常接近的情形。我们感觉这些简化是必须的，这样数据集能够迅速被采用，因为目前大多数方法并没有处理多人姿态估计问题，而且也不人的位置和尺度的搜索。

We consider three metrics as indicators for the pose estimation performance. The widely adopted “PCP” metric [8] that considers a body part to be localized correctly if the estimated body segment endpoints are within 50% of the ground-truth segment length from their true locations. The “PCP” metric has a drawback that foreshortened body parts should be localized with higher precision to be considered correct. We define a new metric denoted as “PCPm” that uses 50% of the mean ground-truth segment length over the entire test set as a matching threshold, but otherwise follows the definition of “PCP”. Finally, we consider the “PCK” metric from [22] that measures accuracy of the localization of the body joints. In [22] the threshold for matching of the joint position to the ground-truth is defined as a fraction of the person bounding box size. We use a slight modification of the “PCK” and define the matching threshold as 50% of the head segment length. We denote this metric as “PCKh”. We choose to use head size because we would like to make the metric articulation independent.

我们为姿态估计的性能定义三种度量标准。一种是广泛采用的PCP度量标准[8]，即如果估计的身体部分的端点距离其真实位置不超过真值部分长度的50%，就认为身体部位定位准确。PCP度量标准有一个缺点，就是透视的身体部位应当用更高的准确度定位才能认为是正确的。我们定义了一种新的度量标准，称为PCPm，使用了真值部分长度在整个测试集上的平均值的50%作为匹配阈值，其他部分与PCP定义一样。最后，我们使用[22]中的PCK度量标准，衡量的是身体关节点的定位准确度。在[22]中，匹配关节点位置的真值的阈值定义为人的边界框大小的一部分。我们对PCK进行略微修改，定义匹配阈值为头部长度的50%。我们称这种度量标准为PCKh。我们使用头部尺寸是因为我们要让这种度量标准与关节连接无关。

## 3. Analysis of the state of the art 目前最好算法的分析

In this section we analyse the performance of leading human pose estimation approaches on our benchmark. We take advantage of our rich annotations and conduct a detailed analysis of various factors influencing the results, such as foreshortening, activity and viewpoint, previously not possible in this detail. The goal of this analysis is to evaluate the robustness of the current approaches in various challenges for articulated pose estimation, identify the existing limitations and stimulate further research advances.

本节中在我们的基准测试上分析了目前最好的姿态估计算法的性能。我们利用了数据集标注丰富的优势，详尽的分析了可能影响结果的各种因素，比如透视、活动类别、视角，这在之前的数据集上是不可能这么详细的。这个分析的目标是，评估目前方法在铰接姿态估计任务中，在各种不同挑战下的稳健性，确定已有的限制并激励进一步的研究进展。

In our analysis we consider two full body and two upper body pose estimation approaches. The full body approaches are the version 1.3 of the flexible mixture of parts (FMP) approach of Yang and Ramanan [22] and the pictorial structures (PS) approach of Pishchulin et al. [15]. The upper body pose estimation approaches are the multimodal decomposable models (MODEC) approach of Sapp et al. [17] and the Armlets approach of Gkioxari et al. [9]. In case of FMP and MODEC we use publicly available code and pre-trained models. The PS model used here corresponds to our best model published in [15]. In case of the Armlets model, the code and pre-trained model provided by the authors correspond to the version from [9] that includes the HOG features only. The performance of our version of Armlets on the “Armlets” dataset is 3.3 PCP lower than the version based on combination of all features.

在我们的分析中，我们考虑两种全身姿态估计方法和两种上半身姿态估计方法。全身方法是Yang和Ramanan [22]的1.3版部位灵活混合(FMP)方法，和Pishchulin等的[15]pictorial结构(PS)方法。上半身姿态估计方法是Sapp等[17]的多模可分解模型(MODEC)方法，和Gkioxari等[9]的Armlets方法。FMP和MODEC方法的情况下，我们使用公开可用的代码和预训练的模型。这里使用的PS模型对应着我们在[15]中发表的最好模型。在Armlets模型中，作者提供的代码和预训练模型对应着[9]中的版本，只包括了HOG特征。我们版本的Armlets方法在在Armlets数据集上的性能，比所有特征结合在一起的版本，低了3.3 PCP。

Note that the approaches considered in this evaluation are the best performing ones in their respective categories. The PS approach achieves the best results to date on LSP that is focused on the strongly articulated people [15]. The Armlets approach is best on the “Armlets” dataset [9] that includes large number of truncation and occlusions, and MODEC is the best on the recent upper body pose estimation dataset “FLIC” [17]. We include the FMP approach that is widely used in the literature and typically shows competitive performance for a variety of settings. In the following experiments we use “PCPm” as our working metric, while also providing results for “PCP” and “PCKh” in the supplementary material. While we observe little performance differences when using each metric, all conclusions obtained during “PCPm”-based evaluation are valid for “PCP” and “PCKh”-based evaluations as well.

注意本次评估中使用的方法都是各自类别中表现最好的方法。PS方法在LSP数据集上得到了迄今为止最好的结果，这个数据集特点是铰接姿态各异的人[15]。Armlets方法在Armlets数据集[9]上效果最好，该数据集包括了大量截断和遮挡，MODEC在最近的上半身姿态估计数据集FLIC[17]上效果最好。FMP方法在各种文献中广泛使用，在各种设置中都能得到有竞争力的性能。下面的试验中，我们使用PCPm作为度量标准，同时也在补充资料中给出PCP和PCKh的结果。我们得到的各种度量标准之间的差异很小，使用PCPm度量得到的结论对于PCP和PCKh也可以得到。

**Overall performance evaluation**. We begin our analysis by reporting the overall pose estimation performance of each approach and summarize the results in Tab. 2. We include both upper- and full body results to enable comparison across different models. The PS approach achieves the best result of 42.3% PCPm, followed by the FMP approach with 38.3% PCPm. On the upper body evaluation, PS performs best with 39.1%, while both MODEC (27.8% PCPm) and Armlets (26.4% PCPm) perform significantly worse.

**总体性能评估**。我们先给出每种方法的总体姿态估计性能，总结在表2中。这里包括了上半身和全身的结果，以在不同模型之间进行比较。PS方法得到了最好的结果42.3% PCPm，然后是FMP方法38.3% PCPm。在上半身方法评估中，PS结果最好39.1%，MODEC和Armlets方法比较都明显更差(27.8%, 26.4% PCPm)。

The interesting outcome of this comparison is that both upper body approaches MODEC and Armlets are outperformed by the full body approaches PS and FMP evaluated on upper body only. This is interesting because significant portion of the dataset (15 %) includes people that have only upper body visible. It appears that the PS and FMP approaches are sufficiently robust to missing parts to produce reliable estimates even in the case of lower body occlusion.

这个比较的结果有趣之处在于，两种上半身方法MODEC和Armlets，都被全身方法PS和FMP超过了，而且是只在上半身结果评估中。由于数据集相当一部分(15%)包含的人只有上半身可见，所以结果才有趣。PS和FMP方法对于遗失的部位非常稳健，即使在下半身遮挡的情况下也可以产生可靠的估计。

Lower part of Tab. 2 shows the results when using provided rough location of person during test time inference. We observe, that while the performance increases for all methods, upper body approaches profit at most, as they heavily depend on correct torso localization. For the sake of fair comparison among the methods, we do not use the rough location in the following experiments. Another interesting outcome is that the achieved performance is substantially lower than current best results on the sports-centric LSP dataset, but comparable to results on the “Armlets” dataset (42.2 PCP on our benchmark (see supplemental) vs. 69.2 on LSP [15] vs. 36.2 PCP on “Armlets”). This suggests that sport activities are not necessary the most difficult cases for pose estimation; challenges such as appearance variability, occlusion and truncation apparently deserve more attention in the future.

表2的下半部分是，在测试时候的推理过程中，使用了给定的人体粗略位置，所得到的结果。我们观察到，虽然所有方法的结果都有所改善，但上半身方法改善最大，因为它们强烈依赖于躯干定位的正确性。为对各种方法进行公平比较，我们在下面的试验中不使用粗略位置信息。另一个有趣的结果是，与在运动类数据集LSP上的结果相比，这些最好的方法的结果都普遍差很多，但和在Armlets数据集上的结果差不多（我们的数据集42.2，LSP 69.2，Armlets 36.2）。这说明运动活动对于姿态估计来说并不是最难的情况，未来的研究中应当更多的关注外表的变化、遮挡和截断等情况。

Table 2. Pose estimation results (PCPm) on the proposed dataset without and with using rough body location (“+ loc” in the table).

Setting | Torso | Upper leg | Lower leg | Upper arm | Fore-arm | Head | Upper body | Full body
--- | --- | --- | --- | --- | --- | --- | --- | ---
Gkioxari et al. [9] | 51.3 | - | - | 28.0 | 12.4 | - | 26.4 | -
Sapp&Taskar [17] | 51.3 | - | - | 27.4 | 16.3 | - | 27.8 | -
Yang&Ramanan [22] | 61.0 | 36.6 | 36.5 | 34.8 | 17.4 | 70.2 | 33.1 | 38.3
Pishchulin et al. [15] | 63.8 | 39.6 | 37.3 | 39.0 | 26.8 | 70.7 | 39.1 | 42.3
Gkioxari et al. [9] + loc | 65.1 | - | - | 33.7 | 14.9 | - | 32.4 | -
Sapp&Taskar [17] + loc | 65.1 | - | - | 32.6 | 19.2 | - | 33.7 | -
Yang&Ramanan [22] + loc | 67.2 | 39.7 | 39.4 | 37.4 | 18.6 | 75.7 | 35.8 | 41.4
Pishchulin et al. [15] + loc | 66.6 | 40.5 | 38.2 | 40.4 | 27.7 | 74.5 | 40.6 | 43.9

### 3.1. Analysis of pose estimation challenges 姿态估计挑战的分析

We now analyse the performance of each approach with respect to the following five factors: part occlusion, fore-shortening, body pose, viewpoint, and activity of the person. For the purpose of this analysis we define quantitative complexity measures that map body image annotations to a real value that relates to the complexity of the image with respect to each factor.

现在我们分析每种方法在以下五种因素影响下的表现：部位遮挡，透视，身体姿态，视角和人的活动。为进行分析，我们定义了量化的复杂度度量，将身体图像标注映射到反应每种因素复杂度的实数值。

Let us denote the annotation of the person by $L = \{L^{pose} ,L^{view} ,L^{vis}\}$, where $L^{pose} = \{l_i, i = 1,...,N\}$ corresponds to the positions of body parts, $L^{view} = \{α_1, α_2, α_3 \}$ are the Euler angles representation of the torso rotation, and $L^{vis} = \{(ρ_i, θ_i), i = 1,...,N\}$ encodes body part visibility via a set of occlusion labels $ρ_i ∈ {0,1}$ and truncation labels $θ_i ∈ {0,1}$.

我们将对人体的标注表示为$L = \{L^{pose} ,L^{view} ,L^{vis}\}$，其中$L^{pose} = \{l_i, i = 1,...,N\}$对应身体部位的位置，$L^{view} = \{α_1, α_2, α_3 \}$是躯干旋转角度的欧拉角表示，$L^{vis} = \{(ρ_i, θ_i), i = 1,...,N\}$编码了身体部位的可见性，采用了遮挡标签集$ρ_i ∈ {0,1}$和截断标签集$θ_i ∈ {0,1}$。

We define the following complexity measures. Pose complexity is measured as the deviation from the mean pose on the entire dataset. We define $m_{pose} (L) = \Pi_{(i,j)∈E} p_{ps} (l_i |l_j )$, where E is a set of body joints and $p_{ps}(l_i |l_j )$ is a Gaussian distribution measuring relative position of the two adjacent body parts using the transformed state-space representation introduced in [7]. Note that $m_{pose}(L)$ corresponds to the likelihood of the pose under the tree structured pictorial structures model [7]. The amount of foreshortening is measured by $m_f(L) = \sum^N_{i=1} |d(l_i)−m_i|/m_i$, where $d(l_i)$ is the length of the body part i, and $m_i$ is the mean length over the entire dataset. The viewpoint complexity is measured by the deviation from the frontal viewpoint: $m_v(L) = \sum^3_{i=1} α_i$. Finally, the amount of occlusion and truncation correspond to the number of occluded and truncated body parts: $m_{occ} = \sum^N_{i=1} ρ_i$, and $m_t = \sum^N_{i=1} τ_i$.

我们定义下述的复杂度度量。姿态复杂度定义为与整个数据集的平均姿态的偏差。我们定义$m_{pose} (L) = \Pi_{(i,j)∈E} p_{ps} (l_i |l_j )$，其中E是身体关节的集合，$p_{ps}(l_i |l_j )$是高斯分布的概率，衡量的是两个邻接的身体部位的相对位置，使用的是[7]中提出的变换状态空间表示。注意$m_{pose}(L)$对应着在[7]中的树状结构的pictorial结构下的姿态可能性。透视程度由$m_f(L) = \sum^N_{i=1} |d(l_i)−m_i|/m_i$度量，其中$d(l_i)$是身体部位i的长度，$m_i$是整个数据集上身体部位的平均长度。视角复杂度由与前视角的偏差度量：$m_v(L) = \sum^3_{i=1} α_i$。最后，遮挡和截断的程度对应着遮挡的和截断的身体部位的数量，即：$m_{occ} = \sum^N_{i=1} ρ_i$, $m_t = \sum^N_{i=1} τ_i$。

**Performance as a function of the complexity measures**。 To visualize the influence of the various factors on pose estimation performance we plot PCPm scores for the images sorted in the order of increasing complexity (see Fig. 4). In general and as expected, the performance drops for all measures as the complexity increases. There are interesting differences however. Body pose complexity clearly influences the performance of all approaches the most. The second most influential factor is the viewpoint of the torso. For upper body pose estimation approaches this factor is equally influential as body pose. The third most influential factors is occlusion while for the full body estimation approaches this is equally influential as the torso orientation. Contrary to our expectation we found that the part length is less influential. Part length and in particular foreshortening effects are considered to be the key difficulties for both pose estimation. Based on this analysis the above mentioned factors have a higher influence on the performance. The least influential factor is truncation having the smallest effect. In the case of upper body estimation the performance even slightly increases as the amount of truncation increases due to two factors. As truncation if more likely for the lower body these approaches suffer less from truncation and also truncated poses are biased towards frontal views for which the methods are more suited. We now discuss and analyze each factor in more detail.

**性能作为复杂度度量的函数**。我们将各种因素对姿态估计性能的影响进行了可视化，画出了PCPm分数与不断增加的复杂度之间的关系（见图4）。总体来说，与预期相符，当复杂度增加的时候，所有度量的性能都下降了。但是也有有趣的不同之处。身体姿态复杂度明显对所有方法的性能影响最大。影响力第二的因素为躯干的视角。对于上半身姿态估计方法，这个因素与身体姿态影响力相当。影响力第三的因素是遮挡，对于全身姿态估计方法来说，这与躯干朝向的影响力是相当的。与我们期望相反的是，身体部位长度的影响是很小的。身体部位长度，尤其是透视效果，一向被认为是两类姿态估计方法的关键难点。基于这个分析，上述的因素对性能影响较大。影响最小的因素是截断。在上半身姿态估计中，随着截断因素的增加，性能甚至略有上升，这有两个原因。如果截断的是身体下半部分，那么这对算法影响很小，而且截断的姿态对于前视角来说是偏移的，这些方法对这个也很适合。下面我们详细分析每种因素。

**Body pose performance**. As stated above the complexity of the pose is a dominating factor for the performance of all considered approaches. For example the PS approach achieves 72.8% PCPm on the 1000 images with lowest pose complexity, compared to 42.3% for the entire dataset. The same is true for the FMP model, 63.4% PCPm on 1000 least pose complex images vs. 38.3% overall.

**身体姿态性能**。如上所述，姿态复杂度是影响所有方法性能的最主要因素。比如，PS方法在最低姿态复杂度的1000幅图像上取得了72.8%的成绩，但在整个数据集上只有42.3%。对于FMP方法也是一样的，在1000幅最低姿态复杂度图像上PCPm为63.4%，整体结果为38.3%。

To highlight variations in performance across different body configurations we cluster the test images according to the body pose and measure performance for each cluster. We repeat this three times, clustering all body joints, only the upper body joints, and finally the lower body joints. In the latter two cases we measure performance on the upper/lower body parts only. These three clusterings correspond both to different types of challenges as well as applications. Furthermore, this allows to directly compare full vs. upper body techniques. We show the average PCPm for all full body clusters with more than 25 examples in Fig. 5 ordering the results from left to right by increasing mean pose complexity. Note the significant variations in performance across different clusters. For example, results on full body clusters vary between 77% and 2% PCPm. The best performance is achieved on clusters with poses similar to the mean pose e.g. clusters 1 and 5 (see Figure 5). Examining clusters with poor performance we immediately discover several failure modes of PS and FMP approaches. Consider the clusters 42 and 43 that correspond to people with slightly foreshortened torso. FMP improves over PS by 14% PCPm on cluster 25 (54% PCPm for PS vs. 68% PCPm for FMP) and by 16% PCMm on cluster 42 (44% PCPm for PS vs. 60% PCPm for FMP), as it can better model torso foreshortening by representing torso as configuration of multiple flexible parts, whereas PS models torso as a single rigid part. Also, the flexibility of FMP model accounts for its better performance on frontal sitting people (cluster 43) where FMP improves over PS by 7% PCPm (46% PCPm for FMP vs. 39% PCPm for PS), mainly due to better modeling of the foreshortened upper legs. However, performance on the sideview sitting people (e.g. clusters 26, 30, 34, 44) is poor for all methods. Another prominent failure mode for all approaches are people facing away from the camera, e.g. cluster 50. Such part configurations are commonly mistaken for the frontal view which leads to a mismatch between left and right body parts resulting in incorrect estimation. These findings demonstrate inability of current methods to reliably discriminate between frontal and backward views of people. Interestingly, upper body approaches outperform full body methods on the full body cluster 31. This is an easy case for the former group of methods due to frontal upright upper body, but is a challenging task for the full body upproaches as legs are hard to estimate in this case. However, both MODEC and Armlets fail on examples when torso start deviating from canonical orientation (e.g. clusters 20, 27, 37). At the same time both full body methods perform better, as they are more robust to the viewpoint changes. Surprisingly, full body methods outperform upper body approaches on “easy” examples (c.f. cluster 1, 3 and 5). We attribute this effect to the correct integration of signals from the legs into a more reliable upper body estimate.

为区分不同身体配置下的性能差异，我们将测试图像根据身体姿态进行聚类，对每个聚类进行性能度量。这个过程重复三次，分别是对所有身体关节点进行聚类，只对上半身关节点进行聚类，和只对下半身节点进行聚类。后两种情况中，我们只对上/下半身进行性能度量。这三种聚类对应着不同类型的挑战和应用。而且，这可以直接对全身/上半身姿态估计方法进行对比。我们显示了所有全身聚类的平均PCPm结果，具体例子数量超过了25个，如图5所示，结果从左到右是按照身体姿态复杂度逐渐增加的顺序。注意不同聚类之间性能的显著差异。比如，全身聚类的结果在77%到2% PCPm之间变化。最好的结果对应的聚类是与平均姿态最接近的聚类，比如聚类1到5（见图5）。检查一下那些性能很差的聚类，我们立刻发现了PS和FMS方法的几种失败模式。考虑聚类42和43的情况，对应着躯干略微透视的情况。在聚类25中，FMP比PS结果高14%(54% PCPm for PS vs. 68% PCPm for FMP)，而在聚类42中高了16%(44% PCPm for PS vs. 60% PCPm for FMP)，这是因为FMP可以将透视的躯干表示成多个灵活部位的配置，而PS将躯干表示为单个的刚性部位。而且，FMP模型的灵活性也可以解释在聚类43上的较好表现，其中超过了PS模型7% PCPm (46% PCPm for FMP vs. 39% PCPm for PS)，这主要是由于对透视的前腿建模较好。但是，坐着的人的侧视角（如聚类26，30，34，44）的情况下所有方法效果都很差。所有的方法另一个明显的失败模式是人没有看镜头的情况，如聚类50。这种部位配置通常都会被误认为是前视角，这会使得左右身体部位的错误匹配，带来错误估计。这些发现证明了现有的方法不能可靠的区分前视角和后视角。有趣的是，在全身聚类31上，上半身方法超过了全身方法的表现。这对于上半身方法来说是个相对容易的情况，因为是正面直立的人体，但对于全身方法来说是一个挑战，因为腿在这种情况下很难估计。但是，当躯干与标准朝向偏离时，MODEC和Armlets方法都开始出现错误，如聚类20，27，37，与此同时两种全身方法的表现更好，因为对于视角变化更稳健。令人惊奇的是，在一些“简单”例子中（如聚类1,3,5），全身方法超过了上半身方法。我们认为这是因为腿部得到的信息正确的整合到了一起，得到了更可靠的上半身估计。

Figure 5. Performance (PCPm) on images clustered by full body pose. Clusters are ordered by increasing mean pose complexity and representatives are shown beneath. Results using upper body and lower body clusters can be found in supplementary material.

**Occlusion and truncation performance**. In Fig. 4 we clearly see difference in how occlusion and truncation influences the results. As expected we observe that the performance is best for fully visible people, but full visibility does not result in success rate similar to the one we observed for the images with simple poses, e.g. PS approach achieves 72.8% PCPm for 1000 most simple poses vs. 60% PCPm for same amount of people with least occlusion. We observe that occlusion results in significant performance drop on the order of 10% PCPm, e.g. in the case of PS approach 19.3% vs. 31.2% PCPm for the forearm with and without occlusion.

**遮挡和截断情况下的表现**。在图4中，我们清楚的看到遮挡和截断是怎样影响结果的。如同预期一样，性能在全部可见的人的情况下最好，但在简单姿态的情况下，完全可见性并不对应着更高的结果，如PS方法在1000幅简单姿态的情况下取得了72.8% PCPm的结果，而在相同数量的人上在最少遮挡的情况下只取得了60%的PCPm。我们观察到遮挡导致显著的性能下降，大约有10%的PCPm，如在PS方法的情况下，有遮挡时19.3% PCPm，没有遮挡时31.2% PCPm。

As mentioned above, truncation showed the least influence overall among the discussed factors. There are at least two reasons. First, the number of images with truncation is limited in our dataset (about 30% of the test data contain truncated people). Second, and more importantly, for truncation one cannot annotate positions of body parts outside of the image. Therefore the standard procedure is to exclude truncated body parts from the evaluation. In that sense approaches that wrongly estimate the position of a truncated body part are not punished for that. This limitation could be addressed by requiring that models have to also report which parts of the body are truncated.

如上所述，截断在上面讨论的因素中影响最小。这至少有两个原因。第一，我们数据集中有截断的图像数量有限（约30%的测试数据包含截断的人）。第二，更重要的是，对于截断的情况，不能在图像之外标注身体部位的位置。所以标准步骤是把截断的身体部位排除在评估之外。在这种意义下，如果方法错误的估计了截断的身体部位的位置，那么这种错误并没有被惩罚。这种局限性是可以解决的，可以要求模型也给出哪些身体部位被截断的结果。

**Viewpoint performance**. We evaluate the pose estimation for various torso viewpoints in two ways. In Fig. 4 we show results using our standard analysis method based on images ordered by deviation from the frontal viewpoint. For a more detailed analysis we quantize the space of viewpoints by clustering training examples according to their 3D torso orientations. We show results for the viewpoint clusters in Fig. 6 ordering them by the number of examples corresponding to each cluster. The number of examples per cluster ranges between 1453 examples for the largest cluster corresponding to the frontal viewpoint, and 53 examples for the viewpoint with extreme torso tilt.

**视角影响下的表现**。我们在不同的躯干视角下以两种方式评估姿态估计方法。如图4所示，我们用标准分析方法得到了分析结果，图像按照与正面视角的偏移程度排序。为了更细节的分析，我们将视角空间量化，根据3D躯干朝向对训练样本进行聚类。我们将视角聚类的结果放到图6中，按照每个聚类中的样本数量进行排序。每个聚类中的样本数量最多的是1453，这是正面视角，对于最极端倾斜的躯干视角，有53个样本。

We observe that in contrast to the full body approaches, viewpoint has profound influence on the performance of the upper body approaches considered in our evaluation. The performance of both Armlets and MODEC approaches drops significantly for non-frontal views.

我们观察到，与全身方法对比，视角对于上半身方法的影响更大一些。Armlets和MODEC方法在非正面视角下性能下降明显。

A per viewpoint evaluation reveals significant performance differences across viewpoints. In Fig. 6 we show the results for the “person centric” annotations that we use throughout experiments in this paper and in addition for the “observer centric” (OC) annotations, in which body limbs are labeled as left/right based on their image location with respect to the torso. Frontal and near-frontal viewpoints are performing best. We observe a large drop in performance for backward facing people when performance is measured in “person centric” manner, which suggests that large portion of incorrect pose estimates for backward views is due to incorrect matching of left/right limbs.

每个视角下的评估给出了各种视角下明显的性能差异。如图6所示，我们给出了“以人为中心”的标注下的结果，这也是我们在本文全部试验中所用的设置；还给出了“观察者为中心”标注下的结果，其中身体肢体左右的标注是基于其相对于躯干的图像位置。前视角和接近前视角的图像表现最好。我们发现对于背对着的人，在使用“以人为中心”标注的情况下，表现下降很多，这说明后视角下错误的姿态估计的很大一部分，是因为左右肢体的错误匹配。

We observe that all approaches handle extreme viewpoints poorly. PS approach is the only one in our evaluation that gracefully handles in-plane rotations (cluster 12), whereas performance of other approaches significantly degrades in that case. Also, PS outperforms other methods in case of extreme torso tilts (e.g. cluster 14). The performance for clusters with extreme torso rotation is on the level of 20 - 30% PCPm for the best method, corresponding to only 2 - 3 out of 10 body parts being localized correctly for such viewpoints.

我们观察到，所有方法在极端视角下表现都很差。PS方法是唯一一种渐进的应对视角旋转的问题的（聚类12），而其他方法的性能在这种情况下下降很多。而且，PS方法在极端躯干倾斜的情况下（如聚类14）表现超过了其他方法。对于躯干极端旋转的聚类，最好的方法也只能达到20%-30% PCPm，这对应着每10个身体部位中只有2-3个在这种视角下被正确定位。

**Part length performance**. Fig. 4 also shows the influence of part length on the performance of each approach. In this context, foreshortening is the most influential aspect and considered an important challenge for articulated pose estimation. The key observation is that the presence or absence of foreshortening has relatively little influence on the result compared to the other factors such as pose and occlusion. The best performing PS model is the most robust to foreshortening compared to other three approaches. For example the performance for the first 4000 images ordered by increasing foreshortening remains nearly constant.

**部位长度影响下的表现**。图4中还给出了每种方法中部位长度对性能的影响。在这种上下文中，透视是影响最大的因素，是铰接姿态估计中的重要挑战。观察到的最重要的结果是，与姿态和遮挡相比，透视的存在与否对结果的影响很小。表现最好的PS模型对透视最为稳健。比如，按照透视程度逐渐增加的顺序，前4000幅图像上的表现几乎是一样的。

**Activity performance**. Finally, we evaluate pose estimation performance as a function of the person activity. To that end we group test images by the activity categories in the hierarchy used for the image collection [1] and compute PCPm for each category. The results are shown in Fig. 7, where we order categories from left to right according to the number of test examples.

We observe strong variation of performance for different activity types. Best results are obtained on the sports- and dancing-centric activities (e.g. “Sports”, “Running”, “Winter Activities” and “Dancing”). Most difficult turn out to be activities that are performed in bulky clothing and involve use of tools (e.g. “Home Repair”) and activities performed in cluttered scenes (e.g. “Fishing and Hunting”). MODEC outperforms all other approaches on the “Self care” activities (examples of activities from this category are “Eating, sitting”, “Hairstyling”, “Grooming” etc. with “Eating, sitting” containing by far the largest number of images.)

**活动类型影响下的表现**。最后，我们讨论人的活动类型对姿态估计性能的影响。为这个目的，我们将测试图像按照活动类型[1]进行分类，计算每个类别的PCPm。结果如图7所示，其中从左到右是按照测试样本的数量排序的。

我们观察到，对于不同的活动类型，性能变化非常强烈。在运动类和舞蹈类的活动中得到了最好的结果（如运动，跑步，冬季活动和舞蹈）。最难的活动种类中一般都穿着笨重的一幅，或使用工具（如家庭修理），或在杂乱的场景中进行的活动（如钓鱼和打猎）。在“自我照顾”这个类别中，MODEC取得了最好的结果（如吃饭、坐着、梳头、打扮等，吃饭、坐着中的图像数量目前是最多的）。

**Retrained models**. To showcase the usefulness of the benchmark as an analysis tool we retrain the PS and FMP models on the training set from our benchmark. To speed up training we consider a subset of 4000 images, which is 4 times as many images as in the LSP and 40 times as many as in the PARSE datasets used by the publicly available PS and FMP models. The results are shown in Tab. 3. FMP significantly benefits from retraining (44.7 PCPm for retrained vs. 38.3 for original). PS achieves slightly better result, although overall improvement due to retraining is smaller (46.1 PCPm for retrained vs. 42.3 PCPm the original).

**重新训练的模型**。为展现我们的基准测试作为分析工具的作用，我们重新训练了PS和FMP模型，在我们的基准测试的训练集中。为加速训练，我们使用了4000图像的子集，这是PS和FMP所使用的LSP数据集的4倍，PARSE数据集的40倍。结果如表3所示。FMP经过重新训练性能显著提升(44.7% PCPm vs 38.3% PCPm)。PS提升的较少一些(46.1% PCPm vs 42.3% PCPm)。

Table 3. Comparison of performance (PCPm) before and after retraining. For PCKh results see supplementary material.

Setting | Torso | Upper leg | Lower leg | Upper arm | Fore-arm | Head | Upper body | Full body
--- | --- | --- | --- | --- | --- | --- | --- | ---
Yang&Ramanan [22] | 61.0 | 36.6 | 36.5 | 34.8 | 17.4 | 70.2 | 33.1 | 38.3
Yang&Ramanan [22] retrained | 69.3 | 39.5 | 38.8 | 43.4 | 27.7 | 74.6 | 42.3 | 44.7
Pishchulin et al. [15] | 63.8 | 39.6 | 37.3 | 39.0 | 26.8 | 70.7 | 39.1 | 42.3
Pishchulin et al. [15] retrained | 68.4 | 42.7 | 42.8 | 42.0 | 29.2 | 76.3 | 42.1 | 46.1

Although performances for FMP and PS are close overall, we observe interesting differences when examining performance at the level of individual activities and viewpoints (thereby exploiting the rich annotations of our benchmark). Results are shown in Fig. 8. We observe that our publicly available PS model is winning by a large margin on the highly articulated categories, such as “Dancing” and “Running”. Retraining the model boosts performance on activities with less articulation but more complex appearance (e.g. “Home Activities”, “Lawn and Garden”, “Bicycling”, and “Occupation”). Our results show that training on the larger amount of more variable data significantly improved robustness of FMP to viewpoint changes. Performance of FMP improves on the difficult viewpoints by a large margin (e.g. for viewpoint cluster 10 improvement is from 17 to 31% PCPm). Retraining improves the performance of PS model on difficult viewpoints as well, although not as dramatically as for FMP, likely because PS already models in-plane rotations explicitly.

虽然FMP和PS的性能总体上比较接近，但是当我们分单个运动类别和视角来观察性能时，可以得到有趣的差别，如图8所示。我们观察到，在高度铰接的类别中，如舞蹈和跑步，公开可用的PS模型反而性能更好。重新训练的模型对于铰接较少但外表更复杂的情况提升更多，如家庭活动、草坪和花园、骑自行车、职业活动等。我们的结果还显示，在更多的变化更大的数据中训练，可以显著改进FMP对视角变化的稳健性。FMP在困难视角上的表现改进了很多（如，对于视角聚类10的改进是从17%到31% PCPm）。重新训练也改进了PS模型在困难视角下的表现，但是改进程度没有FMP高，这可能是因为PS已经对平面内旋转的情况考虑进去了。

## 4. Conclusion 结论

In this work we advance the state of the art in human pose estimation by establishing new qualitatively higher standards for evaluation and analysis of pose estimation methods and demonstrate the most promising research directions for the next years. To that end we propose a novel “MPII Human Pose” benchmark that we collected by leveraging a taxonomy of activities established in the literature. Compared to current datasets our benchmark covers significantly wider range of human poses spanning from householding to recreational activities and sports. Rich labeling of the collected data and a set of developed evaluation tools enable comprehesive analysis which we perform to demonstrate the strengths and weaknesses of the current methods.

在本文中，我们推进了人体姿态估计的前沿，创建了一个新的更高的评估标准，并分析了姿态估计方法，展示了以后最有希望的研究方向。为了这个目的，我们提出了一种新的MPII人体姿态基准测试，利用了已有的活动分类法收集的样本。与现有的数据集相比，我们的基准测试覆盖的人体姿态范围更广，从家庭活动到娱乐活动、体育活动。对收集数据的丰富标注，以及一套高级评估工具，使我们可以进行综合的分析，以展示目前方法的优势和缺点。

Our findings indicate that current methods are significantly challenged by cases outside their comfort zone, such as large torso rotation and loose clothing. From all other factors, pose complexity has the most profound effect on the pose estimation performance. Current methods perform best on activities with simple tight clothing (e.g. in sport scenes), and are challenged by images with complex clothing and background clutter that are typical for many occupational and outdoor activities.

我们的发现说明，目前的方法在脱离其舒适区后，会受到严峻的挑战，比如躯干旋转角度很大，或宽松的衣着条件。我们考虑了很多因素，其中姿态复杂度对姿态估计性能影响最大。目前的方法在简单的紧身着装下的活动中表现最好（如运动场景），但在复杂着装和背景杂乱的情况下受到了挑战，这在很多职业活动和户外活动中都很常见。

We will make the data, rich annotations for training images and evaluation tools publicly available in order to enable detailed analysis of future pose estimation methods. To prevent accidentally tuning on the test set, the annotations for the test images will be withheld and made accessible through an online evaluation tool. In the future we plan to extend our benchmark to joint pose estimation of multiple people and pose estimation in image sequences.

我们会公开数据、丰富的标准和评估工具，使其他人可以详尽的分析未来的姿态估计方法。为防止意外在测试集上过拟合，测试图像的标注不会公开，只能通过一种在线评估工具可用。在未来，我们计划扩展此基准测试，使其可以进行多人姿态估计，和图像序列中的姿态估计。

**Acknowledgements**. This work has been supported by the Max Planck Center for Visual Computing & Communication. The authors are thankful to Steve Hillyer and numerous anonymous Mechanical Turk workers for the help with preparation of the dataset.