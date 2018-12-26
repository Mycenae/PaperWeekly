# CrowdHuman: A Benchmark for Detecting Human in a Crowd

Shao Shuai et al. Megvii Inc (Face++)

## Abstract 摘要

Human detection has witnessed impressive progress in recent years. However, the occlusion issue of detecting human in highly crowded environments is far from solved. To make matters worse, crowd scenarios are still under-represented in current human detection benchmarks. In this paper, we introduce a new dataset, called CrowdHuman, to better evaluate detectors in crowd scenarios. The CrowdHuman dataset is large, rich-annotated and contains high diversity. There are a total of 470K human instances from the train and validation subsets, and 22.6 persons per image, with various kinds of occlusions in the dataset. Each human instance is annotated with a head bounding-box, human visible-region bounding-box and human full-body bounding-box. Baseline performance of state-of-the-art detection frameworks on CrowdHuman is presented. The cross-dataset generalization results of CrowdHuman dataset demonstrate state-of-the-art performance on previous dataset including Caltech-USA,CityPersons, and Brainwash without bells and whistles. We hope our dataset will serve as a solid baseline and help promote future research in human detection tasks.

人体检测近些年来取得了非常大的进展。但是，高度拥挤的环境中人体检测的遮挡问题还远未解决。问题更大的是，目前的人体检测基准测试中，拥挤的情况仍然没有受到重视。本文中，我们给出了一个新的数据集，称为CrowdHuman，可以更好的评估拥挤情况下的检测器。CrowdHuman数据集规模很大，标注丰富，变化多样。训练集和验证集中总计有47万个人体实例，每幅图像中平均有22.6个人体，在数据集中有各种各样的遮挡情况。每个人体实例都进行了三种标注，即头部边界框，人体可见区域边界框和人体完整身体边界框。目前最好的检测框架在CrowdHuman上的检测基准在文中相应给出。CrowdHuman数据集的跨数据集泛化结果展示了在之前的数据集，包括Caltech-USA，CityPersons和Brainwash上的结果是目前最好的结果。我们希望我们的数据集会成为一个坚实的基准，帮助推进人体检测任务的将来研究。

## 1. Introduction 引言

Detecting people in images is among the most important components of computer vision and has attracted increasing attention in recent years [29, 14, 32, 30, 10, 5, 4, 6, 18]. A system that is able to detect human accurately plays an essential role in applications such as autonomous cars, smart surveillance, robotics, and advanced human machine interactions. Besides, it is a fundamental component for research topics like multiple-object tracking [13], human pose estimation [28], and person search [24]. Coupled with the development and blooming of convolutional neural networks (CNNs) [12, 22, 8], modern human detectors [1, 29, 26] have achieved remarkable performance on several major human detection benchmarks.

图像中的人体检测是计算机视觉中最重要的组成部分之一，近年来吸引了越来越多的注意力[29, 14, 32, 30, 10, 5, 4, 6, 18]。能够精确的检测人体的系统在自动驾驶、智能监控、机器人和高级人机互动中都是极其重要的角色。另外，这还是一些研究课题如多目标跟踪[13]，人体姿势估计[28]，和人体搜索[24]的基础组成部分。与CNN的突破性进展相结合[12,22,8]，现代人体检测器[1,29,26]在几个主要的人体检测基准测试中都取得了很好的成绩。

However, as the algorithms improve, more challenging datasets are necessary to evaluate human detection systems in more complicated real world scenarios, where crowd scenes are relatively common. In crowd scenarios, different people occlude with each other with high overlaps and cause great difficulty of crowd occlusion. For example, when a target pedestrian T is largely overlapped with other pedestrians, the detector may fail to identify the boundaries of each person as they have similar appearances. Therefore, detector will treat the crowd as a whole, or shift the target bounding box of T to other pedestrians mistakenly. To make matters worse, even though the detectors are able to discriminate different pedestrians in the crowd, the highly overlapped bounding boxes will also be suppressed by the post process of non-maximum suppression (NMS). As a result, crowd occlusion makes the detector sensitive to the threshold of NMS. A lower threshold may lead to drastically drop on recall, while a higher threshold brings more false positives.

但是，随着算法的改进，需要更具有挑战性的数据集在更复杂的真实世界情景中评估人体检测系统，其中拥挤的情况是相对常见的。在拥挤的情况中，不同的人相互遮挡，高度重叠，形成了群体遮挡的高难度情况。比如，当目标行人T与其他行人大部分重叠时，检测器不能分辨每个人的边界，因为他们样貌很类似。所以，检测器会将整个群体当成一个整体，或者将T的边界框误检测为其他人的。即使检测器能够分辨人群中不同的行人，但高度重叠的边界框会被后续的非最大抑制(NMS)所抑制。结果是，人群遮挡使检测器对NMS的阈值很敏感。较低的阈值会导致召回率急剧下降，而较高的阈值会导致更多的误报。

Current datasets and benchmarks for human detection, such as Caltech-USA [6], KITTI [25], CityPersons [31], and “person” subset of MSCOCO [17], have contributed to a rapid progress in the human detection. Nevertheless, crowd scenarios are still under-represented in these datasets. For example, the statistical number of persons per image is only 0.32 in Caltech-USA, 4.01 in COCOPersons, and 6.47 in CityPersons. And the average of pairwise overlap between two human instances (larger than 0.5 IoU) in these datasets is only 0.02, 0.02, and 0.32, respectively. Furthermore, the annotators for these datasets are more likely to annotate crowd human as a whole ignored region, which cannot be counted as valid samples in training and evaluation.

现在的人体检测数据集和测试基准，如Caltech-USA[6]，KITTI[25]，CityPersons[31]和MSCOCO[17]的“person”子集，为人体检测的快速发展做出了贡献。但是，拥挤人群的情形在这些数据集中仍然没有得到重视。比如，在Caltech-USA中平均每幅图像中人的数量为0.32，在COCOPersons中为4.01，在CityPersons中为6.47。这些数据集中两个人体实例的成对重叠（IOU大于0.5）的平均数量分为别0.02，0.02和0.32。更进一步，这些数据集的标注者更可能将人群标注为一个整体忽略的区域，不能再训练和评估中视作有效样本。

Our goal is to push the boundary of human detection by specifically targeting the challenging crowd scenarios. We collect and annotate a rich dataset, termed CrowdHuman, with considerable amount of crowded pedestrians. CrowdHuman contains 15,000, 4,370 and 5,000 images for training, validation, and testing respectively. The dataset is exhaustively annotated and contains diverse scenes. There are totally 470k individual persons in the train and validation subsets, and the average number of pedestrians per image reaches 22.6. We also provide the visible region bounding-box annotation, and head region bounding-box annotation along with its full body annotation for each person. Fig. 1 shows examples in our dataset compared with those in other human detection datasets.

我们的目标是针对拥挤人群的情景推进人体检测的研究。我们收集并标注了一个大数据集，称为CrowdHuman，有大量的行人人群。CrowdHuman的训练集、验证集和测试集分别包括15000，4370和5000幅图像。数据集进行了完全标注，包含众多场景。在训练集和验证集中共计有47万个人体实例，每幅图中的平均行人数量为22.6。我们还给出了三种标注，包括人体可见区域边界框标注、头部区域边界框标注和人体整体边界框标注。图1所示的是我们数据集和其他人体检测数据集的例子对比。

Figure 1. Illustrative examples from different human dataset benchmarks. The images inside the green, yellow, blue boxes are from the COCO [17], Caltech [6], and CityPersons [31] datasets, respectively. The images from the second row inside the red box are from our CrowdHuman benchmark with full body, visible body, and head bounding box annotations for each person.

To summarize, we propose a new dataset called CrowdHuman with the following three contributions: 总结一下，我们提出了一种新的数据集称为CrowdHuman，有如下三项贡献：

- To the best of our knowledge, this is the first dataset which specifically targets to address the crowd issue in human detection task. More specifically, the average number of persons in an image is 22.6 and the average of pairwise overlap between two human instances (larger than 0.5 IoU) is 2.4, both of which are much larger than the existing benchmarks like CityPersons, KITTI and Caltech.

- 据我们所知，这是第一个针对性解决人体检测任务中拥挤人群问题的数据集。每幅图像中的平均人体数量为22.6，两个人体实例成对重叠（IOU大于0.5）的平均数量为2.4，两项指标都比现有的基准测试（如CityPersons，KITTI和Caltech）要高很多。

- The proposed CrowdHuman dataset provides annotations with three categories of bounding boxes: head bounding-box, human visible-region bounding-box, and human full-body bounding-box. Furthermore, these three categories of bounding-boxes are bound for each human instance.

- 提出的CrowdHuman数据集给出了三种边界框标注：头部边界框，人体可见部分边界框和人体完整部分边界框。更进一步，这三种边界框对每个人体实例都进行了标注。

- Experiments of cross-dataset generalization ability demonstrate our dataset can serve as a powerful pretraining dataset for many human detection tasks. A framework originally designed for general object detection without any specific modification provides state-of-the-art results on every previous benchmark including Caltech and CityPersons for pedestrian detection, COCOPerson for person detection, and Brainwash for head detection.

- 跨数据集泛化能力的试验证明，我们的数据集可以作为很多人体检测任务的预训练数据集。通用目标检测框架，不需要特意改动，在我们的数据集上预训练后，可以在每个之前的基准测试中取得目前最好的结果，包括Caltech，CityPersons的行人检测，COCOPerson的人体检测和Brainwash的头部检测。

## 2. Related Work 相关工作

### 2.1. Human detection datasets. 人体检测数据集

Pioneer works of pedestrian detection datasets involve INRIA [3], TudBrussels [27], and Daimler [7]. These datasets have contributed to spurring interest and progress of human detection, However, as algorithm performance improves, these datasets are replaced by larger-scale datasets like Caltech-USA [6] and KITTI [25]. More recently, Zhang et al. build a rich and diverse pedestrian detection dataset CityPersons [31] on top of CityScapes [2] dataset. It is recorded by a car traversing various cities, contains dense pedestrians, and is annotated with high-quality bounding boxes.

行人检测数据集的先驱工作有INRIA[3]，TudBrussels[27]和Daimler[7]。这些数据集为人体检测算法的发展做出了贡献，但是，当算法性能提升后，这些数据集就被更大规模的数据集像Caltech-USA[6]和KITTI[25]替换了。最近，Zhang等人在CityScapes[2]数据集上构建了一个大型多样化的行人检测数据集CityPersons[31]。这是由一辆车穿行在多个城市中拍摄的，包括密集的行人，标注为高质量的边界框。

Despite the prevalence of these datasets, they all suffer a problem of from low density. Statistically, the Caltech-USA and KITTI datasets have less than one person per image, while the CityPersons has ∼6 persons per image. In these datasets, the crowd scenes are significantly under-represented. Even worse, protocols of these datasets allow annotators to ignore and discard the regions with a large number of persons as exhaustively annotating crowd regions is incredibly difficult and time consuming.

尽管这些数据集很流行，它们都有一个密度低的问题。统计表明，Caltech-USA和KITTI数据集每幅图像人数少于1人，CityPersons每幅图像大约6人左右。在这些数据集中，拥挤人群的场景很少。而且，这些数据集的协议允许标注者忽略并抛弃大量人群聚集的区域，因为完全标注人群区域非常困难，耗时太多。

**Human detection frameworks**. Traditional human detectors, such as ACF [4], LDCF [19], and Checkerboard [32], exploit various filters based on Integral Channel Features (IDF) [5] with sliding window strategy.

**人体检测框架**。传统人体检测器，比如ACF[4]，LDCF[19]和Checkerboard[32]，探索了基于IDF[5]的各种滤波器和滑窗策略。

Recently, the CNN-based detectors have become a predominating trend in the field of pedestrian detection. In [29], self-learned features are extracted from deep neural networks and a boosted decision forest is used to detect pedestrians. Cai et al. [1] propose an architecture which uses different levels of features to detect persons at various scales. Mao et al. [18] propose a multi-task network to further improve detection performance. Hosang et al. [9] propose a learning method to improve the robustness of NMS. Part-based models are utilized in [20, 33] to alleviate occlusion problem. Repulsion loss is proposed to detect persons in crowd scenes [26].

最近，基于CNN的检测器已经在行人检测中成为主流趋势。在[29]中，从深度神经网络中提取出了自学习特征，用boosted decision tree来检测行人。Cai等人[1]提出了一个框架使用不同层次的特征在不同尺度上来检测人体。Mao等[18]提出了一个多任务网络来进一步改进检测性能。Hosang等[9]提出了一种学习方法来改进NMS的稳健性。[20,33]利用了基于组件的模型来缓解遮挡的问题。[26]提出了repulsion loss来检测拥挤人群场景中的人。

## 3. CrowdHuman Dataset

In this section, we describe our CrowdHuman dataset including the collection process, annotation protocols, and informative statistics.

在这个部分中，我们给出CrowdHuman数据集，包括收集过程，标注协议，和信息统计。

### 3.1. Data Collection 数据收集

We would like our dataset to be diverse for real world scenarios. Thus, we crawl images from Google image search engine with ∼ 150 keywords for query. Exemplary keywords include “Pedestrians on the Fifth Avenue”, “people crossing the roads”, “students playing basketball” and “friends at a party”. These keywords cover more than 40 different cities around the world, various activities (e.g., party, traveling, and sports), and numerous viewpoints (e.g., surveillance viewpoint and horizontal viewpoint). The number of images crawled from a keyword is limited to 500 to make the distribution of images balanced. We crawl ∼60,000 candidate images in total. The images with only a small number of persons, or with small overlaps between persons, are filtered. Finally, ∼ 25,000 images are collected in the CrowdHuman dataset. We randomly select 15,000, 4,370 and 5,000 images for training, validation, and testing, respectively.

我们希望数据集包含真实世界多种多样的情景。所以，我们从谷歌图像搜索爬取数据时，使用了大约150个关键词进行检索。一些关键词例子包括“第五大道上的行人”，“过马路的行人”，“打篮球的学生”和“聚会上的朋友”。这些关键词覆盖了世界上40个不同的城市，不同的活动（如聚会，旅行，运动），很多视角（如监控视角和水平视角）。一个关键词爬取的图像数量限制在500，以均衡图像分布。我们爬取了大约6万幅图像作为候选。包含人数较少的图像，人与人之间重叠较少的图像，都被过滤掉了。最后，CrowdHuman数据集收集了约25000幅图像。我们随机选择了15000、4370、5000幅图像分别作为训练、验证和测试用。

### 3.2. Image Annotation 图像标注

We annotate individual persons in the following steps. 我们用如下步骤标注单个人。

- We annotate a full bounding box of each individual exhaustively. If the individual is partly occluded, the annotator is required to complete the invisible part and draw a full bounding box. Different from the existing datasets like CityPersons, where the bounding boxes annotated are generated via drawing a line from top of the head and the middle of feet with a fixed aspect ratio (0.41), our annotation protocol is more flexible in real world scenarios which have various human poses. We also provide bounding boxes for human-like objects, e.g., statue, with a specific label. Following the metrics of [6], these bounding-boxes will be ignored during evaluation.

- 我们对每个实例都详细标注了完全边界框。如果单个实例被部分遮挡，那么就要求标注者去补全不可见部分，画出一个完整的边界框。现有的数据集，如CityPersons，标注的边界框是这样生成的，从头部上面到脚的中间画一条线，固定纵横比(0.41)，我们的标注协议更灵活，可以包括人体很多姿势。我们还给人形目标提供边界框，如雕塑，并提供特别的标签。遵循[6]的度量标准，这些边界框在评估时会被忽略。

- We crop each annotated instance from the images, and send these cropped regions for annotators to draw a visible bounding box.

- 我们从图像中剪切出每个标注的例子，并将这些剪切出的区域给标注者用来画一条可见的边界框。

- We further send the cropped regions to annotate a head bounding box. All the annotations are double-checked by at least one different annotator to ensure the annotation quality.

- 我们进一步将这些剪切出的区域标注出一个头部的边界框。所有标注都至少由另一位标注者进行二次检查以确保标注质量。

Fig. 2 shows the three kinds of bounding boxes associated with an individual person as well as an example of annotated image.

图2所示的是一个人体实例对应的三种边界框，以及一幅标注的图像例子。

Figure 2. (a) provides an illustrative example of our three kinds of annotations: Head Bounding-Box, Visible Bounding-Box, and Full Bounding-Box. (b) is an example image with our human annotations where magenta mask illustrates the ignored region.

We compare our CrowdHuman dataset with previous datasets in terms of annotation types in Table 1. Besides from the popular pedestrian detection datasets, we also include the COCO [17] dataset with only a “person” class. Compared with CrowdHuman, which provides various types of annotations, Caltech and CityPersons have only normalized full bounding boxes and visible boxes, KITTI has only full bounding boxes, and COCOPersons has only visible bounding boxes. More importantly, none of them has head bounding boxes associated with each individual person, which may serve as a possible means to address the crowd occlusion problem.

我们比较了CrowdHuman数据集和之前的数据集，如表1所示。除了流行的行人检测数据集，我们还包括了COCO[17]数据集的“person”类别。CrowdHuman提供了多种标注，而Caltech和CityPersons只有归一化的整体边界框和可见部分边界框，KITTI只有整体边界框，COCOPersons只有可见部分边界框。更重要的是，其余数据集都没有头部边界框，CrowdHuman每个个体实例都对应一个边界框，这是解决人群遮挡问题的一种可能方法。

Table 1. Comparison of different annotation types for the popular human detection benchmarks. †: Aligned to a certain ratio.

| | Caltech | KITTI | CityPersons | COCOPersons | CrowdHuman
--- | --- | --- | --- | --- | ---
Full BBox | Y | Y | Y† | N | Y
Visible BBox | Y | N | Y | Y | Y
Head BBox | N | N | N | N | Y

### 3.3. Dataset Statistics 数据集统计

**Dataset Size**. The volume of the CrowdHuman training subset is illustrated in the first three lines of Table 2. In a total of 15,000 images, there are ∼340k person and ∼99k ignore region annotations in the CrowdHuman training subset. The number is more than 10x boosted compared with previous challenging pedestrian detection dataset like CityPersons. The total number of persons is also noticeably larger than the others.

**数据集规模**。CrowdHuman训练子集的容量如表2中的前3行所述。CrowdHuman训练子集共计有1.5万幅图像，标注了约34万个人体实例，约9.9万个忽略区域。与之前的行人检测数据集如CityPersons相比，数量多了10倍。人体的总计数量也比其他数据集多了很多。

Table 2. Volume, density and diversity of different human detection datasets. For fair comparison, we only show the statistics of training subset. 人体检测数据集的容量、密度和多样性。为公平对比，我们只给出训练子集的统计。

| | Caltech | KITTI | CityPersons | COCOPersons | CrowdHuman
--- | --- | --- | --- | --- | ---
images | 42,782 | 3,712 | 2,975 | 64,115 | 15,000
persons | 13,674 | 2,322 | 19,238 | 257,252 | 339,565
ignore regions | 50,363 | 45 | 6,768 | 5,206 | 99,227
person/image | 0.32 | 0.63 | 6.47 | 4.01 | 22.64
unique persons | 1,273 | <2,322 | 19,238 | 257,252 | 339,565

**Density**. In terms of density, on average there are ∼22.6 persons per image in CrowdHuman dataset, as shown in the fourth line of Table 2. We also report the density from the existing datasets in Table 3. Obviously, CrowdHuman dataset is of much higher crowdness compared with all previous datasets. Caltech and KITTI suffer from extremely low-density, for that on average there is only ∼1 person per image. The number in CityPersons reaches ∼7, a significant boost while still not dense enough. As for COCOPersons, although its volume is relatively large, it is insufficient to serve as a ideal benchmark for the challenging crowd scenes. Thanks to the pre-filtering and annotation protocol of our dataset, CrowdHuman can reach a much better density.

**密度**。在密度上，CrowdHuman数据集中平均每幅图中有约22.6个人体实例，如表2第4行所示。我们在表3中列出了已有数据集的密度。显然，CrowdHuman数据集与其他数据集相比，人体实例密度要大的多。Caltech和KITTI密度非常低，平均每幅图像不到1个人。CityPersons的密度接近7，增长很大，但仍然不够密集。对于COCOPersons来说，虽然其容量相对较大，但仍不能成为理想的人群场景的测试基准。多亏了我们数据集的预滤除和标注协议，CrowdHuman可以达到足够高的密度。

Table 3. Comparison of the human density against the widely used human detection dataset. The first column refers to the number of human instances in the image. 与其他广泛使用的人体检测数据集在人体密度上的比较；第1列表示图像中的人体实例。

person/img ≥ | Caltech | KITTI | CityPersons | COCOPersons | CrowdHuman
--- | --- | --- | --- | --- | ---
1 | 7839 18.3% | 969 26.1% | 2482 83.4% | 64115 100.0% | 15000 100.0%
2 | 3257 7.6% | 370 10.0% | 2082 70.0% | 39283 61.3% | 15000 100.0%
3 | 1265 3.0% | 273 7.4% | 1741 58.5% | 28553 44.5% | 14996 100.0%
5 | 282 0.7% | 164 4.4% | 1225 41.2% | 18775 29.3% | 14220 94.8%
10 | 36 0.1% | 19 0.5% | 610 20.5% | 9604 15.0% | 10844 72.3%
20 | 0 0.0% | 0 0.0% | 227 7.6% | 0 0.0% | 5907 39.4%
30 | 0 0.0% | 0 0.0% | 94 3.2% | 0 0.0% | 3294 21.9%

**Diversity**. Diversity is an important factor of a dataset. COCOPersons and CrowdHuman contain people in unlimited poses in a wide range of domains, while Caltech, KITTI and CityPersons are all recorded by a car traversing on streets. The number of identical persons is also critical. As reported in the fifth line in Table 2, this number amounts to ∼33k in CrowdHuman while images in Caltech and KITTI are not sparsely sampled, resulting in less amount of identical persons.

**多样性**。多样性是数据集的重要因素。COCOPersons和CrowdHuman中的人姿势不限，领域宽广，而Caltech、KITTI和CityPersons都是由车辆在街道上录制的。相同人的数量也是很关键的。如表2中第5行中所示，在CrowdHuman中数量达到了约3.3万(?)，而在Caltech和KITTI数据集中并不是稀疏取样的，得到的相同的人的数量很少。

**Occlusion**. To better analyze the distribution of occlusion levels, we divide the dataset into the “bare” subset (occlusion ≤ 30%), the “partial” subset (30% < occlusion ≤ 70%), and the “heavy” subset (occlusion > 70%). In Fig. 3, we compare the distribution of persons at different occlusion levels for CityPersons(The statistics is computed without group people). The bare subset and partial subset in CityPersons constitute 46.79% and 24.19% of entire dataset respectively, while the ratios for CrowdHuman are 29.89% and 32.13%. The occlusion levels are more balanced in CrowdHuman, in contrary to those in CityPersons, which have more persons with low occlusion.

**遮挡**。为更好的分析遮挡程度的分布，我们将数据集分成“基本不遮挡”子集（遮挡小于30%），“部分遮挡”子集（遮挡大于30%小于70%），和“严重遮挡”子集（遮挡大于70%）。如图3所示，我们比较了CityPersons和沃恩数据集在不同遮挡程度下的人体分布（统计数据没有计算group people）。基本不遮挡的子集和部分遮挡的子集在CityPersons中分别占整个数据集的46.79%和24.19%，而在CrowdHuman中则为29.89%和32.13%。遮挡程度在CrowdHuman中更为均衡，而在CityPersons中，更多的人体是低遮挡的。

Figure 3. Comparison of the visible ratio between our CrowdHuman and CityPersons dataset. Visible Ratio is defined as the ratio of visible bounding box to the full bounding box.

We also provide statistics on pair-wise occlusion. For each image, We count the number of person pairs with different intersection over union (IoU) threshold. The results are shown in Table 4. In average, few person pairs with an IoU threshold of 0.3 are included in Caltech, KITTI or COCOPersons. For CityPersons dataset, the number is less than one pair per image. However, the number is 9 for CrowdHuman. Moreover, There are averagely 2.4 pairs whose IoU is greater than 0.5 in the CrowdHuman dataset. We further count the occlusion levels for triples of persons. As shown in Table 5, such cases can be hardly found in previous datasets, while they are well-represented in CrowdHuman.

我们还给出了成对遮挡的统计。对于每幅图像，我们数出不同IOU阈值的人体成对遮挡数量。结果如表4所示。平均来说，在Caltech、KITTI或COCOPersons中，IOU阈值0.3的人体对很少。在CityPersons数据集中，每幅图像数量少于1。但是，在CrowdHuman中数量为9。而且，在CrowdHuman数据集中，IOU大于0.5的平均有2.4对。我们还进一步统计了三人遮挡的情况。如表5所示，这种情况在之前的数据集里很少发现，而在CrowdHuman中则有不少。

Table 4. Comparison of pair-wise overlap between two human instances.

pair/img | Cal | City | COCO | CrowdHuman
--- | --- | --- | --- | ---
iou>0.3 | 0.06 | 0.96 | 0.13 | 9.02
iou>0.4 | 0.03 | 0.58 | 0.05 | 4.89
iou>0.5 | 0.02 | 0.32 | 0.02 | 2.40
iou>0.6 | 0.01 | 0.17 | 0.01 | 1.01
iou>0.7 | 0.00 | 0.08 | 0.00 | 0.33
iou>0.8 | 0.00 | 0.02 | 0.00 | 0.07
iou>0.9 | 0.00 | 0.00 | 0.00 | 0.01

Table 5. Comparison of high-order overlaps among three human instances.

pair/img | Cal | City | COCO | CrowdHuman
--- | --- | --- | --- | ---
iou>0.1 | 0.02 | 0.30 | 0.02 | 8.70
iou>0.2 | 0.00 | 0.11 | 0.00 | 2.09
iou>0.3 | 0.00 | 0.04 | 0.00 | 0.51
iou>0.4 | 0.00 | 0.01 | 0.00 | 0.12
iou>0.5 | 0.00 | 0.00 | 0.00 | 0.03

## 4. Experiments 实验

In this section, we will first discuss the experiments on our CrowdHuman dataset, including full body detection, visible body detection and head detection. Meanwhile, the generalization ability of our CrowdHuman dataset will be evaluated on standard pedestrian benchmarks like Caltech and CityPersons, person detection benchmark on COCOPersons, and head detection benchmark on Brainwash dataset. We use FPN [15] and RetinaNet [16] as two baseline detector store present the two-stage algorithms and one-stage algorithms, respectively.

在这一部分，我们首先讨论在CrowdHuman数据集上的试验，包括完整人体检测，可见人体检测和头部检测。同时，CrowdHuman数据集的泛化能力将在标准行人数据集如Caltech和CityPersons上进行评估，也在人体检测数据集如COCOPersons上进行评估，也在头部检测标准测试的Brainwash数据集上进行评估。我们分别使用FPN[15]和RetinaNet[16]作为两阶段算法和一阶段算法的基准检测器。

### 4.1. Baseline Detectors 基准检测器

Our baseline detectors are Faster R-CNN [21] and RetinaNet [16], both based on the Feature Pyramid Network (FPN) [15] with a ResNet-50 [8] back-bone network. Faster R-CNN and RetinaNet are both proposed for general object detection, and they have dominated the field of object detection in recent years.

我们的基准检测器为Faster R-CNN[21]和RetinaNet[16]，这都是基于FPN[15]的，骨干网络为ResNet-50[8]。Faster R-CNN和RetinaNet都是作为通用目标检测提出来的，而且已经主导了近几年的目标检测领域。

### 4.2. Evaluation Metric 评估标准

The training and validation subsets of CrowdHuman can be downloaded from our website. In the following experiments, our algorithms are trained based on CrowdHuman train subset and the results are evaluated in the validation subset. An online evaluation server will help to evaluate the performance of the testing subset and a leaderboard will be maintained. The annotations of testing subset will not be made publicly available.

CrowdHuman的训练和验证集可以从网站上下载。在下面的实验中，我们的算法是由CrowdHuman训练集训练的，结果在验证集中进行评估。在线的评估服务器将会评估在测试集上的表现，并维护一个排行榜。测试集的标注将不会开放。

We follow the evaluation metric used for Caltech [6], denoted as mMR, which is the average log miss rate over false positives per-image ranging in [0.01, 1]. mMR is a good indicator for the algorithms applied in the real world applications. Results on ignored regions will not considered in the evaluation. Besides, Average Precision (AP) and recall of the algorithms are included for reference.

我们遵循Caltech[6]的评估准则，即mMR，也就是误报率为[0.01,1]的每个图像的平均log漏报率。mMR是应用在真实世界的算法的很好的指示器。被忽略的区域的结果在评估时不会考虑。另外，算法的平均精度AP和召回率也记录作为参考。

### 4.3. Implementation Details 实现细节

We use the same setting of anchor scales as [15] and [16]. For all the experiments related to full body detection, we modify the height v.s. width ratios of anchors as {1 : 1,1.5 : 1,2 : 1,2.5 : 1,3 : 1} in consideration of the human body shape. While for visible body detection and human head detection, the ratios are set to {1 : 2,1 : 1,2 : 1}, in comparison with the original papers. The input image sizes of Caltech and CityPersons are set to 2× and 1× of the original images according to [31]. As the images of CrowdHuman and MSCOCO are both collected from the Internet with various sizes, we resize the input so that their short edge is at 800 pixels while the long edge should be no more than 1400 pixels at the same time. The input sizes of Brainwash is set as 640 × 480.

我们使用[15]和[16]中相同的锚尺度设置。对于所有与整体人体检测相关的试验，我们修正锚窗的高宽比为{1:1, 1.5:1, 2:1, 2.5:1, 3:1}，以适应人体形状。而对于可见人体检测和头部检测，比例设置为{1:2, 1:1, 2:1}。根据[31]，Caltech和CityPersons的输入图像尺寸设置为原图像的2倍和1倍。因为CrowdHuman和MSCOCO的图像都是从互联网上收集的，尺寸不一，我们改变图像尺寸，使其短边为800像素，同时长边不应超过1400像素。Brainwash的输入大小设置为640×480。

We train all datasets with 600k and 750k iterations for FPN and RetinaNet, respectively. The base learning rate is set to 0.02 and decreased by a factor of 10 after 150k and 450k for FPN, and 180k and 560k for RetinaNet. The Stochastic Gradient Descent (SGD) solver is adopted to optimize the networks on 8 GPUs. A mini-batch involves 2 images per GPU, except for CityPersons where a mini-batch involves only 1 image due to the physical limitation of GPU memory. Weight decay and momentum are set to 0.0001 and 0.9. We do not finetune the batch normalization [11] layers. Multi-scale training/testing are not applied to ensure fair comparisons.

对于FPN和RetinaNet，我们训练所有的数据集迭代次数分别为60万次和75万次。基准学习速率为0.02，对于FPN，在15万次和45万次后除以10，对于RetinaNet在18万次和56万次后除以10。采用SGD在8GPU上优化网络。Mini-batch大小为每GPU 2幅图像，在CityPersons上是个例外，由于GPU内存的限制所以为每GPU 1幅图像。权值衰减和动量设置为0.0001和0.9。我们没有精调批归一化层[11]。公平起见，没有进行多尺度训练/测试。

### 4.4. Detection results on CrowdHuman 数据集上的检测结果

**Visible Body Detection**. As the human have different poses and occlusion conditions, the visible regions may be quite different for each individual person, which brings many difficulties to human detection. Table 6 illustrates the results for the visible part detection based on FPN and RetinaNet. FPN outperforms RetinaNet in this case. According to Table 6, the proposed CrowdHuman dataset is a challenging benchmark, especially for the state-of-the-art human detection algorithms. The illustrative examples of visible body detection based on FPN are shown in Fig. 5.

**可见人体检测**。由于人体有多种姿态和遮挡情况，对于每个人体实例来说，可能会有非常不同的可见区域，这会给人体检测带来很多困难。表6给出了基于FPN和RetinaNet的可见部分检测结果。FPN在这种情况中超过了RetinaNet的表现。根据表6，我们提出的CrowdHuman数据集是一个有挑战的数据集，尤其是对最新的人体检测算法。基于FPN的可见人体检测的结果如图5所示。

Table 6. Evaluation of visible body detection on CrowdHuman benchmark.

| | Recall | AP | mMR
--- | --- | --- | --- | ---
FPN [15] | 91.51 | 85.60 | 55.94
RetinaNet [16] | 90.96 | 77.19 | 65.47

**Full Body Detection**. Detecting full body regions is more difficult than detecting the visible part as the detectors should predict the occluded boundaries of the full body. To make matters worse, the ground-truth annotation might be suffered from high variance caused by different decision-makings by different annotators.

**整体人体检测**。检测人体整体区域比检测可见人体要更难，因为检测器要预测整体人体被遮挡的边缘。而真值标记还可能因为不同标注者的不同决策面临很大的变化，这使问题更糟糕。

Different from the visible part detection, the aspect ratios of the anchors for the full body detection are set as [1.0,1.5,2.0,2.5,3.0] to make the detector tend to predict the slim and tall bounding boxes. Another important thing is that the RoIs are not clipped into the limitation of the image boundaries, as there are many full body bounding boxes extended out of images. The results are shown in Table 7 and the illustrative examples of FPN are shown in Fig. 4. Similar to the Visible body detection, FPN has a significant gain over RetinaNet.

与可见人体检测不同，整体人体检测的锚窗的宽高比设置为[1.0,1.5,2.0,2.5,3.0]，这样检测器才会倾向于去预测又细又高的边界框。另一个重要的事是RoI不受图像边缘的界限的影响，因为又很多整体人体边界框延展到图像之外。结果如表7所示，FPN的结果如图4所示。与可见人体检测类似，FPN比RetinaNet结果要好。

Table 7. Evaluation of full body detection on CrowdHuman benchmark.

| | Recall | AP | mMR
--- | --- | --- | ---
FPN [15] | 90.24 | 84.95 | 50.42
RetinaNet [16] | 93.80 | 80.83 | 63.33
FPN on Caltech | 99.76 | 89.95 | 10.08
FPN on CityPersons | 97.97 | 94.35 | 14.81

In Table 7, we also report the FPN pedestrian detection results(The results are evaluated on the standard reasonable set) on Caltech, i.e., 10.08 mMR, and CityPersons, i.e., 14.81 mMR. It shows that our CrowdHuman dataset is much challenging than the standard pedestrian detection benchmarks based on the detection performance.

在表7中，我们还给出了FPN在Caltech和CityPersons的行人检测结果，即10.08mMR和14.81mMR（结果在标准集上评估）。结果显示，CrowdHuman数据集比标准行人检测测试基准要更又挑战性。

**Head Detection**. Head is one of the most obvious parts of a whole body. Head detection is widely used in the practical applications such as people number counting, face detection and tracking. We compare the results of FPN and RetinaNet as shown in Table 8. The illustrative examples of head detection on CrowdHuman by FPN detector are shown in Fig. 6.

**头部检测**。头部是身体最明显的一个部位。头部检测在实际中有广泛的应用，如人数统计，人脸检测和跟踪。我们比较了FPN和RetinaNet的结果，如表8所示。FPN在CrowdHuman上的检测结果如图6所示。

Table 8. Evaluation of Head detection on CrowdHuman benchmark.

| | Recall | AP | mMR
--- | --- | --- | ---
FPN [15] | 81.10 | 77.95 | 52.06
RetinaNet [16] | 78.43 | 71.36 | 60.64

### 4.5. Cross-dataset Evaluation 跨数据集评估

As shown in Section 3, the size of CrowdHuman dataset is obviously larger than the existing benchmarks, like Caltech and CityPersons. In this section, we evaluate that the generalization ability of our CrowdHuman dataset. More specifically, we first train the model on our CrowdHuman dataset and then finetune it on the visible body detection benchmarks like COCOPersons [17], full body detection benchmarks like Caltech [6] and CityPersons [31], and head detection benchmarks like Brainwash [23]. As reported in Section 4.4, FPN is superior to RetinaNet in all three cases. Therefore, in the following experiments, we adopt FPN as our baseline detector.

如第3节所示，CrowdHuman数据集明显比现有的测试基准数据集要大，如Caltech和CityPersons。在本节中，我们评估CrowdHuman数据集的泛化能力。确切的说，我们首先在CrowdHuman数据集上训练模型，然后一些测试基准上精调，如可见人体检测的COCOPersons[17]，整体人体检测基准的Caltech[6]和CityPerson[31]，和头部检测基准的Brainwash[23]。如同在4.4节中报告的一样，FPN在所有情况中都比RetinaNet好。所以，在以下的试验中，我们采取FPN作为我们的基准检测器。

**COCOPersons**. COCOPersons is a subset of MSCOCO from the images with groundtruth bounding box of “person”. The other 79 classes are ignored in our evaluation. After the filtering process, there are 64115 images from the trainval minus minival for training, and the other 2639 images from minival for validation. All the persons in COCOPersons are annotated as the visible body with different type of human poses. The results are illustrated in Table 9. Based on the pretraining of our CrowdHuman dataset, our algorithm has superior performance on the COCOPersons benchmark against the one without CrowdHuman pretraining.

**COCOPersons**。COCOPersons是MSCOCO的一个子集，只包含“person”这一类，其他79类在我们的评估中都被忽略掉。在过滤过程后，在trainval去掉minival集作训练的有64115幅图像，minival中的2639幅图像作为验证集。COCOPersons中的所有person都是作为可见人体标记的，其姿势可能不同。结果如表9所示。在CrowdHuman数据集上进行预训练后，我们的算法在COCOPersons测试标准上有了更好的表现。

Table 9. Experimental results on COCOPersons.
Train-set | Recall | AP | mMR
--- | --- | --- | ---
COCOPersons | 95.57 | 83.83 | 41.89
Crowd⇒COCO | 95.87 | 85.02 | 39.79

**Caltech and CityPersons**. Caltech and CityPersons are widely used benchmarks for pedestrian detection, both of them are usually adopted to evaluate full body detection algorithms. We use the reasonable set for Caltech dataset where the object size is larger than 50 pixels. Table 10 and Table 11 show the results on Caltech and CityPersons, respectively. We compare the algorithms in the first part of the tables with:

**Caltech and CityPersons**。Caltech和CityPersons是广泛使用的行人检测基准测试集，两个都经常用于评估整体人体检测算法。我们使用Caltech数据集中目标大小大于50像素的集合。表10和表11给出了在Caltech和CityPersons上的结果。我们在表格的第一部分这样比较算法：

- FPN trained on the Caltech 在Caltech上训练的FPN
- FPN trained on CityPersons 在CityPersons上训练的FPN
- FPN trained on CrowdHuman 在CrowdHuman上训练的FPN
- FPN model pretrained on CrowdHuman and then finetuned on the corresponding target training set 在CrowdHuman上预训练然后再对应的目标训练集上精调的FPN

Also, state-of-art algorithms on Caltech and CityPersons are reported in the second part of tables as well. To summarize, the results illustrated in Table 10 and Table 11 demonstrate that our CrowdHuman dataset can serve as an effective pretraining dataset for pedestrian detection task on Caltech and CityPersons (The evaluation is based on 1× scale) for full body detection.

同时，目前最好的算法在Caltech和CityPersons上的结果在表格第2部分给出。总结起来，表10和表11中的结果证明了，我们的CrowdHuman数据集可以作为行人检测的有效的预训练数据集，然后在Caltech和CityPersons上作整体人体检测。

Table 10. Experimental results on Caltech dataset.

Train-set | Recall | AP | mMR
--- | --- | --- | ---
Caltech | 99.76 | 89.95 | 10.08
CityPersons | 99.05 | 85.81 | 14.69
CrowdHuman | 99.88 | 90.58 | 8.81
Crowd⇒Calt | 99.88 | 95.69 | 3.46
CityPersons⇒Calt [31] | - | - | 5.1
Repulsion [26] | - | - | 4.0
[18] | - | - | 5.5

Table 11. Experimental reslts on CityPersons.

Train-set | Recall | AP | mMR
--- | --- | --- | ---
Caltech | 87.21 | 65.87 | 45.52
CityPersons | 97.97 | 94.35 | 14.81
CrowdHuman | 98.73 | 98.10 | 21.18
Crowd⇒City | 97.78 | 95.58 | 10.67
CityPersons [31] | - | - | 14.8
Repulsion [26] | - | - | 13.2

**Brainwash**. Brainwash [23] is a head detection dataset whose images are extracted from the video footage at every 100 seconds. Following the step of [23], the training set has 10,917 images with 82,906 instances and the validation set has 500 images with 3318 instances. Similar to visible body detection and full body detection, Brainwash dataset is evaluated to validate the generalization ability of our CrowdHuman dataset for head detection.

**Brainwash**。Brainwash[23]是一个头部检测数据集，其图像是从视频片段中每隔100秒一帧提取出的。遵循[23]中的步骤，训练集包括10917幅图像，82906个实例，验证集有500幅图像，3318个实例。与可见人体检测、整体人体检测类似，Brainwash数据集的评估是验证我们CrowdHuman数据集在头部检测上的泛化能力。

Table 12 shows the results of head detection task on Brainwash dataset. By using the FPN as the head detector, the performance is already much better than the state-of-art in [23]. On top of that, pretraining on the CrowdHuman dataset further boost the result by 2.5% of mMR, which validates the generalization ability of our CrowdHuman dataset for head detection.

表12所示的是在Brainwash数据集上的头部检测结果。使用了FPN作为头部检测器，其结果已经比[23]中的最好结果要好很多。在此之上，在CrowdHuman数据集上预训练进一步提升了2.5%的mMR，这验证了CrowdHuman数据集在头部检测上的泛化能力。

Table 12. Experimental results on Brainwash.

Train-set | Recall | AP | mMR
--- | --- | --- | ---
Brainwash | 98.52 | 95.74 | 19.77
Crowd⇒Brain | 98.66 | 96.15 | 17.24
[23] | - | 78.0 | -

## 5. Conclusion 结论

In this paper, we present a new human detection benchmark designed to address the crowd problem. There are three contributions of our proposed CrowdHuman dataset. Firstly, compared with the existing human detection benchmark, the proposed dataset is larger-scale with much higher crowdness. Secondly, the full body bounding box, the visible bounding box, and the head bounding box are annotated for each human instance. The rich annotations enables a lot of potential visual algorithms and applications. Last but not least, our CrowdHuman dataset can serve as a powerful pretraining dataset. State-of-the-art results have been reported on benchmarks of pedestrian detection benchmarks like Caltech and CityPersons, and Head detection benchmark like Brainwash. The dataset as well as the code and models discussed in the paper will be released.

在本文中，我们给出了一个新的人体检测基准，其设计是为了解决人群问题。提出的CrowdHuman数据集有三方面的贡献。第一，与现有的人体检测基准测试比较，给出的数据集更大、人群密度高的多。第二，对每个人体实例都进行了整体人体边界框、可见边界框和头部边界框标注。这些丰富的标注使得很多潜在的视觉算法和应用成为可能。最后，我们的CrowdHuman数据集可以作为一个有效的预训练数据集。在行人检测的基准测试中已经给出了最好结果，如Caltech和CityPersons，还有头部检测的基准如Brainwash。本文讨论的这个数据集以及代码、模型将会放出。