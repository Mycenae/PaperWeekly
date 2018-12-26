# Microsoft COCO: Common Objects in Context

Tsung-Yi Lin et al.

## Abstract 摘要

We present a new dataset with the goal of advancing the state-of-the-art in object recognition by placing the question of object recognition in the context of the broader question of scene understanding. This is achieved by gathering images of complex everyday scenes containing common objects in their natural context. Objects are labeled using per-instance segmentations to aid in precise object localization. Our dataset contains photos of 91 objects types that would be easily recognizable by a 4 year old. With a total of 2.5 million labeled instances in 328k images, the creation of our dataset drew upon extensive crowd worker involvement via novel user interfaces for category detection, instance spotting and instance segmentation. We present a detailed statistical analysis of the dataset in comparison to PASCAL, ImageNet, and SUN. Finally, we provide baseline performance analysis for bounding box and segmentation detection results using a Deformable Parts Model.

我们给出了一个新的数据集，其目的是推进目标识别的最高水平，方法是将目标识别的问题放在更广阔的场景理解的上下文中。我们通过收集复杂的日常场景中包含自然上下文普通目标的图片。目标标注采用每个实例的分割的方式，有助于精确的目标定位。我们的数据集包含91类目标的图像，四岁的孩子可以很容易的识别这些图像。数据集包括32.8万张图像，250万标注的目标实例，数据集的建立利用了广泛众包工作者，使用了新的类别检测、实例发现和实例分割用户界面。我们给出了数据集的详细统计分析，并与PASCAL、ImageNet和SUN数据集进行了比较。最后，我们给出了DPM模型的边界框和分割检测结果的基准性能分析。

## 1 INTRODUCTION 引言

One of the primary goals of computer vision is the understanding of visual scenes. Scene understanding involves numerous tasks including recognizing what objects are present, localizing the objects in 2D and 3D, determining the objects’ and scene’s attributes, characterizing relationships between objects and providing a semantic description of the scene. The current object classification and detection datasets [1], [2], [3], [4] help us explore the first challenges related to scene understanding. For instance the ImageNet dataset [1], which contains an unprecedented number of images, has recently enabled breakthroughs in both object classification and detection research [5], [6], [7]. The community has also created datasets containing object attributes [8], scene attributes [9], keypoints [10], and 3D scene information [11]. This leads us to the obvious question: what datasets will best continue our advance towards our ultimate goal of scene understanding?

计算机视觉的一个基本目标是视觉场景的理解。场景理解涉及到大量任务，包括识别存在什么目标，2D、3D目标定位，确定目标和场景的属性，确定目标间的联系特征，给出场景的语义描述。现在的目标分类和目标检测数据集[1,2,3,4]帮助我们探索了场景理解的第一个挑战。比如ImageNet数据集[1]，包括的图像数量前所未有的多，这使目标分类和目标检测研究都取得了突破[5,6,7]。一些社区创造的数据集包含了目标属性[8]，场景属性[9]，关键点[10]和3D场景信息[11]。这带来了一个明显的问题：什么样的数据集能够最好的促进我们朝向场景理解的终极目标的进步呢？

We introduce a new large-scale dataset that addresses three core research problems in scene understanding: detecting non-iconic views (or non-canonical perspectives [12]) of objects, contextual reasoning between objects and the precise 2D localization of objects. For many categories of objects, there exists an iconic view. For example, when performing a web-based image search for the object category “bike,” the top-ranked retrieved examples appear in profile, unobstructed near the center of a neatly composed photo. We posit that current recognition systems perform fairly well on iconic views, but struggle to recognize objects otherwise – in the background, partially occluded, amid clutter [13] – reflecting the composition of actual everyday scenes. We verify this experimentally; when evaluated on everyday scenes, models trained on our data perform better than those trained with prior datasets. A challenge is finding natural images that contain multiple objects. The identity of many objects can only be resolved using context, due to small size or ambiguous appearance in the image. To push research in contextual reasoning, images depicting scenes [3] rather than objects in isolation are necessary. Finally, we argue that detailed spatial understanding of object layout will be a core component of scene analysis. An object’s spatial location can be defined coarsely using a bounding box [2] or with a precise pixel-level segmentation [14], [15], [16]. As we demonstrate, to measure either kind of localization performance it is essential for the dataset to have every instance of every object category labeled and fully segmented. Our dataset is unique in its annotation of instance-level segmentation masks, Fig. 1.

我们提出了一种新的大规模数据集，解决场景理解中的三个核心研究问题：检测非经典视角[12]的目标，目标间的上下文推理和目标的2D精确定位。对于很多类别的目标，都有一个标志性视角。比如，当进行基于网络的bike目标类别图像搜索，排名最前的样本是bike的外形，没有遮挡，置于图像接近中央的位置。我们假设目前的识别系统在标志性视角可以识别的很好，但其他视角识别性能较差，比如部分遮挡，在目标群当中[13]，这正反应了实际的日常场景。我们在试验中验证这些情况；当用日常场景评估时，用我们的数据集训练出的模型表现比用之前的数据集训练的要好。另一个挑战是找到包含多个目标的自然图像。由于存在目标很小，或模糊的外形的情况，多个目标的情况只能在存在上下文的情况下确定。为推进上下文推理的研究，表现场景的图像[3]非常必要，而不是那些只包含目标的图像。最后，我们认为场景分析的一个核心部分是得到目标空间分布的详细情况。一个目标的空间位置可以粗略的用一个边界框[2]来定义，或用精确的像素级分割定义[14,15,16]。如我们展示的一样，不论衡量哪种定位性能，数据集都必须标注或完整分割每个目标类别的每个实例。我们的数据集在实例级分割掩模标注中是唯一的，如图1所示。

Fig. 1: While previous object recognition datasets have focused on (a) image classification, (b) object bounding box localization or (c) semantic pixel-level segmentation, we focus on (d) segmenting individual object instances. We introduce a large, richly-annotated dataset comprised of images depicting complex everyday scenes of common objects in their natural context.

图1：之前的数据集聚焦在(a)图像分类(b)目标边界框定位(c)像素级的语义分割，我们聚焦在(d)分割目标实例个体。我们提出了一个大型富标准的数据集，包含的图像都是展现常见目标自然上下文的日常场景的。

To create a large-scale dataset that accomplishes these three goals we employed a novel pipeline for gathering data with extensive use of Amazon Mechanical Turk. First and most importantly, we harvested a large set of images containing contextual relationships and noniconic object views. We accomplished this using a surprisingly simple yet effective technique that queries for pairs of objects in conjunction with images retrieved via scene-based queries [17], [3]. Next, each image was labeled as containing particular object categories using a hierarchical labeling approach [18]. For each category found, the individual instances were labeled, verified, and finally segmented. Given the inherent ambiguity of labeling, each of these stages has numerous tradeoffs that we explored in detail.

为创建满足这三个目标的大型数据集，我们采用新的收集数据的流程，广泛的应用了Amazon Mechanical Turk。第一，我们获得了大量包含上下文关系和非经典目标视角的图像。我们完成这个任务使用了一个非常简单但有效的技术，即查询成对的连接在一起的目标的图像，而图像都是由基于场景的查询获得的[17,3]。下一步，每个图像都使用层次化标记方法[18]标注了包含特定类别的目标。对于发现的每个类别，实例个体都进行了标注、确认，最后进行了分割。由于标注的内在模糊性，每个阶段都有很多折中，我们会详细讨论。

The Microsoft Common Objects in COntext (MS COCO) dataset contains 91 common object categories with 82 of them having more than 5,000 labeled instances, Fig. 6. In total the dataset has 2,500,000 labeled instances in 328,000 images. In contrast to the popular ImageNet dataset [1], COCO has fewer categories but more instances per category. This can aid in learning detailed object models capable of precise 2D localization. The dataset is also significantly larger in number of instances per category than the PASCAL VOC [2] and SUN [3] datasets. Additionally, a critical distinction between our dataset and others is the number of labeled instances per image which may aid in learning contextual information, Fig. 5. MS COCO contains considerably more object instances per image (7.7) as compared to ImageNet (3.0) and PASCAL (2.3). In contrast, the SUN dataset, which contains significant contextual information, has over 17 objects and “stuff” per image but considerably fewer object instances overall.

MS COCO数据集包含91种普通目标类别，其中82种有超过5000个标注的个体，如图6所示。数据集总计有32.8万幅图像，250万标记的实例。与流行的ImageNet数据集[1]相比，COCO类别更少，但每个类别实例更多。这有助于可以精确2D定位的模型的学习。数据集比PASCAL[2]和SUN[3]每个类别的实例也多出非常多。另外，我们的数据集与其他数据集的关键区别在于，每幅图中的标注样本是有助于学习上下文信息的，如图5所示。MS COCO包含每幅图中的目标实例数(7.7)与ImageNet(3.0)和PASCAL(2.3)相比多出很多。相比之下，SUN数据集，虽然包含大量上下文信息，每幅图包含17个目标和"stuff"，但总计的实例数量更少。

An abridged version of this work appeared in [19]. [19]是本文的一个删节版。

## 2 RELATED WORK 相关工作

Throughout the history of computer vision research datasets have played a critical role. They not only provide a means to train and evaluate algorithms, they drive research in new and more challenging directions. The creation of ground truth stereo and optical flow datasets [20], [21] helped stimulate a flood of interest in these areas. The early evolution of object recognition datasets [22], [23], [24] facilitated the direct comparison of hundreds of image recognition algorithms while simultaneously pushing the field towards more complex problems. Recently, the ImageNet dataset [1] containing millions of images has enabled breakthroughs in both object classification and detection research using a new class of deep learning algorithms [5], [6], [7].

在计算机视觉的研究历史中，数据集一直扮演着关键的角色。它们不仅提供了训练和评估算法的途径，还把研究推向新的更具挑战性的方向。真值标注立体和光流图像数据集[20,21]帮助刺激了这些领域的研究兴趣。早期的目标识别数据集[22,23,24]演化利用了数百个图像识别算法的直接对比，同时将这个领域推向更复杂的问题。最近，包含数百万张图像的ImageNet数据集[1]使得目标分类和检测研究的突破成为可能，它们都使用了一种新的深度学习算法[5,6,7]。

Datasets related to object recognition can be roughly split into three groups: those that primarily address object classification, object detection and semantic scene labeling. We address each in turn.

与目标识别有关的数据集可以大致分成三组：主要解决目标分类、目标检测和语义场景标记的。我们依次进行讨论。

**Image Classification**。 The task of object classification requires binary labels indicating whether objects are present in an image; see Fig. 1(a). Early datasets of this type comprised images containing a single object with blank backgrounds, such as the MNIST handwritten digits [25] or COIL household objects [26]. Caltech 101 [22] and Caltech 256 [23] marked the transition to more realistic object images retrieved from the internet while also increasing the number of object categories to 101 and 256, respectively. Popular datasets in the machine learning community due to the larger number of training examples, CIFAR-10 and CIFAR-100 [27] offered 10 and 100 categories from a dataset of tiny 32×32 images [28]. While these datasets contained up to 60,000 images and hundreds of categories, they still only captured a small fraction of our visual world.

**图像分类**。目标分类的任务需要二值标签来指示目标是否在图像中存在，见图1(a)。早期的这种数据集其组成图像是包含单个目标空白背景的，比如MNIST手写数字[25]，或COIL家居物品[26]。Caltech 101 [22]和Caltech 256 [23]标志着向更实际的目标图像的转变，它们都是从互联网上获得的，同时将目标类别从101增加到256。机器学习团体中流行的数据集取决于训练样本的多少，CIFAR-10和CIFAR-100 [27]给出了10类和100类的32×32小图像[28]。这些数据集包含最多6万幅图像，数百个类别，但它们仍然只包括我们视觉世界的很小一部分。

Recently, ImageNet [1] made a striking departure from the incremental increase in dataset sizes. They proposed the creation of a dataset containing 22k categories with 500-1000 images each. Unlike previous datasets containing entry-level categories [29], such as “dog” or “chair,” like [28], ImageNet used the WordNet Hierarchy [30] to obtain both entry-level and fine-grained [31] categories. Currently, the ImageNet dataset contains over 14 million labeled images and has enabled significant advances in image classification [5], [6], [7].

最近，ImageNet[1]在数据集规模上有了极大的跃升。他们提出创建的数据集包含22000类，每类500-1000张图片。之前的数据集包含入门级的类别[29],比如类别“狗”或“椅子”，如[28]，但ImageNet使用WordNet Hierachy[30]来获得入门级的和细粒度的[31]类别。现在，ImageNet数据集包括1400万标注图像，使图像分类取得了大幅度进展[5,6,7]。

**Object detection**. Detecting an object entails both stating that an object belonging to a specified class is present, and localizing it in the image. The location of an object is typically represented by a bounding box, Fig. 1(b). Early algorithms focused on face detection [32] using various ad hoc datasets. Later, more realistic and challenging face detection datasets were created [33]. Another popular challenge is the detection of pedestrians for which several datasets have been created [24], [4]. The Caltech Pedestrian Dataset [4] contains 350,000 labeled instances with bounding boxes.

**目标检测**。检测一个目标需要指出属于某一特定类别的目标是存在的，并在图像中定位出它。目标的位置一般由边界框表示，如图1(b)。早期的算法聚焦在人脸检测[32]，使用的是各种临时数据集。后来创建出了更真实和具有挑战性的人脸检测数据集[33]。另一个流行的挑战是行人检测，已经建立了几个数据集[24,4]。Caltech行人数据集[4]包含35万个用边界框标注的实例。

For the detection of basic object categories, a multiyear effort from 2005 to 2012 was devoted to the creation and maintenance of a series of benchmark datasets that were widely adopted. The PASCAL VOC [2] datasets contained 20 object categories spread over 11,000 images. Over 27,000 object instance bounding boxes were labeled, of which almost 7,000 had detailed segmentations. Recently, a detection challenge has been created from 200 object categories using a subset of 400,000 images from ImageNet [34]. An impressive 350,000 objects have been labeled using bounding boxes.

对于基本目标类别的检测，从2005年到2012年，学术界一直致力于建立和维护一系列基准数据集，这些数据集被广泛采用。PASCAL VOC[2]数据集包含20个目标类别，11000幅图像，标记了超过27000个目标实例的边界框，其中大约7000个有详细的分割。最近，用ImageNet的一个包含200类、40万幅图像的子集创建了一个检测挑战赛，其中35万个目标已经用边界框进行了标记。

Since the detection of many objects such as sunglasses, cellphones or chairs is highly dependent on contextual information, it is important that detection datasets contain objects in their natural environments. In our dataset we strive to collect images rich in contextual information. The use of bounding boxes also limits the accuracy for which detection algorithms may be evaluated. We propose the use of fully segmented instances to enable more accurate detector evaluation.

由于很多物体的检测，比如太阳镜，手机，或椅子，高度依赖上下文信息，所以检测数据集中的目标处于自然环境中这非常重要。在我们的数据集中，我们努力收集了包含丰富上下文信息的图像。边界框的使用也限制了检测算法评估的准确度。我们提出使用完全分割的实例来使得更精确的检测器评估成为可能。

**Semantic scene labeling**. The task of labeling semantic objects in a scene requires that each pixel of an image be labeled as belonging to a category, such as sky, chair, floor, street, etc. In contrast to the detection task, individual instances of objects do not need to be segmented, Fig. 1(c). This enables the labeling of objects for which individual instances are hard to define, such as grass, streets, or walls. Datasets exist for both indoor [11] and outdoor [35], [14] scenes. Some datasets also include depth information [11]. Similar to semantic scene labeling, our goal is to measure the pixel-wise accuracy of object labels. However, we also aim to distinguish between individual instances of an object, which requires a solid understanding of each object’s extent.

**语义场景标记**。在一个场景中标记语义目标的任务需要图像中的每个像素都归属于一个类别，如天空，椅子，地板等等。与检测任务对比，目标的单个实例不需要被分割，如图1(c)所示。这样一些单个实例很难定义的目标也可以被标记，如草、街道或墙。室内场景[11]和室外场景[35,14]的数据集都有。一些数据集也包括深度信息[11]。与语义场景标记类似，我们的目的是衡量像素级精确度的目标标记。但是，我们还想要分辨一类目标的单个实例，这就需要完全理解每个目标的外延。

A novel dataset that combines many of the properties of both object detection and semantic scene labeling datasets is the SUN dataset [3] for scene understanding. SUN contains 908 scene categories from the WordNet dictionary [30] with segmented objects. The 3,819 object categories span those common to object detection datasets (person, chair, car) and to semantic scene labeling (wall, sky, floor). Since the dataset was collected by finding images depicting various scene types, the number of instances per object category exhibits the long tail phenomenon. That is, a few categories have a large number of instances (wall: 20,213, window: 16,080, chair: 7,971) while most have a relatively modest number of instances (boat: 349, airplane: 179, floor lamp: 276). In our dataset, we ensure that each object category has a significant number of instances, Fig. 5.

SUN数据集是为场景理解创建的一个新数据集，结合了目标检测和语义场景标记的性质。SUN包含WordNet字典[30]中的908个场景类别，每个目标都进行了分割。这3819个目标类别包括了普通的目标检测数据集（人，椅子，车），也包括了语义场景标记的数据集（墙，天空，地板）。由于数据集中的图像反应了各种场景类别，每个目标类别中的实例数量展现出长尾效应的现象。即，一些类别有很多数量的实例（墙20213，窗户16080，椅子7971），而大多数类别实例数量都相对不多（船349，飞机179，地灯：276）。在我们的数据集中，我们确保每个目标类别都有很多数量的实例，如图5所示。

**Other vision datasets**. Datasets have spurred the advancement of numerous fields in computer vision. Some notable datasets include the Middlebury datasets for stereo vision [20], multi-view stereo [36] and optical flow [21]. The Berkeley Segmentation Data Set (BSDS500) [37] has been used extensively to evaluate both segmentation and edge detection algorithms. Datasets have also been created to recognize both scene [9] and object attributes [8], [38]. Indeed, numerous areas of vision have benefited from challenging datasets that helped catalyze progress.

**其他视觉数据集**。数据集刺激了计算机视觉很多领域的发展。一些著名的数据集包括包括用于立体视觉[20]、多视角立体视觉[36]和光流[21]的Middlebury数据集。Berkeley分割数据集(BSDS500)[37]在评估分割和边缘检测算法时被广泛使用。也创建了数据集用来进行场景属性识别[9]和目标属性识别[8,38]。确实，大量视觉领域都从这些有挑战性的数据集中受益，它们促进了这个过程。

## 3 IMAGE COLLECTION 图像收集

We next describe how the object categories and candidate images are selected. 下面我们讨论一下怎样选择目标类别和候选图像。

### 3.1 Common Object Categories 普通目标类别

The selection of object categories is a non-trivial exercise. The categories must form a representative set of all categories, be relevant to practical applications and occur with high enough frequency to enable the collection of a large dataset. Other important decisions are whether to include both “thing” and “stuff” categories [39] and whether fine-grained [31], [1] and object-part categories should be included. “Thing” categories include objects for which individual instances may be easily labeled (person, chair, car) where “stuff” categories include materials and objects with no clear boundaries (sky, street, grass). Since we are primarily interested in precise localization of object instances, we decided to only include “thing” categories and not “stuff.” However, since “stuff” categories can provide significant contextual information, we believe the future labeling of “stuff” categories would be beneficial.

目标类别的选择是一件重要的事情。类别必须形成所有类别的一个代表性集合，与实际应用相关，并且发生的频率要很高，足够形成一个大型数据集。其他重要决定还包括是否要包含“thing”和“stuff”类别[39]，是否要包含细粒度[31,1]和目标部件类别。“Thing”类别是指那些实例个体很容易标记的类别，如人、椅子、车，而“stuff”类别是指那些没有明确边界的物质和目标，如天空、街道、草。由于我们主要对目标实例的精确定位感兴趣，我们决定只收集“thing”类别，而不包括“stuff”类别。但是，由于“stuff”类别可以提供很多上下文信息，我们相信将来对“stuff”类别进行标注是有益的。

The specificity of object categories can vary significantly. For instance, a dog could be a member of the “mammal”, “dog”, or “German shepherd” categories. To enable the practical collection of a significant number of instances per category, we chose to limit our dataset to entry-level categories, i.e. category labels that are commonly used by humans when describing objects (dog, chair, person). It is also possible that some object categories may be parts of other object categories. For instance, a face may be part of a person. We anticipate the inclusion of object-part categories (face, hands, wheels) would be beneficial for many real-world applications.

目标类别的明确性可以变化很大。比如，狗可能是属于“哺乳动物”，“狗”或“德国牧羊犬”的。为使每个类别包括足够多数量的实例，形成可以实际应用的图像集合，我们决定数据集只包含入门级的类别，即人描述目标时通常会使用的类别标签（狗，椅子，人）。也可能出现一些目标类别是其他目标类别的一部分。比如，人脸这个类别是人的一部分。我们希望目标部件类别（人脸，手部，轮子）能够对很多真实世界应用有益。

We used several sources to collect entry-level object categories of “things.” We first compiled a list of categories by combining categories from PASCAL VOC [2] and a subset of the 1200 most frequently used words that denote visually identifiable objects [40]. To further augment our set of candidate categories, several children ranging in ages from 4 to 8 were asked to name every object they see in indoor and outdoor environments. The final 272 candidates may be found in the appendix. Finally, the co-authors voted on a 1 to 5 scale for each category taking into account how commonly they occur, their usefulness for practical applications, and their diversity relative to other categories. The final selection of categories attempts to pick categories with high votes, while keeping the number of categories per super-category (animals, vehicles, furniture, etc.) balanced. Categories for which obtaining a large number of instances (greater than 5,000) was difficult were also removed. To ensure backwards compatibility all categories from PASCAL VOC [2] are also included. Our final list of 91 proposed categories is in Fig. 5(a).

我们使用几个源来收集入门级“things”目标类别。我们首先合并了PASCAL VOC[2]的类别和1200个最经常使用的表示视觉可辨认的目标的词汇[40]。为进一步增加我们的候选类别集合，我们问了几个4到8岁的孩子，对所看到的室内和室外场景中的目标进行命名。最后的272个候选可以在附录中找到。最后，本文的作者们对每个类别进行投票计分，按照是否经常出现、对实际应用的用处、与其他类别关联的多样性投1-5分。最后的类别选择是最高投票得分的类别，但要保持每个超类（动物，交通工具，家具等）的类别数量均衡。难以收集到很多数量实例（多于5000）的类别也去除掉了。为保证后向兼容，PASCAL VOC[2]的所有类别都包括了。我们确定的最后91个类别如图5(a)所示。

### 3.2 Non-iconic Image Collection 非典型视角图像的收集

Given the list of object categories, our next goal was to collect a set of candidate images. We may roughly group images into three types, Fig. 2: iconic-object images [41], iconic-scene images [3] and non-iconic images. Typical iconic-object images have a single large object in a canonical perspective centered in the image, Fig. 2(a). Iconic-scene images are shot from canonical viewpoints and commonly lack people, Fig. 2(b). Iconic images have the benefit that they may be easily found by directly searching for specific categories using Google or Bing image search. While iconic images generally provide high quality object instances, they can lack important contextual information and non-canonical viewpoints.

确定了目标类别的列表，我们下一个目标是收集候选图像集。我们粗略的将图像分成三类，如图2：典型的目标图像[41]，典型的场景图像[3]和非典型图像。典型目标图像在图像中央以经典视角包含单个很大的目标，如图2(a)。典型的场景图像以经典视角进行拍摄，经常是没有人的，如图2(b)。典型图像可以用google和bing的图像搜索直接搜索特定类别很容易找到。典型图像一般给出高质量的目标实例，可以没有重要的上下文信息和非经典视角。

Our goal was to collect a dataset such that a majority of images are non-iconic, Fig. 2(c). It has been shown that datasets containing more non-iconic images are better at generalizing [42]. We collected non-iconic images using two strategies. First as popularized by PASCAL VOC [2], we collected images from Flickr which tends to have fewer iconic images. Flickr contains photos uploaded by amateur photographers with searchable metadata and keywords. Second, we did not search for object categories in isolation. A search for “dog” will tend to return iconic images of large, centered dogs. However, if we searched for pairwise combinations of object categories, such as “dog + car” we found many more non-iconic images. Surprisingly, these images typically do not just contain the two categories specified in the search, but numerous other categories as well. To further supplement our dataset we also searched for scene/object category pairs, see the appendix. We downloaded at most 5 photos taken by a single photographer within a short time window. In the rare cases in which enough images could not be found, we searched for single categories and performed an explicit filtering stage to remove iconic images. The result is a collection of 328,000 images with rich contextual relationships between objects as shown in Figs. 2(c) and 6.

我们的目标是数据集中的大部分图像是非典型的，如图2(c)。已经证明，包含更多非典型图像的数据集泛化能力更好[42]。我们用两种策略来收集非典型图像。第一，就像PASCAL VOC[2]一样，我们从Flickr收集图像，其中典型图像数量较少。Flickr上很多都是业余摄影师拍摄的图片上传的，其中包含可以搜索的元数据和关键字。第二，我们没有单独搜索某个目标类别。单独搜索“狗”很容易出现图像中间有一个很大的狗的典型图像。但是，如果我们搜索两个类别的组合，如“狗＋车”，我们会发现很多非典型图像。这些图像一般还不止包括搜索的这两个类别，而且还包括很多别的类别。为进一步补充我们的数据集，我们还搜索场景/目标类别对，详见附录。单个摄影师短期内拍摄的图片，我们最多下载5幅。在很少的情况中，找不到足够多的照片，我们搜索单个类别，然后加入一个过滤步骤去掉典型图像。最后得到的结果是32.8万幅图像，其目标上下文关联非常丰富，如图2(c)和图6所示。

## 4 IMAGE ANNOTATION 图像标注

We next describe how we annotated our image collection. Due to our desire to label over 2.5 million object instances, the design of a cost efficient yet high quality annotation pipeline was critical. The annotation pipeline is outlined in Fig. 3. For all crowdsourcing tasks we used workers on Amazon’s Mechanical Turk (AMT). Our user interfaces are described in detail in the appendix. Note that, since the original version of this work [19], we have taken a number of steps to further improve the quality of the annotations. In particular, we have increased the number of annotators for the category labeling and instance spotting stages to eight. We also added a stage to verify the instance segmentations.

下一步我们阐述一下如何标注我们的图像集。由于我们需要标注超过250万个目标实例，所以设计一种经济又高质量的标注过程非常关键。标注过程如图3所示。对于所有的众包任务，我们使用AMT上的工作者。我们的用户界面详见附录。注意，自从这个工作的最初版本[19]，我们就采取了几个步骤来进一步改进标注质量。特别的，我们增加了类别标记和实例定位阶段的标注者到8人。我们还增加了核实实例分割的阶段。

Fig. 3: Our annotation pipeline is split into 3 primary tasks: (a) labeling the categories present in the image (§4.1), (b) locating and marking all instances of the labeled categories (§4.2), and (c) segmenting each object instance (§4.3).

图3：我们的标注过程分成3个主要的任务：(a)标记图像中存在的类别(§4.1) (b)定位并标记标注的类别的所有实例(§4.2) (c)分割每个目标实例(§4.3)

### 4.1 Category Labeling 类别标记

The first task in annotating our dataset is determining which object categories are present in each image, Fig. 3(a). Since we have 91 categories and a large number of images, asking workers to answer 91 binary classification questions per image would be prohibitively expensive. Instead, we used a hierarchical approach [18].

标注数据集的第一个任务是确定每幅图像中有哪些目标类别，如图3(a)。由于我们有91类，而且有大量的图像，让工作者对每幅图像都回答91个二值分类问题将会成本非常高。我们采用的是一种层次化的方法[18]。

We group the object categories into 11 super-categories (see the appendix). For a given image, a worker was presented with each group of categories in turn and asked to indicate whether any instances exist for that super-category. This greatly reduces the time needed to classify the various categories. For example, a worker may easily determine no animals are present in the image without having to specifically look for cats, dogs, etc. If a worker determines instances from the super-category (animal) are present, for each subordinate category (dog, cat, etc.) present, the worker must drag the category’s icon onto the image over one instance of the category. The placement of these icons is critical for the following stage. We emphasize that only a single instance of each category needs to be annotated in this stage. To ensure high recall, 8 workers were asked to label each image. A category is considered present if any worker indicated the category; false positives are handled in subsequent stages. A detailed analysis of performance is presented in §4.4. This stage took ∼20k worker hours to complete.

我们将目标类别归类成11个超类（见附录）。对于给定图像，工作者轮流给定类别组，要指出是否存在这个超类中的任何实例。这极大的降低了对众多类别进行分类所需的时间。比如，一个工作者可以很容易的确定，图像中没有动物，而不需要逐个去核实狗、猫等的类别。如果一个工作者确定存在一个超类（动物）的实例，对于存在的每个子类别（狗，猫，等等），工作者必须将类别的图标拖拽到图像中，放在类别实例上。这些图标的放置，对于下一步很关键。我们强调，在这个步骤中，只需要标注每个类别的一个实例。为确保召回率高，安排了8个工作者来标记每幅图像。如果任一工作者认为存在某类别，那么就认为这个类别存在；误报在后续的阶段进行处理。§4.4详细分析了性能。这个阶段耗费了大约2万人工小时。

### 4.2 Instance Spotting 实例定位

In the next stage all instances of the object categories in an image were labeled, Fig. 3(b). In the previous stage each worker labeled one instance of a category, but multiple object instances may exist. Therefore, for each image, a worker was asked to place a cross on top of each instance of a specific category found in the previous stage. To boost recall, the location of the instance found by a worker in the previous stage was shown to the current worker. Such priming helped workers quickly find an initial instance upon first seeing the image. The workers could also use a magnifying glass to find small instances. Each worker was asked to label at most 10 instances of a given category per image. Each image was labeled by 8 workers for a total of ∼10k worker hours.

下一阶段，一幅图像中目标类别的所有实例都要标注，如图3(b)。前一阶段每个工作者标记了一个类别的一个实例，但可能存在目标的多个实例。所以，对于每幅图像，工作者要在前一阶段发现的特定类别的每个实例上放置一个十字。为提高召回率，前一阶段发现的实例的位置是展示给现在这个阶段的工作者的。这个准备工作帮助工作者第一眼看到图像时就迅速的找到初始的实例。工作者还可以使用放大镜工具还找到小的实例。每个工作者在每幅图像中最多标记给定类别的10个实例。每幅图像都由8位工作者标记，共计耗费1万人工小时。

### 4.3 Instance Segmentation 实例分割

Our final stage is the laborious task of segmenting each object instance, Fig. 3(c). For this stage we modified the excellent user interface developed by Bell et al. [16] for image segmentation. Our interface asks the worker to segment an object instance specified by a worker in the previous stage. If other instances have already been segmented in the image, those segmentations are shown to the worker. A worker may also indicate there are no object instances of the given category in the image (implying a false positive label from the previous stage) or that all object instances are already segmented.

最后一个阶段是分割每个目标实例的工作，如图3(c)。Bell等[16]开发了图像分割应用的用户界面，我们修改了这个优秀的工作，在这个阶段进行应用。我们的界面要求工作者分割在上一阶段指定的目标实例。如果图像中别的实例已经被分割出来，这些分割就显示给现在的工作者。一个工作者也可以指出图像中没有指定类别的目标实例（这意味着前一阶段的工作为误报），或所有的目标实例都已经被分割出来了。

Segmenting 2,500,000 object instances is an extremely time consuming task requiring over 22 worker hours per 1,000 segmentations. To minimize cost we only had a single worker segment each instance. However, when first completing the task, most workers produced only coarse instance outlines. As a consequence, we required all workers to complete a training task for each object category. The training task required workers to segment an object instance. Workers could not complete the task until their segmentation adequately matched the ground truth. The use of a training task vastly improved the quality of the workers (approximately 1 in 3 workers passed the training stage) and resulting segmentations. Example segmentations may be viewed in Fig. 6.

分割250万个目标实例是非常耗时的工作，每1000个分割耗费22个人工小时。为最小化代价，每个实例都只有一个工作者进行分割。但是，当第一次完成任务时，大多数工作者只会给出粗糙的实例轮廓。所以，我们要求所有工作者对于每个目标类别都完成一个训练任务。训练任务要求工作者分割一个目标实例，只有当分割与真值足够匹配时，才算完成任务。训练任务的使用极大改善了工作者和分割结果的质量（大约只有1/3的工作者通过了训练阶段）。分割的例子见图6。

While the training task filtered out most bad workers, we also performed an explicit verification step on each segmented instance to ensure good quality. Multiple workers (3 to 5) were asked to judge each segmentation and indicate whether it matched the instance well or not. Segmentations of insufficient quality were discarded and the corresponding instances added back to the pool of unsegmented objects. Finally, some approved workers consistently produced poor segmentations; all work obtained from such workers was discarded.

训练任务筛选掉了大多数不合格的工作者，我们还对每个分割过的实例进行了验证步骤来确保高质量。多个工作者(3-5)来评价每个分割，指出分割是否很好的匹配了实例。分割的质量不够高，那么就抛弃之，对应的实例再重新加入未分割的目标池。最后，一些通过的工作者会持续的生成低劣的分割；这种工作者的所有工作都会被抛弃掉。

For images containing 10 object instances or fewer of a given category, every instance was individually segmented (note that in some images up to 15 instances were segmented). Occasionally the number of instances is drastically higher; for example, consider a dense crowd of people or a truckload of bananas. In such cases, many instances of the same category may be tightly grouped together and distinguishing individual instances is difficult. After 10-15 instances of a category were segmented in an image, the remaining instances were marked as “crowds” using a single (possibly multipart) segment. For the purpose of evaluation, areas marked as crowds will be ignored and not affect a detector’s score. Details are given in the appendix.

对于包含10个或更少给定类别的目标实例的图像，每个实例都分别分割（注意再一些图像中最多分割出来了15个实例）。偶尔实例数量非常高；比如，考虑一群非常密集的人或一大车香蕉。在这种情况下，很多相同类别的实例可能紧密的靠在一起，分辨单个实例可能比较困难。在一幅图像中分割了10-15个这个类别的实例后，剩下的实例用单个分割（可能包括多部分）标记为“群”。为了有效评估，标记为群的区域被忽略掉，不影响检测器的分数。详见附录。

### 4.4 Annotation Performance Analysis 标注性能分析

We analyzed crowd worker quality on the category labeling task by comparing to dedicated expert workers, see Fig. 4(a). We compared precision and recall of seven expert workers (co-authors of the paper) with the results obtained by taking the union of one to ten AMT workers. Ground truth was computed using majority vote of the experts. For this task recall is of primary importance as false positives could be removed in later stages. Fig. 4(a) shows that the union of 8 AMT workers, the same number as was used to collect our labels, achieved greater recall than any of the expert workers. Note that worker recall saturates at around 9-10 AMT workers.

我们分析众包工作者的工作质量，在类别标记任务中，是通过与细致的专家工作者比较，如图4(a)。我们比较了7为专家工作者（文章的共同作者）和1-10位AMT工作者的集合的工作结果的精确率和召回率。真值是通过计算专家工作的多数投票。对于这个任务，召回率是最重要的，因为误报可以在后面的阶段中去除。图4(a)展示了8个AMT工作者的集合就会得到高于任何专家工作者的召回率，这和我们使用的工作者数量一致，而且注意，工作者的召回率在9-10个工作者的时候就饱和了。

Fig. 4: Worker precision and recall for the category labeling task. (a) The union of multiple AMT workers (blue) has better recall than any expert (red). Ground truth was computed using majority vote of the experts. (b) Shows the number of workers (circle size) and average number of jobs per worker (circle color) for each precision/recall range. Most workers have high precision; such workers generally also complete more jobs. For this plot ground truth for each worker is the union of responses from all other AMT workers. See §4.4 for details.

Object category presence is often ambiguous. Indeed as Fig. 4(a) indicates, even dedicated experts often disagree on object presence, e.g. due to inherent ambiguity in the image or disagreement about category definitions. For any unambiguous examples having a probability of over 50% of being annotated, the probability all 8 annotators missing such a case is at most .5^8 ≈ .004. Additionally, by observing how recall increased as we added annotators, we estimate that in practice over 99% of all object categories not later rejected as false positives are detected given 8 annotators. Note that a similar analysis may be done for instance spotting in which 8 annotators were also used.

目标类别的存在与否经常是含糊的。如图4(a)所指出的，即使是专家工作者也经常对于目标存在与否持不同意见，比如，由于图像内在的含糊性，或类别定义的不同意见。对于任何明确的例子，如果有超过50%的概率被标注，那么8个标注者都错过了这个目标的概率大概是0.5^8≈0.004。另外，标注者增加，召回率上升，观察到这个现象，我们估计实践中所有目标类别的99%都被8个标注者检测到了，除了那些后来发现是误报而拒绝的。注意对于实例定位也可以作类似的分析，其中也使用了8个标注者。

Finally, Fig. 4(b) re-examines precision and recall of AMT workers on category labeling on a much larger set of images. The number of workers (circle size) and average number of jobs per worker (circle color) is shown for each precision/recall range. Unlike in Fig. 4(a), we used a leave-one-out evaluation procedure where a category was considered present if any of the remaining workers named the category. Therefore, overall worker precision is substantially higher. Workers who completed the most jobs also have the highest precision; all jobs from workers below the black line were rejected.

最后，图4(b)重新检查了AMT工作者在更大的图像集中做类别标记的精确率和召回率。工作者的数量（圆圈大小）和平均每个工作者的工作数量（圆圈颜色）在每个精确率/召回率上都进行了测试。与图4(a)不同的是，我们使用了leave-one-out评估过程，其中如果任何一个工作者命名了这个类别，那么就认为这个类别存在。所以，总体的工作者精确度基本是更高的。工作者完成最多工作的也同时有最高的精确率；低于黑线的工作者的所有工作都被拒了。

### 4.5 Caption Annotation 标题标注

We added five written caption descriptions to each image in MS COCO. A full description of the caption statistics and how they were gathered will be provided shortly in a separate publication.

MS COCO中的每幅图像，我们都加上了5条书写的标题描述。标题统计的完整描述和怎样收集的将会在另外的文章中简短给出。

## 5 DATASET STATISTICS 数据集统计

Next, we analyze the properties of the Microsoft Common Objects in COntext (MS COCO) dataset in comparison to several other popular datasets. These include ImageNet [1], PASCAL VOC 2012 [2], and SUN [3]. Each of these datasets varies significantly in size, list of labeled categories and types of images. ImageNet was created to capture a large number of object categories, many of which are fine-grained. SUN focuses on labeling scene types and the objects that commonly occur in them. Finally, PASCAL VOC’s primary application is object detection in natural images. MS COCO is designed for the detection and segmentation of objects occurring in their natural context.

下一步，我们分析MS COCO数据集的性质与其他几个流行数据集之间的对比。这包括ImageNet[1]，PASCAL VOC 2012[2]和SUN[3]。这几个数据集在规模、标记的类别和图像风格上都很不一样。创建ImageNet是为了覆盖大量目标类别，很多类别是细粒度的。SUN聚焦在标注场景类型的图像，以及其中的常见目标。最后，PASCAL VOC的主要应用是在自然图像的目标检测中。MS COCO的创建是用于自然场景上下文的目标检测与分割。

The number of instances per category for all 91 categories is shown in Fig. 5(a). A summary of the datasets showing the number of object categories and the number of instances per category is shown in Fig. 5(d). While MS COCO has fewer categories than ImageNet and SUN, it has more instances per category which we hypothesize will be useful for learning complex models capable of precise localization. In comparison to PASCAL VOC, MS COCO has both more categories and instances.

91类每类的实例数量见图5(a)。数据集的摘要包括目标类别数目和每个类别中实例的数量如图5(d)所示。虽然MS COCO的类别数量少于ImageNet和SUN，但每个类别中的实例数量更多，我们假设这应当对复杂的能够精确定位的模型有帮助。与PASCAL VOC相比，MS COCO的类别数量和实例数量都要更多。

An important property of our dataset is we strive to find non-iconic images containing objects in their natural context. The amount of contextual information present in an image can be estimated by examining the average number of object categories and instances per image, Fig. 5(b, c). For ImageNet we plot the object detection validation set, since the training data only has a single object labeled. On average our dataset contains 3.5 categories and 7.7 instances per image. In comparison ImageNet and PASCAL VOC both have less than 2 categories and 3 instances per image on average. Another interesting observation is only 10% of the images in MS COCO have only one category per image, in comparison, over 60% of images contain a single object category in ImageNet and PASCAL VOC. As expected, the SUN dataset has the most contextual information since it is scene-based and uses an unrestricted set of categories.

我们数据集的一个重要性质是，我们努力找到了自然场景上下文的包含目标的非典型图像。一幅图像中上下文信息的含量可以通过检查每幅图像中的平均目标类别数量和实例数量估计得到，如图5(b,c)。对于ImageNet我们画的对象是目标检测的验证集，因为训练数据只标记了单个目标。我们的数据集每幅图像包含3.5个类别，7.7个实例。对比起来，ImageNet和PASCAL VOC每幅图像平均类别数都少于2个，平均实例数都少于3个。另一个有趣的观察结果是，MS COCO中只有10%的图像每幅图像只有一个目标类别，对比起来，在ImageNet和PASCAL VOC中，超过60%的图像只有一个单独目标类别。而SUN数据集的上下文信息最多，因为这个数据集是基于场景的，使用的目标类别没有限制。

Finally, we analyze the average size of objects in the datasets. Generally smaller objects are harder to recognize and require more contextual reasoning to recognize. As shown in Fig. 5(e), the average sizes of objects is smaller for both MS COCO and SUN.

最后，我们分析了数据集中目标的平均尺寸。一般来说，较小的目标更难识别，需要更多的上下文推理来识别。如图5(e)所示，MS COCO和SUN数据集的目标平均尺寸更小一些。

Fig. 5: (a) Number of annotated instances per category for MS COCO and PASCAL VOC. (b,c) Number of annotated categories and annotated instances, respectively, per image for MS COCO, ImageNet Detection, PASCAL VOC and SUN (average number of categories and instances are shown in parentheses). (d) Number of categories vs. the number of instances per category for a number of popular object recognition datasets. (e) The distribution of instance sizes for the MS COCO, ImageNet Detection, PASCAL VOC and SUN datasets.

## 6 DATASET SPLITS 数据集分离

To accommodate a faster release schedule, we split the MS COCO dataset into two roughly equal parts. The first half of the dataset was released in 2014, the second half will be released in 2015. The 2014 release contains 82,783 training, 40,504 validation, and 40,775 testing images (approximately 50% train, 25% val, and 25% test). There are nearly 270k segmented people and a total of 886k segmented object instances in the 2014 train+val data alone. The cumulative 2015 release will contain a total of 165,482 train, 81,208 val, and 81,434 test images. We took care to minimize the chance of near-duplicate images existing across splits by explicitly removing near duplicates (detected with [43]) and grouping images by photographer and date taken.

为能更快的发布数据集，我们将MS COCO数据集分成两个大致相同大小的部分。数据集第一半在2014年放出，第二半将在2015年放出。2014版包括82783训练图像，40504验证图像，和40775测试图像（大约50%训练，25%验证和25%测试）。在2014训练验证集数据中，有接近27万分割的人和共计88.6万分割的目标实例。追加的2015版将包括共计165482训练图像，81208验证图像，81434测试图像。我们小心的使分离数据集间的接近重复的图像的概率最小化，方法是显式的去除接近重复的图像（用[43]的方法检测），用摄影师和拍摄时间对图像进行分组。

Following established protocol, annotations for train and validation data will be released, but not for test. We are currently finalizing the evaluation server for automatic evaluation on the test set. A full discussion of evaluation metrics will be added once the evaluation server is complete.

遵循确定的协议，训练和验证数据的标注将会放出，但测试数据没有。我们现在正在最后调试评估服务器以自动在测试集上评估。一旦评估服务器准备结束，评估度量标准的完整讨论将会给出。

Note that we have limited the 2014 release to a subset of 80 categories. We did not collect segmentations for the following 11 categories: hat, shoe, eyeglasses (too many instances), mirror, window, door, street sign (ambiguous and difficult to label), plate, desk (due to confusion with bowl and dining table, respectively) and blender, hair brush (too few instances). We may add segmentations for some of these categories in the cumulative 2015 release.

注意我们的2014版只包括80个类别。我们没有对以下11个类别进行分割：帽子，鞋子，眼睛（实例太多），镜子，窗户，门，街道标志（有歧义，而且难以标记），盘子，桌子（由于分别会和碗和餐桌混淆），搅拌机，梳子（实例太少）。我们在将来的2015版中可能会增加这些类别中一些的分割。

## 7 ALGORITHMIC ANALYSIS 算法分析

**Bounding-box detection**. For the following experiments we take a subset of 55,000 images from our dataset (These preliminary experiments were performed before our final split of the dataset intro train, val, and test. Baselines on the actual test set will be added once the evaluation server is complete.) and obtain tight-fitting bounding boxes from the annotated segmentation masks. We evaluate models tested on both MS COCO and PASCAL, see Table 1. We evaluate two different models. DPMv5-P: the latest implementation of [44] (release 5 [45]) trained on PASCAL VOC 2012. DPMv5-C: the same implementation trained on COCO (5000 positive and 10000 negative images). We use the default parameter settings for training COCO models.

**边界框检测**。下面的试验，我们从数据集中取出了包含55000图像的子集（这些初步试验是在数据集分离成训练集、验证集和测试集之前进行的，一旦评估服务器准备完毕，在真正的测试集上的基准结果就会给出），从标注的分割掩模中得到了紧致边界框。我们评估的模型在MS COCO和PASCAL上都进行了测试，见表1。我们评估了两个不同的模型，DPMv5-P：[44]的最新实现，在PASCAL 2012上进行的训练；DPMv5-C：同样的实现，在COCO数据集上训练（5000个正样本，10000个负样本）。我们使用默认的参数设置在COCO数据集上训练。

TABLE 1: Top: Detection performance evaluated on PASCAL VOC 2012. DPMv5-P is the performance reported by Girshick et al. in VOC release 5. DPMv5-C uses the same implementation, but is trained with MS COCO. Bottom: Performance evaluated on MS COCO for DPM models trained with PASCAL VOC 2012 (DPMv5-P) and MS COCO (DPMv5-C). For DPMv5-C we used 5000 positive and 10000 negative training examples. While MS COCO is considerably more challenging than PASCAL, use of more training data coupled with more sophisticated approaches [5], [6], [7] should improve performance substantially.

If we compare the average performance of DPMv5-P on PASCAL VOC and MS COCO, we find that average performance on MS COCO drops by nearly a factor of 2, suggesting that MS COCO does include more difficult (non-iconic) images of objects that are partially occluded, amid clutter, etc. We notice a similar drop in performance for the model trained on MS COCO (DPMv5-C).

如果我们比较DPMv5-P在PASCAL VOC和MS COCO上的平均表现，我们发现在MS COCO上的平均表现下降了一半，这意味着MS COCO确实包含了更困难（非典型）的图像，其中的目标是部分遮挡的，在杂乱之中等的情况。我们注意到在MS COCO上训练的模型(DPMv5-C)也有类似的性能下降。

The effect on detection performance of training on PASCAL VOC or MS COCO may be analyzed by comparing DPMv5-P and DPMv5-C. They use the same implementation with different sources of training data. Table 1 shows DPMv5-C still outperforms DPMv5-P in 6 out of 20 categories when testing on PASCAL VOC. In some categories (e.g., dog, cat, people), models trained on MS COCO perform worse, while on others (e.g., bus, tv, horse), models trained on our data are better.

在PASCAL VOC或MS COCO上训练的模型的检测性能效果可以通过比较DPMv5-P和DPMv5-C来分析得到。它们使用的都是同样的实现，但是不一样的训练数据。表1显示在PASCAL VOC上测试时，20个类中的6个DPMv5-C比DPMv5-P的检测性能更好。在一些类别中（如狗，猫，人），在MS COCO上训练的模型效果更差，而其他的类（如，公交车，电视，马），在我们的数据上训练的模型更好一些。

Consistent with past observations [46], we find that including difficult (non-iconic) images during training may not always help. Such examples may act as noise and pollute the learned model if the model is not rich enough to capture such appearance variability. Our dataset allows for the exploration of such issues.

与之前的观察一致[46]，我们发现在训练时包含困难（非典型）的图像并不一定总是有助于提高性能。如果模型没有丰富到能够捕捉这些形状和变化，这些样本可能会成为噪声，污染学习好的模型。我们的数据集允许探索这种问题。

Torralba and Efros [42] proposed a metric to measure cross-dataset generalization which computes the ‘performance drop’ for models that train on one dataset and test on another. The performance difference of the DPMv5-P models across the two datasets is 12.7 AP while the DPMv5-C models only have 7.7 AP difference. Moreover, overall performance is much lower on MS COCO. These observations support two hypotheses: 1) MS COCO is significantly more difficult than PASCAL VOC and 2) models trained on MS COCO can generalize better to easier datasets such as PASCAL VOC given more training data. To gain insight into the differences between the datasets, see the appendix for visualizations of person and chair examples from the two datasets.

Torralba and Efros [42]提出一种度量标准来衡量跨数据集的泛化能力，其计算模型在一个数据集上训练但在另一个数据集上测试的“性能下降”。DPMv5-P模型在两个数据集上的性能差异为12.7% mAP，而DPMv5-C只有7.7% mAP的差异。而且，在MS COCO数据集上的总体性能更低。这些观察支撑以下两点假设：1) MS COCO比PASCAL VOC更难；2) 如果给定更多训练数据，在MS COCO数据集上训练的模型在更简单的数据集如PASCAL VOC上的泛化能力更好。为洞悉数据集间的差异，附录中有两个数据集中人和椅子的样本的可视化效果。

**Generating segmentations from detections**. We now describe a simple method for generating object bounding boxes and segmentation masks, following prior work that produces segmentations from object detections [47], [48], [49], [50]. We learn aspect-specific pixel-level segmentation masks for different categories. These are readily learned by averaging together segmentation masks from aligned training instances. We learn different masks corresponding to the different mixtures in our DPM detector. Sample masks are visualized in Fig. 7.

**从检测中生成分割**。我们现在描述一种简单的方法来生成目标边界框和分割掩模，遵循的是从目标检测结果中生成分割的已有工作[47,48,49,50]。我们学习不同类别的特定角度的像素级分割掩模。通过将对齐的训练实例的分割掩模一起平均，就可以学习到。我们学习DPM检测器对应不同混合的不同掩模。图7是掩模例子的可视化。

Fig. 7: We visualize our mixture-specific shape masks. We paste thresholded shape masks on each candidate detection to generate candidate segments.

**Detection evaluated by segmentation**. Segmentation is a challenging task even assuming a detector reports correct results as it requires fine localization of object part boundaries. To decouple segmentation evaluation from detection correctness, we benchmark segmentation quality using only correct detections. Specifically, given that the detector reports a correct bounding box, how well does the predicted segmentation of that object match the ground truth segmentation? As criterion for correct detection, we impose the standard requirement that intersection over union between predicted and ground truth boxes is at least 0.5. We then measure the intersection over union of the predicted and ground truth segmentation masks, see Fig. 8. To establish a baseline for our dataset, we project learned DPM part masks onto the image to create segmentation masks. Fig. 9 shows results of this segmentation baseline for the DPM learned on the 20 PASCAL categories and tested on our dataset.

**通过分割评估检测**。即使假设检测器可以得到正确的结果，分割是一个很有挑战性的任务，因为需要目标部件的边缘精细定位。为使分割的评估与检测的正确性无关，我们只使用正确的检测结果来作为分割质量的基准。特别的，如果检测器给出了正确的边界框，该目标预测的分割与真值分割匹配的如何呢？作为正确检测的准则，我们加入了标准的要求，即预测和真值框间的IOU至少为0.5。我们然后度量预测和真值的分割掩模的IOU，如图8所示。为确定我们的数据集的基准，我们将学习好的DPM部件掩模投影到图像中，以生成分割掩模。图9所示的是在20类PASCAL上学习得到的DPM模型在我们的数据集上测试的这种分割基准的结果。

Fig. 8: Evaluating instance detections with segmentation masks versus bounding boxes. Bounding boxes are a particularly crude approximation for articulated objects; in this case, the majority of the pixels in the (blue) tight-fitting bounding-box do not lie on the object. Our (green) instance-level segmentation masks allows for a more accurate measure of object detection and localization.

Fig. 9: A predicted segmentation might not recover object detail even though detection and ground truth bounding boxes overlap well (left). Sampling from the person category illustrates that predicting segmentations from top-down projection of DPM part masks is difficult even for correct detections (center). Average segmentation overlap measured on MS COCO for the 20 PASCAL VOC categories demonstrates the difficulty of the problem (right).

## 8 DISCUSSION 讨论

We introduced a new dataset for detecting and segmenting objects found in everyday life in their natural environments. Utilizing over 70,000 worker hours, a vast collection of object instances was gathered, annotated and organized to drive the advancement of object detection and segmentation algorithms. Emphasis was placed on finding non-iconic images of objects in natural environments and varied viewpoints. Dataset statistics indicate the images contain rich contextual information with many objects present per image.

我们给出了一种新的数据集，用于目标检测和目标分割，目标是日常生活中在自然环境中的目标。耗费了超过7万人工小时，搜集了巨大的目标实例集，进行了标注和组织，以驱动目标检测和分割算法的研究。特别强调寻找非典型图像，包含自然环境和变化视角的目标。数据集统计数字表明，图像包含丰富的上下文信息，每幅图像包括很多个目标。

There are several promising directions for future annotations on our dataset. We currently only label “things”, but labeling “stuff” may also provide significant contextual information that may be useful for detection. Many object detection algorithms benefit from additional annotations, such as the amount an instance is occluded [4] or the location of keypoints on the object [10]. Finally, our dataset could provide a good benchmark for other types of labels, including scene types [3], attributes [9], [8] and full sentence written descriptions [51]. We are actively exploring adding various such annotations.

更进一步标注我们的数据集有几个有希望的方向。我们目前只标记了“things”，但标记“stuff”会提供更明显的上下文信息，以助力目标检测。很多目标检测算法都从更多的标注中受益，比如实例被遮挡的部分[4]，或目标的关键点的位置[10]。最后，我们的数据集可以为其他类型的标记提供很好的基准，包括场景类型的[3]，属性[9,8]，和整句描述[51]。我们积极探索增加各种这样的标注。

To download and learn more about MS COCO please see the project website. MS COCO will evolve and grow over time; up to date information is available online.

访问我们的项目网站来下载和学习MS COCO。数据集会随着时间烟花并增长；线上信息是最新的信息。

**Acknowledgments**. Funding for all crowd worker tasks was provided by Microsoft. P.P. and D.R. were supported by ONR MURI Grant N00014-10-1-0933. We would like to thank all members of the community who provided valuable feedback throughout the process of defining and collecting the dataset.

## APPENDIX OVERVIEW 附录概览

In the appendix, we provide detailed descriptions of the AMT user interfaces and the full list of 272 candidate categories (from which our final 91 were selected) and 40 scene categories (used for scene-object queries).

附录中，我们给出了AMT用户界面的详细描述，和272个候选类别的完整列表（从中选出了我们最后的91类），和40个场景类别（用来进行场景目标查询）。

