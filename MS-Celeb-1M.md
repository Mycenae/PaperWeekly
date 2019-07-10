# MS-Celeb-1M: A Dataset and Benchmark for Large-Scale Face Recognition

Yandong Guo et al. Microsoft Research

## Abstract 摘要

In this paper, we design a benchmark task and provide the associated datasets for recognizing face images and link them to corresponding entity keys in a knowledge base. More specifically, we propose a benchmark task to recognize one million celebrities from their face images, by using all the possibly collected face images of this individual on the web as training data. The rich information provided by the knowledge base helps to conduct disambiguation and improve the recognition accuracy, and contributes to various real-world applications, such as image captioning and news video analysis. Associated with this task, we design and provide concrete measurement set, evaluation protocol, as well as training data. We also present in details our experiment setup and report promising baseline results. Our benchmark task could lead to one of the largest classification problems in computer vision. To the best of our knowledge, our training dataset, which contains 10M images in version 1, is the largest publicly available one in the world.

本文中，我们设计了一个基准测试任务，提供了相关的数据集，进行人脸图像的识别，并与知识库中对应的实体键值联系起来。具体的，我们提出了一个基准测试任务来从脸部图像来识别一百万名人，使用的是某个个体在网上所有可能收集到的脸部图像作为训练数据。知识库提供的丰富信息帮助消除了歧义，改进识别准确率，并对各种真实世界应用有所贡献，比如图像加标题和新闻视频分析。与这个任务相关的，我们设计并给出了具体的衡量集合，评估准则，以及训练数据。我们还详细给出了我们的试验设置，并给出了很有希望的基准结果。我们的基准测试任务可能是计算机视觉任务中最大的分类任务之一。据我们所知，我们的训练集，在第一版中包含了10 millions图像，是世界上最大的公开可用集。

**Keywords**: Face recognition, large scale, benchmark, training data, celebrity recognition, knowledge base 人脸识别，大规模，基准测试，训练集，名人识别，知识库

## 1 Introduction 引言

In this paper, we design a benchmark task as to recognize one million celebrities from their face images and identify them by linking to the unique entity keys in a knowledge base. We also construct associated datasets to train and test for this benchmark task. Our paper is mainly to close the following two gaps in current face recognition, as reported in [1]. First, there has not been enough effort in determining the identity of a person from a face image with disambiguation, especially at the web scale. The current face identification task mainly focuses on finding similar images (in terms of certain types of distance metric) for the input image, rather than answering questions such as “who is in the image?” and “if it is Anne in the image, which Anne?”. This lacks an important step of “recognizing”. The second gap is about the scale. The publicly available datasets are much smaller than that being used privately in industry, such as Facebook [2,3] and Google [4], as summarized in Table 1. Though the research in face recognition highly desires large datasets consisting of many distinct people, such large dataset is not easily or publicly accessible to most researchers. This greatly limits the contributions from research groups, especially in academia.

本文中，我们设计了一个基准测试任务，从人脸图像中识别一百万个名人，并与知识库中的唯一实体键值关联起来。我们还构建了相关的数据集，对这个基准测试任务进行训练和测试。我们的文章主要填补了目前人脸识别中的两个空白，如[1]中所述。第一，从人脸图像中毫无疑义的确定一个人的身份，这方面的努力还不够，尤其是在网络上。目前的人脸识别任务主要关注的是，对输入图像找到类似的人脸（以某种距离度量），而不是回答“图像中的是谁？”和“如果图像中的人是Anne，哪个Anne？”的问题。这缺少识别的重要步骤。第二个空白是规模的问题。公开可用的数据集非常小，工业私用的如Facebook和Google则大的多，如表1所示。人脸识别的研究非常期望数据集中包含很多不同的人，这样的大型数据集对于大多数研究者来说不是那么容易公开可用。这极大的限制了研究组织的贡献，尤其是学术上的。

Table 1. Face recognition datasets

Dataset | Available | people | images
--- | --- | --- | ---
IJB-A [17] | public | 500 | 5712
LFW [7,8] | public | 5K | 13K
YFD [14] | public | 1595 | 3425 videos
CelebFaces [15] | public | 10K | 202K
CASIA-WebFace [16] | public | 10K | 500K
Ours | public | 100K | about 10M
Facebook | private | 4K | 4400K
Google | private | 8M | 100-200M

Our benchmark task has the following properties. First, we define our face recognition as to determine the identity of a person from his/her face images. More specifically, we introduce a knowledge base into face recognition, since the recent advance in knowledge bases has demonstrated incredible capability of providing accurate identifiers and rich properties for celebrities. Examples include Satori knowledge graph in Microsoft and “freebase” in [5]. Our face recognition task is demonstrated in Fig. 1.

我们的基准测试任务有以下性质。第一，我们定义我们的人脸识别为，从其脸部图像中确定人的身份。更具体的，我们提出了一个知识库进行人脸识别，由于知识库最近的进展已经展现了极好的能力，可以对名人给出精确的标识和丰富的性质，这样的例子包括微软的Satori知识图谱和[5]中的freebase。我们的人脸识别任务如图1所示。

Fig. 1. An example of our face recognition task. Our task is to recognize the face in the image and then link this face with the corresponding entity key in the knowledge base. By recognizing the left image to be “Anne Hathaway” and linking to the entity key, we know she is an American actress born in 1982, who has played Mia Thermopolis in The Princess Diaries, not the other Anne Hathaway who was the wife of William Shakespeare. Input image is from the web.

Linking the image with an entity key in the knowledge base, rather than an isolated string for a person’s name naturally solves the disambiguation issue in the traditional face recognition task. Moreover, the linked entity key is associated with rich and comprehensive property information in the knowledge base, which makes our task more similar to human behavior compared with traditional face identification, since retrieving the individual’s name as well as the associated information naturally takes place when humans are viewing a face image. The rich information makes our face recognition task practical and beneficial to many real applications, including image search, ranking, caption generation, image deep understanding, etc.

将图像与知识库的实体键值关联起来，而不是孤立的人名字符串，这就很自然的解决了传统识别任务中消除歧义的问题。而且，关联的实体键值与知识库丰富广泛的特性信息相关联，使得我们的任务，在与传统人脸识别相比较时，与人类行为更类似，因为当人类观察一幅人脸图像时，很自然的会得到个体的姓名以及相关联的信息。丰富的信息使我们的人脸识别任务对很多真实应用更实用更有益处，包括图像搜索，排序，标题生成，图像深度理解，等。

Second, our benchmark task targets at recognizing celebrities. Recognizing celebrities, rather than a pre-selected private group of people, represents public interest and could be directly applied to a wide range of real scenarios. Moreover, only with popular celebrities, we can leverage the existing information (e.g. name, profession) in the knowledge base and the information on the web to build a large-scale dataset which is publicly available for training, measurement, and re-distributing under certain licenses. The security department may have many labeled face images for criminal identification, but the data can not be publicly shared.

第二，我们的基准测试任务的目标是识别名人。识别名人，而不是预先选定的一群人，代表了公众的兴趣，可以直接应用于很多真实场景。而且，只有受欢迎的名人，我们可以利用知识库中现有的信息（如，名字，职业），和网络上的信息来构建一个大型数据集，可以公开用于训练、测量和在一些许可下进行重新分发。安全部门可能有很多标注的人脸图像进行罪犯识别，但数据不能公开分享。

Third, we select one million celebrities from freebase and provide their associated entity keys, and encourage researchers to build recognizers to identify each people entity. Considering each entity as one class may lead to, to the best of our knowledge, the largest classification problem in computer vision. The clear definition and mutually exclusiveness of these classes are supported by the unique entity keys and their associated properties provided by the knowledge base, since in our dataset, there are a significant amount of celebrities having same/similar names. This is different from generic image classification, where to obtain a large number of exclusive classes with clear definition itself is a challenging and open problem [6].

第三，我们从freebase中选择了一百万个名人，给出他们相关的实体键值，并鼓励研究者构建识别器，识别每个人的实体。考虑到每个实体都是一类，据我们所知，这会带来计算机视觉中最大的分类问题。这些类的清晰的定义和互相之间的排斥，是由知识库中的唯一实体键值及其相关的特性所支持的，因为在我们的数据集中，有很多名人有相同/类似的名字。这与通用图像分类是不同的，通用图像分类中，得到大量互斥的有清晰定义的类别，是一个有挑战和开放的问题。

The large scale of our problem naturally introduces the following attractive challenges. With the increased number of classes, the inter-class variance tends to decrease. There are celebrities look very similar to each other (or even twins) in our one-million list. Moreover, large intra-class variance is introduced by popular celebrities with millions of images available, as well as celebrities with very large appearance variation (e.g., due to age, makeups, or even sex reassignment surgery).

我们问题的极大规模带来了下列有吸引力的挑战。随着类别数量的增加，类间变化一般会降低。在我们一百万的列表中，有一些名人看起来互相很类似（甚至是双胞胎）。而且，这些名人共有上百万张图像，这带来了更大的类内变化，尤其是哪些有很大外貌变化的名人（如，因为年龄、化妆，甚至是变性手术导致的）。

In order to evaluate the performance of our benchmark task, we provide concrete measurement set and evaluation protocol. Our measurement set consists of images for a subset of celebrities in our one-million celebrity list. The celebrities are selected in a way that, our measurement set mainly focuses on popular celebrities to represent the interest of real application and users, while the measurement set still maintains enough (about 25%) tail celebrities to encourage the performance on celebrity coverage. We manually label images for these celebrities carefully. The correctness of our labeling is ensured by deep research on the web content, consensus verification, and multiple iterations of carefully review. In order to make our measurement more challenging, we blend a set of distractor images with this set of carefully labeled images. The distractor images are images of other celebrities or ordinary people on the web, which are mainly used to hide the celebrities we select in the measurement.

为评估我们的基准测试任务的性能，我们提供了具体的衡量集合和评估方法。我们的衡量集合是由我们的一百万名人列表的子集的图像构成的。这些名人选择的方式是，我们的度量集合主要聚焦在流行的名人上，代表了真实应用和用户的兴趣，同时度量集合仍然保持足够的（大约25%）的tail celebrities以鼓励名人覆盖上的性能。我们手工仔细标注这些名人的图像。标注的正确性的是得到保证的，通过对网络内容的深度研究，审查验证和多次仔细的迭代检查。为使我们的度量更有挑战性，我们掺杂了一些干扰图像，也是经过仔细标注的图像。这些干扰图像是网络上的其他名人或普通人，主要用于隐藏选择的度量集合中的名人。

Along with this challenging yet attractive large scale benchmark task proposed, we also provide a very large training dataset to facilitate the task. The training dataset contains about 10M images for 100K top celebrities selected from our one-million celebrity list in terms of their web appearance frequency. Our training data is, to the best of our knowledge, the largest publicly available one in the world, as shown in Table 1. We plan to further extend the size in the near future. For each of the image in our training data, we provide the thumbnail of the original image and cropped face region from the original image (with/without alignment). This is to maximize the convenience for the researchers to investigate using this data.

与这个有挑战性的吸引人的大规模基准测试集合一起，我们还提出了一个非常大规模的训练数据集，以帮助这个任务。训练数据集包含10 million图像，是100万名人列表中最有名的100K个名人，以其网络出现频率计。据我们所知，我们的训练数据是世界上最大的公开可用的数据集，如表1所示。我们计划在将来进一步拓展其规模。对我们训练数据集中的每幅图像，我们会给出原图的缩略图，并将人脸从原始图像中剪切出来（有/无对齐）。这是为了研究最大其便利性，来研究使用这些数据。

With this training data, we trained a convolutional deep neural network with the classification setup (by considering each entity as one class). The experimental results show that without extra effort in fine-tuning the model structure, we recognize 44.2% of the images in the measurement set with the precision 95% (hard case, details provided in section 4). We provide the details of our experiment setup and experimental results to serve as a very promising baseline in section 4.

在这些训练数据的基础上，我们用分类设置训练了一个卷积深度神经网络（将每个实体当做一个类别）。试验结果表明，不需要额外精调模型结构，我们识别出了度量集中44.2%的图像，精度95%（高难情况下，详见第4部分）。我们给出试验设置的细节和试验结果，作为非常有希望的基准，见第4部分。

**Contribution Summary**. Our contribution in this paper is summarized as follows. 本文中我们的贡献如下所述。

- We design a benchmark task: to recognize one million celebrities from their face images, and link to their corresponding entity keys in freebase [5]. 我们设计一个基准测试任务：为从人脸图像中识别一百万名人，并与freebase中相应的实体键值关联起来。
- We provide the following datasets: (1) One million celebrities selected from freebase with corresponding entity keys , and a snapshot for freebase data dumps; (2) Manually labeled measurement set with carefully designed evaluation protocol; (3) A large scale training dataset, with face region cropped and aligned (to the best of our knowledge, the largest publicly available one). 我们给出了下列数据集：(1)从freebase中选出的一百万名人，及其对应的实体键值，还有freebase数据dump的快照；(2)手工标注的度量集，包含仔细设计的评估方案；(3)大规模训练数据集，其人脸区域是剪切出来并对齐的（据我们所知，这是公开可用的最大数据集）。
- We provide promising baseline performance with our training data to inspire more research effort on this task. 我们使用训练数据，给出了很有希望的基准性能，来激励这个任务的更多研究。

Our benchmark task could lead to a very large scale classification problem in computer vision with meaningful real applications. This benefits people in experimenting different recognition models (especially fine-grained neural network) with the given training/testing data. Moreover, we encourage people to bring in more outside data and evaluate experimental results in a separate track.

我们的基准测试任务，是计算机视觉中最大规模的分类任务，还是有意义的真实应用。这对进行不同识别模型的人都是有益的，尤其是细粒度神经网络。而且，我们鼓励人们带来更多的外部数据，在外部赛道中评估试验结果。

## 2 Related works 相关工作

Typically, there are two types of tasks for face recognition. One is very well-studied, called face verification, which is to determine whether two given face images belong to the same person. Face verification has been heavily investigated. One of the most widely used measurement sets for verification is Labeled Faces in the Wild (LFW) in [7,8], which provides 3000 matched face image pairs and 3000 mismatched face image pairs, and allows researchers to report verification accuracy with different settings. The best performance on LFW datasets has been frequently updated in the past several years. Especially, with the “unrestricted, labeled outside data” setting, multiple research groups have claimed higher accuracy than human performance for verification task on LFW [4,9].

一般来说，有两种类型的人脸识别任务。一种研究的很多，称为人脸验证，即确实两个给定的人脸属于相同的人。人脸验证的研究非常多。人脸验证最广泛使用的度量集是LFW，包含3000对匹配的人脸图像对，和3000对不匹配的人脸图像对，使得研究者可以用不同的设置给出验证准确率。在LFW数据集上的最好性能在过去几年中不断刷新。尤其是，使用“不受限，室外标记数据”的设置，多个研究小组声称在LFW上的验证准确率超过了人类表现。

Recently, the interest in the other type of face recognition task, face identification, has greatly increased [9,10,11,3]. For typical face identification problems, two sets of face images are given, called gallery set and query set. Then the task is, for a given face image in the query set, to find the most similar faces in the gallery image set. When the gallery image set only has a very limited number (say, less than five) of face images for each individual, the most effective solution is still to learn a generic feature which can tell whether or not two face images are the same person, which is essentially still the problem of face verification. Currently, the MegaFace in [11] might be one of the most difficult face identification benchmarks. The difficulty of MegaFace mainly comes from the up-to one million distractors blended in the gallery image set. Note that the query set in MegaFace are selected from images from FaceScrub [12] and FG-NET [13], which contains 530 and 82 persons respectively.

最近，另一种人脸识别任务，人脸鉴定，研究也非常的多。对于典型的人脸鉴定问题，会给定两个人脸图像的集合，称为gallery集和query集。具体任务是，对于一幅query集的给定人脸图像，找到gallery图像集中的最相似的人脸。当gallery集，每个个体的人脸图像数量很少（如，少于5幅）时，最有效的解决方案仍然是学习一个通用的特征，辨别两幅人脸图像是否是同一个人，即仍然是人脸验证的问题。目前，MegaFace可能是最困难的人脸鉴定基准测试之一。MegaFace的困难主要来自于在gallery图像集中混入了接近一百万干扰图像。注意，MegaFace中的query集是从FaceScrub和FG-NET中选出的图像，分别包含530人和82人。

Several datasets have been published to facilitate the training for the face verification and identification tasks. Examples include LFW [7,8], Youtube Face Database (YFD) [14], CelebFaces+ [15], and CASIA-WebFace [16]. In LFW, 13000 images of faces were collected from the web, and then carefully labeled with celebrities’ names. The YFD contains 3425 videos of 1595 different people. The CelebFace+ dataset contains 202, 599 face images of 10, 177 celebrities. People in CelebFaces+ and LFW are claimed to be mutually exclusive. The CASIA-WebFace [16] is currently the largest dataset which is publicly available, with about 10K celebrities, and 500K images. A quick summary is listed in Table 1.

有几个数据集可以方便人脸验证和鉴定任务的训练。如LFW，Youtube Face Database(YFD)，CelebFaces+，CASIA-WebFace。在LFW中，从网络中收集了13000人脸图像，然后仔细的标注了名人的名字。YFD包含了1595个不同的人的3425视频。CelebFace+数据集包含10177个名人的202599人脸图像。CelebFaces+和LFW中的人据称是互不包含的。CASIA-WebFace是目前最大的公开可用数据集，有10K名人，500K图像。见表1中的归纳。

As shown in Table 1, our training dataset is considerably larger than the publicly available datasets. Another uniqueness of our training dataset is that our dataset focuses on facilitating our celebrity recognition task, so our dataset needs to cover as many popular celebrities as possible, and have to solve the data disambiguation problem to collect right images for each celebrity. On the other hand, the existing datasets are mainly used to train a generalizable face feature, and celebrity coverage is not a major concern for these datasets. Therefore, for the typical existing dataset, if a name string corresponds to multiple celebrities (e.g., Mike Smith) and would lead to ambiguous image search result, these celebrities are usually removed from the datasets to help the precision of the collected training data [18].

如表1所示，我们的训练数据集比公开可用的数据集都大了好多。我们的训练数据集，另一个唯一之处是，聚焦在名人识别的任务上，所以我们的数据集需要覆盖尽可能多的受欢迎的名人，并必须解决消除疑义的问题，为每个名人收集正确的图像。另一方面，现有的数据集主要用于训练一个可泛化的人脸特征，名人覆盖面不是这些数据集的主要考虑。所以，对于典型的现有数据集来说，如果一个人名字符串对应多个名人（如，Mike Smith），就会得到有歧义的图像搜索结果，这些名人通常不在数据集中，以提高收集的训练数据的精确度。

## 3 Benchmark construction 基准测试构成

Our benchmark task is to recognize one million celebrities from their face images, and link to their corresponding entity keys in the knowledge base. Here we describe how we construct this task in details.

我们的基准测试任务是从其人脸图像中识别一百万名人，并与知识库中的对应实体键值关联起来。这里我们详细描述一下，我们怎样构建这个任务。

### 3.1 One million celebrity list 一百万名人列表

We select one million celebrities to recognize from a knowledge graph called freebase [5], where each entity is identified by a unique key (called machine identifier, MID in freebase) and associated with rich properties. We require that the entities we select are human beings in the real world and have/had public attentions.

我们从知识图谱freebase[5]中选择了一百万名人来识别，其中每个实体都通过一个唯一的键值来识别（称为机器识别码，在freebase中为MID），关联着丰富的性质。我们的要求是，我们选择的实体是真实世界的人类，而且有公众的关注。

The first step is to select a subset of entities (from freebase [5]) which correspond to real people using the criteria in [1]. In freebase, there are more than 50 million topics capsulated in about 2 billion triplets. Note that we don’t include any person if his/her facial appearance is unknown or not clearly defined.

第一步是（从freebase中）用[1]中的准则选择实体子集，对应着真实的人。在freebase中，在2 billion个三元组中有超过50 million个主题。注意，如果某人的人脸特征是未知的，或没有清晰定义的，那么我们就不收藏这个人。

The second step is to rank all the entities in the above subset according to the frequency of their occurrence on the web [1]. We select the top one million entities to form our celebrity list and provide their entity keys (MID) in freebase. We concern the public attention (popularity on the web) for two reasons. First, we want to align our benchmark task with the interest of real applications. For applications like image search, image annotations and deep understanding, and image caption generation, the recognition of popular celebrities would be more attractive to most of the users than ordinary people. Second, we include popular celebrities so that we have better chance to obtain multiple authority images for each of them to enable our training, testing, and re-distributing under certain licenses.

第二步是在上述子集中对所有实体进行等级区分，根据他们在网络上出现的频率。我们选择最高的一百万个实体，构成我们的名人列表，并给出其在freebase中的实体键值(MID)。我们关注公众注意力（网上的流行程度），主要有两个原因。第一，我们希望我们的基准测试任务是接近真实应用的。对于图像搜索、图像标注和深度理解、图像标题生成这样的应用，识别流行的名人是非常有用的，至少比识别普通人有用。第二，我们使用流行的名人，所以我们可以得到更多的图像，这样才可以更好的进行训练、测试，以及在某种授权下分发。

We present the distribution of the one million celebrities in different aspects including profession, nationality, age, and gender. In our one million celebrity list, we include persons with more than 2000 different professions (Fig. 2 (a)), and come from more than 200 distinct countries/regions (Fig. 2 (b)), which introduces a great diversity to our data. We cover all the major races in the world (Caucasian, Mongoloid, and Negroid). Moreover, as shown in Fig. 2 (c), we cover a large range of ages in our list. Though we do not manually select celebrities to make the profession (or gender, nationality, age) distribution uniform, the diversity (gender, age, profession, race, nationality) of our celebrity list is guaranteed by the large scale of our dataset. This is different from [17], in which there are about 500 subjects so the manual balancing over gender distribution is inevitable.

我们给出一百万名人的多种属性，包括职业，国籍，年龄和性别。在我们的一百万名人列表中，有超过2000个不同的职位（图2(a)），超过200个不同的国家/地区（图2(b)），这使我们的数据有非常大的多样性。我们包括世界上的主要人种(Caucasian, Mongoloid, and Negroid)。而且，如图2(c)所示，我们的列表中人的年龄分布也非常广泛。虽然我们没有手工选择所有名人，以使其职业（或性别、国籍和年龄）分布更加均匀，我们的名人列表的多样性是由数据集的巨大规模来确保的。这与[17]不同，其中有约500个主题，这样就必须手动对性别的分布进行均衡。

Note that our property statistics are limited to the availability of freebase information. Some celebrities in our one million list do not have complete properties. If a certain celebrity does not have property A available in freebase, we do not include this celebrity for the statistic calculation of the property A.

注意，我们的属性统计是仅限于freebase中可用的信息的。在我们的一百万名人列表中，一些名人没有完整的属性信息。如果某名人没有freebase中可用的性质A，那么我们在统计性质A时就不包括这个名人。

Fig. 2. Distribution of the properties of the celebrities in our one-million list in different aspects. The large scale of our dataset naturally introduces great diversity. As shown in (a) and (b), we include persons with more than 2000 different professions, and come from more than 200 distinct countries/regions. The figure (c) demonstrates that we don’t include celebrities who were born before 1846 (long time before the first rollfilm specialized camera “Kodak” was invented [19]) and covers celebrities of a large variance of age. In (d), we notice that we have more females than males in our one-million celebrity list. This might be correlated with the profession distribution in our list.

### 3.2 Celebrity selection for measurement 进行度量选择的名人

In order to evaluate the recognition performance on the one million celebrities obtained in the last subsection, we build up a measurement set which includes a set of carefully labeled images blended with another set of randomly selected face images as distractors. The measurement set construction is described in details in the following subsections, while the evaluation protocol is described in Section 4.

为评估上一节中得到的一百万名人的识别性能，我们构建了一个度量集，包括了仔细标记的图像，还混杂了随机选择的人脸图像作为干扰用。度量集的构建在下面的小节中详细叙述，评估方案则在第4部分描述。

For the labeled images, we sample a subset of celebrities (Currently there are 1500. We will increase the number of celebrities in our measurement set in the future.) from the one-million celebrity list due to limited labeling resource. The sampling weight is designed in a way that, our measurement set mainly focuses on top celebrities (rank among the top in the occurrence frequency list) to represent the interest of real applications and users, yet maintain a certain amount of tail celebrities (celebrities not mentioned frequently on the web, e.g., from 1 to 10 times in total) to guarantee the measurement coverage over the one-million list.

对于标注的图像，由于有限的标注资源，我们从一百万名人列表中取样得到了一个名人子集（目前有1500人，未来我们会在度量集中增加名人数量）。采样的权重在设计时，使度量集主要聚焦于最有名的名人（在出现频率列表中排名靠前的），以代表真实应用和用户的兴趣，而且还保持了一定数量的tail名人（网上出现频率没那么高的名人，如总计出现了1到10次的），以确保度量集在一百万列表中的覆盖程度。

More specifically, let $f_i$ denote the number of documents mentioned the i-th celebrity on the web. Following the method in [1], we set the probability for the i-th celebrity to get selected to be proportional to $f_i^'$, defined as,

更具体的，令$f_i$表示第i为名人在网上被提到的文档的数量。采用[1]中的方法，我们设第i个名人被选中的概率与$f'_i$成正比，定义为

$$f'_i = f_i^{\frac{1}{\sqrt 5}}$$(1)

where the exponent $1/\sqrt5$ is obtained empirically to include more celebrities with small f. 其中指数$1/\sqrt5$是通过经验得到的，可以用小的f包含更多的名人。

Though it seems to be a natural solution, we do not set the sampling weights to be proportional to $f_i$, since this option will make our measurement set barely contain any celebrities from the bottom 90% in our one-million list (ordered by $f_i$). The reason is that the distribution of f is very long-tailed. More than 90% of the celebrities have f smaller than 30, while the top celebrities have f larger than one million. We need to include sufficient number of tail celebrities to encourage researchers to work on the hard cases to improve the performance from the perspective of recognition coverage. This is the reason that we applied the adjustment in (1).

我们没有将采样权重设置与$f_i$成正比，这似乎是一个很自然的选择，因为这个选项会让我们的度量集几乎不会包含列表中任何后90%的名人（以$f_i$进行排序）。其原因是，f的分布是非常长尾的。超过90%的名人的f小于30，而最高名人的f超过一百万。我们需要包含足够多的tail名人，以鼓励研究者对这样的困难情况进行研究，改进识别覆盖率角度上的性能。这也是我们在(1)中进行调整的原因。

With the sampling weight f' in (1) applied, our measurement set still mainly focuses on the most popular celebrities, while about 25% of the celebrities in our measurement set come from the bottom 90% in our one-million celebrity list (ordered by f). If we do not apply the adjustment in (1), but just use f as the sampling weight, less than 10% of the celebrities in the measurement set come from the bottom 90% in our one-million celebrity list.

应用了(1)中的采样权重f'后，我们的度量集仍然主要聚焦于最受欢迎的名人，而我们度量集中25%的名人是从一百万名人列表中底部90%来的（按f进行排序）。如果我们不使用(1)中的调整，而只使用f作为采样权重，那么度量集中来自一百万名人列表中后面90%的则只有少于10%。

Since the list of the celebrities in our measurement set is not exposed(We publish the images for 500 celebrities, called development set, while hold the rest 1000 for grand challenges), and our measurement set contains 25% of the celebrities in our measurement set come from the bottom 90%, researchers need to include as many celebrities as possible (not only the popular ones) from our one-million list to improve the performance of coverage. This pushes the scale of our task to be very large.

由于我们的度量集中的名人列表没有公开（我们公开500位名人的图像，称为开发集，而保留剩下的1000人进行比赛），我们的度量集包含25%的名人来自后面90%，研究者需要包含尽可能多的名人（不仅仅是最流行的那些），以改进覆盖性能。这使我们的任务的规模变得非常大。

### 3.3 Labeling for measurement 为度量进行的标注

After we have the set of celebrities for measurement, we provide two images for each of the celebrity. The correctness of our image labeling is ensured by deep research on the web content, multiple iterations of carefully review, and very rigorous consensus verification. Details are listed as follows.

在我们有了名人度量集之后，我们为每个名人准备了两幅图像。我们的图像标注的正确性通过对网络内容的深度研究、多次仔细的审查和非常严格的审查验证得到保证。细节如下。

**Scraping**. Scraping provides image candidates for each of the celebrities selected for the measurement set. Though in the end we provide only two images per celebrity for evaluation, we scraped about 30 images per celebrities. During the scraping procedure, we applied different search queries, including the celebrity’s name, name plus profession, and names in other languages (if available). The advantages of introducing multiple variations of the query used for each celebrity is that with multiple queries, we have better chance to capture the images which are truly about the given celebrity. Moreover, the variation of the query and scraping multiple images also brings in the diversity to the images for the given celebrity. Especially for the famous celebrities, the top one image returned by search engine is typically his/her representative image (frontal facial image with high quality), which is relatively easier to recognize, compared with the other images returned by the search engine. We increase the scraping depth so that we have more diverse images to be recognized for each of the celebrity.

**爬取**。爬取为每个选作度量集的名人提供图像候选。虽然到最后，每个名人只有两幅图像进行评估，但我们为每位名人准备了约30幅图像。在scraping的过程中，我们应用了不同的搜索请求，包括名人的名称，名称加职业，和其他语言的名称（如果可用的话）。为每个名人使用多个查询的优势是，我们可以收集到关于给定名人的真正相关的图像。而且，查询的变化和准备多幅图像也为给定的名人带来了图像的多样性。尤其是对于有名的名人，搜索引擎返回的第一位的图像，一般是他/她的代表性图像（高质量正面照），与搜索引擎返回的其他图像比较，会很容易识别出来。我们增加scraping深度，这样就有需要识别的更多图像了。

**Label**. Labeling picks up the images which are truly about the given celebrity. As shown in Fig.3, for each given celebrity, we (all the authors) manually label all the scraped image candidates to be truly about this celebrity or not. Extreme cautious was applied. We have access to the page which contains the scraped image to be labeled. Whenever needed, the judge (the authors) is asked to visit the original page with the scraped image and read the page content to guide his/her labeling. The rich information on the original page benefits the quality of the labeling, especially for a lot of the hard cases. Each of the image-celebrity entity pair was judged by at least two persons. Whenever there is a conflict, the two judges review together and provide the final decision based on verbal discussion. In total, we have about 30K images labeled, spent hundreds of hours.

**标记**。标注选出真正关于给定名人的图像。如图3所示，对于每个给定的名人，我们（所有作者）手工标注所有爬取到的图像，是否是真的关于这个名人，这其中会非常的小心。我们可以访问要标记的爬取图像的页面。不论什么时候需要，评判者（即作者）都会被要求去访问爬取图像的原始页面，阅读页面的内容以指导他/她的标注。原始页面的丰富内容对标记的质量是有帮助的，尤其是对于标记困难的情况，帮助很大。每个图像-名人的实体对至少有两个人进行判断。不论什么时候有冲突，两个评判者都会一起进行检查，进行讨论以给出最后的决定。总计我们有30K图像进行了标记，花费了数百小时。

Fig. 3. Labeling GUI for “Chuck Palhniuk”. (partial view) As shown in the figure, in the upper right corner, a representative image and a short description is provided. For a given image candidate, judge can label as “not for this celebrity” (red), “yes for this celebrity” (green), or “broken image” (dark gray).

In our measurement set, we select two images for each of the celebrity to keep the evaluation cost low. We have two subset (each of them have the same celebrity list), described as follows. 在我们的度量集中，我们为每个名人选择两幅图像，使评估代价较低。我们有两个子集（每个都包含相同数量的名人），如下所述。

- **Random set**. The image in this subset is randomly selected from the labeled images. One image per celebrity. This set reveals how many celebrities are truly covered by the models to be tested. **随机集**。这个子集中的图像是从标记的图像中随机选择的。每个名人一幅图像。这个集合反应了要测试的模型真正覆盖了多少名人。

- **Hard set**. The image in this subset is the one (from the labeled images) which is the most different from any images in the training dataset. One image per celebrity. This set is to evaluate the generalization ability of the model. **困难集**。这个子集中的图像是（标记的图像中）与任何训练数据集图像中最不一样的图像。每个名人一幅图像。这个集合是为了评估模型的泛化能力。

Then, we blend the labeled images with images from other celebrities or ordinary people. The evaluation protocol is introduced in details in the next section.

然后，我们将标记的图像与其他名人或普通人的图像混合到一起。评估方案在下一节中仔细介绍。

## 4 Celebrity recognition 名人识别

In this section, we set up the evaluation protocol for our benchmark task. Moreover, in order to facilitate the researchers to work on this problem, we provide a training dataset which is encouraged (optional) to use. We also present the baseline performance obtained by using our provided training data. We also encourage researchers to train with outside data and evaluate in a separate track.

本节中，我们设置好我们基准测试任务的评估方案。而且，为方便研究者对这个问题进行研究，我们提供了一个鼓励使用的训练数据集。我们还给出使用我们的训练数据得到的基准性能。我们还鼓励研究者使用外部数据训练，并在另外分离的赛道上评估。

### 4.1 Evaluation Protocol 评估方案

We evaluate the performance of our proposed recognition task in terms of precision and coverage (defined in the following subsection) using the settings described as follows. 我们以精度和覆盖率（定义见下面的小节），评估提出的识别任务的性能，使用的设置也在下面进行叙述。

**Setup**. We setup our evaluation protocol as follows. For a model to be tested, we collect the model prediction for both the labeled image and distractors in the measurement set. Note that we don’t expose which images in the measurement are labeled ones or which are distractors. This setup avoids human labeling to the measurement set, and encourages researchers to build a recognizer which could robustly distinguish one million (as many as possible) people faces, rather than focusing merely on a small group of people.

**设置**。我们的评估方案如下。对于要测试的模型，我们对度量集中的标记图像和干扰图像都进行预测。注意，我们不会标识出度量集中的图像，哪个是标记的图像，哪个是干扰图像。这种设置防止了度量集的人为标记，鼓励研究者构建可以稳健的区分一百万人脸（尽可能多）的识别器，而不是只针对一小部分人。

Moreover, during the training procedure, if the researcher leverages outside data for training, we do not require participants to exclude celebrities in our measurement from the training data set. Our measurement still evaluate the generalization ability of our recognition model, due to the following reasons. There are one million celebrities to be recognized in our task, and there are millions of images for some popular celebrities on the web. It is practically impossible to include all the images for every celebrity in the list. On the other hand, according to section 4.2, the images in our measurement set is typically not the representative images for the given celebrity (e.g., the top one searching result). Therefore the chance to include the measurement images in the training set is relatively low, as long as the celebrity list in the measurement set is hidden. This is different from most of the existing face recognition benchmark tasks, in which the measurement set is published and targeted on a small group of people. For these traditional benchmark tasks, the evaluation generalization ability relies on manually excluding the images (from the training set) of all the persons in the measurement set (This is mainly based on the integrity of the participants).

而且，在训练过程中，如果研究者利用外部数据进行训练，我们不要求参与者将度量中的名人从训练集中排除掉。我们的度量还会评估识别模型的泛化能力，主要原因如下。我们的任务中要识别一百万名人，在网上对于一些流行的名人，有数百万图像。实践上不可能包含列表上每个名人的所有图像。另一方面，根据4.2节，度量集中的图像一般并不是给定名人的代表性的图像（如，搜索结果的第一个）。所以度量图像包含在训练集中的概率是相对很低的，只要度量集中的名人列表是隐藏的。这与大多数现有的人脸识别基准测试任务都是不同的，现有的基准测试中度量集是针对一小群人发布的。对于这些传统的基准测试任务，评估泛化能力依靠（从训练集中）手工排除度量集中的所有人。

**Evaluation metric**. In the measurement set, we have n images, denoted by $\{x_i\}^n_{i=1}$. The first m images {$x_i | i = 1, 2, 3, ..., m$} are the labeled images for our selected celebrities, while the rest {$x_i | i = m + 1, ..., n$} are distractors. Note that we hide the order of the images in the measurement set. **评估度量标准**。在度量集中，我们有n幅图像，表示为$\{x_i\}^n_{i=1}$。前m幅图像{$x_i | i = 1, 2, 3, ..., m$}是选定名人的标记图像，而剩下的{$x_i | i = m + 1, ..., n$}是干扰图像。注意我们隐藏了在度量集中图像的顺序。

For the i-th image, let $g(x_i)$ denote the ground truth label (entity key obtained by labeling). For any model to be tested, we assume the model to output {$ĝ(x_i), c(x_i)$} as the predicted entity key of the i-th image, and its corresponding prediction confidence. We allow the model to perform rejection. That is, if $c(x_i) < t$, where t is a preset threshold, the recognition result for image $x_i$ will be ignored. We define the precision with the threshold t as,

对于第i幅图像，令$g(x_i)$表示真值标签（从标签得到的实体键值）。对于任意要测试的模型，我们假设模型的输出为{$ĝ(x_i), c(x_i)$}，是第i幅图像的预测实体键值，和对应的预测置信度。我们允许模型拒绝，即，如果$c(x_i) < t$，其中t是预设的阈值，对于图像$x_i$的识别结果将会被忽略。我们定义阈值t时的精确度为：

$$P(t) = \frac {|\{x_i | ĝ(x_i) = g(x_i) ∧ c(x_i) ≥ t, i = 1, 2, ..., m\}|}{|\{x_i | c(x_i) ≥ t, i = 1, 2, ..., m\}|}$$(2)

where the nominator is the number of the images of which the prediction is correct (and confidence score is larger than the threshold). The denominator is the number of images (within the set ${x_i}^m_{i=1}$) which the model does have prediction (not reject to recognize).

其中分子是预测结果是正确的图像数量（置信度高于阈值），分母是模型有预测结果的图像数量（在集合${x_i}^m_{i=1}$中，且不没有拒绝识别）。

The coverage in our protocol is defined as 我们的方案的覆盖率定义为

$$C(t) = \frac {|\{x_i | c(x_i) ≥ t, i = 1, 2, ..., m\}|}{m}$$(3)

For each given t, a pair of precision P(t) and coverage C(t) can be obtained for the model to be tested. The precision P(t) is a function of C(t). Our major evaluation metric is the maximum of the coverage satisfying the condition of precision, P(t) ≥ $P_{min}$. The value of $P_{min}$ is 0.95 in our current setup. Other metrics and analysis/discussions are also welcomed to report. The reason that we prefer a fixed high precision and measure the corresponding coverage is because in many real applications high precision is usually more desirable and of greater value.

对于给定的t值，对要测试的模型，可以得到一对精度P(t)和覆盖率C(t)。精度P(t)是C(t)的函数。我们的主要评估度量标准是在满足精度条件下的最大覆盖率，P(t) ≥ $P_{min}$。我们目前的设置中$P_{min}$为0.95。其他度量标准和分析/讨论也欢迎进行进行报道。我们倾向于固定的高精度，并度量对应的覆盖率的原因是，在很多实际应用中，高精确度通常更受欢迎，价值也更大。

### 4.2 Training dataset 训练数据集

In order to facilitate the above face recognition task we provide a large training dataset. This training dataset is prepared by the following two steps. First, we select the top 100K entities from our one-million celebrity list in terms of their web appearance frequency. Then, we retrieve approximately 100 images per celebrity from popular search engines.

为方便上述人脸识别任务，我们给出了一个大型训练数据集。这个训练数据集的准备有下面两个步骤。第一，我们从一百万名人列表中，以其网络出现频率为依据选择最高的100K实体。然后，我们从流行的搜索引擎中每个名人检索大约100幅图像。

We do not provide training images for the entire one-million celebrity list for the following considerations. First, limited by time and resource, we can only manage to prepare a dataset of top 100K celebrities as a v1 dataset to facilitate the participants to quickly get started. We will continuously extend the dataset to cover more celebrities in the future. Moreover, as shown in the experimental results in the next subsection, this dataset is already very promising to use. Our training dataset covers about 75% of celebrities in our measurement set, which implies that the upper bound of recognition recall rate based on the provided training data cannot exceed 75%. Therefore, we also encourage the participants, especially who are passionate to break this 75% upper bound to treat the dataset development as one of the key problems in this challenge, and bring in outside data to get higher recognition recall rate and compare experimental results in a separate track. Especially, we encourage people label their data with entity keys in the freebase snapshot we provided and publish, so that different dataset could be easily united to facilitate collaboration.

我们没有为整个一百万名人列表都提供训练图像，主要是因为以下考虑。首先，受时间和资源所限，我们只能为排名最高的100K个名人准备数据集，作为v1数据集，以方便参与者快速开始。我们会持续拓展这个数据集，将来覆盖更多的名人。而且，如下一小节中展示的试验结果一样，这个数据集已经使用起来效果很好了。我们的训练数据集在度量集中覆盖了75%的名人，这暗示了基于给定的训练数据的识别召回率的上限不会超过75%。因此，我们也鼓励参与者，尤其是那些很希望打破这个75%上限的人，将数据集开发作为这个挑战的关键问题，将外部数据引入以得到更高的识别召回率，在另外的赛道上比较试验结果。尤其是，我们鼓励人们用freebase中我们给出并发布的snapshot实体键值标记其数据，这样不同的数据集可以很容易的统一到一起，以方便协作。

On example in our training dataset is shown in Figure 4. As shown in the figures, same celebrity may look very differently in different images. In Figure 4, we see images for Steve Jobs (m.06y3r) when he was about 20/30 years old, as well as images when he was about 50 years old. The image at row 2, column 8 (in green rectangle) in Figure 4 is claimed to be Steve Jobs when he was in high school. Notice that the image at row 2, column 3 in Figure 4, marked with red rectangle is considered as a noise sample in our dataset, since this image was synthesized by combining one image of Steve Jobs and one image of Ashton Kutcher, who is the actor in the movie “Jobs”.

我们训练数据集中的一个例子如图4所示。如图所示，相同的名人在不同的图像中看起来非常不一样。在图4中，我们看到了乔布斯在20/30岁时的样子，也看到了他在50岁时的样子。图4中第2行第8列的图像（绿色矩形框）据说是乔布斯在高中时的样子。注意，第2行第3列中的图像，用红色矩形框标记的，是我们数据集中的一个噪声样本，因为图像是用一幅乔布斯的图像和一幅Ashton Kutcher的图像合成得到的，他是电影《乔布斯》中的演员。

Fig. 4. Examples (subset) of the training images for the celebrity with entity key m.06y3r (Steve Jobs). The image marked with a green rectangle is claimed to be Steve Jobs when he was in high school. The image marked with a red rectangle is considered as a noise sample in our dataset, since it is synthesized by combining one image of Steve Jobs and one image of Ashton Kutcher, who is the actor in the movie “Jobs”.

(a) Original Image (b) Aligned Face Image

As we have mentioned, we do not manually remove the noise in this training data set. This is partially because to prepare training data of this size is beyond the scale of manually labeling. In addition, we have observed that the state-of-the-art deep neural network learning algorithm can tolerate a certain level of noise in the training data. Though for a small percentage of celebrities their image search result is far from perfect, more data especially more individuals covered by the training data could still be of great value to the face recognition research, which is also reported in [18]. Moreover, we believe that data cleaning, noisy label removal, and learning with noisy data are all good and real problems that are worth of dedicated research efforts. Therefore, we leave this problem open and do not limit the use of outside training data.

就像我们提到的那样，我们没有将训练数据集中的这个噪声图像手工去除。这部分是因为，准备这样大规模的训练数据超过了手工标记的规模。另外，我们观察到，目前最好的深度神经网络学习算法可以容忍训练数据集中一定程度的噪声。虽然对于一小部分名人，他们的图像搜索结果远未达到完美，更多的数据，尤其是训练数据覆盖的更多个体，仍然对于人脸识别研究是有很大价值的，这在[18]中也有报道。而且，我们相信，数据清晰，含噪标签去除，含噪数据的学习，都是很好的真实问题，值得仔细的进行研究。因此，我们没有处理这个问题，也没有限制外部训练数据的使用。

### 4.3 Baseline 基准

There are typically two categories of methods to recognize people from face images. One is template-based. For methods in this category, a gallery set which contains multiple images for the targeted group of people is pre-built. Then, for the given image in the query set, the most similar image(s) in the gallery set (according to some certain metrics or in pre-learned feature space) is retrieved, and the annotation of this/these similar images are used to estimate the identity of the given query image. When the gallery is not very large, this category of methods is very convenient for adding/removing entities in the gallery since the face feature representation could be learned in advance. However, when the gallery is large, a complicated index needs to be built to shorten the retrieval time. In this case, the flexibility of adding/removing entities for the methods in this category vanishes. Moreover, the accuracy of the template-based methods highly relies on the annotation accuracy in the gallery set. When there are many people in the targeted group, accurate annotation is beyond human effort and could be a very challenging problem itself.

一般有两类方法来从人脸图像中识别人。第一是基于模板的。在这类方法中，有一个gallery集，内建包含了目标人群的多幅图像。然后，对于query集中的给定图像，取出gallery集中最相似的图像（根据某种度量或在预学习的特征空间），这些类似的图像的标注被用来估计给定的query图像的身份。当gallery集不是很大时，这类方法是对于在gallery集中增加/去除实体是非常方便的，因为人脸特征表示可以提前学习到。但是，当gallery集很大时，需要构建一个复杂的索引来缩短查询时间。在这种情况下，增加/删除实体的灵活性就没有了。而且，基于模板方法的准确性高度依赖于gallery集中标注准确性。当目标人群数量很多，准确的标注仅靠人力是不够的，这本身就是一个很大的问题。

We choose the second category, which is a model-based method. More specifically, we model our problem as a classification problem and consider each celebrity as a class. 我们选择了第二类方法，这是基于模型的方法。具体的，我们将问题建模为一个分类问题，将每个名人看做一类。

In our experiment, we trained a deep neural network following the network structure in [20]. Training a deep neural network for 100K celebrities is not a trivial task. If we directly train the model from scratch, it is hard to see the model starts to converge even after a long run due to the large number of categories. To address this problem, we started from training a small model for 500 celebrities, which have the largest numbers of images for each celebrity. In addition, we used the pre-trained model from [20] to initialize this small model. This step is optional, but we observed that it helps the training process converge faster. After 50, 000 iterations, we stopped to train this model, and used it as a pre-trained model to initialize the full model of 100K celebrities. After 250, 000 iterations, with learning rate decreased from the initial value 0.01 to 0.001 and 0.0001 after 100, 000 and 200, 000 iterations, the training loss decrease becomes very slow and indiscernible. Then we stopped the training and used the last model snapshot to evaluate the performance of celebrity recognition on our measurement set. The experimental results (on the published 500 celebrities) are shown in Fig. 5 and Table 2.

在我们的试验中，我们参考[20]中的网络结构训练了一个深度神经网络。为100K个名人训练一个深度神经网络不是一件小事。如果我们直接从头训练这个模型，由于种类众多，甚至在很长时间内都看不到模型开始收敛。为解决这个问题，我们为500个名人训练一个小型模型，我们选择人脸图像数量最多的500人。另外，我们使用[20]中的预训练模型来初始化这个小模型。这一步骤是可选的，但我们观察到这会帮助训练过程收敛的更快。在50K次迭代后，我们停止了模型的训练，将其用作预训练模型来初始化100K名人的完整模型。在250K次迭代后，学习速率从初始值0.01，在100K次和200K次迭代后分别降低到0.001和0.0001，训练损失的降低变得很慢很慢、不可分辨。然后我们停止了训练，用最后的模型快照，在度量集上来评估名人识别的性能。试验结果（在公布的500名人上）如图5和表2所示。

Table 2. Experimental results on the 500 published celebrities

| | Coverage@Precision 99% | Coverage@Precision 95%
--- | --- | ---
Hard Set | 0.052 | 0.442
Random Set | 0.606 | 0.728

Fig. 5. Precision-coverage curve with our baseline model

The promising results can be attributed to the deep neural network capability and the high quality of image search results thanks for years of improvement in image search engines. However, the curves also shows that the task is indeed very challenge. To achieve both high precision and high recall, a great amount of research efforts need to be spent on data collection, cleaning, learning algorithm, and model generalization, which are valuable problems to computer vision researchers.

得到的结果还不错，这是深度神经网络容量的功劳，也是图像搜索结果质量很高的原因。但是，曲线也表明了，这个任务确实非常有挑战性。为得到高准确率和高召回率，还需要大量的工作，进行数据收集、数据清洗、学习算法、模型泛化，这都是计算机视觉中的重要问题。

## 5 Discussion and Future work 讨论和未来工作

In this paper, we have defined a benchmark task which is to recognize one million celebrities in the world from their face images, and link the face to a corresponding entity key in a knowledge base. Our face recognition has the property of disambiguation, and close to the human behavior in recognizing images. We also provide concrete measurement set for people to evaluate the model performance easily, and provide, to the best of our knowledge, the largest training dataset to facilitate research in the area.

在本文中，我们定义了一个基准测试任务，即从人脸图像中识别世界上的一百万名人，并将人脸与一个知识库中的对应实体键值联系起来。我们的人脸识别有消除歧义的性质，在识别图像中与人类行为接近。我们还给出了具体的度量集，研究者很容易就可以评估其模型性能。据我们所知，我们提供的训练数据集是世界上最大的数据集，可以方便这个领域的研究。

Beyond face recognition, our datasets could inspire other research topics. For example, people could adopt one of the cutting-edge unsupervised/semi-supervised clustering algorithms [21] [22] [23] [24] on our training dataset, and/or develop new algorithms which can accurately locate and remove outliers in a large, real dataset. Another interesting topic is the to build estimators to predict a person’s properties from his/her face images. For example, the images in our training dataset are associated with entity keys in knowledge base, of which the gender information (or other properties) could be easily retrieved. People could train a robust gender classifier for the face images in the wild based on this large scale training data. We look forward to exciting research inspired by our training dataset and benchmark task.

在人脸识别之外，我们的数据集也可以为其他研究课题提供灵感。比如，人们可以在我们的训练数据集上使用最先进的无监督/半监督聚类算法，也可以在这个大型的真实数据集上开发新的算法，准确的定位并去除离群值。另一个有趣的课题是从人脸图像中预测一个人的属性。比如，我们训练数据集中的图像是与知识库中的实体键值联系到一起的，其性别信息（或其他性质）可以很容易提取到。人们可以基于这个大规模训练数据集训练一个稳健的性别分类器。我们期待在训练数据集和基准测试之上可以有很好的研究结果。