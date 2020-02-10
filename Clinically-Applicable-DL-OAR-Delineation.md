# Clinically applicable deep learning framework for organs at risk delineation in CT images

Xiaohui Xie et al. University of California, DeepVoxel Inc.

## 0. Abstract

Radiation therapy is one of the most widely used therapies for cancer treatment. A critical step in radiation therapy planning is to accurately delineate all organs at risk (OARs) to minimize potential adverse effects to healthy surrounding organs. However, manually delineating OARs based on computed tomography images is time-consuming and error-prone. Here, we present a deep learning model to automatically delineate OARs in head and neck, trained on a dataset of 215 computed tomography scans with 28 OARs manually delineated by experienced radiation oncologists. On a hold-out dataset of 100 computed tomography scans, our model achieves an average Dice similarity coefficient of 78.34% across the 28 OARs, significantly outperforming human experts and the previous state-of-the-art method by 10.05% and 5.18%, respectively. Our model takes only a few seconds to delineate an entire scan, compared to over half an hour by human experts. These findings demonstrate the potential for deep learning to improve the quality and reduce the treatment planning time of radiation therapy.

放射治疗是癌症治疗最广泛使用的疗法之一。放射治疗计划的一个关键步骤是准确的勾画所有OAR，以最小化对周围健康器官可能的副作用。但是，基于CT的手动勾画的OARs非常耗时，容易出错。这里，我们提出一种基于DL的自动勾画头颈部OARs模型，在215个CT scans的数据集上进行了训练，有28个OARs，由有经验的放射肿瘤学家进行的手工勾画。在保留的100个CT scans的数据集上，我们的模型得到的28个OARs的平均Dice相似系数为78.34%，明显超过了人类专家，以及之前最好的方法，幅度分别为10.05%和5.18%。我们的模型只需要几秒来勾画整个scan，而由人类专家则需要超过半小时。这些发现说明DL有很大潜力来改进质量，减少放射治疗的计划时间。

## 1. Introduction

Radiation therapy (RT) is an important treatment option for many types of cancer. It can be used as an adjuvant treatment before or after surgery, or as a radical treatment administered jointly with chemotherapy[1–10]. However, irradiation can lead to damage of normal organs (‘organs at risk’, OARs), especially in treating head and neck cancers, owing to the complex anatomical structures and dense distribution of important organs in this area. Damaging normal organs in the head and neck can result in a series of complications, such as xerostomia, oral mucosal damage, laryngeal oedema, dysphagia, difficulty in opening the mouth, visual deterioration, hearing loss and cognitive impairment. Recently, as the efficacy of radiotherapy for head and neck cancers has been steadily improving, irradiation complications affecting patients’ quality of life have received increasing attention. Research on mitigating the toxic side effects of RT can have important clinical and social implications[11–18].

放射治疗RT是很多类型癌症的一个重要治疗选项。RT可以用于手术前后的一种辅助治疗方法，或与化疗一起，作为一种激进的治疗方法。但是，放射会导致对正常器官的伤害，尤其在对头颈部肿瘤的治疗中，主要因为在这个区域中，有非常复杂的解剖结构和重要器官的密集分布。伤害头颈部的正常器官可以导致一系列的并发症，比如口腔干燥，口腔黏膜损伤，喉水肿，吞咽困难，张口困难，视力变差，听力损伤和认知损伤。最近，由于头颈部肿瘤的放射治疗效率稳步提高，影响病人生活质量的放射并发症得到越来越多的关注。减缓RT副作用的研究，可以有重要的临床和社会意义。

A key step involved in mitigating the side effects of irradiation during RT planning is to accurately delineate all OARs so that they can be protected during radiotherapy. OAR delineation is typically done manually by radiation oncologists based on computed tomography (CT) scans, which can take significant time and effort depending on the number of OARs to be delineated and the complexity of the local anatomical structures. In the case of head and neck, the treatment range of many tumours in the area, such as nasopharyngeal carcinoma, is relatively large and covers a large number of OARs. Manual delineation is especially daunting because of the anatomical complexity of the area. Computational tools that automatically segment anatomical regions can greatly alleviate clinicians’ manual efforts, if these tools can delineate OARs accurately and within a reasonable amount of time.

在RT计划中，减缓放射的副作用的一个重要步骤是，准确的勾画所有OARs，这样在治疗的时候就可以得到保护。OAR勾画一般是由放射肿瘤学家在CT scans上手工进行的，这需要耗费很多时间和努力，看需要勾画多少OARs以及局部解剖结构的复杂度而定。在头颈部的情况下，这个区域中很多肿瘤的治疗范围，如nasopharyngeal carcinoma，是非常大的，覆盖很多OARs。由于这个区域的解剖复杂度很高，手工勾画尤其使人望而却步。自动分割解剖结构的计算工具可以极大的缓解临床医生的手工耗时，但是这些工具需要准确的勾画OARs，花费的时间也不能太长。

A number of computational methods have been proposed to delineate OARs from CT or magnetic resonance imaging (MRI) images. Traditional OAR delineation methods are mostly atlas-based, producing OAR segmentation by aligning new images to a fixed set of manually labelled image templates. However, atlas-based methods have some limitations. First, they are computationally intensive and often take many minutes or even hours to complete, depending on hardware and implementation. Second, because of the reliance on templates, they cannot adequately account for anatomical variations naturally occurring among patients or due to the growth of tumours[19–36].

已经提出了很多计算方法来在CT或MRI中勾画OARs。传统OAR勾画方法主要是基于atlas的，通过将新图像与手动标注过的图像模版集合进行对齐，来生成OAR分割。但是，基于atlas的方法有一些限制。首先，它们计算量很大，需要消耗很多分钟甚至几小时来完成，这要看硬件情况和实现方法。第二，由于依赖模版，他们不会自然的考虑到不同病人之间解剖结构的变化，或由于肿瘤生长导致的解剖结构变化。

Deep learning provides an alternative solution to the OAR delineation problem[37]. Deep convolutional neural networks (DCNNs) can learn to extract a hierarchy of complex image features directly from data and utilize these image features to segment anatomical structures, without mapping an image to templates. DCNNs have been successfully applied to segment objects in natural images, as well as biomedical images from CT, MRI or microscopy. There are existing works that apply DCNN to head and neck OAR delineation. Most of these apply deep learning to segment OARs from cropped local patches around regions of interest, which is achieved either manually or based on outputs from image registrations by mapping to templates[38–47] or by delineating OARs one slice at a time[48].

DL可以给出另一种OAR勾画方法。DCNNs可以学习直接从数据中提取出复杂的图像特征，并利用这些图像特征来分割解剖结构，而不需要将图像映射到模板中。DCNNs已经成功的应用于自然图像的目标分割，以及CT，MRI和显微等生物图像的分割。已经有一些工作将DCNN用于头颈部OAR的勾画。多数将深度学习用于从ROI附近的局部图像块中分割OARs，这要么是手动的，或基于图像配准的输出得到的，将图像映射到模版，或一次从一个slice中勾画OARs。

Recently, delineating OARs directly from whole-volume CT images has also been attempted and shown better performance than atlas-based methods[49–52]. However, these studies were limited in scope and/or scale, with only a few OARs delineated and a limited number of samples tested. More importantly, previous studies have not examined the clinical utility of these methods and to what extent these methods might actually help clinicians.

最近，直接从whole-volume CT图像勾画OARs也有一些尝试，得到了比基于atlas的方法更好的结果。但是，这些研究受限于scope或scale，只勾画了几个OARs，测试的样本数量也很有限。更重要的是，之前的研究没有研究这些方法的临床功能，以及这些方法能在多大程度上真正帮助到临床医生。

In this study, we sought to address the questions described above by proposing a new deep learning model that can delineate a comprehensive set of 28 OARs in the head and neck area (Fig. 1), trained with 215 CT samples collected and carefully annotated by experienced radiation oncologists with over 10 years of experience. The accuracy of our model was compared to both previous state-of-the-art methods and a radiotherapy practitioner. Moreover, we deployed our deep learning model in actual RT planning of new patient cases and evaluated the clinical utility of the model.

在这项研究中，我们尝试解决上述问题，提出了一种新的深度学习模型，可以勾画头颈部的28个OARs，用215个CT样本，经过有10年以上经验的熟练的放射肿瘤学家仔细勾画。我们模型的准确性与之前最好的方法和一个放射治疗医生进行比较。而且，我们将我们的深度学习模型部署到了真正的新病人的RT计划中，评估了模型的临床用途。

Our deep learning model addresses some key constraints of the existing deep learning methods used in this area. Most of the existing deep learning methods for OAR segmentation are based on U-Net-like network architectures[53,54]. U-Net consists of an encoder and decoder architecture, with lateral connections concatenating low-level and high-level feature maps. The integrated image features infuse both low-level image information and high-level semantic content, which has proven very effective for precise voxel-wise classification. However, without constraining the rough location and size of an OAR, U-Net often produces many false positives (called pseudo hot-spots), scattered voxels lying outside the normal ranges of OARs (Supplementary Fig. 1). These false positives are especially problematic in actual clinical applications, because it would take a significant amount of time and effort to manually remove them from the delineation map and, to a certain extent, it might offset the benefit received from correct predictions. In addition, there are also implementation issues when applying U-Net to whole-volume images. Training U-Net-like models based on whole-volume input requires specialized graphics processing units (GPUs) with large memory. For these reasons, most of the existing deep learning methods have focused on segmenting OARs from local image patches or on a slice-by-slice basis.

我们的深度学习模型对现有深度学习方法在这个领域中应用的一些关键限制进行了处理。多数现有的进行OAR分割的深度学习方法是基于U-Net类的网络架构的。U-Net由编码器和解码器架构组成，还有横向连接将底层和高层特征拼接了起来。综合的图像特征将底层图像信息和高层语义信息结合了起来，这已经被证实了，对于精确的逐体素的分类是非常有效的。但是，没有约束一个OAR的大致位置和大小的话，U-Net通常会产生很多假阳性结果（称为伪热点），还有在正常OARs范围之外散落的像素（见附录材料图1）。这些假阳性结果在实际的临床应用中有很多问题，因为需要耗费挺多时间和努力来从勾画图中手动去除掉，在一定程度上，可能会使得正确预测的好处打了折扣。除此以外，在将U-Net应用到whole-volume图像时，还有实现的问题。在whole-volume输入的基础上，训练类U-Net的模型，需要特殊的大显存GPU。基于这些原因，大多数现有的深度学习方法都关注的是，从局部图像块中，或逐个切片分割OARs。

To address the challenges outlined above, we propose a novel end-to-end deep learning framework, named Ua-Net (attention- modulated U-Net), to segment 28 OARs in the head and neck area from whole-volume CT images. The proposed framework consists of two stages: OAR detection and OAR segmentation. The first stage includes an OAR detection network, the objective of which is to identify the approximate location and size of each OAR[55] and to suppress false-positive predictions outside the normal range of each OAR. The second stage includes an OAR segmentation network, which utilizes the results of the first stage as a guide and focuses on regions containing OARs to derive a fine-scale segmentation of each individual OAR. This attention mechanism allows us to perform local contrast normalization to enhance image features for improving the segmentation of OAR boundaries with low contrast. The networks of the two stages share the same underlying feature extraction backbone, based on three-dimensional (3D) convolutions, allowing weight sharing and significantly reducing GPU memory cost. With the two-stage design, it is feasible to train and test the entire model end-to-end directly on whole-volume images, using easily accessible commodity GPUs.

为处理上述挑战，我们提出了一种新的端到端的深度学习框架，即Ua-Net（注意力调制的U-Net），从头颈部的whole-volume CT图像中分割28个OARs。提出的框架包含两个阶段：OAR检测和OAR分割。第一阶段包含一个OAR检测网络，其目标是识别每个OAR的大致位置和大小，以抑制在正常范围外的每个OAR的false-positive预测。第二个阶段包含一个OAR分割网络，利用第一阶段的结果作为指引，聚焦在含有OAR的区域中，推导出单个OAR的细尺度分割。注意力机制使我们可以进行局部对比度归一化，以增强图像特征，改进分割低对比度OAR的边缘。两阶段网络共享同样的特征提取骨干网络，基于3D卷积，可以进行权重共享，并显著降低GPU显存消耗。在两阶段设计的基础上，就可以在whole-volume图像上对整个模型进行端到端的直接训练和测试，只需要使用普通的GPUs。

In the following, we describe the detailed design of our model, the data that we have collected to train and test our model and the evaluation results by comparing it to state-of-the-art methods and a radiation oncologist. Furthermore, we present a study on the clinical utility of our method. In particular, we report how much time the clinicians can actually save when assisted by our model during RT planning. The work presented here provides a study of a clinically applicable deep learning model for OAR delineation in the head and neck area.

下面，我们描述一下我们模型的详细设计，我们收集的用于训练和测试模型的数据，以及与目前最好的方法和放射肿瘤学家的比较评估结果。而且，我们给出了我们方法的临床的功用研究。特别是，我们给出了在RT计划的时候，在我们的模型的帮组下，临床医师可以实际节省的时间。本文给出了一种可以临床应用的头颈部OAR勾画深度学习模型。

## 2. Data

We used three datasets in this study (Table 1). Dataset 1 contains an in-house collection of 175 CT scans with 28 OARs annotated (Fig. 1) in each scan. We randomly split the data into 145 cases for training and 30 cases for testing. Dataset 2 contains 140 CT scans from two different sources available at The Cancer Imaging Archive (TCIA)[56]. We manually delineated all 28 OARs in each of these scans and randomly split the data, with half reserved for training and the other half for testing. Altogether, this brings the total number of samples used for training to 215 and the total number of samples for testing to 100. Dataset 3 is a publicly available dataset with predetermined training (33 cases) and test (15 cases) sets, each with nine OARs annotated. It was used primarily for evaluation purposes.

我们在这个研究中使用了三个数据集（表1）。数据集1包含公司内部收集的175 CT scans，每个scan标注了28个OARs。我们随机将这些数据分割为145例训练集，和30集测试集。数据集2包含140 CT scans，有两个TCIA(The Cancer Imaging Archive)的数据源。我们手动标注了每个scan的所有28个OARs，并随机分割这些数据一半用于训练，另外一半用于测试。总计，就有215例进行训练的数据，测试样本共有100例。数据集3是一个公开可用的数据集，预先确定好了训练集（33例）和测试集（15例），每个标注了9个OARs。这基本上是用于评估的目的。

Table 1. Datasets used in this study

| | Data Source | No. of OARs annotated | Train | Test
--- | --- | --- | --- | ---
Dataset 1 | In-house | 28 | 145 | 30
Dataset 2 | HNC[58] | 28 | 18 | 17
| | HNPETCT[59] | 28 | 52 | 53
Dataset 3 | PDDCA[60] | 9 | 33 | 15
Total | | | 248 | 115

Dataset 1 contains CT scans from the head and neck areas of patients and the delineation of OARs in each scan by radiation oncologists. The data include 175 CT scans from patients with head and neck cancer (Table 2), who undertook RT from 2016 to 2018. The CT scans were generated by GE Discovery 590 RT with the following scanning conditions: bulb voltage 120kV, current 300mA, scan layer thickness 2.5 mm, scan field of view (SFOV) 50 cm, and scan range from top of the skull to tracheal carina.

数据集1包含的CT scans是头颈部的病人，每个scan中OAR的勾画是由放射肿瘤学家完成的。数据包含头颈部肿瘤的175个病人的CT scans（表2），在2016-2018年进行的放射治疗。这些CT scans是由GE Discovery 590 RT生成的，扫描条件如下：管电压120kV，管电流300mA，扫描层厚2.5mm，扫描FOV 50cm，扫描范围为颅顶到气管隆突。

Table 2. Characteristics of the in-house collected data

| | Train | Test
--- | --- | ---
No. of patients (CTs) | 145 | 30
Patient average age (years) | 61.0 | 65.5
Gender | 
Male | 112 | 24
Female | 33 | 6
Tumor site |
Nasopharynx | 25 | 7
Hypopharynx | 14 | 7
Larynx | 18 | 2
Nasal cavity | 3 | 1
Brain | 50 | 10
Oropharynx | 7 | 1
Parotid gland | 4 | 0
Other | 24 | 2

The CT scans were manually delineated by a radiation oncologist with more than 10 years of experience, using the Varian Eclipse 10.0 system for delineation and following the guidelines of ref. [57], with assistance of MRI images from the same patients when necessary. The delineations were further reviewed and revised by a second expert with more than 30 years of clinical experience in RT. We call the manual delineations generated in this way the ‘gold standard’ for both training and testing purposes, while being mindful of the caveat that there are potential subjective variants or even errors within the annotation.

CT scans由超过10年的放射肿瘤学家进行手动勾画，使用的是Varian Eclipse 10.0系统进行勾画，使用[57]的勾画指南，在需要的时候，使用同样病人的MRI进行协助勾画。勾画还会进一步由另一个专家进行检查并修改，这位专家超过30年放射治疗临床经验。我们称这种方法产生的手动勾画为训练和测试目的的金标准，同时也要留意到在这些标注中，会有潜在的主观的变化甚至是错误。

A clinically relevant set of 28 OARs were delineated in each CT scan: brachial plexus, brain stem, constrictor naris, ears (left and right), eyes (left and right), hypophysis, larynx, lenses (left and right), mandible, optic chiasm, optic nerves (left and right), oral cavity, parotids (left and right), submandibular gland left (SMG L), submandibular gland right (SMG R), spinal cord, sublingual gland, temporal lobes (left and right), thyroid, temporomandibular joint left (TMJ L), temporomandibular joint right (TMJ R) and trachea (Fig. 1).

在每个CT scan中临床有重大作用的28个OARs集进行了勾画：brachial plexus, brain stem, constrictor naris, ears (left and right), eyes (left and right), hypophysis, larynx, lenses (left and right), mandible, optic chiasm, optic nerves (left and right), oral cavity, parotids (left and right), submandibular gland left (SMG L), submandibular gland right (SMG R), spinal cord, sublingual gland, temporal lobes (left and right), thyroid, temporomandibular joint left (TMJ L), temporomandibular joint right (TMJ R) and trachea (图1)。

We randomly split the dataset into a training set consisting of 145 CT scans and a test set consisting of 30 CT scans. We verified that the distributions of gender, age and tumour sites were roughly equal between the training and test sets (see Table 2). The training set was used to train our model, and the test set was used for evaluation and was not seen by the model during training.

我们将数据集随机分割成训练集和测试集，分别包含145和30个CT scans。我们验证了性别、年龄和肿瘤位置的分布在训练和测试集中是大致等同的（见表2）。训练集用于训练我们的模型，测试集用于评估，模型在训练时并没有遇到这些模型。

Patient identities and other clinical information were removed from the data. Non-clinicians had access only to image data and corresponding OAR labels. The usage of the data for this study was reviewed and approved by an institutional review board (IRB) responsible for overseeing the human subject data, before the beginning of the study.

病人的ID和其他临床信息从数据中移除掉了。非临床人员只能访问图像数据及对应的OAR标签。本研究的数据的用途是经过检查过的，经过IRB批准的，IRB是负责在研究之前审查人类相关的数据。

Dataset 2 consists of CT scans from two sources: Head–Neck Cetuximab (HNC)[58] and Head–Neck–PET–CT (HNPETCT)[59], both available at TCIA[56]. HNC consists of image data from a clinical trial for stage III and IV head and neck carcinomas, while HNPETCT consists of imaging data from four different institutions in Québec with histologically proven head and neck cancer. We followed the same procedure as described in generating Dataset 1 to annotate 28 OARs in each of CT scans.

数据2包含的CT scans有两个来源：HNC[58]和HNPETCT[59]，都在TCIA[56]中可用。HNC包含的图像数据是第三和第四期的头颈部癌症的临床试验病人，而HNPETCT包含的图像数据是Québec的四个不同机构的，都已经被证明是头颈部癌症的病人。我们按照生成数据集1的同样方法，在每个CT scans中标注28个OARs。

Dataset 3 consists of CT scans from a public dataset called the Public Domain Database for Computational Anatomy (PDDCA), used in the head and neck auto segmentation challenge at the 2015 MICCAI conference[60]. A total of nine OARs were annotated in this dataset: brain stem, mandible, optic chiasm, optic nerve (left and right), parotid (left and right) and submandibular gland (left and right).

数据集3是公开数据集的CT scans，为Public Domain Database for Computational Anatomy (PDDCA)，用于2015 MICCAI[60]的头颈部自动分割挑战赛。数据集中标注了9个OARs：brain stem, mandible, optic chiasm, optic nerve (left and right), parotid (left and right) and submandibular gland (left and right).

## 3. Model performance

Ua-Net is an end-to-end deep learning model for OAR delineation, composed of two sub-networks: (1) OAR detection, the objective of which is to identify the approximate location and size of each OAR and (2) OAR segmentation, which extracts fine-scale image features and performs image segmentation with attention focused on individual OARs (Fig. 2). It receives whole-volume images as input and outputs predicted masks of all 28 OARs at once. It follows the general U-Net-like structure for feature extraction, consisting of an encoder (a sequence of downsampling blocks for extracting semantically more complex features) and a decoder (a sequence of upsampling blocks for increasing the resolutions of image features for fine-scale segmentation). Each feature extraction block is composed of several residual sub-blocks, all based on 3D convolution to make use of 3D image features in volumetric CT images.

Ua-Net是一个端到端的深度学习模型，可以进行OAR勾画，包括两个子网络：(1)OAR检测，目标是识别每个OAR的大致位置和大小；(2)OAR分割，提取细节图像特征，对每个OAR进行注意力机制的分割（图2）。网络以whole-volume图像作为输入，一次性输出所有28个OARs的预测掩膜。在特征提取时采用一般性的U-Net类的网络结构，包含一个编码器（下采样模块序列，提取语义越来越复杂的特征），和一个解码器（上采样模块序列，增加图像特征分辨率，进行细节分割）。每个特征提取模块都由几个残差子模块组成，所有的都是基于3D卷积的，利用CT体图像的3D图像特征。

A major difference between Ua-Net and the conventional U-Net model[53] is that Ua-Net utilizes the OAR detection module to first identify regions containing OARs, and then upsamples image features only within the detected OAR regions, instead of the whole volume as in U-Net. The two-stage design enables the model to focus its attention on extracting high-resolution image features surrounding OARs, with the advantage of reducing false-positive predictions outside the normal range of OARs and substantially cutting down the GPU memory consumption required for the upsampling step. A detailed description of the model is provided in the Methods.

Ua-Net与传统U-Net模型的主要区别是，Ua-Net利用了OAR检测模块来首先识别包含OARs的区域，然后只在检测到OAR的区域中将图像特征上采样，而不是像U-Net中在整个图像中进行操作。两阶段的设计使得模型可以将注意力聚焦在提取OAR附近的高分辨率图像特征，优势是减少了在OAR正常范围之外的假阳性预测，极大的降低了上采样步骤中的GPU内存消耗。模型的详细描述在Methods部分给出。

**Performance metrics**. We used the volumetric Dice similarity coefficient (DSC)[60] and the 95th percentile Hausdorff distance (95% HD)[60,61], the two most commonly used metrics in this field, to evaluate the quality of OAR delineation. The OAR segmentation results of our model are represented by 28 binary masks, one for each OAR. Each binary mask is a 3D array of the same size as the input CT images, with values of either 0 or 1, indicating whether the underlying voxel is part of the corresponding OAR. Let Mp and Mg be the set of voxels with value 1 in the predicted and gold standard masks, respectively. The DSC is defined as DSC = 2|Mp ∩ Mg|/(|Mp| + |Mg|), measuring the volumetric overlap between the two masks. In addition to DSC, we also measured the HD between the boundaries of two masks. Let Cp and Cg denote the contours of the predicted and gold standard masks, respectively. Define max HD to be max{h(Cp, Cg), h(Cg, Cp)}, where h(Cp, Cg) = max_{a∈Cp} min_{b∈Cg} ||a-b||. Because max HD is very sensitive to outliers, a more commonly used metric, 95% HD, which measures the 95th percentile distance between two contours, is used instead.

**性能指标**。我们使用了体DSC和95% Hausdorff距离这两个最常用的这个领域的度量，以评估OAR勾画的质量。我们模型的OAR分割结果表示为28个二值掩膜，每个OAR一个。每个二值掩膜都是与输入CT图像相同大小的3D阵列，其值为0或1，表示每个体素是否属于对应OAR。令Mp和Mg分别表示预测的值为1的体素，和金标准掩膜。DSC的定义为DSC = 2|Mp ∩ Mg|/(|Mp| + |Mg|)，度量的是两个掩膜的体交叠。除了DSC，我们还计算了两个掩膜的边缘的HD。令Cp和Cg分别表示预测的和金标准掩膜的边缘。定义max HD为max{h(Cp, Cg), h(Cg, Cp)}，其中h(Cp, Cg) = max_{a∈Cp} min_{b∈Cg} ||a-b||。由于max HD对于离群点非常敏感，所以更经常使用的是95% HD，度量的是两个边缘的95%距离。

To assess the clinical utility of our model, we calculated and compared the time that a radiation oncologist spent on delineating OARs, either from scratch or by modifying the delineation results from the model.

为得到我们模型的临床功用，我们计算并与一个放射肿瘤学家在勾画OAR的时间进行对比，比较了从头开始勾画的耗时，与修改我们模型的勾画结果的耗时。

**Comparison with state-of-the-art methods**. The Ua-Net model was trained on 215 CT scans in the training set (from Datasets 1 and 2; see Supplementary Methods for model training details). Next, we evaluate its segmentation accuracy on the test set, which includes 30 CT scans from Dataset 1 and 70 CT scans from Dataset 2. Because the two test sets are from difference sources, the performance evaluations are reported in separate tables (Tables 3 and 4 for Dataset 1 and Supplementary Table 5 for Dataset 2) for the purpose of assessing the robustness of our model across different data sites.

**与目前最好方法的比较**。Ua-Net模型在215 CT scans的训练集上进行训练（来自数据集1和2；模型训练细节详见补充资料中的Methods部分）。下一步，我们在测试集上评估其分割准确率，测试集包含数据集1的30 CT scans和数据集2的70 CT scans。因为这两个测试集是不同源的，其评估性能在两个表格中给出（表3和表4是数据集1的，附属材料中的表5是数据集2的），这样就评估了我们的模型在不同的数据上的稳健性。

We first compared Ua-Net to a state-of-the-art deep learning model called AnatomyNet[51], which has been shown to be able to significantly outperform traditional atlas-based methods for OAR delineation in terms of both accuracy and speed. AnatomyNet is representative of U-Net-like models for OAR segmentation, but stands out from the rest of the deep learning models in that it can segment OARs directly on whole-volume CT images instead of extracted local patches. To ensure consistent comparisons, we trained both models on the same training dataset using the same procedure.

我们首先比较了Ua-Net与目前最好的深度学习模型，称为AnatomyNet[51]，这种模型与传统的基于atlas进行OAR勾画的方法比较，在准确率和速度上都有明显优势。AnatomyNet是典型的U-Net类的OAR分割模型，但与其他深度学习模型相比有区别，因为可以直接从whole-volume CT图像中直接分割OAR，而不是从提取出的局部块中进行分割。为确保比较的一致性，我们对两个模型都在同样的训练集上进行了训练，采用的方法过程也是一样的。

In terms of DSC, Ua-Net outperformed AnatomyNet in 27 out of 28 OARs, with an average improvement of 4.24% (Table 3) on Dataset 1, and in 28 out of 28 OARs, with an average improvement of 5.7% (Supplementary Table 5) on Dataset 2. Ua-Net performed particularly better on anatomies that are difficult to delineate under normal contrast conditions, such as the optic chiasm and sublingual gland, probably due to the local contrast normalization mechanism implemented in Ua-Net. Ua-Net performed slightly worse than AnatomyNet on the right ear, the difference is relatively small.

以DSC进行比较，28个OARs中，Ua-Net在数据集1中，有27个都超过了AnatomyNet，平均提高了4.24%（表3），在数据集2上，所有28个都超过了，平均提高了5.7%。Ua-Net在那些正常对比度条件下很难进行勾画的解剖结构上，表现的尤其好，比如视交叉和sublingual gland，可能是因为Ua-Net中实现的局部对比度归一化机制的原因。Ua-Net在右耳朵上的表现比AnatomyNet略差，其区别相对较小。

The advantage of Ua-Net over AnatomyNet is even more obvious when evaluated in terms of the HD, with the average 95% HD decreasing from 21.96 mm to 6.21 mm. As shown in Table 4, AnatomyNet is prone to produce false positives outside the normal range of OARs, which is expected because its segmentation is performed on whole-volume images instead of localized OAR regions (as in Ua-Net). These false positives were small in terms of the number of voxels, having less negative effects on DSC, which measures volumetric overlap, but they significantly increased the HD.

Ua-Net比AnatomyNet的优势，在以HD为指标进行评估时，更为明显，95%HD平均由21.96mm降低到6.21mm。如表4所示，AnatomyNet很容易在OARs正常范围之外产生假阳性，这是在预期之内的，因为其分割是在whole-volume图像中的，而不是在局部的OAR区域中的（Ua-Net则是）。这些假阳性以体素数量来说是很少的，对DSC的负面影响较小，因为衡量的是体积重叠情况，但在HD的衡量下则非常大。

Next, we compared Ua-Net to multi-atlas segmentation (MAS), a classical OAR delineation method based on image registration (see Supplementary Methods for details of the MAS method). In both datasets, MAS generated significantly lower scores (15.56% lower average DSC score on Dataset 1 and 23.16% lower on Dataset 2 compared to our model), indicating that the classical method is not as competitive as the deep learning-based methods.

下面，我们将Ua-Net与multi-atlas分割(MAS)的方法进行比较，这是一种经典的基于图像配准的OAR勾画方法（见附录材料中Methods中的MAS方法细节）。在两个数据集中，MAS得到的分数都很低（与我们的模型相比，在数据集1中，低了15.56%，在数据集2中，低了23.16%），说明经典方法与深度学习方法相比，是没那么好的。

Table 3 DSC comparison on the test set of dataset 1

OAR | MAS | anatomyNet | Ua-Net | Human | Human_a
--- | --- | --- | --- | --- | ---
Brachial plexus | 30.38 ± 15.63 | 50.41 ± 8.08 | **56.15 ± 10.83** | 33.03 ± 7.83 | 33.03 ± 7.83
Brain stem | 82.25 ± 7.47 | 82.63 ± 4.57 | **86.25 ± 3.86** | 83.25 ± 4.63 | 83.47 ± 4.35
Constrictor naris | 66.38 ± 8.21 | 73.68 ± 7.56 | **75.46 ± 6.13** | 62.34 ± 8.63 | 62.34 ± 8.63
Ear L | 70.38 ± 14.94 | 76.68 ± 5.00 | **77.28 ± 4.25** | 43.57 ± 12.63 | 43.57 ± 12.63
Ear R | 70.03 ± 15.57 | **78.77 ± 5.77** | 78.64 ± 6.35 | 39.71 ± 10.81 | 39.71 ± 10.81

Table 4 average 95th percentile HD comparison on the test set of Dataset 1

OAR | MAS | anatomyNet | Ua-Net | Human | Human_a
--- | --- | --- | --- | --- | ---
Brachial plexus | 30.73 ± 25.99 | 37.97 ± 36.44 | **18.27 ± 14.53** | 43.20 ± 18.19 | 43.20 ± 18.19
Brain stem | 6.62 ± 2.99 | 5.30 ± 1.36 | **4.75 ± 1.58** | 5.04 ± 1.28 | 4.89 ± 1.22
Constrictor naris | 9.49 ± 8.78 | 8.14 ± 6.20 | **5.71 ± 3.34** | 12.92 ± 7.57 | 12.92 ± 7.57
Ear L | 9.81 ± 19.85 | 25.88 ± 76.28 | **5.04 ± 1.35** | 11.08 ± 3.49 | 11.08 ± 3.49
Ear R | 9.01 ± 16.04 | 23.28 ± 68.48 | **4.67 ± 1.42** | 13.44 ± 3.31 | 13.44 ± 3.31

Finally we compared the performance of our model to previous state-of-the-art results on Dataset 3 (PDDCA). Table 5 contains a summary of previously reported delineation results, evaluated in terms of DSC on nine OARs from the Dataset 3 test set. Ua-Net obtained the best delineation results on eight out of nine OARs, achieving an average DSC score of 81.23% across the nine OARs, higher than all previous methods.

最后将我们的模型与之前最好的结果在数据集3上进行了比较(PDDCA)。表5包含了之前给出的勾画结果的总结，在数据集3上的9个OARs上以DSC进行了评估。Ua-Net在9个OARs中的8个上得到了最好的勾画结果，在9个OARs上取得的平均DSC有81.23%，比之前所有的方法都要高。

**Comparison with human experts**. Having demonstrated that Ua-Net performed better than both the classical and state-of-the-art deep learning methods for OAR delineation, we proceeded to compare its performance to manual delineations produced by human experts. For this purpose, we enlisted a radiation oncologist with over 10 years of professional experience, who was not involved in annotating either the training or test datasets. The radiation oncologist manually delineated the 28 OARs on the 30 CT scans in the test set from Dataset 1 in accordance with the normal professional procedure, but without consulting other professionals or seeking assistance from additional data sources such as MRI images.

**与人类专家的比较**。已经证明了，Ua-Net比经典的和目前最好的OAR勾画方法都要好，然后我们继续与人类专家的手动勾画进行比较。为此，我们征集了一个放射肿瘤学家列表，都有超过10年的专业经验，但都没有对训练集或测试集进行过标注。放射肿瘤学家手动在数据集1的测试集的30个CT scans上，根据正常的专业程序勾画了28 OARs，但没有参考其他专业知识，也没有额外的数据如MRI的协助支持。

In terms of DSC, both Ua-Net and AnatomyNet performed better than the human expert. Ua-Net outperformed the human expert on 27 out of 28 OARs, with an average improvement of 10.15%. The human expert’s delineation had the lowest DSC scores on optic chiasm (28.61), brachial plexus (33.03) and sublingual gland (35.16), highlighting the challenge of manually delineating these organs, which are small in size and have a relatively low contrast in CT images. The gap between the human expert and our deep learning model delineation was smaller when the results were evaluated using the HD. Both the human expert and Ua-Net did substantially better than AnatomyNet, for the reasons explained above. Ua-Net did better than the human expert (with smaller 95% HD) on most of the OARs (22 out of 28), lowering the average 95% HD from 8.01 mm to 6.28 mm (a 21% reduction). Because the HD is very sensitive to outliers, it was a more challenging metric for models than human experts, whose mistakes were mostly confined to regions around OARs.

以DSC相比，Ua-Net和AnatomyNet都超过了人类专家。Ua-Net在28个OARs中的27个上超过了人类专家，平均提高了10.15%。人类专家的勾画在optic chiasm (28.61), brachial plexus (33.03)和sublingual gland (35.16)上DSC最低，说明手动勾画这些器官挑战很大，因为这些器官很小，而且在CT图像中对比度相对很低。人类专家与我们的深度学习模型的勾画，在用HD进行评估时，差距较小。人类专家与Ua-Net都比AnatomyNet效果要好的多，原因上面已经说过了。Ua-Net比人类专家在大部分OARs上（28个中的22个）的表现都要好（95% HD要好），将95% HD从8.01mm降到了6.28mm（降低了21%）。因为HD对于离群点非常敏感，对于模型来说是一个更有挑战的度量，因为人类专家的错误主要是在OARs附近的区域的。

In real clinical practice, clinicians may also reference MRI images during the delineation process. To benchmark the delineation quality of clinicians in a real clinical setting, we also asked the same clinician to update the delineation results guided by inputs from both CT and MRI images. We observed a noticeable improvement in the delineation quality of several OARs, especially those with low CT image contrast such as optic chiasm and optic nerves. This led to an increase of the average DSC to 72.17%. We should note that the lower score is probably contributed by multiple factors, including inter-observer variation[28,37,57,62], a common issue in OAR delineation, as well as skills and experiences.

在真实的临床实践中，临床医生在勾画过程中可能会参考MRI图像。为对临床医生在真实临床设置下的勾画质量进行基准测试，我们还要求同一个临床医生在有MRI协助的情况下更新勾画结果。我们在几个OARs中观察到了一些较为明显的勾画质量改进，尤其是那些在CT图像中对比度较低的，如视交叉和视神经。这会带来DSC平均值增加到72.17%，我们应当注意到更低的分数可能是因为多个因素造成的，保护观察者之间的差异[28,37,57,62]，这是OAR勾画中的普通问题，还包括技能和经验的原因。

Altogether, the experimental results described above suggest that (1) the two-stage model implemented in Ua-Net is beneficial for improving the performance of deep learning models in terms of DSC, and substantially so in terms of the HD, and (2) the model is also capable of providing better delineation performance than the human expert. Note that our model can complete the entire delineation process of a case within a couple of seconds (Supplementary Table 7). By contrast, the human expert took, on average, 34min to complete one case, highlighting the significant advantage of the deep learning model (Table 6).

总体来说，上述试验结果说明：(1)Ua-Net中实现的两阶段模型对于改进深度学习模型的DSC性能是有好处的，在HD性能的改进上更大；(2)模型可以比人类专家给出更好的勾画结果。注意，我们的模型可以在几秒钟之内完成一个病例的整个勾画过程。比较起来，人类专家要花34min完成一个病例，这说明深度学习模型有很大的优势。

Table 6 Time comparison for an oncologist delineating using two approaches

| | Tumor category | Mode 1(min) | Mode 2(min) | t | p
--- | --- | --- | --- | --- | ---
PA1 | NPC | 35 | 13
PA2 | NPC | 33 | 11
PA3 | NPC | 30 | 13
PA4 | NPC | 38 | 20
PA5 | HD | 36 | 11
PA6 | BM | 34 | 12
PA7 | NPC | 32 | 8
PA8 | NPC | 33 | 14
PA9 | BM | 35 | 15
PA10 | BM | 30 | 14
Average | | 33.60 ± 2.55 | 13.10 ± 3.14 | 21.68 | 4.5 × 10^−9

NPC, nasopharyngeal carcinoma; HD, Hodgkin disease; MD, brain metastasis. Mode 1 refers to delineation performed completely manually, without any assistance of computational tools, and mode 2 refers to delineation done by modifying the delineation result generated by the proposed method. t denotes the paired t-test statistic and P denotes the P value.

**Clinical performance**. Having demonstrated that our model can do better than both the state-of-the-art method and the human expert, we next sought to find out its clinical utility, that is, to what extent the model can actually help clinicians. For this purpose, we conducted a study to compare the time spent by radiation oncologists in delineating OARs under two modes—without or with assistance from our model. In the first mode, the delineation was performed completely manually from scratch. In the second mode, the delineation results of all 28 OARs from our model were provided to the clinician, who would then verify the results and revise incorrect delineations when necessary. The overall work time in this case includes the time spent on verifying the results, as well as the time spent on modifying the model’s predictions.

**临床性能**。证明了我们的模型可以比之前最好的模型和人类专家都要优秀，我们下一步要发现其临床用处，即，模型在多大程度上可以帮助到临床医生。为此，我们进行了一个研究，比较放射肿瘤学家在两种模式下勾画OARs所需的时间，即完全手动勾画，或在我们模型帮助下进行勾画。在第一种模式下，勾画是完全从头手动勾画的。在第二种模式下，我们模型所有28个OARs的勾画结果首先给临床医生，然后对医生对这些结果进行核实，并在必要的时候修改不正确的结果。这个病例的总体需要时间包括花在确认结果的时间中，以及花在修改模型预测的时间上。

Ten new CT scans from real-life radiotherapy planning were studied. We recorded the time spent by an experienced radiation oncologist to delineate 28 OARs in each of these 10 cases, operating under the two modes described above. To ensure the quality of delineation, all delineation results were checked and confirmed by a second radiation oncologist.

我们研究了10个真实的放射治疗计划中的新CT scans。我们记录了一个有经验的放射肿瘤学家在这10个病例中对28个OARs进行勾画所需的时间，包括上面所述的两种模式。为确保勾画质量，所有勾画结果都有另一个放射肿瘤学家进行确认。

Without assistance from the model, the radiation oncologist spent on average 33.6 ± 2.55 min to delineate one case. By contrast, when assisted by our model, the delineation time was substantially reduced, reaching an average value of 13.1 ± 3.14 min (Table 6), representing an approximately 61% reduction in time. A paired t-test confirmed that the differences were statistically significant (P = 4.5 × 10−9). We note that the radiation oncologist accepted the model delineation results on most OARs without any need of modification. Most of the modification time was spent on the brachial plexus and temporal lobes, two OARs with relatively large volumes.

在没有模型帮助的情况下，放射肿瘤学家平均花费33.6 ± 2.55 min来勾画一个病例。对比之下，在我们的模型帮助下，勾画时间大幅缩短，平均时间为13.1 ± 3.14 min (表6)，大约减少了61%的时间。成对的t-test确认了，差异在统计上是非常明显的(P = 4.5 × 10−9)。我们注意到，放射肿瘤学家在很多OARs上不需要任何修改就可以接受模型勾画结果。多数修改的时间都是在brachial plexus和temporal lobes上，这两个OARs占据的体积较大。

This study confirmed the clinical utility of our model, demonstrating that the model can generalize well on previously unseen cases, and can save a clinician’s time by as much as 61%, when offered as a tool to assist the clinician’s manual delineation.

本研究确认了我们的模型的临床用处，证明了我们可以在之前没见过的案例中泛化的很好，用作临床医生手动勾画的时间时，可以节省大约61%的时间。

**Visualization**. We randomly selected two representative CT scans from the holdout test set for visualizing the delineation quality of the proposed method. Figure 3 shows the model prediction and manual delineation results of 28 OARs in axial planes. Figure 4 shows an unusual case where the head is tilted. In both cases, the model was able to generate delineations that matched well the results produced by human experts.

**可视化**。我们从保留的测试集中随机的选择了两个有代表性的CT scans，展示我们提出的方法的勾画结果。图3展示了28个OARs的模型预测和手动勾画结果。图4展示了一个不常见的情况，其头部是歪的。在两种情况下，模型生成的勾画都与人类专家的结果有很好的匹配。

## 4. Discussion

In this study, we have presented a new deep learning model to automatically delineate OARs in the head and neck area. We have demonstrated that the new model improves the delineation accuracy over the state-of-the-art method by 5.18% in terms of the DSC score, and substantially more in terms of the HD. In addition, we have shown that the model also performs better than a radiation oncologist, achieving 10.15% higher DSC and 1.80mm lower 95% HD, when averaged across 28 OARs.

本研究中，我们提出了一个新的深度学习模型，可以自动勾画头颈部的OARs。我们已经证明了，新模型与目前最好的方法相比，可以将勾画准确率改进5.18% DSC分数，以HD评价还会提高更高。另外，我们还表明，模型比一个放射肿瘤学家性能更好，28个OARs的平均性能，DSC高10.15%，95% HD低了1.80mm。

The success of the model can be attributed to its two-stage design. Different from most existing deep learning models in this field, which are based on U-Net or its variants, Ua-Net first identifies regions containing OARs and then focuses on extracting image features around these focal areas. There are several advantages with this design. First, it allows the model to perform local contrast normalization within each detected anatomy, which we notice has a significant impact on delineating anatomies with low CT image contrast. In addition, it becomes more efficient to train and force the model to learn better features to segment boundaries of OARs, because the segmentation loss function is now confined to the local OAR regions. Second, the design significantly reduces false positives, effectively eliminating outliers away from the normal range of OARs. This is reflected in the improvement of the HD scores. Third, the design cuts down GPU memory consumption and is more computationally efficient. Training deep neural nets on volumetric CT images is computationally intensive and requires hardware with large GPU memory, which has become a bottleneck in many deep learning applications in this field. In models developed for OAR delineation, most of the GPU memory consumption occurs at the last few layers, where the image feature maps are upsampled to have the same spatial resolution as the original CT image. Our model only upsamples feature maps containing OARs, and thus is able to drastically cut down GPU memory consumption. With our model, it becomes feasible to delineate all 28 OARs from whole-volume CT images using only commodity GPUs (for example, with 11Gb memory), which we believe is important for the method to be able to be deployed in actual clinics.

模型的成功可以归功于其两阶段设计。这个领域现有的多数深度学习模型都是基于U-Net或其变体的，Ua-Net与其不同，首先识别出包含OARs的区域，然后在这些聚焦区域中提取特征。这种设计有几种优势。首先，可以使得模型在每个检测到的解剖结构区域内，进行局部对比度归一化，这在勾画低CT对比度有显著的影响。而且，模型的训练效率更高，可以迫使模型学习到更好的特征，以分割OARs的边缘，因为分割的损失函数现在局限在局部OAR区域。第二，这个设计显著降低了假阳性结果，有效的消除了离群点。这从HD分数的改进中可以反映出来。第三，模型的设计降低了GPU显存的消耗，计算上更高效。在体CT图像上训练深度神经网络，需要很大的计算量，需要大显存的GPU，这在这个领域的很多深度学习应用中都是瓶颈。在为OAR勾画开发的模型中，多数GPU显存的消耗都在最后几层网络中，图像特征图经过上采样，与原始CT图像有相同的空间分辨率。我们的模型只对有OARs的特征图进行上采样，所以可以极大的降低GPU显存消耗。使用我们的模型，从whole-volume CT图像中使用商用GPUs（如，显存11GB）勾画所有28个OARs变得可行了，我们相信这是很重要的，可以部署到实际的临床环境中。

This study has several important limitations. First, only CT images were used by our model to delineate OARs. Some anatomies, such as the optic chiasm, have low contrast on CT and are difficult to delineate based on the CT modality alone. It is important to integrate images from other modalities (for example, MRI) into the deep learning framework to further improve the delineation accuracy. Second, although we have taken great care in generating gold standard annotations (see Data section), these annotations were still carried out manually by human experts, with the caveat of potential subjective variations and even errors. We showed that an independent human expert only reached an average DSC score of 70.38% (72.17% if the human experts reference MRI images in addition to CT) on the test of Dataset 1 (Table 3). Some of these discrepancies can be attributed to the inter-observer variation among experts[28,37,57,62], although they have been trained to follow the same delineation guidelines and procedures. In this regard, further improving the annotation quality and generating an industry-wide standardized dataset will be necessary in the future. Nonetheless, our study suggested that the deep learning model provides an attractive solution for standardizing the delineation process and ensuring consistent results across institutions and individuals. Third, the dataset used here is relatively small for deep network training. In dealing with this constraint, we have limited the number of layers and the number of free parameters to control the complexity of our model, and have augmented the training data through affine and elastic transformation (Supplementary Methods). However, more data collected from a more diverse set of sources will be needed to improve the cross-domain adaptation and generalization of the model.

本研究有几个重要的局限。首先，我们的模型只使用了CT图像来勾画OARs。一些解剖结构，比如视交叉，在CT中对比度很低，只用CT很难勾画。将其他模态的图像（如MRI）整合到深度学习框架中，以改进勾画准确率，是非常重要的。第二，虽然我们费了很多努力生成金标准标注，这些标注仍然是由人工专家手动进行的，有潜在的主观差异甚至是错误的风险。我们表明，一个单独的人类专家只能达到70.38% DSC的平均水平（如果人类专家参考了MRI图像，则可以达到72.17%），这是在数据集1的测试集上得到的结果。一些差异可以归因于专家中不同观察者之间的差异，但实际上他们的训练是按照同样的勾画指南和过程进行的。就这一点而言，未来进一步改进标注质量，生成工业范围内的标准化数据集是很有必要的。尽管如此，我们的研究表明，深度学习模型为标准化勾画过程提供了有吸引力的解决方案，可以保证在机构和个人之间得到一致的结果。第三，这里使用的数据集，对于深度学习训练是相对较小的。要处理这种局限，我们限制了网络的层数和自由参数的数量，以控制我们模型的复杂度，对训练数据进行了仿射变换和弹性变换的增广。但是，需要从更多数据源收集更多数据，以改进cross-domain adaptation和模型的泛化能力。

In summary, we have demonstrated that our proposed deep learning approach can accurately delineate OARs in the head and neck with an accuracy comparable to an experienced radiation oncologist. It is clinically applicable and can already save approximately two-thirds of a clinician’s time spent on OAR delineation. With further improvements on model and data, it is conceivable that the time-consuming OAR delineation process critical for RT planning may be fully automated by deep learning methods.

总结起来，我们证明了我们提出的深度学习模型可以准确的头颈部的OARs，得到的准确率可以与熟练的放射肿瘤学家相媲美。这是可以在临床上应用的，已经节省了大约2/3的临床医生的OAR勾画时间。随着对模型和数据的进一步改进，可以相信，对放射治疗计划非常关键的非常耗时的OAR勾画过程，可以通过深度学习模型得到完全的自动化。

## 5. Methods

Ua-Net consists of two submodules—one for OAR detection and the other for OAR segmentation. The goal of the OAR detection module is to identify the location and size of each OAR from the CT images, while the goal of the OAR segmentation module is to further segment OARs within each detected OAR region. The overall diagram of the network architecture is shown in Fig. 2.

Ua-Net包含两个子模块 - 一个进行OAR检测，另一个进行OAR分割。OAR检测模块的目的是从CT图像中识别每个OAR的位置和大小，而OAR分割模块的目的是在每个检测到的OAR区域中进一步分割OAR。网络架构的整体框图如图2所示。

**OAR detection module**. The OAR detection module receives a whole-volume CT image as input (with dimensions D × H × W, denoting depth, height and width, respectively) and extracts images features through a sequence of downsampling blocks, followed by an upsampling block. Each downsampling block is composed of two residual sub-blocks, all based on 3D convolution, reducing the resolution by one-half along each axis after each downsampling. The last downsampling block is upsampled to a final feature map of 64 channels with size D/8 × H/8 × W/8 (feature_map_8 in Fig. 2) through transposed convolution and by concatenating feature maps of the same size from the corresponding downsampling block. OAR candidate screening is carried out based on this final feature map, with one head for bounding box regression and one head for binary classification (to be detailed in the following). Detected OAR candidate bounding boxes further undergo a 3D ROI-pooling[63] step to generate feature maps of fixed sizes, which are then used for further bounding box regression and for multi-class classification to identify the class label associated with each OAR.

**OAR检测模块**。OAR检测模块的输入为whole-volume CT图像（维度为D × H × W,，分别表示深度、高度和宽度），通过下采样模块提取图像特征，然后跟着一个上采样模块。每个下采样模块都有两个残差子模块组成，全都是基于3D卷积的，在每次下采样后都将分辨率沿着每个轴降低一半。最后一个下采样模块进行上采样，通过转置卷积并从对应的下采样模块拼接同样大小的特征图，得到的最终特征图为64通道，大小为D/8 × H/8 × W/8（图2中的feature_map_8）。OAR候选筛选基于这个最后的特征图来进行，一个头进行边界框回归，一个头进行3D ROI-pooling以生成固定大小的特征图，然后用于进一步边界框回归和进一步的分类，以识别每个OAR相关的类别标签。

To generate OAR candidates, we branch the final feature map of the detection module (feature_map_8 in Fig. 2) into two separate heads—one for bounding box regression and the other for binary classification, with each head undergoing
3 × 3 × 3 convolution followed by 1 × 1 × 1 convolution. Each bounding box is represented by a rectangular cuboid, defined by six parameters t = (x, y, z, d, h, w) with (x, y, z) denoting its centre and (d, h, w) its depth, height and width in original CT image coordinates. Overall, 12 anchors are used to generate OAR candidates at each sliding window. The selection of the anchors and their sizes are described in the Supplementary Methods.

为生成OAR候选，我们将检测模块的最终特征图（图2中的feature_map_8）分成两个分支头，一个进行边界框回归，另一个进行二值分类，每个头都进行3 × 3 × 3的卷积，然后进行1 × 1 × 1的卷积。每个边界框都表示为一个矩形立方体，由于6个参数t = (x, y, z, d, h, w)定义，(x, y, z)表示其中心，(d, h, w)表示其深度，高度和宽度，这都是以CT原始坐标系计算的。再每个滑窗中，总计用了12个锚框以生成OAR候选，每个锚框的选择及其大小在附加材料的Methods中进行描述。

The anchors produce a list of candidate bounding boxes. Let $t_i ∈ R^6$ be the bounding box parameter associated with the ith anchor, predicted by the regression head, and P_i be the probability of the anchor being an OAR, predicted by the classification head. We minimize a multi-task loss function

锚框生成了一个候选边界框列表。令$t_i ∈ R^6$是第i个锚框相关的边界框参数，由回归头预测，P_i是锚框为OAR的概率，这由分类头预测。我们对如下多类别损失函数进行最小化

$$L_d(\{p_i\}, \{t_i\}) = \frac {1}{N_{cls}} \sum_i L_{cls} (P_i, P_i^*) + λ \frac {1} {N_{reg}} \sum_i L_{reg} (t_i, t_i^*)$$(1)

where the first term is a classification loss, the second term is a regression loss and λ is a hyper parameter balancing the two losses (set to 1 in this study). $N_{cls}$ and $N_{reg}$ are the total number of anchors included in classification and regression loss calculation, respectively. $P_i^*$ is 0 if the ith anchor does not contain any OAR and 1 otherwise. $t^*_i$ is the ground truth box parameter. Both $t_i$ and $t^*_i$ are parameterized relative to the size of the anchor box (see Supplementary Methods for details). We use weighted binary focal loss for $L_{cls}$ (ref. [64]) and smooth $l_1$ loss for $L_{reg}$.

其中第一项是分类损失，第二项是回归损失，λ是超参数，用于平衡两个损失（这里设置为1）。$N_{cls}$和$N_{reg}$分别是在分类和回归损失计算中涉及到的锚框数量。如果第i个锚框中没有OAR，则$P_i^*$为0，否则为1。$t^*_i$是真值框的参数。$t_i$和$t^*_i$与锚框大小相关的参数。我们在$L_{cls}$中使用加权二值focal损失，在$L_{reg}$中使用smooth $l_1$损失。

To assign a class label to each OAR proposal, we apply an ROI-pooling step[63] to image features extracted from the feature_map_8 in regions specified by its predicted bounding box to derive a feature map with fixed dimensions. Two fully connected layers are subsequently applied to classify each OAR proposal into one of 29 classes (28 anatomies plus one background) and to further regress coordinates and size offsets of its bounding box. We minimize a similar multi-task loss function as equation (1), with $L_{cls}$ replaced by a weighted cross entropy loss of 29 classes, while the regression loss remains the same. The final output of the OAR detection network is the predicted bounding box ($\hat x, \hat y, \hat z, \hat d, \hat h, \hat w$) in the original image coordinates, with the corresponding class label $\hat c$ for each OAR.

为给每个OAR建议指定一个类别标签，我们对从feature_map_8中在预测的边界框之内提取出的图像特征进行ROI-pooling步骤，以推导出固定尺寸的特征图。然后用两个全连接层来对每个OAR候选进行29类的分类（28个解剖结构加上一个背景），并进一步对其边界框的坐标和大小进行回归。我们对与式(1)类似的多任务损失函数进行最小化，将$L_{cls}$替换为29类的加权交叉熵损失，回归损失仍然是一样的。OAR检测网络的最终输出是预测的边界框($\hat x, \hat y, \hat z, \hat d, \hat h, \hat w$)，在原始图像坐标系中的坐标，包括每个对应的OAR的类别标签$\hat c$。

**OAR segmentation module**. The goal of the segmentation module is to segment each of the OAR regions returned by the detection module. The module takes the bounding box and the class label of each OAR as input, and produces a binary mask to delineate the OAR in the original image resolution. It starts by cropping feature maps from the feature_map_8, the location and size of which are specified by the predicted bounding box. The cropped feature maps are subsequently upsampled by a sequence of upsampling blocks to derive a final set of feature maps in the original CT resolution (that is, from 1/8× to 1× resolution). Each upsampling block is composed of a trilinear upsampling, followed by a 3D convolution and local contrast normalization. To incorporate fine-scale local image features, we also crop image features from the feature maps derived by the downsampling blocks (in the detection module) in regions specified by the bounding box, and concatenate them into the feature maps of the corresponding upsampling blocks. The final segmentation feature map consists of 64 channels with size $\hat d × \hat h × \hat w$, the same as the dimensions of the predicted bounding box. Finally we apply a 1 × 1 × 1 3D convolution (chosen according to the class label $\hat c$) to this final feature map, followed by sigmoid transformation, to generate the predicted mask m, a set indexed by voxel coordinates, with $m_i$ denoting the probability of voxel i being the foreground of the OAR. The same procedure is applied to each detected OAR within a CT scan. The final predicted mask $m^c$ associated with OAR c is taken to be the union of all $m_i$ whose predicted OAR class label is c.

**OAR分割模块**。分割模块的目的是对由检测模块返回的每个OAR区域进行分割。这个模块以每个OAR的边界框和类别标签为输入，生成一个二值掩膜以在原始图像分辨率中勾画OAR。模块开始先从feature_map_8中将图像特征进行剪切，其位置和大小由预测的边界框进行指定。剪切过的图像特征然后由上采样模块进行上采样，推导得到原始CT分辨率下的特征图集（即，从1/8x到1x分辨率）。每个上采样模块都由三次线性上采样，3D卷积和局部对比度归一化组成。为将细节局部图像特征考虑进去，我们还剪切了下采样模块（在检测模块中）的特征图，区域就是边界框，并将其拼接到对应的上采样模块中。最终的分割特征图包含64个通道，大小为$\hat d × \hat h × \hat w$，与预测的边界框的维度相同。最后我们对这个最终的特征图使用1 × 1 × 1卷积（根据类别标签$\hat c$选择的），然后进行sigmoid变换，以生成预测的掩膜m，其下标与体素下标一样，$m_i$表示体素i是OAR的概率。对CT scan中每个检测到的OAR都进行同样的过程。最终预测的掩膜$m^c$（与OAR c相关）是所有预测为类别c的OAR的$m_i$的并集。

The segmentation loss associated with one CT scan is defined to be 一个CT scan相关的分割损失定义为：

$$L_s = \sum_{c=1}^{28} I(c) (1 - ϕ(m^c, g^c))$$

where I(c) is an indicator function, taking 1 if OAR c is detected by the detection module and zero otherwise. g^c denotes the ground truth binary mask of OAR c: g^c_i = 1 if voxel i is within the OAR and zero otherwise. ϕ(m, g) computes a soft Dice score between the predicted mask m and the ground truth g:

其中I(c)是指示函数，如果检测模块检测到OAR c，则为1，否则为0。g^c表示OAR c的真值二值掩膜，如果体素i在OAR中则为g^c_i = 1，否则为0。ϕ(m, g)计算的是预测的掩膜m和真值g之间的软Dice分数：

$$ϕ(m, g) = \frac {\sum_{i=1}^N m_i g_i} {\sum_{i=1}^N m_i g_i + α \sum_{i=1}^N m_i (1-g_i) + β \sum_{i=1}^N (1-m_i) g_i + ϵ}$$

where i is a voxel index and N denotes the total number of voxels. The terms $\sum_{i=1}^N m_i (1-g_i)$ and $\sum_{i=1}^N (1-m_i) g_i$ can be understood as soft false positives and soft false negatives, respIectively. Parameters α and β control the weights of penalizing false positives and false negatives, and were set to be 0.5 in this study. The ε term was added to ensure the numerical stability of the loss function.

其中i是体素索引，N表示体素总量。项$\sum_{i=1}^N m_i (1-g_i)$和$\sum_{i=1}^N (1-m_i) g_i$可以分别理解为软的假阳性和软的假隐性。参数α和β控制的权重是用来惩罚假阳性和假阴性的，本研究中设为0.5。ε项的加入是为了确保损失函数的数值稳定性。

**Local contrast normalization**. To facilitate the training of the segmentation module, we apply a response normalization step that we refer to as ‘local contrast normalization’ to feature maps of the upsampling blocks. It standardizes each 3D response map to zero mean and unit variance across all voxels. More specifically, let $x ∈ R^{C × D × H × W}$ be a feature map with C channels and of dimension D × H × W. The local contrast normalization step transforms the map to $y ∈ R^{C × D × H × W}$ with

**局部对比度归一化**。为帮助分割模块的训练，我们对上采样模块的特征图应用了一个响应归一化步骤，我们称之为“局部对比度归一化”。它将每个3D响应图的所有体素标准化到零均值和单位方差。更具体的，令$x ∈ R^{C × D × H × W}$为C通道的特征图，维度为D × H × W。局部对比度归一化步骤将特征图变换到$y ∈ R^{C × D × H × W}$，采用下式的变换：

$$y_{cijk} = \frac {x_{cijk} - μ_c} {\sqrt{σ_c^2+ϵ}}$$

where μ_c and σ_c are the mean and standard deviation of the voxel intensities within the c-th channel of the feature map x. We found that the local contrast normalization step can not only facilitate the training by making it converge faster, but also improve segmentation accuracy (see Supplementary Discussion for details).

其中μ_c和σ_c是特征图x的第c通道的体素灰度的均值和标准差。我们发现，局部对比度归一化步骤不仅可以通过使其更快收敛而促进训练，还可以改进分割准确率（详见附加材料的讨论）。

At a high level, Ua-Net shares similarities with Mask-RCNN[65] and the feature pyramid network (FPN)[66]. However, the overall objectives and implementation details are quite different. First, the ultimate goal of Ua-Net is segmentation, while the goal of Mask-RCNN is both object detection and segmentation. As a result, Mask-RCNN has two equally important parallel heads—one for detection and the other for segmentation after ROI aligning. By contrast, Ua-Net is a two-stage model with only the segmentation head in the second stage. Second, Ua-Net is designed to perform segmentation on the original image resolution. It is different from FPN in that (1) it does not perform multi-scale segmentation as would be the case for FPN and (2) its bottom-up (upsampling) path and lateral feature concatenation from the top-down path only involve cropped image/feature map regions containing the detected OARs, unlike FPN, where both bottom-up and top-down paths are on whole images/feature maps. These design considerations allow the model to focus its attention on the fine and detailed segmentation of each individual OAR.

从高层来说，Ua-Net与Mask RCNN、FPN有很多相似点。但，其总体目标和实现细节是非常不一样的。首先，Ua-Net的最终目标是分割，而Mask RCNN的目标是目标检测和分割。结果是，Mask RCNN有两个同样重要的并行头 - 一个进行检测，另一个在ROI对齐后进行分割。对比起来，Ua-Net是一个两阶段模型，只在第二阶段有分割头。第二，Ua-Net的设计是在原始图像分辨率上进行分割。这与FPN是不一样的，即(1)其不进行多尺度分割，而FPN有多尺度的任务；(2)其自下而上（上采样）的路径和自上而下的横向特征拼接，只与含有检测到的OARs的剪切的图像/特征图区域相关，而FPN与之不同，其自下而上和自上而下的路径是在整幅图像/特征图上的。这些设计考虑使模型可以聚焦在其注意力在细节上，仔细的分割每个单独的OAR。

## 6. Data availability

Because of patient privacy, access to the training data in Dataset 1 will be granted on a case by case basis on submission of a request to the corresponding authors. The availability of the train data is subject to review and approval by IRB. The test data in Dataset 1 is available for non-commercial research purpose at https:// github.com/uci-cbcl/UaNet#Data. The CT images for Dataset 2 are freely available at https://doi.org/10.7937/K9/TCIA.2015.7AKGJUPZ and https://doi.org/10.7937/ K9/TCIA.2017.8oje5q00; we provide the annotated dataset (freely available) for non-commercial use at https://github.com/uci-cbcl/UaNet#Data. Dataset 3 is freely available at http://www.imagenglab.com/newsite/pddca/.

## 7. Code availability

Code for the algorithm development, evaluation and statistical analysis is freely available for non-commercial research purposes (https://github.com/uci-cbcl/UaNet).