# A Statistical Approach to Texture Classification from Single Images

Manik Varma and Andrew Zisserman, VGG

## 0. Abstract

We investigate texture classification from single images obtained under unknown viewpoint and illumination. A statistical approach is developed where textures are modelled by the joint probability distribution of filter responses. This distribution is represented by the frequency histogram of filter response cluster centres (textons). Recognition proceeds from single, uncalibrated images and the novelty here is that rotationally invariant filters are used and the filter response space is low dimensional.

我们研究了在未知视角和光照条件下得到的单幅图像的纹理分类问题。提出了一种统计方法，对纹理采用滤波器响应的联合概率分布进行了建模。这个分布是由滤波器响应簇中心的频率直方图表示的。对单幅、未校准的图像进行识别，这里的创新是，采用了旋转不变的滤波器，滤波器响应空间是低维的。

Classification performance is compared with the filter banks and methods of Leung and Malik [IJCV 2001], Schmid [CVPR 2001] and Cula and Dana [IJCV 2004] and it is demonstrated that superior performance is achieved here. Classification results are presented for all 61 materials in the Columbia-Utrecht texture database.

分类性能与滤波器组、Leung and Malik、Schmid和Cula and Dana的方法进行了比较，证明得到了很好的性能。对Columbia-Utrecht纹理数据集中的61种材质进行了分类，给出了结果。

We also discuss the effects of various parameters on our classification algorithm – such as the choice of filter bank and rotational invariance, the size of the texton dictionary as well as the number of training images used. Finally, we present a method of reliably measuring relative orientation co-occurrence statistics in a rotationally invariant manner, and discuss whether incorporating such information can enhance the classifier’s performance.

我们还讨论了分类算法中各种参数的作用，比如滤波器组和旋转不变性的选项，texton字典的大小，以及使用的训练图像的数量。最后，我们提出了一种方法，可以以一种旋转不变的方式可靠的度量相对方向共现统计结果，并讨论了将这种信息结合进来，是否可以增强分类器的性能。

**Keywords**: material classification, 3D textures, textons, filter banks, rotation invariance

## 1. Introduction

In this paper, we investigate the problem of classifying materials from their imaged appearance, without imposing any constraints on, or requiring any a priori knowledge of, the viewing or illumination conditions under which these images were obtained. Classifying textures from single images under such general conditions is a very demanding task.

本文中，我们研究了根据材质的图像对其进行分类的问题，对成像的视角或光照条件不施加任何限制，也不需要任何先验知识。在这种通用条件下，从单幅图像中对纹理进行分类，是要求非常高的任务。

A texture image is primarily a function of the following variables: the texture surface, its albedo, the illumination, the camera and its viewing position. Even if we were to keep the first two parameters fixed, i.e. photograph exactly the same patch of texture every time, minor changes in the other parameters can lead to dramatic changes in the resultant image (see figure 1). This causes a large variability in the imaged appearance of a texture and dealing with it successfully is one of the main tasks of any classification algorithm. Another factor which comes into play is that, quite often, two textures when photographed under very different imaging conditions can appear to be quite similar, as is illustrated by figure 2. It is a combination of both these factors which makes the texture classification problem so hard.

纹理图像基本是下面的变量的函数：纹理表面，其反射率，光照，相机及其视角。即使我们固定前两个参数，即，每次都对同样的纹理块进行拍照，其他参数很小的变化，也会得到非常不一样的图像（见图1）。这导致得到的纹理图像变化很大，对其进行成功分类，是一个主要任务。另一个可能的影响因素可能是，两种纹理在非常不同的成像条件下，会得到非常类似的图像，如图2所示。这些因素的组合，导致纹理分类问题非常困难。

A statistical learning approach to the problem is developed and investigated in this paper. Textures are modelled by the joint distribution of filter responses. This distribution is represented by texton (cluster centre) frequencies, and textons and texture models are learnt from training images. Classification of a novel image proceeds by mapping the image to a texton distribution and comparing this distribution to the learnt models. As such, this procedure is quite standard (Leung and Malik, 2001), but the originality comes in at two points: first, texton clustering is in a very low dimensional space and is also rotationally invariant. The second innovation is to classify textures from single images while representing each texture class by a small set of models.

本文提出研究了这个问题的一种统计学习方法。纹理用滤波器响应的联合分布进行建模。这个分布是用纹理基元的频率来表示，纹理基元和纹理是从训练图像中学习得到的。一幅新的图像的分类，是首先将图像映射到纹理基元分布，将其分布与学习到的模型进行比较。这样，这个过程是非常标准的，但其创新之处在于：第一，纹理基元的聚类是在很低维的空间进行的，是旋转不变的；第二是，用单幅图像对纹理进行分类，用少数模型来表示每个纹理类别。

Our approach is most closely related to those of Leung and Malik (Leung and Malik, 2001), Schmid (Schmid, 2001) and Cula and Dana (Cula and Dana, 2004). Leung and Malik’s method is not rotationally invariant and requires as input a set of registered images acquired under a (implicitly) known set of imaging conditions. Schmid’s approach is rotationally invariant but the invariance is achieved in a different manner to ours, and texton clustering is in a higher dimensional space. Cula and Dana classify from single images, but the method is not rotationally invariant and their algorithm for model selection differs from the one developed in this paper. These points are discussed in more detail subsequently.

我们的方法与Leung and Malik，Schmid和Cula and Dana的方法紧密相关。Leung and Malik的方法不是旋转不变的，需要输入一系列配准的图像，成像条件是（隐式）已知的。Schmid的方法是旋转不变的，但其不变性获得方式与我们不同，纹理基元聚类是在更高维的空间的。Cula and Dana从单幅图像进行分类，但这个方法不是旋转不变的，其模型选择的算法与本文的不同。这些点在后续进行更详细的讨论。

The paper is organised as follows: in section 2, the basic classification algorithm is developed within a rotationally invariant framework. The clustering, learning and classification steps of the algorithm are described, and the performance of four filter sets is compared. The sets include those used by Schmid (Schmid, 2001), Leung and Malik (Leung and Malik, 2001), and two rotationally invariant sets based on maximal filter responses. In section 3, methods are developed which minimise the number of models used to characterise the various texture classes. Section 4 then deals with various modifications and generalisations of the basic algorithm. In particular, the effect of the choice of texton dictionary and training images upon the classifier is investigated. Finally, the issue of whether information is lost by using only the first order statistics of rotationally invariant filter responses is discussed. A method for reliably measuring the relative orientation co-occurrence of textons is presented in order to incorporate second order statistics into the classification scheme.

本文组织如下：在第2部分中，提出了带有旋转不变性的基本分类算法。描述了算法的聚类，学习和分类步骤，比较了四个滤波器集合的性能。这些集合包括，Schmid使用的，Leung and Malik使用的，和两种基于maximal滤波器响应的旋转不变的集合。在第3部分中，提出了最小化模型数量来描述各种纹理类别特征的方法。第4部分对基本算法进行各种改动和泛化。特别的，研究了纹理基元字典选择和训练图像的选择的效果。最后，讨论了只使用旋转不变滤波器的一阶统计量，是否损失了信息的问题。。提出了一种可靠的测量纹理相对方向的共现的方法，以将二阶统论量纳入到分类方案中。

All experiments are carried out on the Columbia-Utrecht (CUReT) database (Dana et al., 1999), the same database used by (Cula and Dana, 2004; Leung and Malik, 2001). It is demonstrated that the classifier developed here achieves performance superior to that of (Cula and Dana, 2004) and (Leung and Malik, 2001), while requiring only a single image as input and with no information (implicit or explicit) about the illumination and viewing conditions. The CUReT database contains 61 textures, and each texture has 205 images obtained under different viewing and illumination conditions. The variety of textures in this database is shown in figure 3. Results are reported for all 61 textures. A preliminary version of these results appeared in (Varma and Zisserman, 2002).

所有的试验都是在Columbia-Utrecht (CUReT)数据集上进行的，其他作者也使用了这个数据集。证明了这里提出的分类器得到了最好的效果，而只需要一幅图像作为输入，不需要光照和视角条件的信息。CUReT数据集包含61种纹理，每种纹理包含205幅图像，是在不同的视角和光照条件下获得的。这个数据集中纹理的多样性如图3所示。对所有61种纹理都给出了结果。

### 1.1 Background

Most of the early work on material classification tended to view texture as albedo variation on a flat surface – thereby ignoring all surface normal effects which play a major role when imaging conditions vary. Recently, however, focus has been placed on these surface normal, or 3D, effects. Chantler et al. (Chantler et al., 2002a; Chantler et al., 2000; Chantler et al., 2002b) and Penirschke et al. (Penirschke et al., 2002) have studied the effect of change in illumination on textures and have developed photometric stereo based classification algorithms.

材质分类以前的工作倾向于将纹理视为平坦表面的反射率变化，因此忽略了所有表面法线的效果，这在成像条件变化的时候，会起到很大作用。但是最近，开始关注这些表面法线，或3D的效果。Chantler等和Penirschke等研究了对纹理的光照变化的效果，提出了基于立体光度的分类算法。

Dana et al. (Dana et al., 1999), realising the need for a large texture database which captured the variation of imaged appearances with change in viewpoint and illumination, created the Columbia-Utrecht (CUReT) database. Dana and Nayar (Dana and Nayar, 1998; Dana and Nayar, 1999) developed parametric models based on surface roughness and correlation lengths which were tested on sample textures from the CUReT database. However, no significant classification results were presented.

Dana等意识到需要一个大型纹理数据集，在视角和光照变化的时候，对不同的纹理进行成像，因此创造了Columbia-Utrecht (CUReT)数据集。Dana and Nayar提出了基于表面粗糙度和相关长度的参数模型，在CUReT数据集上的纹理样本上进行了测试。但是，没有给出明显的分类结果。

Leung and Malik (Leung and Malik, 2001) were amongst the first to seriously tackle the problem of classifying textures under varying viewpoint and illumination. In particular, they made an important innovation by giving an operational definition of a texton. They defined a 2D texton as a cluster centre in filter response space. This not only enabled textons to be generated automatically from an image, but also opened up the possibility of a universal set of textons for all images. To compensate for 3D effects, they proposed 3D textons which were cluster centres of filter responses over a stack of images with representative viewpoints and lighting. In the learning stage of their classification algorithm, 20 images of each texture were geometrically registered and mapped to a 48 dimensional filter response space. The registration was necessary because the clustering that defined the texton was in the stacked 20 × 48 = 960 dimensional space (i.e. the textons were 960-vectors), and it was important that each filter be applied at the same texture surface point as camera pose and illumination varied. In the classification stage, 20 novel images of the same texture were presented. However, these images also had to be registered and more significantly had to have the same order as the original 20 (i.e. they had to be taken from images with similar viewpoint and illumination to the original). In essence, the viewpoint and lighting were being supplied implicitly by this ordering. Leung and Malik also developed an MCMC algorithm for classifying a single image under known imaging conditions. However, the classification accuracy of this method was not as good as that achieved by the multiple image method.

Leung and Malik是第一个严肃的处理在不同视角和光照下分类纹理的问题的。特别是，他们有一个重要的创新，给出了纹理基元一个运算上的定义。他们将2D纹理基元定义为滤波响应空间的聚类中心。这不仅使纹理基元可以从图像中自动生成，而且还开启了对所有图像的统一纹理基元集的可能。为对3D效果进行补偿，他们提出了3D纹理基元，是滤波器对很多典型视角和光照的纹理图像的响应的聚类中心。在其分类算法的学习阶段，每种纹理的20幅图像进行几何对齐，映射到一个48维滤波器响应空间。对齐是必须的，因为在纹理基元上定义的聚类是在堆叠的20 × 48 = 960维空间上（即，纹理基元是960维的向量），在相机姿态和光照变化的时候，每个滤波器应用到相同的纹理表面点，这就非常重要。在分类阶段，给出同样纹理的20幅新的图像。但是，这些图像也必须要对齐，也要和原始的20幅图像有相同的顺序（即，要和之前的图像有相同的视角和光照）。实际上，视角和光照要通过这种顺序隐式的给出。Leung and Malik还提出了一种MCMC算法，在给定的成像条件下对单幅图像进行分类。但是，这种方法的分类准确率，与多图像方法相比，并不太好。

Cula and Dana (Cula and Dana, 2004) presented an algorithm based on Leung and Malik’s framework but capable of classifying single images without requiring any a priori information. Using much the same filter bank as Leung and Malik, they showed how to achieve results comparable to (Leung and Malik, 2001) but using 2D textons generated from single images instead of registered image stacks. We compare the performance of our algorithm with theirs in section 3.

Cula and Dana提出了一种基于Leung and Malik框架的算法，可以在不需要先验知识的条件下，对单幅图像进行分类。使用的滤波器组与Leung and Malik一样，他们展示了使用单幅图像生成的2D纹理基元，来得到类似的结果，而不需要对齐的图像组。我们在第3部分中与其算法进行了性能比较。

Suen and Healy (Suen and Healey, 2000) used correlation functions across multiple colour bands to determine basis textures for each texture class. They assumed that, for every texture image picked from a given class, the correlation function for that image could be represented as a linear combination of the basis texture correlation functions of that class. A nearest neighbour classifier employing the sum of squared differences metric was used. The number of basis images for a particular texture class also provided information about the dimensionality of that class. The main drawback of their algorithm was its heavy reliance on colour rather than purely on texture. While colour provides a very strong cue for discrimination, it can also be misleading due to the colour constancy problem (Funt et al., 1998). The classifier developed in this paper does not use colour information at all but rather normalises the images and filter responses so as to achieve partial invariance to changes in illuminant intensity.

Suen and Healy使用多色段之间的相关函数，来对每个纹理类别确定基纹理。他们假设，对从给定类别选择的每个纹理图像，这幅图像的相关函数，可以表示为这个类别基纹理相关函数的线性组合，并使用采用了均方差和的最近邻分类器。对一个特定的纹理类别，基图像的数量也给出了这个类别的维度的信息。其算法的主要缺点是，对颜色有很强的依赖性，而不是单纯依赖于纹理。虽然色彩给出了很强的线索，但由于色彩一致性问题，也会带来误导的问题。本文中提出的分类器没有使用色彩信息，而是对图像和滤波器响应进行了归一化，这样可以得到对光照亮度变化的部分不变性。

## 2. The Basic Algorithm

Weak classification algorithms based on the statistical distribution of filter responses have been particularly successful of late (Cula and Dana, 2004; Konishi and Yuille, 2000; Leung and Malik, 2001; Schmid, 2001). Our classification algorithm too is one such and, as is customary amongst weak classifiers, is divided into a learning stage and a classification stage. In the learning stage, training images are convolved with a filter bank to generate filter responses (see figure 4). Exemplar filter responses are chosen as textons (via K-Means clustering (Duda et al., 2001)) and are used to label each filter response, and thereby every pixel, in the training images. The histogram of texton frequencies is then used to form models corresponding to the training images (see figure 5). In the classification stage, the same procedure is followed to build the histogram corresponding to the novel image. This histogram is then compared with the models learnt during training and is classified on the basis of the comparison (see figure 6). A nearest neighbour classifier is used and the χ2 statistic employed to measure distances. The histograms should be normalised to sum to unity, but this is not required in our case as all training and testing images have the same number of pixels.

近年来，基于滤波器响应的统计分布的弱分类算法非常成功。我们的分类算法也是属于这种的，和其他弱分类器一样，分成学习阶段和分类阶段。在学习阶段，训练图像和一组滤波器进行卷积，生成滤波器响应（见图4）。典型的滤波器响应选为纹理基元，用于标记每个滤波器响应，因此也就标记了训练阶段的每个像素。纹理基元直方图频率用于形成对应于训练图像的模型（见图5）。在分类阶段，用同样的过程来构建新图像的直方图。这个直方图然后与训练阶段学习到的模型比较，在比较的基础上进行分类（见图6）。使用的是最近邻分类器，采用χ2统计量来度量距离。直方图应当归一化为和为1，但在我们的情况下不需要，因为所有的训练和测试图像有相同数量的像素。

In the following subsections, we describe the filters and algorithmic steps in more detail. Classification results are presented on the CUReT database, and compared with those of Leung and Malik (Leung and Malik, 2001) and Cula and Dana (Cula and Dana, 2004).

下面的小节中，我们详细描述了滤波器和算法步骤。分类结果在CUReT数据集上给出，并与其他的成功方法比较。

### 2.1 Rotationally Invariant Fitlers

In this subsection, we introduce the rotationally invariant filter sets that are used in the classification algorithm. We also describe two other filter sets that will be used in classification comparisons in subsection 2.4. The aspects of interest are the dimension of the filter space, and whether the filter set is rotationally invariant or not.

本小节中，我们提出了旋转不变滤波器组，用于分类算法。我们还描述了两种其他的滤波器组，在2.4节中用于分类结果比较。感兴趣的部分是滤波器空间的维度，以及滤波器组是否是旋转不变的。

The four filter sets that will be compared are: those of Leung and Malik (Leung and Malik, 2001) which are not rotationally invariant; those of Schmid (Schmid, 2001) which are; and two reduced sets of filters based on using the maximum response (which are again rotationally invariant). Filter sets will be assessed by their classification performance using textons clustered in their response spaces.

要进行比较的四种滤波器组是：Leung and Malik，不是旋转不变的；Schmid，是旋转不变的；基于使用最大响应的两种蜕化滤波器组，也是旋转不变的。滤波器组会通过分类性能进行评估，分类时使用在响应空间聚类的纹理基元。

#### 2.1.1 The Leung-Malik (LM) set

The LM set consists of 48 filters, partitioned as follows: first and second derivatives of Gaussians at 6 orientations and 3 scales making a total of 36; 8 Laplacian of Gaussian filters; and 4 Gaussians. The scale of the filters range between σ = 1 and σ = 10 pixels. They are shown in figure 7.

LM包括48个滤波器，分成下面的组：6个方向和3个尺度上的高斯函数的一阶和二阶导数，这共有36个滤波器；8个LoG滤波器；和4个高斯滤波器。滤波器的尺度范围从σ = 1到σ = 10像素。如图7所示。

#### 2.1.2. The Schmid (S) set

The S set consists of 13 rotationally invariant filters of the form

S集包含13个旋转不变的滤波器，形式为

$$F(r, σ, τ) = F_0(σ, τ) + cos(\frac {πτr} {σ}) e^{-\frac {r^2} {2σ^2}}$$

where F0(σ, τ) is added to obtain a zero DC component with the (σ, τ) pair taking values (2,1), (4,1), (4,2), (6,1), (6,2), (6,3), (8,1), (8,2), (8,3), (10,1), (10,2), (10,3) and (10,4). The filters are shown in figure 8. As can be seen all the filters have rotational symmetry.

其中F0(σ, τ)的加入是为了获得0直流部分，(σ, τ)对的值为(2,1), (4,1), (4,2), (6,1), (6,2), (6,3), (8,1), (8,2), (8,3), (10,1), (10,2), (10,3)和(10,4)。滤波器如图8所示。可以看出，所有滤波器都有旋转对称性。

#### 2.1.3. The Maximum Response (MR) sets

The MR8 filter bank consists of 38 filters but only 8 filter responses. The filter bank contains filters at multiple orientations but their outputs are “collapsed” by recording only the maximum filter response across all orientations. This achieves rotation invariance. The filter bank is shown in figure 9 and consists of a Gaussian and a Laplacian of Gaussian both with σ = 10 pixels (these filters have rotational symmetry), an edge filter at 3 scales (σx,σy)={(1,3), (2,6), (4,12)} and a bar filter at the same 3 scales. The latter two filters are oriented and, as in LM, occur at 6 orientations at each scale. Measuring only the maximum response across orientations reduces the number of responses from 38 (6 orientations at 3 scales for 2 oriented filters, plus 2 isotropic) to 8 (3 scales for 2 filters, plus 2 isotropic).

MR8滤波器组包含38个滤波器，但只有8个滤波器响应。滤波器组包含多个方向的滤波器，但其输出只记录了在所有方向的最大滤波器响应。这就获得了旋转不变性。滤波器组如图9所示，包含一个高斯函数和一个LoG函数，σ = 10（这些滤波器有旋转对称性），在3个尺度(σx,σy)={(1,3), (2,6), (4,12)}上的边缘滤波器，和同样3个尺度上的bar滤波器。后面两种滤波器都是有方向的，如同在LM中一样，在每个尺度上都有6个方向。只测量所有方向上的最大响应，将响应数量从38降到了8。

The MR4 filter bank is a subset of the MR8 filter bank where the oriented edge and bar filters occur at a single fixed scale (σx = 4, σy = 12).

MR4滤波器组是MR8滤波器组的子集，其中有向边缘和bar滤波器只在固定尺度上(σx = 4, σy = 12)。

The motivation for introducing these MR filters sets is twofold. The first is to overcome the limitations of traditional rotationally invariant filters which do not respond strongly to oriented image patches and thus do not provide good features for anisotropic textures. However, since the MR sets contain both isotropic filters as well as anisotropic filters at multiple orientations they are expected to generate good features for all types of textures. Additionally, unlike traditional rotationally invariant filters, the MR sets are also able to record the angle of maximum response. This enables us to compute higher order co-occurrence statistics on orientation and such statistics may prove useful in discriminating textures which appear to be very similar. We return to this in subsection 4.2.

引入这些MR滤波器有两个原因。第一是克服传统旋转不变滤波器的限制，即对有向图像块响应不强，因此对各项异性纹理没有提供很好的特征。但是，由于MR集包含各向同性和多个方向的各向异性滤波器，它们应当可以对各种类型的纹理都产生很好的特征。另外，与传统旋转不变的滤波器不同，MR集也可以记录最大响应的角度。这使我们可以对方向计算更高阶的共线统计量，这些统计量在区分外观很类似的纹理中会比较有用。我们在4.2节中对此进行讨论。

The second motivation arises out of a concern about the dimensionality of the filter response space. Quite apart from the extra processing and computational costs involved, the higher the dimensionality, the harder the clustering problem. In general, not only does the number of cluster centres needed to cover the space rise dramatically, so does the amount of training data required to reliably estimate each cluster centre. This is mitigated to some extent by the fact that texture features are sparse and can lie in lower dimensional subspaces. However, the presence of noise and the difficulty in finding and projecting onto these lower dimensional subspaces can counter these factors. Therefore, it is expected that the MR filter banks should generate more significant textons not only because of improved clustering in a lower dimensional space but also because rotated features are correctly mapped to the same texton.

第二个原因是滤波器响应空间维度的考虑。除了额外的处理和计算代价，维度越高，聚类问题就越难。总体上，不仅需要覆盖空间的聚类中心的数量会极大增加，而且用于可靠的估计每个聚类中心的训练数据数量也要极大增加。纹理特征是稀疏的，可以在低维子空间上，这一定程度上缓解了这个问题。但是，噪声的存在，和找到并投影到这些更低维的子空间上的困难，可以对付这些因素。因此，MR滤波器组期望生成更明显的纹理基元，不仅因为在低维空间上改进了聚类，而且因为旋转特征正确的映射到相同的基元。

### 2.2 Pre-processing

The following pre-processing steps are applied before going ahead with any learning or classification. 在学习或分类之前，先进行下面的预处理步骤。

First, before convolving with any of the filter banks, a central 200×200 texture region is cropped and retained from every image and the extraneous background data discarded. All processing is done on these cropped regions and they are converted to grey scale and intensity normalised to have zero mean and unit standard deviation. This normalisation gives invariance to global (i.e. across the entire region) affine transformations in the illumination intensity.

首先，在与任何滤波器组进行卷积之前，从每幅图像中剪切出中心的200×200纹理区域并保留下来，其余的背景数据都丢弃掉。所有的处理都是在这个剪切出的区域上进行，要将其转换到灰度图，灰度进行归一化，其均值为0，方差为1。这种归一化对光照亮度的全局（即，整个区域范围内的）仿射变换给出了不变性。

Second, all 4 filter banks are L1 normalised so that the responses of each filter lie roughly in the same range. In more detail, each filter Fi in the filter bank is divided by ||Fi||1 so that the filter has unit L1 norm. This helps vector quantization, when using Euclidean distances, as the scaling for each of the filter response axes becomes the same (Malik et al., 2001).

第二，所有4个滤波器组都进行了L1归一化，这样每个滤波器的响应都在大致相同的范围内。细节上，滤波器组中的每个滤波器Fi都除以||Fi||1，这样滤波器有单位L1范数。当使用欧式距离时，这对矢量量化有帮助，因为每个滤波器响应轴的缩放都变得一样了。

Third, following (Fowlkes et al., 2002; Malik et al., 2001) and motivated by Weber’s law, the filter response at each pixel x is (contrast) normalised as 第三，按照文献，受Weber定律推动，滤波器在每个像素x处的响应都归一化为

$$F(x) ← F(x) [log (1 + L(x)/0.03)] /L(x)$$

where L(x) = ||F(x)||2 is the magnitude of the filter response vector at that pixel. 其中L(x) = ||F(x)||2是滤波器响应向量在那个像素上的幅度。

### 2.3 Textons by Clustering

We now consider clustering the filter responses in order to generate a texton dictionary. This dictionary will subsequently be used to define texture models based on texton frequencies learnt from training images.

我们现在考虑对滤波器响应进行聚类，以生成纹理基元字典。这个字典后续会用于定义基于纹理模型，是基于从训练图像中学习到的纹理基元频率。

For each filter set, we adopt the following procedure for computing a texton dictionary: A selection of 13 images is chosen randomly for each texture (these images sample the variations in illumination and viewpoint), the filter responses over all these images are aggregated, and 10 texton cluster centres computed using the standard K-Means algorithm (Duda et al., 2001). The learnt textons for each texture are then collected into a single dictionary. For example, if there are 5 texture classes then the dictionary will contain 50 textons. Examples of the textons for the S, LM and MR8 filter banks are shown in figure 10.

对每个滤波器组，我们采用下面的过程计算纹理基元字典：从每种纹理中随机选择13幅图像（不同的光照和视角），滤波器在所有这些图像上的响应聚积在一起，用标准K-均值算法计算得到10个纹理基元聚类中心。每种纹理学习得到的纹理基元，收集到一个字典中。比如，如果有5个纹理类别，那么字典会包含50个纹理基元。S，LM和MR8滤波器组的纹理基元的例子如图10所示。

Our clustering task is considerably simpler than that of Leung and Malik, and Cula and Dana (who use essentially the same filter bank) as we are able to cluster in low, 4 and 8, dimensional spaces. This compares to 13 dimensional for S, and 48 dimensional for LM (we are not considering 3D textons at this point where the dimensionality is 960).

我们的聚类任务比其他任务要简单很多（但使用的基本上是相同的滤波器组），因为我们可以在低维，4维和8维空间上进行聚类。比较起来，S滤波器组是13维，LM是48维（这里我们没有考虑3D纹理基元，其维度是960）。

Concerning the rotation properties of the LM and MR textons, consider a texture and an (in plane) rotated version of the same texture. Corresponding features in the original and the rotated texture will map to the same point in MR filter space, but to different points in LM. It is therefore expected that more significant clusters will be obtained in the rotationally invariant case. Secondly, for the LM filter set, which is not rotationally invariant, it would be expected that its textons can not classify a rotated version of a texture unless the rotated version is included in the training set (both of these points are demonstrated in figure 11).

有关LM和MR纹理基元的旋转性质，考虑一个纹理和平面内旋转的同样纹理。原始纹理和旋转纹理的对应特征，在MR滤波器空间会映射到同样一个点，但在LM空间中则是不同的点。因此在旋转不变的情况下，会期望得到更显著的聚类。然后，对于LM滤波器组，并不是旋转不变的，因此其纹理基元不能对旋转版的纹理进行正确分类，除非旋转版的也在训练集之中（如图11所示）。

This establishes that there is an advantage in being rotationally invariant as rotated versions of the same texture can be represented by one histogram, while several are required for the LM textons. However, there is still the possibility that rotation invariance has the disadvantage that two different textures (which are not rotationally related) have the same histogram. We address this point next, where we compare classification rates over a variety of textures.

这说明了，旋转不变是有优势的，因为同样纹理的旋转版可以用同一直方图进行表示，而对于LM纹理基元，则需要用几个直方图进行表示。但是，旋转不变性也可能有缺点，即两个不同的纹理（并不是旋转相关的）有相同的直方图。我们下一步再比较不同纹理的分类准确率时再处理这一点。

### 2.4 Classification Method and Comparison Results

In this subsection we perform three experiments to assess texture classification rates over 92 images for each of 20, 40 and 61 texture classes respectively. The first experiment, where we classify images from 20 textures, corresponds to the setup employed by Cula and Dana (Cula and Dana, 2004). The second experiment, where 40 textures are classified, is modelled on the setup of Leung and Malik (Leung and Malik, 2001). In the third experiment, we classify all 61 textures present in the Columbia-Utrecht database. The 92 images are selected as follows: for each texture in the database, there are 118 images where the viewing angle θv is less than 60 degrees. Out of these, only those 92 are chosen for which a sufficiently large region could be cropped across all texture classes.

在这个小节中，我们进行三个试验，分别对20，40和61类纹理评估纹理分类准确率，每个类别92幅图像。第一个试验，我们对20类纹理进行分类，对应着Cula and Dana采用的设置。第二个试验，对40类纹理进行分类，对应着Leung and Malik的试验设置。在第三个试验中，我们对所有61类纹理进行分类。92幅图像按如下方法选择：对数据集中的每个纹理，有118幅图像其视角θv小于60度。在这其中，只选择了92幅，需要有足够大的区域来在所有纹理类别中进行剪切。

Each experiment consists of three stages: texton dictionary generation; model generation, where texture models are learnt from training images; and, classification of novel images. The 92 images for each texture are partitioned into two, disjoint sets. Images in the first (training) set are used for dictionary and model generation, classification accuracy is only assessed on the 46 images for each texture in the second (test) set.

每个试验包含三个阶段：纹理基元字典生成；模型生成，从训练图像中学习得到纹理模型；新图像的分类。每类纹理的92幅图像分成两个集合，训练集用于字典和模型生成，测试集用于评估分类准确率（46幅图像）。

Each of the 46 training images per texture defines a model for that class as follows: the image is mapped (vector quantized) to a texton distribution (histogram). Thus, each texture class is represented by a set of 46 histograms. An image from the test set is classified by forming its histogram and then choosing the closest model histogram learnt from the training set. The distance function used to define closest is the χ2 statistic (Press et al., 1992).

每类纹理有46幅训练图像，每幅图像为这个类别定义了一个模型：图像映射到一个纹理基元分布（直方图）。因此，每个纹理类别都由46个直方图进行表示。测试集的一幅图像进行分类时，先形成其直方图，选择从训练集学习的最仅仅的模型直方图。用于定义最近的距离函数是χ2统计量。

In all three experiments we follow both (Cula and Dana, 2004) and (Leung and Malik, 2001), and learn the texton dictionary from 20 textures (using the procedure outlined before in section 2.3). The particular textures used are specified in figure 7 of (Leung and Malik, 2001).

在所有三个试验中，我们都按照两篇文章中的设置进行，从20类纹理中学习纹理基元字典。使用的特别纹理如Leung and Malik的图7所示。

In the first experiment, 20 novel textures are chosen (see figure 19a in (Cula and Dana, 2004) for a list of the novel textures) and 20×46 = 920 novel images are classified in all. In the second experiment, the 40 textures specified in figure 7 of (Leung and Malik, 2001) are chosen and a total of 40 × 46 = 1840 novel images classified. Finally, in the third experiment, all 61 textures in the Columbia-Utrecht database are classified using the same procedure. The results for all three experiments are presented in table I.

在第一个试验中，选择了20个新的纹理，共计分类了20×46 = 920幅新的图像。在第二个试验中，选择了40个纹理类别，分类了40 × 46 = 1840幅图像。最后，在第三个试验中，所有61类纹理都使用相同的过程进行分类。三个试验的结果如表1所示。

#### 2.4.1. Discussion

Two points are notable in these results. First, the MR8 and S filters outperform the LM filters. This is a clear indicator that a rotationally invariant description is not a disadvantage (i.e. salient information for classification is not lost). Second, the fact that MR8 does better than S and LM is also evidence that it is detecting better features, for both isotropic and anisotropic textures, and that clustering in a lower dimensional space can be advantageous. The MR4 filter bank loses out because it only contains filters at a single scale and hence can’t extract such rich features. What is also very encouraging with these results is that as the number of texture classes increases there is only a small decrease in the accuracy of the classifier.

结果中有两点值得关注。第一，MR8和S滤波器组的性能超过了LM滤波器组。这明显说明，旋转不变的描述并不是劣势（即，对于分类的显著信息并没有丢失）。第二，MR8比S和LM要好，说明其检测到了更好的特征，对各向同性和各向异性纹理都是，在更低维空间进行聚类是有优势的。MR4滤波器组性能没那么好，因为其只包含了一个尺度的滤波器，因此不能提取处更丰富的特征。这些结果中令人鼓舞的是，随着分类纹理类别增加，分类器的准确率只有很小的下降。

## 3. Reducing the number of models

In this section, our objective is to reduce the number of training models required to characterise each texture class. In the previous section, the number of models was the same as the number of training images (and in effect (Leung and Malik, 2001) used 20 models/images for every texture). Here, we want to reduce the number of models to that appropriate for each class, independent of the number of training images.

本节中，我们的目标是降低训练模型的数量，这些模型用于描述每个纹理类别的特征。前一节中，模型的数量与训练图像的数量相同（Leung and Malik对每种纹理都使用了20个模型/图像）。这里，我们想减少模型的数量，与训练图像的数量无关。

One would expect that the number of different models that are needed to characterise a texture is a function of how much the texture changes in appearance with imaging conditions, i.e. it is a function of the material properties of the texture. For example, if a texture is isotropic then the effect of varying the lighting azimuthal angle will be less pronounced than for one that is anisotropic. Thus, other parameters (such as relief profile) being equal, fewer models would be required for the isotropic texture (than the anisotropic) to cover the changes due to lighting variation. This is demonstrated in figure 12.

应当期待的是，描述一种纹理所需要的不同的模型的数量，应当是随着成像条件的不同，纹理外观的变化程度的函数，即，是纹理的材质属性的函数。比如，如果一种纹理是各项同性的，那么变化光照角度的效果，就没有各项异性的那么明显。因此，其他参数一样的情况下，对各项同性的纹理应该需要更少的模型，就可以覆盖光照变化导致的变化。如图12所示。

However, if we are selecting models for the express purpose of classification, then another parameter, the inter class image variation, also becomes very important in determining the number of models. For example, even if a texture varies considerably with changing imaging conditions it can be classified accurately using just a few models if all the other textures look very different from it. Conversely, if two textures look very similar then many models may be needed to distinguish between them even if they do not show much variation individually.

但是，如果我们为分类的目标选择模型，那么另一个参数，类别间图像变化，在确定模型数量上也变得非常重要。比如，即使一个纹理在变化成像条件时变化很大，如果所有其他纹理看起来都非常不一样，那么只使用几个模型就可以正确分类。相反的，如果两个纹理看起来很一样，那么就需要很多模型来对其进行区分。

Broadly speaking, there are two major approaches to the problem of model reduction. In the first, various concepts from the Machine Learning literature can be used to select a subset of the models while maximising some criteria of classification and generalisation. The second approach is geometric and focuses on building descriptors invariant to imaging conditions so as to reduce the number of models needed.

广义来说，减少模型有两个主要的方法。首先，机器学习中的各种概念都可以用于选择模型的一个子集，同时最大化分类和泛化的一些规则。第二种方法是几何的，关注的是构建对成像条件不变的描述子，这样就可以减少需要的模型数量。

### 3.1 Model Selection

Many Machine Learning techniques have been developed to reduce the number of models in a classification algorithm. One of the simplest examples (Duda et al., 2001), for a nearest neighbour classifier, is to remove each model for which all the neighbouring models belong to the same class. This can be done safely as these models make no contribution in determining the classification boundaries (as can be seen from the Voronoi tessellation). However, in practise this has often been found not to lead to a substantial reduction in the number of models. It is also possible to reduce the number of models by completely switching classifiers. For instance, Support Vector Machines (Cristianini and Shawe-Taylor, 2000; Hayman et al., 2004; Kim et al., 2002; Scholkopf and Smola, 2002), and perhaps more appropriately Relevance Vector Machines (Tipping, 2001), are both capable of reducing the number of models while providing good generalisation.

很多机器学习技术可以用于减少分类算法中需要的模型数量。对一个最近邻分类器，一个最简单的例子是，去掉属于同一类别的所有相邻模型。这样做很安全，因为这些模型对确定分类边界没有任何贡献。但是，在实践中，这通常会极大的减少模型数量。也可能完全改变分类器，带来模型数量的减少。比如，SVM，更合适的是RVM，都可以降低模型数量的同时保证很好的泛化。

In this subsection, we investigate two schemes for model reduction in a nearest neighbour classifier framework. Both these schemes take into account the inter and intra class image variation. Two types of experiments are performed for either method. In the first, models are selected only from the training set and classification results reported only on the test set. In the second type, the classification experiments are modified slightly so as to maximise the total number of images classified. Following (Cula and Dana, 2004), if only M models per texture are used for training, then the rest of the 46 − M training images are added to the 46 test images so that a total of 92−M images are classified per material. For example, when classifying 61 textures, if only M = 10 models are used on average then a total of 82 images per texture are classified giving a total of 82 × 61 = 5002 test images. This is done so as to be able to make accurate comparisons with (Cula and Dana, 2004). The texton dictionary used in all experiments is the same as the one in the previous section and has 200 textons.

在这个小节中，我们研究最近邻分类器框架下的两种模型缩减的方案。这两种方案都考虑了类内和类间图像变化。对每种方法进行两类试验。第一类，只从训练集中选择模型，只对测试集给出分类结果。在第二类中，分类试验进行了略微修改，对分类图像的总数量达到最大化。按照Cula and Dana，如果每类纹理只使用M个模型进行训练，那么对剩下的46-M幅训练图像就加入到46幅测试图像中，这样对每种材质共对92-M幅图像进行分类。比如，当对61类纹理进行分类时，如果平均只用M=10个模型，那么每类总计82幅图像进行分类，测试图像总计就有82 × 61 = 5002幅。这样就可以很准确的与Cula and Dana进行比较。所有试验中所用的纹理基元词典与前一节相同，有200个纹理基元。

#### 3.1.1. K-Medoid algorithm

Each histogram may be thought of as a point in R^N, where N is the number of bins in the histogram, so that the models for a particular texture class simply consist of a set of points in R^N space. Given a distance function between two points, in our case χ2, the set of points corresponding to a texture’s models may be clustered into representative centres, and the set of points then replaced by the centres. There are many choices that can be made at this point, for example whether to cluster only within a texture class, or to take into account other classes when clustering, or to cluster the histograms of all the training images irrespective of class (i.e. all the training images taken from all the texture classes). Here we only investigate the last case.

每个直方图都可以认为是R^N中的一个点，其中N是直方图中bins的数量，这样特定纹理类别的模型就是R^N空间中的点集。给定两个点之间的距离函数，在我们情况中χ2，对应一个纹理模型的点集可以聚类成有代表性的中心，这样点集就可以被中心所替代。在这个点上可以有很多选项，比如，是在一个纹理类别中进行聚类，或在聚类时考虑到其他类别，或不分类别的对所有训练图像的直方图进行聚类。这里我们只研究最后一种情况。

The clustering is implemented using the K-Medoid algorithm. This is a standard clustering algorithm (Kaufman and Rousseeuw, 1990) where the update rule always moves the cluster centre to the nearest data point in the cluster, but does not merge the points as in the case of the more popular K-Means. The K-Means algorithm can only be applied to points within a texture class. It can not be applied across classes as it merges data points and thus the resultant cluster centres can not be identified uniquely with individual textures. This is not a problem with the K-Medoid algorithm as the cluster centres are always data points themselves. Table II lists the results of classifying 20 textures using the four different filter banks with K = 60, 120 and 180, resulting in an average of 3, 6 and 9 models per texture.

聚类是用K-Medoid算法实现的。这是一种标准的聚类算法，其更新准则是将聚类中心移向聚类中最近的数据点，但并不合并这个数据点，在更流行的K-Means中则是合并这个数据点。K-Means算法只可以应用于一个纹理类别中的点。不能跨类别应用，因为会合并数据点，因为得到的聚类中心对单个纹理不能唯一识别。在K-Medoid算法中这就不是一个问题，因为聚类中心永远是数据点。表2列出了分类20类纹理的结果，使用了4种不同的滤波器组，K=60，120和180，最后每种纹理平均得到了3，6，9个模型。

For MR8, the classification rate with 9 K-Medoid selected models per texture is almost as good as the 97.83% obtained using all 46 models (see column 1 in table I). In the first type of experiment (table IIa) an accuracy of 93.55% is achieved while the second type (table IIb) obtains an accuracy of 93.59% while classifying many more test images. However, clustering does have the disadvantage that very similar models are aggregated into a single cluster even if they come from different texture classes. Similarly, many clusters centres, rather than just one, might be used to represent models which are spread apart even if they belong to the same texture class. Both these shortcomings can be overcome by using a greedy algorithm which prunes the list of models on the basis of classification boundaries.

对于MR8，每种纹理用9个K-Medoid选择的模型得到的分类准确率，几乎与使用所有46个模型得到的准确率97.83%一样（见表1的列1）。在第一类试验中（表2a），获得了93.55%的准确率，而第二类试验（表2b）获得了93.59%的准确率，这里分类了更多的测试图像。但是，聚类确实有一个劣势，即非常类似的模型会聚积成一个聚类，即使是属于不同的纹理类别。类似的，会用很多聚类中心来表示模型（而不是一个），很分散，即使它们都属于同一纹理类别。这些缺陷可以通过使用贪婪算法来克服，基于分类边界来对模型列表进行修剪。

#### 3.1.2. Greedy algorithm

An alternative to the K-Medoid clustering algorithm is a greedy algorithm, based on the post-processing step of the reduced nearest neighbour rule (Gates, 1972; Toussaint, 2002), designed to maximise the classification accuracy while minimising the number of models used. The algorithm is initialised by setting the number of models equal to the number of training images available. Then, at each iteration step, one model is discarded. This model is chosen to be the one for which the classification accuracy decreases the least when it is dropped. This iteration is repeated until no more models are left. Note that while the algorithm is constrained to select models only from the training set, classification performance is being assessed on the test set. This emulates the setup of (Cula and Dana, 2004) where the model reduction algorithm has access to both training and test images for each texture class and should therefore facilitate a faithful comparison with their work. However, it must be emphasised that in real world classification, the test set is not available for inspection to the training set and in such situations it is preferable to subdivide the training set further into model learning and validation sets.

K-Medoid聚类算法的一个替代是贪婪算法，是基于缩减版最近邻原则的后处理步骤的，设计用于最大化分类准确率，同时使所用的模型数量最小化。算法的初始化，是通过设置模型数量为可用的训练图像数量。然后，在每个迭代步骤中，丢弃一个模型。模型的选择，是丢弃后准确率下降最小的那个。这个迭代重复进行，直到不剩下几个模型。注意，算法局限于只从训练集选择模型，但性能评估是在测试集上的。这模仿的是Cula and Dana的试验设置，那里模型缩减算法可以选择每个纹理类别的训练图像和测试图像，因此可以和他们的工作有一个公平的比较。但是，必须强调，在真实世界的分类中，测试集不能用于检视训练集，因此在这种情况下，训练集最好进一步分成模型学习和验证集。

Table III lists the results of classifying 20 textures using the four different filter banks. It is very interesting to note that the classification accuracy obtained using 9 models can actually be better than that obtained using all 46 models (see column 1 in table I). In table IIIa, this implies that using a fewer number of models can improve performance and that the greedy algorithm is good at rejecting noisy or outlier models. In table IIIb, this also indicates that most of the training images being added to the test set are being classified correctly.

表3给出了使用四种不同的滤波器组对20类纹理进行分类的结果。非常有趣的是，使用9个模型得到的分类准确率实际上可以比使用所有46个模型的还要高。在表3a中，这意味着使用较少数量的模型可以改进性能，贪婪算法擅长于去掉含噪和外围的模型。在表3b中，这也说明，多数训练集图像可以加入到测试集中去，也可以得到正确分类。

Figure 13 shows the resultant classification accuracy versus number of models for the four filter banks when classifying 20, 40 and 61 textures. For MR8, a very respectable classification rate of over 97% correct is achieved using on an average only 9 models per texture, even when all 61 classes are included. Figure 14 shows the 9 textures that were assigned the most models as well as the 9 textures that were assigned the least models while classifying all 61 textures.

图13给出了得到的分类准确率与模型数量在四种滤波器组下的关系，包含分类20类、40类和61类三种情况。对于MR8，当每类纹理平均使用9个模型时，在61种纹理都进行分类时，可以得到非常好的97%的准确率。图14展示了在分类61种纹理时，指定了最多数量模型的9种纹理，和指定数量最少模型的9种纹理。

#### 3.1.3. Discussion

The results for both the K-Medoid and the Greedy algorithms, while using the MR8 filter bank, compare very favourably with those reported in (Cula and Dana, 2004) and (Leung and Malik, 2001). In the case where there are 20 textures to be classified, the K-Medoid algorithm has a classification accuracy of 93.59% while using, on average, 9 models per texture class while the Greedy algorithm achieves an accuracy of 98.80%. In contrast, for the same 20 textures, Cula and Dana obtain a classification rate of 71% while using 8 models per texture class (by taking the most significant image from each texture and using a manifold merging procedure). This increases marginally to 72% if 11 models are used per texture (see figure 19b and table 4 in (Cula and Dana, 2004)). Note that the comparison is not exact since we classify only 92 − 9 = 83 images per texture class as compared to the 156 − {8, 11} classified by Cula and Dana. Hence, (Cula and Dana, 2004) classify many more images, some of which might be quite hard to categorise correctly because of the oblique viewing angle.

使用MR8滤波器组时，K-Medoid和贪婪算法的结果，与Cula and Dana和Leung and Malik的结果相比都非常好。在有20类纹理需要分类的情况下，K-Medoid算法的分类准确率为93.59%，平均每个纹理类别需要9个模型，而贪婪算法的准确率为98.80%。对比起来，对于同样的20类纹理，Cula and Dana的分类准确率为71%，平均每个纹理类别使用了8个模型（从每种纹理中选取最显著的图像，使用了一个manifold合并过程）。如果每种纹理使用11个模型，则准确率稍微提升到了72%。注意这个比较并不是很准确的，因为每个纹理类别我们只分类了92-8=83幅图像，，而Cula and Dana则分类了156 − {8, 11}幅图像。因此，Cula and Dana分类的图像数量更多，一些分类起来很困难，因为视角非常特别。

Nevertheless, there is a significant level of difference between the performance of the K-Medoid and the Greedy algorithms on one hand and the manifold method of (Cula and Dana, 2004) on the other. This is primarily due to the fact that the methods developed here take into account both the inter class variation, as well as intra class variation. The models that Cula and Dana learn are general models and not geared specifically towards classification. They ignore the inter class variability between textures and concentrate only on the intra class variability. The models for a texture are selected by first projecting all the training and test images into a low dimensional space using PCA. A manifold is fitted to these projected points, and then reduced by systematically discarding those points which least affect the “shape” of the manifold. The points which are left in the end correspond to the model images that define the texture. Since the models for a texture are chosen in isolation from the other textures, their algorithm ignores the inter class variation between textures.

尽管如此，K-Medoid和贪婪算法的性能，和Cula and Dana的manifold方法性能相比，还是有非常大的差异的。这主要是因为，这里提出的方法，考虑了类间的差异，以及类内的差异。Cula and Dana学习的是通用模型，并不是专用于分类的。他们忽略了纹理间的类间变化，只关注了类内变化。一种纹理的模型的选择，首先将所有训练和测试图像用PCA投影到低维空间上。将这些投影的点拟合成一个manifold，然后将形成这个manifold影响最小的点丢弃掉。留下来的点对应的模型图像，定义了这个纹理。由于一个纹理的模型的选择与其他纹理是无关的，其算法忽略了纹理间的类间变化。

For 40 textures, Leung and Malik report an accuracy rate of 95.6% for classifying multiple (20) images using, in effect, 20 models per texture class. For single image classification under known imaging conditions, using 4 models per texture class results in a drop in the accuracy rate to 87% (as computed for 5 test images per texture). The MR8 filter bank achieves 95.6% accuracy on the same textures using only 5.9 models per texture, and furthermore achieves 98.06% accuracy using, on average, 8.25 models per texture.

对于40种纹理，Leung and Malik给出的准确率是95.6%，每种纹理使用了20个模型，分类多个图像(20)。对于已知成像条件的单幅图像分类，每种纹理使用4个模型，准确率下降到87%（每个纹理5个测试图像）。MR8滤波器组在同样的纹理中获得了95.6%的准确率，每类纹理只使用了5.9个模型，如果每类纹理平均使用8.25个模型，则准确率会进一步提升到98.06%。

### 3.2 Pose Normalization

In this subsection we discuss some geometric approaches to model reduction. In theory, these approaches are valid only in the absence of 3D effects, i.e. for planar textures where illumination does not play a major role, and where a 3D rotation and translation of the texture is equivalent to an affine transformation of its image. However, in practise, these methods are quite robust.

本节中，我们讨论模型缩减的一些几何方法。理论上，这些方法只在有3D效果的时候才可用，即，对于平面纹理，光照的角色并不是那么重要，纹理的3D旋转和平移，等价于图像的一个仿射变换。但是，实践中，这些方法是很稳健的。

The fundamental idea is to incorporate some level of geometric invariance into a model. This will ultimately allow us to be invariant to changes in the camera viewpoint and thereby reduce the number of models required to characterise a texture. The use of rotationally invariant filters is already a first step in this direction but the problem of scale still needs to be resolved (we are ignoring perspective effects for the moment). One approach could be to extend the MR sets to take the maximum response not only over all orientations but over all scales or over all affine transformations of the basic filter, but that is not investigated here. Instead we investigate the method of pose normalisation.

基本的思想是，给模型中增加一定程度的几何不变性。这会使得对相机视角的变化具有不变性，因此降低描述一个纹理所需要的模型数量。旋转不变滤波器的使用，已经事这个方向的第一步，但尺度的问题仍然需要解决（我们现在忽略视角的效果）。一种方法可以是，将MR集拓展到不仅是所有方向的最大响应，而且还是所有尺度或所有基本滤波器的仿射变换的最大响应，但这里并没有进行研究。这里我们研究的是，姿态归一化的方法。

In (Schaffalitzky and Zisserman, 2001) it was demonstrated that, provided a texture has sufficient directional variation, it can be pose normalised by maximising the isotropy of its gradient second moment matrix (a method originally suggested in (Lindeberg and G˚arding, 1994)). The method is applicable in the absence of 3D texture effects.

在文献中证明了，如果一个纹理的足够的方向变化，可以通过最大化其梯度二阶动量矩阵的isotropy进行姿态归一化。这种可以在没有3D纹理效果的情况下进行应用。

Here we investigate if this normalisation can be used to at least reduce the effects of changing viewpoint, and hence provide tighter clusters of the filter responses, or better still reduce the number of models needed to account for viewpoint change.

这里我们研究了，这种归一化是否可以用于降低变换视角的效果，因此得到滤波器响应的更紧凑的聚类，或可以减少用于应对视角变化的模型的数量。

In detail, if the normalisation is successful, then for moderate changes in the viewing angle, two such “pose normalised” images of the same texture should differ from each other by only a similarity transformation. If there are no major 3D scale effects, the responses of a rotationally invariant filter bank (MR or S) to these images should be much the same. A preliminary investigation shows that this is indeed the case for suitable textures.

在细节中，如果归一化是成功的，那么对于视角的一般变化，同一纹理的两个这种姿态归一化过的图像，其之间的差别应当仅仅就是一个相似性变换。如果没有明显的3D尺度效果，一个旋转不变的滤波器组(MR or S)对这些图像的响应，就应当是一样的。初步的研究表明，对于合适的纹理来说，确实就是这种情况。

Figure 15 shows results for two textures - Plaster A and Rough Plastic. Twelve images of each texture are selected to have similar photometric appearance (i.e. constant illumination conditions), but monotonically varying viewing angle. The graph shows the χ2 distance between the texton histogram of one of the images (selected as the model image) and the rest, before and after pose normalisation. As can be seen, the χ2 distance is reduced for the pose normalised images. This in turn translates to better classification as well. On experiments on 4 textures, using the same 12 image set and one model per texture, the classification rate increased from 81.81% before pose normalization to 93.18% afterwards.

图15给出了Plaster A和Rough Plastic两种纹理的结果。每种纹理选择了12幅图像，有着相似的光学效果（即，光照条件一样），但视角单调变化。图中给出的是一幅图像（选为模型图像）的纹理基元直方图和其余图像的χ2距离，对应两种情况即姿态归一化前和后。可以看出，对于姿态归一化过后的图像，χ2距离降低了，这样就会得到更好的分类效果。在对4种纹理的试验中，使用同样的12幅图像的集合，每种纹理一个模型，分类准确率在姿态归一化前是81.81%，归一化后提升到93.18%。

One drawback of this method is that the proposed normalisation is global rather than local. Not only would local normalisation be more robust but it would also allow the method to be extended to textures which are not globally planar but which can be approximated as being locally planar. Realising this, (Lazebnik et al., 2003b; Lazebnik et al., 2003a) proposed alternative methods of generating local, affine invariant, texture features. In their framework, certain interest regions are first detected in texture images using the Laplacian and Harris detectors. Each of these regions is then scale and pose normalised locally. Spin images are then used instead of filter banks to generate rotationally invariant features for each region. Their results are very encouraging though no direct comparison is possible as their experiments are not carried out on the CUReT database. One point of concern however, is the reliance on the detection of blob like interest regions as there exist many textures which do not exhibit such markings.

这种方法的一个缺陷是，提出的归一化是全局的，而不是局部的。局部归一化会更加稳健，而且方法可以拓展到不是全局平面的纹理，即近似为局部平面的纹理。意识到这个，有文献提出了其他方法生成局部，仿射不变的纹理特征。在他们的框架中，特定的，首先在纹理图像中检测出特定的感兴趣区域，使用的是Laplacian和Harris检测器。每个这些区域然后进行局部的缩放和姿态归一化。然后用spin图像，而不是滤波器组，来对每个区域生成旋转不变的特征。其结果是非常鼓舞人心的，但不能进行直接的比较，因为其试验不是在CURet数据集上进行的。但是，一个考虑的点是，依赖于检测blob类的感兴趣区域，但很多纹理并没有这种特征。

## 4. Generalisations

In this section, we investigate the various generalisations and modifications that can be made to the basic classification algorithm. In subsection 4.1, we study the effect of some of the more important parameters on our classifier. In particular, the effect of the choice of texton dictionary and training images is investigated. We also look at how scaling the images impacts performance. Finally, the issue of whether information is lost by using just the first order statistics of rotationally invariant filter responses is discussed in section 4.2. A method for reliably measuring relative orientation texton co-occurrence is presented in order to incorporate second order statistics into the classification scheme.

本节中，我们研究对基本分类算法的各种泛化和改动。在4.1节中，我们研究分类器中一些重要参数的效果。特别是，研究了选择纹理基元字典和训练图像的效果。我们还研究了图像的缩放怎样影响性能。最后，在4.2节中研究了，只使用旋转不变滤波器响应的一阶统计量，是否损失了信息的问题。给出了一种可以可靠的度量相对方向纹理基元共现的方法，这样就可以纳入二阶统计信息到分类方案中。

### 4.1 Algorithm parameter variations and the issue of scale

In this subsection, various parameters of the algorithm are varied and the effect on the classification performance determined. We first calculate a benchmark classification rate and then vary the images in the training set and also the size of the texton dictionary to see how performance is affected.

在这个小节中，算法的各种参数进行变化，观察了分类性能的变化。我们首先计算了一个基准测试分类准确率，然后对训练集的图像和纹理基元字典的大小进行了变化，观察了性能受到了怎样的影响。

For the benchmark case, the texton dictionary is built by learning 10 textons from each of the 61 textures (using the procedure described in subsection 2.3) to have a total of 610 textons. The 46 training images per texture from which the models will be generated are chosen by selecting every alternate image from the set of 92 available. Under these conditions, the MR8 filter bank achieves a classification accuracy of 96.93% using 46 models per texture for all 61 textures. On running the greedy algorithm the classification accuracy increases to 98.3% using, on average, only 8 models per texture. This defines the benchmark rate.

对于基准测试的情况，纹理基元的字典的构建，是对61种纹理每个学习10个纹理基元（学习过程如2.3节所述），所以总计有610个纹理基元。每种纹理选择46幅训练图像，从中生成模型，这些图像的选择是从92幅可用的图像中每隔一幅选一个。在这些条件下，MR8滤波器组对所有61种纹理，每种纹理选择46个模型，得到的分类准确率为96.93%。在运行贪婪算法后，准确率提升到98.3%，每种纹理平均只选择8个模型。这定义了基准测试准确率。

We now investigate the effect of choice of textons on the classification performance. First we reduce the number of textons by learning 10 textons from only 31 randomly chosen textures to get a dictionary of 310 textons, and then repeat the experiment of section 2. The classification rate decreased only slightly from the benchmark to 98.19%.

我们现在研究纹理基元的选择对分类准确率的影响。首先我们我们降低纹理基元的数量，从31个随机选择的纹理中每种纹理学习10个纹理基元，得到了310个纹理基元的字典，然后重复第2节的试验。分类准确率只有略微的下降，为98.19%。

The number of textons in the dictionary can be further reduced by merging textons which lie very close to each other in filter response space. The texton dictionary can be pruned down from 310 to 100 by selecting 80 of the most distinct textons (i.e. those textons that didn’t have any other textons lying close by) and then running K-Means, with K = 20, on the rest. This procedure entailed another slight decrease in the classification accuracy to 97.38%. These results indicate that the pruned dictionaries are still universal (Leung and Malik, 2001), i.e. texton primitives learnt from some randomly chosen texture classes can be used to successfully characterise other classes as well.

字典中纹理基元的数量可以进一步降低，将滤波器响应中非常接近的纹理基元合并成一个。纹理基元字典可以从310修剪到100，即选择80个最明显的纹理基元（附近没有其他纹理基元的），然后对其他纹理基元采用K-Means算法得到K=20个纹理基元。这个过程使得分类准确率进一步略微下降了一点，到97.38%。结果表明，修剪的字典仍然是非常通用的，即从随机选择的纹理类别中学习到的纹理基元原语，也可以在其他类别中很成功的应用。

We now increase the size of the texton dictionary to see if classification improves accordingly. Table IV gives a summary of the results. The best performance is obtained with a dictionary of 2440 textons when the classification accuracy is 97.43% using 46 models per texture. On running the greedy algorithm, the number of models used is reduced to, on average, 7.14 per texture. If the unused training images are added to the test set, the classification rate improves to 98.61%.

我们现在增加纹理基元字典的大小，看分类准确率是否也相应的提高。表4给出了结果的总结。最好的结果，是纹理基元词典的大小为2440时，分类准确率是97.43%，每种纹理使用46个模型。在运行贪婪算法后，使用的模型数量降低到平均每种纹理7.14个。如果没使用的训练图像增加到测试集中，分类准确率提升到98.61%。

Essentially we are comparing different representations of the joint probability distribution of filter responses in terms of their classification performance. A set of textons can be thought of as adaptively partitioning the space of filter responses into bins (determined by the Voronoi diagram) and a histogram of texton frequencies can be equated to a probability distribution over filter responses (Varma and Zisserman, 2005). In such a situation, the number of bins should not be too few otherwise the approximation to the true PDF will be poor nor should there be too many bins so as to prevent over-fitting.

其实我们比较的是滤波器响应的联合概率分布的不同表示，表现为其分类性能。一个纹理基元集合，可以认为是将滤波器响应空间自适应的分割成了bins，这样纹理基元频率的直方图，就等价于滤波器响应的概率分布。在这种情况下，bins的数量不应当太小，否则对真实PDF的近似就会很差，bins的数量也不应当很多，可以防止过拟合。

As can be seen in table IV there is a point beyond which increasing the number of textons actually decreases performance as the data is now being over fitted. This can be used to automatically select the appropriate number of textons for a given problem by partitioning the data into a training and validation set and then choosing the texton dictionary which maximises classification on the validation set.

从表4中可以看出，纹理基元数量超过一定点之后，实际上会降低分类性能，因为这时候数据就过拟合了。这可以用于对给定问题的纹理基元数量的自动选择，将数据分成训练集和验证集，然后选择在验证集上最大化分类准确率的纹理基元。

We now turn to the choice of training images. It could be argued that the results presented here are biased as the training set has been chosen by including every alternate image from the set of 92 available per texture. We address this issue by repeating the classification experiment but with the training images chosen randomly. The dictionary of 2440 textons generated previously is used and the experiment repeated 50,000 times. Figure 16 shows the distribution of classification results when 46 images were chosen randomly from every texture class to form the training set while table V provides a summary of the results for varying sizes of the training set. The mean classification accuracy when the 46 models were chosen randomly was 97.28% which is very similar to the 97.43% obtained when the 46 images were chosen by including every alternate image. This shows that our experimental setup is not biased and that we are not over fitting to the data.

我们现在转到训练图像的选择上。可以认为，这里得到的结果是有偏置的，因为训练的选择是将每种纹理中可用的92幅图像每隔一个选一个。我们对这个问题的处理方法是，重复分类试验，但训练图像随机选择。我们使用生成2440纹理基元字典，试验重复了50000次。图16给出了每个纹理类别随机选择46幅图像形成训练集的分类结果分布，表5给出了训练集大小变化的结果总结。当46个模型是随机选择的，平均分类准确率是97.28%，这与前面的46幅图像是隔一个选一个得到的结果97.43%是非常类似的。这表明，我们的试验设置并没有偏置，对数据并没有过拟合。

In summary, the best classification rate achieved, while classifying all 61 textures, was 98.61% obtained when 2440 textons were used and the worst rate was 97.38% when only 100 textons were used. These results are listed in table VI. We can therefore conclude that our algorithm is robust and relatively insensitive to the choice of training image set and texton vocabulary with the classification rate not being affected much by changes in these parameters.

总结起来，在对所有61类纹理进行分类时，得到的最佳分类准确率是98.61%，使用2440个纹理基元，最差是97.38%，这时只使用了100个纹理基元。结果如表6所示。因此我们得出结论，我们的算法是稳健的，对训练图像集和纹理基元字典的选择并没有那么敏感，这些参数的变化对分类准确率的影响不算很大。

Finally, a word about scale. It may be of concern that the MR4 filter bank does not have filters at multiple scales and hence will be unable to handle scale changes successfully. To test this, 25 images from 14 texture classes were artificially scaled, both up and down, by a factor of 3. The classification experiment was repeated using the original, normal sized, filter banks and texton dictionaries. We found that as long as models from the scaled images were included as part of the texture class definition, classification accuracy was virtually unaffected and classification rates of over 97% were achieved. However, if the choice of models was restricted to those drawn from the original sized images, then the classification rate dropped to 17%. It is evident from this that filter bank and texton vocabulary are sufficient, and it is the model that must be extended (see figure 17).

最后，关于尺度。MR4滤波器组没有多尺度上的滤波器，因此不能成功的处理尺度变化。为验证，对14个纹理类别中的25幅图像进行人工缩放，包括缩小和放大，系数为3。分类试验用原始的正常大小的滤波器组和纹理基元字典进行重复。我们发现，只要缩放图像中的模型只要是纹理类别定义的一部分，分类准确率就基本没有影响，可以得到超过97%的分类准确率。但是，如果模型的原则还是那些原始大小的图像，那么分类准确率就会下降到17%。很明显，滤波器组和纹理基元字典是很充分的，是模型必须要进行拓展（见图17）。

### 4.2. Orientation co-occurrence

The classification scheme, up to this stage, has only used information about first order texton statistics (i.e. their frequency and not a measure of their co-occurrence). However, recent research into texture driven content-based image retrieval (Schmid, 2001) has shown that a hierarchical system which uses co-occurrence of textons over a spatial neighbourhood can lead to good results. Therefore, in this subsection, we investigate whether incorporating such second order statistics can improve classification performance on the CUReT database.

到这个时候，分类方案目前只使用了纹理基元的一阶统计量（即，其频率，而没有其共现）。但是，最近关于纹理驱动的基于内容的图像索引的研究表明，使用在空间邻域中使用纹理基元共现的层次系统可以得到很好的结果。因此，在这一节中，我们研究一下，纳入二阶统计量，是否可以在CUReT数据集上改进分类准确率。

As was seen in the previous subsection, classification on the basis of texton frequency information alone is already very good and rates of over 97% can be achieved. What is also interesting is that, of the images that were misclassified, the correct texture class was ranked within the top 5 most of the times. Figure 18 shows how similar one of the misclassified novel images is to both the top ranked, but incorrect, texture model and the second ranked, but correct, model. Since the MR8 filter bank is rotationally invariant, there is the possibility that some of these misclassifications are due to two different texture classes, which are not rotationally related, being mapped to the same texton frequency distribution. Therefore, we focus on the question of whether incorporating second order texton statistics, in the form of co-occurrence of angles, can improve classification (though the method developed here is general and can also be applied to spatial co-occurrence).

前一小节中可以看到，基于纹理基元频率信息的分类效果已经非常好了，可以得到超过97%的准确率。有趣的是，那些误分类的图像，正确的纹理类别通常排在前5的类别之中。图18展示了一个误分类的图像，与排名第一的不正确的纹理模型，和排名第二的正确的模型，是多么的相似。由于MR8滤波器组是旋转不变的，有可能这种误分类是因为两种不同的纹理类别，并不是旋转相关的，但是映射到了相同的纹理基元频率分布。因此，我们关注的是，将二阶纹理基元统计量纳入进来，即角度共现，是否可以改进分类准确率（这里提出的方法是通用的，可以用于空间共现中）。

#### 4.2.1. Reliably measuring a relative orientation co-occurrence statistic

Given a texton in an image labelling, the objective is to measure the relative angle of occurrence of surrounding textons, that lie within a circular neighbourhood, with respect to the given texton. Certain difficulties have to be overcome in order to reliably measure this relative angle co-occurrence. Firstly, the angles of occurrence of the textons have to be measured robustly. Conventionally, working in a match filter paradigm, the orientation of a feature (such as an edge or a bar) is determined to be the angle of maximum response of a filter designed to match that feature. However, features can occur at multiple angles at the same point and, as such, it is difficult to assign them a particular orientation (see figure 19). For instance, an edge filter will have a maximal response at two orientations when matching a corner and choosing one edge orientation over the other will lead to instabilities. Note that these instabilities do not affect the MR representation because only the value of the response (not its angle) is significant — if the same value occurs at two orientations the orientation corresponding to the maximum response is unstable, but the maximum response is not. Here we use the orientated filter (of MR8) that has the maximum response to determine the orientation.

在图像标签中给定一个纹理基元，其目标是周围纹理基元的相对角度共现，在一个环形邻域对给定纹理基元的。要可靠的度量这个相对角度共现，必须要克服一些困难。第一，纹理基元的共现角度必须稳健的度量。传统上是用匹配滤波器的方案进行的，一个特征（比如边缘或bar）的角度，是设计用于匹配这个特征的滤波器的最大响应角度。但是，特征在相同的点上可以出现多次，这样，就很难给其指定一个特定方向（见图19）。比如，边缘滤波器在对角点进行匹配时，会在两个方向上有最大响应，选择一个方向而放弃另一个，就会导致不稳定。注意，这种不稳定性不会影响MR表示，因为只有响应的值（而并不是角度）是明显的，如果相同的值在两个方向出现，那么对应最大响应的方向是不稳定的，但最大响应则不是不稳定的。这里我们使用MR8的有最大响应的有向滤波器来确定方向。

Returning to relative orientation, a robust representation can be obtained if the magnitude of the filter response at each angle (normalised so that the sum of magnitudes squared over all angles is unity) is treated as a confidence measure in the feature occurring at that orientation. Thus, in our case, this normalised magnitude vector will be a 6 vector representing the confidence that the given feature occurs at the 6 angles corresponding to the orientations present in the MR8 filter bank (though a richer representation can be obtained using approximated steerable kernels and interpolation (Perona, 1992)). The relative angles between two features, which is invariant to rotation, can now be calculated by computing the cross-correlation between their normalised magnitude vectors. Given a central texton, we can compute the frequency with which other textons occur at various relative angles to it by forming the sum of the cross-correlations between the normalised magnitude vectors of the central texton and the surrounding textons. Essentially, this is computing (via soft binning) the count of how many times a neighbouring texton occurs at a given angle relative to the central texton. To maintain rotational invariance, the surrounding textons come from a circular neighbourhood with a predefined radius, centred around the given texton.

回来相对方向的问题上，如果滤波器响应在每个角度的幅度认为是特征在这个方向发生的置信度度量，就可以得到一个稳健的表示。因此，在我们的情况中，这种归一化的幅度相邻就是6个矢量，表示给定特征在6个角度发生的置信度，这些方向对应着MR8滤波器组中的方向。两种特征的相对角度对于旋转是不变的，可以通过计算其归一化幅度向量的交叉相关得到。给定一个中央纹理基元，我们可以计算其他纹理基元在各种相对角度发生的频率。。。。

#### 4.2.2. Extending the classification algorithm

Now that a co-occurrence 6-vector can be associated with every texton in an image labelling, the classification algorithm can be extended to use the joint distribution of filter responses and co-occurrence vectors. Just as filter responses were clustered into filter response textons in section 2.3, co-occurrence vectors can be clustered to find exemplars as well, and a dictionary of co-occurrence vector textons can be formed. Textons from this dictionary can be used to label the co-occurrence vectors for a given image. The model for a training image then becomes the joint histogram of the frequency of occurrence of filter response textons and co-occurrence vector textons. Thus, a model is an K_fr×K_cv matrix M where K_fr is the number of filter response textons and K_cv is the number of co-occurrence vector textons. Each entry M_ij in this matrix represents the probability of filter response texton K_fr_i and orientation co-occurrence texton K_cv_j occurring together in the training image. This is somewhat similar to the co-occurrence representation of (Schmid, 2001). To classify a novel image, its joint histogram is built and is then compared to all the models using χ2 over all elements of the M matrix. Thus, the essence of the classifier remains the same, the only extension is that joint distribution of filter response and cooccurrence textons are used rather than just the histogram of filter response textons. Hence, we get to add extra information and yet retain all the benefits of our existing classification scheme.

现在在一幅图像中，对每个纹理基元，可以关联一个共现6矢量，分类算法可以利用滤波器响应的联合分布和共现矢量。就像在2.3节中，滤波器响应聚类成滤波器响应纹理基元，共现向量也可以进行聚类，以找到范本，可以形成共现向量纹理基元的字典。从这个字典得到的纹理基元可以用于对给定图像标记其共现向量。一个训练图像的模型，就变成了滤波器响应纹理基元和共现矢量纹理基元的联合直方图。因此，一个模型就是一个K_fr×K_cv矩阵M，其中K_fr是滤波器响应纹理基元的数量，K_cv是共现矢量纹理基元的数量。矩阵中的每个入口M_ij表示滤波器响应纹理基元K_fr_i和方向共现纹理基元K_cv_j在训练图像中共同发生的概率。这与Schmid的共现表示有些类似。为分类一个新图像，其联合直方图需要进行计算，然后与所有模型进行比较。因此，分类器基本还是保持一样的，唯一的拓展是使用了滤波器响应和共现纹理基元的联合分布。因此，我们加入了额外的信息，而且保持了我们现有分类方案的所有好处。

#### 4.2.3. Experimental Setup and Results

The orientation co-occurrence texton dictionary is created by clustering the co-occurrence vectors (calculated for a particular radius of the circular neighbourhood) from the same set of 13 training images per texture that were used to generate the filter response texton dictionary. The filter responses and co-occurrence vectors of the training images are then labelled using the two texton dictionaries. Finally, the models are built by forming the frequencies, in the K_fr × K_cv texton space, of the joint occurrence of the filter response textons and the orientation co-occurrence textons.

方向共现纹理基元字典的创建，是对共现矢量进行聚类，每类纹理采用了13幅训练图像，也用于生成滤波器响应纹理基元字典。训练图形的滤波器响应和共现矢量然后用两个纹理基元字典进行标记。最后，模型通过形成频率来构建，在K_fr × K_cv纹理基元空间，是滤波器响应纹理基元和方向共现纹理基元的联合发生的情况。

Obviously, the choice of K_fr and K_cv is important as K_fr × K_cv equals the number of bins and therefore determines how accurately the joint PDF is approximated. However, we cannot choose K_fr = 610 as had been done previously, because the number of bins becomes too large and we start over-fitting the data (see table VII (a)-(c)). A lower value, such as K_fr = 30, was found to be more appropriate. Table VII (d)-(f) lists the classification results obtained for various values of the radius when K_cv is also set to 30. The performance, using the joint representation, is better than using just 30 filter response textons or just 30 co-occurrence vector textons. Though it is worse than if 900 filter response textons were used without any co-occurrence. If the radius is kept fixed and K_cv varied then the performance of the joint representation, predictably, first increases, reaches a maximum and then falls (though in no case is it ever able to surpass the performance achieved using an equivalent number of filter response textons alone).

显然，K_fr和K_cv的选择是很重要的，因为K_fr × K_cv等于bins的数量，因此确定了联合PDF近似的准确性。但是，我们不能像之前做的那样选择K_fr = 610，因为bins的数量太大了，会开始对数据过拟合。我们发现小一些的值K_fr = 30是比较合适的。表7d-f列出了分类准确率结果，其中的半径值有几种选择，K_cv值也设为30。使用联合表示的性能，比只使用30个滤波器响应纹理基元，或只使用30个共现矢量纹理基元的要好，但比使用900个滤波器响应纹理基元的要差一些。如果半径保持固定，K_cv变化，联合表示的性能会提升，达到最高，然后掉落。

These results indicate, that at least for this dataset, the density of filter response textons is the best measure of discrimination and that orientation co-occurrence does not help much in classification (similar results were found for spatial co-occurrence as well). They also confirm that rotational invariance is advantageous and that no significant information is being lost in this case by using a rotationally invariant filter bank.

这些结果表明，至少对于这个数据集，滤波器响应纹理基元是区分性的最好度量，方向共现对分类并没有多少帮助（空间共现也是一样的）。这也确认了，旋转不变性是非常好的，使用旋转不变滤波器组并没有明显的信息损失。

## 5. Conclusions

In this paper, we have tackled the problem of texture classification and have demonstrated how single images can be classified using a few models without requiring any information about their imaging conditions. This is a substantial improvement over previous work which required multiple images obtained under known conditions. We have also introduced rotationally invariant, low dimensional, maximum response filter banks which were shown to have superior performance as compared to traditional filters due to enhanced feature detection and clustering. Moreover, we presented two novel methods for reducing the number of models needed to characterise textures and again demonstrated their superiority over existing algorithms. It was also shown that the proposed classification scheme is robust to the choice of training images and texton dictionaries. Finally, we concluded that even though the classifier can be extended by incorporating second order statistics this does not lead to an improvement in the overall classification. This implies that using only the frequency distribution of textons is sufficient and that no significant information is being lost by employing rotationally invariant filter banks for this database.

本文中，我们处理纹理分类的问题，证明了单个图像可以使用几个模型进行分类，不需要其成像条件的信息。与之前的工作相比，这是一个显著的进步。我们还提出了旋转不变的低维最大响应滤波器组，与传统滤波器组相比有更好的性能，因为其特征检测和聚类能力都很强。而且，我们提出了两种新的方法降低需要的模型数量，仍然可以很好的描述纹理，并证明了在现有的算法中性能仍然非常好。还证明了，提出的分类方案对于悬链图像和纹理基元字典的选择是稳健的。最后，我们得出结论，虽然分类器可以拓展，加入二阶统计量，但这对性能改进并没有多少帮助。这意味着，只使用纹理基元的频率分布，就足够了，采用旋转不变滤波器组对这个数据集并没有明显的信息损失。

This research has benefited greatly from the availability of the Columbia-Utrecht database. The CUReT database is a considerable improvement over the previously used Brodatz collection (Brodatz, 1966), though it also has some limitations. Its main advantages are that it has many real world textures photographed under varying image conditions, and the effects of specularities, shadowing and other surface normal variations are evident. The limitations of the CUReT database are mainly in the way the images have been photographed and the choice of textures. For the former, there is no significant scale change for most of the textures and limited in-plane rotation. As regards choice of texture, the most serious drawback is that multiple instances of the same texture are present for only a very few of the materials, so intra-class variation cannot be investigated. Hence, it is difficult to make generalisations.

这个研究受益于CRUeT数据集的可用性。该数据集比之前使用的有很大改进，但仍然有一些局限。其主要的优势是，有很多真实世界的纹理，在不同的图成像条件下得到图像，很多效果是很明显的。其局限是，图像成像的方式和纹理的选择。对于前者，多数纹理并没有明显的尺度变化，而且局限于平面内旋转。至于纹理的选择，最严重的缺点是，只有几种材质有相同纹理的多个样本，所以类内变化不能进行研究。因此，很难进行泛化。

The time is now right for a yet more demanding database which overcomes the above limitations, and also includes non-planar surfaces. 是时候有一个更强大的数据集来克服上述局限，并包含非平面的表面。