# GANs for image-to-image translation on multi-contrast MR images - a comparison of CycleGAN and UNIT

Per Welander et al. Linkoping University, Sweden

## 0. Abstract

In medical imaging, a general problem is that it is costly and time consuming to collect high quality data from healthy and diseased subjects. Generative adversarial networks (GANs) is a deep learning method that has been developed for synthesizing data. GANs can thereby be used to generate more realistic training data, to improve classification performance of machine learning algorithms. Another application of GANs is image-to-image translations, e.g. generating magnetic resonance (MR) images from computed tomography (CT) images, which can be used to obtain multimodal datasets from a single modality. Here, we evaluate two unsupervised GAN models (CycleGAN and UNIT) for image-to-image translation of T1- and T2-weighted MR images, by comparing generated synthetic MR images to ground truth images. We also evaluate two supervised models; a modification of CycleGAN and a pure generator model. A small perceptual study was also performed to evaluate how visually realistic the synthesized images are. It is shown that the implemented GAN models can synthesize visually realistic MR images (incorrectly labeled as real by a human). It is also shown that models producing more visually realistic synthetic images not necessarily have better quantitative error measurements, when compared to ground truth data. Code is available at https://github.com/simontomaskarlsson/GAN-MRI.

在医学成像中有一个共同问题，就是非常昂贵而且耗时，才能得到高质量的健康目标和疾病目标的数据。GANs是一种深度学习方法，已经用于合成数据。GANs因为可以用于生成更真实的训练数据，来改进机器学习算法的分类性能。GANs的另一个应用是图像到图像的转化，如从CT图像生成MRI图像，这可以用于从单模态的数据得到多模态的数据集。这里，我们评估了两类无监督的GAN模型(CycleGAN和UNIT)进行图像到图像的转换，转换成T1加权和T2加权的MR图像，并将生成的合成MR图像与真值图像相比较。我们还评估了两个监督模型；一个CycleGAN的变体和一个纯生成器模型。还进行了一个小型感知研究，以评估合成图像视觉上有多真实。试验表明，实现的GAN模型可以合成视觉上非常真实的MR图像（人会将其错误的标注成真实图像）。试验还表明，实现的GAN模型，产生的视觉上很真实的合成数据，与真实数据相比，不一定在量化误差度量上更好。代码已开源。

## 1. Introduction

Deep learning has been applied in many different research fields to solve complicated problems [1], made possible through parallel computing and big datasets. Acquiring a large annotated medical imaging dataset can be rather challenging for classification problems (e.g. discriminating healthy and diseased subjects), as one training example then corresponds to one subject [2]. Data augmentation, e.g. rotation, cropping and scaling, is normally used to increase the amount of training data, but can only provide limited alternative data. A more advanced data augmentation technique, generative adversarial networks (GANs) [3], uses two competing convolutional neural networks (CNNs); one that generates new samples from noise and one that that discriminates samples as real or synthetic. The most obvious application of a GAN in medical imaging is to generate additional realistic training data, to improve classification performance (see e.g. [4] and [5]). Another application is to use GANs for image-to-image translation, e.g. to generate computed tomography (CT) data from magnetic resonance (MR) images or vice versa. This can for example be very useful for multimodal classification of healthy and diseased subjects, where several types of medical images (e.g. CT and MRI) are combined to improve sensitivity (see e.g. [6] and [7]).

深度学习已经在很多研究领域用于解决复杂的问题，这些也离不开并行计算和大数据。大型标注医学数据集的获得会非常昂贵（对于分类问题，如区分健康目标和疾病目标），因为一个训练样本就对应着一个对象。数据扩充，如旋转，剪切和缩放，常常用于增加训练数据的数量，但只能得到有限的数据。一种更先进的数据扩充技术是GANs，使用两个竞争的CNNs；一个从噪声中生成新样本，另一个判别这个样本是真实的还是合成的。GAN在医学成像中最明显的应用是，生成额外的真实训练数据，以改进分类性能。另一个应用是，使用GANs进行图像到图像的转换，如，从MR图像中生成CT数据，或反之。比如，这可能对健康对象和疾病对象的多模态分类非常有用，这里利用多类别的医学图像（如，CT和MRI）结合起来以改进敏感性。

To use GANs for image-to-image translation in medical imaging is not a new idea. Nie et al. [8] used a GAN to generate CT data from MRI. Yang et al. [9] recently used GANs to improve registration and segmentation of MR images, by generating new data and using multimodal algorithms. Similarly, Dar et al. [10] demonstrate how GANs can be used for generation of a T2-weighted MR image from a T1-weighted image. However, since GANs have only recently been proposed for image-to-image translation, and new GAN models are still being developed, it is not clear what the best GAN model is and how GANs should be evaluated and compared. We therefore present a small comparison for image-to-image translation of T1- and T2-weighted MR images. Compared to previous work [9, 10] which used conditional GANs (cGAN) [11], we show results for our own Keras implementations of CycleGAN [12] and UNIT [13], see https://github.com/simontomaskarlsson/GAN-MRI for code.

使用GANs进行医学图像中的图像到图像的转换，这并不是一个新思想。Nie等[8]使用GAN从MRI生成CT。Yang等[9]最近使用GANs来改进MR图像的配准和分割结果，其方法是生成新数据并使用多模态算法。类似的，Dar等[10]使用GANs从T1加权的图像来生成T2加权的图像。但是，由于GANs提出用于图像到图像的转换时间不长，并不断的提出新的GANs模型，还不是很清楚什么样的GAN模型是最好的，GANs怎样进行评估和对比。因此，我们对T1加权和T2加权的MR图像的图像到图像转换进行了小规模比较。与之前使用cGAN的工作相比，我们给出了自己的Keras实现的CycleGAN和UNIT。

## 2. Method

### 2.1. GAN model selection and implementation

Several different GAN models were investigated in a literature study [12, 13, 14, 15, 16]. Two models stood out among the others in synthesizing realistic images in high resolution; CycleGAN [12] and UNIT [13]. Training of neural networks is commonly supervised, i.e. the training requires corresponding ground truth to each input sample. In image-to-image translation this means that paired images from both source and target domain are needed. To alleviate this constraint, CycleGAN and UNIT can work with unpaired training data.

研究了几种不同的GAN模型。有两类模型非常适合于合成高分辨率真实图像：CycleGAN和UNIT。神经网络训练通常是有监督的，如，训练需要每个输入样本和对应的真实数据。在图像到图像的转换中，这意味着需要源领域和目标领域的成对的图像。为缓解这个约束，CycleGAN和UNIT可以使用不成对训练数据工作。

Two different variants of the CycleGAN model were implemented, CycleGAN_s and CycleGAN. Including the ground truth image in the training should intuitively generate better results, since the model then has more information about how the generated image should appear. To investigate this, CycleGAN_s was implemented and trained supervised, i.e. by adding the mean absolute error (MAE) between output and ground truth data. To investigate how the adversarial and cyclic loss contribute to the model, Generators_s was also implemented. It consists of the generators in CycleGAN and is only trained in a supervised manner with a MAE loss using the ground truth images, it does not include the adversarial or the cyclic loss. A Simple baseline model was also implemented for comparison, it consists of only two convolutional layers.

我们实现了两种不同的CycleGAN变体，CycleGAN_s和CycleGAN。在训练中包含真值图像，从直觉上来说，可以得到更好的结果，因为模型会得到生成的图像应当是什么样子的更多信息。为研究这个，CycleGAN_s是进行的有监督训练，即加入了输出与真值数据的MAE。为研究对抗损失与循环损失对模型贡献如何，还实现了Generators_s。这是由CycleGAN中的生成器组成的，只以一种有监督的方式来训练，使用真值图像计算了MAE，这并没有包括对抗损失或循环损失。我们还实现了一个Simple基准，以进行比较，这个模型只包含2个卷积层。

### 2.2. Evaluation

The dataset used in the evaluation is provided by the Human Connectome project[17,18](https://ida.loni.usc.edu/login.jsp). We used (paired) T1- and T2-weighted images from 1113 subjects (but note that CycleGAN and UNIT can be trained using unpaired data). All the images have been registered to a common template brain, such that they are in the same position and of the same size. We used axial images (only slice 120) from the segmented brains. The data were split into a training set of 900 images in each domain. The remaining 213 images, in each domain, were used for testing. Using an Nvidia 12 GB Titan X GPU, training times for CycleGAN and UNIT were 419 and 877 seconds/epoch, respectively. The models were on average trained using 180 epochs. The generation of synthetic images took 0.0176 and 0.0478 ms per image for CycleGAN and UNIT, respectively.

评估中使用的数据集由Human Connectome project提供。我们使用1113个对象的（成对的）T1-和T2加权图像（注意，CycleGAN和UNIT可以使用不成对数据训练）。所有图像都与一个常见的模板脑进行了配准，这样它们都在同样的位置上，有同样的大小。我们从分割的脑中使用轴向图像（只有120个切片的数据）。数据在每个领域中都分成900幅训练图像，每个领域中剩下的213幅图像，都用于测试。使用Nvidia 12 GB Titan X GPU，CycleGAN和UNIT的训练时间分别为每轮数据419s和877s。模型平均使用180轮数据的训练。合成图像的生成，使用CycleGAN和UNIT分别耗时每幅图像0.0176ms和0.0478ms。

The two GAN models were compared using quantitative and qualitative methods. All quantitative results; MAE, mutual information (MI), and peak signal to noise ratio (PSNR), are based on the test dataset. Since the MR images can naturally differ in intensity, each image is normalized before the calculations by division of the standard deviation and subtraction of the mean value.

使用了定量和定性的方法比较了两个GAN模型。所有量化的结果，如MAE，互信息，PSNR，都是基于测试数据集的。由于MR图像在亮度上有差异是很自然的，每幅图像在计算之前都进行了归一化，减去了均值，并除以标准差。

To visually evaluate a synthetic image compared to a real image can be difficult if the differences are small. A solution to the visual inspection is to instead visualize a relative error between the real image and the synthetic image. This is done by calculating the absolute difference between the images, and dividing it by the real image. These calculations are done on images normalized in the same manner as for the quantitative evaluation, and the error is the relative absolute difference.

从视觉上评估合成图像，并与真实图像进行比较，如果其差异比较小，则会很困难。视觉检查的一种方法是，对真实图像和合成图像之间的相对差异进行可视化。这可以计算图像之间的绝对差异，并除以真实图像。这些计算是在同样归一化的图像上进行的，与量化评估中的一样，误差是相对绝对误差。

Determining if the synthetic MR images are visually realistic or not was done via a perceptual study by one of the authors (Anders Eklund). The evaluator received T1- and T2- weighted images where 96 of them were real and 72 were synthetic, the evaluator then had to determine if each image was real or synthetic. The real and synthetic images were equally divided between the two domains. Images from Generators_s and Simple were not evaluated since it is obvious that the images are synthetic, due to the high smoothness. Evaluating anatomical images is a complicated task best performed by a radiologist. The results presented in this paper should therefore only be seen as an indicator of the visual quality.

判断合成MR图像是否是真实的，是通过一个感知研究进行的。评估者得到一些T1加权和T2加权的图像，其中96幅是真的，72幅是合成的，评估者要判断每幅图像是真的或是合成的。真实和合成图像评估分成两个领域。从Generators_s和Simple中得到的图像没有评估，因为很明显图像是合成的，因为非常平滑。评估解剖图像是很复杂的任务，放射科医生最擅长这个。本文的结果因此只能看作是视觉质量的指示。

## 3. Results

Quantitative results and results from the perceptual study are shown in Figure 1. The Generators_s model outperforms the other models in all quantitative measurements.The worst performing model on all quantitative measurements, besides MI on T2 images, is the Simple model (despite its supervised nature equivalent to Generators_s). The performance of CycleGAN, CycleGAN_s and UNIT is similar. With just a few exceptions, the quantitative performance is better for T1 images. The opposite is however shown in the perceptual study where more synthetic T1 images are labeled as synthetic compared to T2. Opposite results are in the perceptual study attained for CycleGAN and UNIT, where UNIT shows the best performance for T1 images and CycleGAN shows the best performance for T2 images.

图1给出了定量结果和视觉研究的结果。Generators_s模型在所有量化度量中超过了其他模型。在所有量化度量中，表现最差的模型，除了在T2上的MI，就是Simple模型了（虽然其监督训练的本质与Generators_s是等价的）。CycleGAN, CycleGAN_s和UNIT的性能是类似的。对于T1图像的量化性能更好一些，只有几个例外。在感知研究中，则是相反的情况，与T2比起来，更多的合成T1图像被判断为合成的。对于CycleGAN和UNIT结果的感知研究是相反的，其中UNIT对T1图像效果更好，CycleGAN对T2图像效果最好。

Fig. 1: Quantitative error measurements: (a) MAE, (b) PSNR and (c) MI, for the compared GAN models. The results in (d) are the total scores of all GAN models in the perceptual study, and the results in (e) are for each specific model. Labeling T2-weighted images as real or synthetic is harder due to the fact that T2 images are darker by nature.

The quantitative superiority of Generators_s does not correspond to the visual realness shown in Figure 2. The supervised training results in an unrealistic, smooth appearance seen in the MR images from Generators_s and the Simple model, where the Simple model also fails in the color mapping of the cerebrospinal fluid. The GAN models trained using an adversarial loss generate more realistic synthetic MR images.

Generators_s在量化结果上的优势，与其视觉真实性并不相符，如图2所示。监督训练即Generators_s和Simple模型，得到了不真实的平滑MR图像效果，其中Simple模型在脑脊液的色彩映射中也是失败的。用对抗损失训练的GAN模型，生成了更真实的合成MR图像。

Fig. 2: Synthetic images from the evaluated GAN models. The real images shown at the top are inputs that the synthetic images are based on, this is clarified by the white arrows. The real T2 image is the input that generated the synthetic T1 images and vice versa. The images are the same slice from a single subject. This means that the top images are ground truth for the images below them. The colorbar belongs to the images in the left and right columns, which are calculated as the relative absolute difference between the synthetic and the ground truth image. T1 results are shown in the far left column, and T2 results in the far right column.

The relative absolute error images in Figure 2 show a greater error for the synthetic T2 images compared to the synthetic T1 images. Synthetic T1 images especially have problems at the edges, whereas errors in the T2 images appear all over the brain.

图2中的相对绝对误差图像表明，合成T2图像的误差更大一些，合成T1图像的较小。合成T1图像在边缘处有一些问题，而T2图像中的误差在全脑中遍布。

## 4. Discussion

### 4.1. Quantitative comparison

During training the Generators_s model uses MAE as its only loss function, which creates a model where the goal is to minimize the MAE. The model does this well compared to other models, as shown in Figure 1. The Simple model, which similarly to Generators_s is only trained using the MAE loss, has the highest error among the models. The Simple model only has two convolutional layers and Generators_s has, similar to the CycleGAN generators, 24 convolutional layers. This indicates that the architecture in the Simple model is not sufficiently complex for the translation.

As expected, the CycleGAN_s model shows a slight improvement in MAE for T2 images compared to CycleGAN. However, the results are not significantly better than CycleGAN and the MAE on T1 images is in fact better for CycleGAN. The CycleGAN and UNIT show similar results and it is difficult to argue why one or the other performs slightly better than the other one.

Figure 1c shows that T1 images have a higher MI value than T2 images. This can be correlated to the results from MAE where a larger error was generated from the T2 images. An explanation to why the Simple model has a higher MI score than the majority of models for T2 images, is that T1 and T2 images from the same subject contain very similar information. Since the Simple model only changes the pixel intensity, the main information is preserved.

### 4.2. Qualitative comparison

From the perceptual study it was shown that the synthetic images have a visually realistic appearance, since synthetic images were classified as real. T2 images were more difficult to classify than T1 images and the reason for the difference can be that the synthetic T2 images had a more realistic appearance, but also the darker nature of T2 images (which for example makes it more difficult to determine if the noise is realistic or not).

感知研究表明，合成图像视觉上非常真实，因为合成图像被归类到了真实这类中。T2图像比T1图像更难分类，这个差异的原因可能是，合成T2图像看起来很真实但T2图像更暗一些（比如，这使其更难判断，噪声是真的还是假的）。

The large error on the edges of the synthetic brain images in Figure 2 can be explained by the fact that each brain has a unique shape, and that T2 images are bright for CSF. Areas where there is an intensity change, e.g. CSF and white matter, seem to be more difficult for the models to learn, this might also be due to differences between subjects.

图2中的合成脑部图像，其边缘处误差较大，这可以解释为，每个大脑都有其特定的形状，而且T2图像的CSF更明亮。有亮度变化的区域，如，CSF和白质，似乎模型更难学习这些部分，这也可能是因为个体之间的差异。

The CycleGAN_s penalizes appearance different from the ground truth, since it uses the MAE loss during training, which forces it to another direction, closer to the smooth appearance of the images from the Generators_s model. If the aim of the test would instead be to evaluate how similar the synthetic images are to the ground truth, the translated images from CycleGAN_s may give better results.

CycleGAN_s对与真值之间的误差进行惩罚，因为在训练时使用了MAE损失，这使其转入另一个方向，与Generators_s模型生成的平滑图像更接近。如果测试的目标是评估合成图像与真值图像有多类似，那么CycleGAN_s转化的图像可能给出更好的结果。

From the results in Figure 2 it is obvious that the supervised training, using MAE, pushes the generators into producing smooth synthetic brain images. Another loss function would probably alter the results, but since it is difficult to create mathematical expressions for assessing how realistic an image is, obtaining visually realistic results using supervised methods is a problematic task. The adversarial loss created by the GAN framework allows the discriminator to act as the complex expression, which results in visually realistic images created from the GAN models.

从图2中的结果可以看出，监督训练由于使用了MAE，很明显使得生成器产生了平滑的合成脑图像。另一个损失函数可能会改变这个结果，但由于很难给出数学表达式，评估图像有多真实，所以使用监督方法得到视觉上真实的结果，是一个很有问题的任务。GAN框架创建的对抗损失，使得判别器的作用就是这个复杂的表示，这样就从GAN模型中得到了视觉上相似的图像。

If the aim was to create images that are as similar to ground truth images as possible, the quantitative measurements would be more applicable. It is clear that even if a model such as Simple has a relatively good score in the quantitative measurements, it does not necessarily generate visually realistic images. This indicates that solely determining if an image is visually realistic can not be done with the used metrics.

如果目标是生成尽可能与真值图像接近的图像，量化度量可能就更可行。很清楚的是，即使是Simple这样的模型，在量化度量中都有相对较好的分数，但并没有生成视觉上很真实的图像。这说明，判断图像视觉上是否真实，不太能用度量标准来完成。

### 4.3. Future work

It has been shown, via a perceptual study, that CycleGAN and UNIT can be used to generate visually realistic MR images. The models performed differently in generating images in the different domains, and training CycleGAN in an unsupervised manner is a better alternative if the aim is to generate as visually realistic images as possible.

通过一个视觉研究表明，CycleGAN和UNIT可以用于生成视觉上真实的MR图像。模型在不同的领域生成图像时，表现也不同，如果目标是生成视觉上尽可能真实的图像，那么以无监督的方式训练CycleGAN是一个很好的选择。

A suggestion for future work is to investigate if GANs can be used for data augmentation (e.g. for discriminating healthy and diseased subjects). This would also provide information regarding if the model which creates the most visually realistic images, or the model which performs best in the quantitative evaluations, is the most suitable to use. Here we have only used 2D GANs, but 3D GANs [8, 19] can potentially yield even better results, at the cost of a longer processing time and an increased memory usage.

未来工作的一个建议是，研究GANs是否可以用于数据扩充（如，判别健康目标和疾病目标）。这还可以给出下面的信息，即模型是否给出视觉上最真实的图像，或模型在定量评估中哪个表现最好，是否最适合使用。这里我们只使用了2D GANs，而3D GANs还可能得到更好的结果，但代价是更长的处理时间，和内存使用增加。