# An Image is Worth 16X16 Words: Transformers for Image Recognition at Scale

Alexey Dosovitskiy et. al. Google Research, Brain Team

## 0. Abstract

While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

Transformer架构已经成为了NLP任务的实际标准，其在计算机视觉中的应用仍然很少。在视觉中，注意力要么是与卷积网络联合应用的，要么是用于替代卷积网络的特定部分，同时保持其总体结构不变。我们证明了，这种对CNNs的依赖是不必要的，纯粹的Transformer架构直接应用于图像块的序列，在图像分类任务中可以得到很好的结果。当大量数据预训练后，迁移到多个中型或小型图像识别基准测试(ImageNet, CIFAR-100, VTAB, etc.)中时，ViT与目前最好的CNN相比，得到了非常好的结果，同时需要进行训练的资源要少的多。

## 1. Introduction

Self-attention-based architectures, in particular Transformers (Vaswani et al., 2017), have become the model of choice in natural language processing (NLP). The dominant approach is to pre-train on a large text corpus and then fine-tune on a smaller task-specific dataset (Devlin et al., 2019). Thanks to Transformers’ computational efficiency and scalability, it has become possible to train models of unprecedented size, with over 100B parameters (Brown et al., 2020; Lepikhin et al., 2020). With the models and datasets growing, there is still no sign of saturating performance.

基于自注意力的架构，特别是Transformers，已经是NLP中的模型选择。主要的方法是，在一个大型文本语料库中进行预训练，然后再一个更小的任务专用的数据集上进行精调。Transformers的计算效率和可扩展性都很好，所以可能训练前所未有大小的模型，超过100B参数。随着模型和数据集的增长，仍然没有性能饱和的迹象。

In computer vision, however, convolutional architectures remain dominant (LeCun et al., 1989; Krizhevsky et al., 2012; He et al., 2016). Inspired by NLP successes, multiple works try combining CNN-like architectures with self-attention (Wang et al., 2018; Carion et al., 2020), some replacing the convolutions entirely (Ramachandran et al., 2019; Wang et al., 2020a). The latter models, while theoretically efficient, have not yet been scaled effectively on modern hardware accelerators due to the use of specialized attention patterns. Therefore, in large-scale image recognition, classic ResNet-like architectures are still state of the art (Mahajan et al., 2018; Xie et al., 2020; Kolesnikov et al., 2020).

但是，在计算机视觉中，CNN架构仍然是主流。受到NLP成功的启发，很多工作尝试将CNN类的架构与自注意力结合起来，一些完全取代了卷积。后面这个模型在理论上是高效的，但是还没有在现代硬件加速器上有效的缩放，因为用了专有的注意力模式。因此，在大规模图像识别中，经典的类ResNet架构仍然是目前最好的模型。

Inspired by the Transformer scaling successes in NLP, we experiment with applying a standard Transformer directly to images, with the fewest possible modifications. To do so, we split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. Image patches are treated the same way as tokens (words) in an NLP application. We train the model on image classification in supervised fashion.

受Transformer在NLP中成功的启发，我们将标准Transformer直接应用到图像中进行试验，修改尽可能的少。为此，我们将图像分成图像块，将这些块的线性嵌入的序列作为输入送入Transformer。图像块与NLP中的tokens（词语）的处理方式一样。我们以有监督的方式在图像分类上训练这个模型。

When trained on mid-sized datasets such as ImageNet without strong regularization, these models yield modest accuracies of a few percentage points below ResNets of comparable size. This seemingly discouraging outcome may be expected: Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data.

当在中等规模的数据集如ImageNet上进行训练，在没有很强的正则化时，这些模型得到了一般的准确率，比类似大小的ResNets模型准确率低了几个百分点。这种似乎令人丧气的结果，是可以预期的：Transformers缺少一些CNNs内在的inductive biases，比如平移同变性和局部性，因此训练数据不够时，泛化效果并不好。

However, the picture changes if the models are trained on larger datasets (14M-300M images). We find that large scale training trumps inductive bias. Our Vision Transformer (ViT) attains excellent results when pre-trained at sufficient scale and transferred to tasks with fewer datapoints. When pre-trained on the public ImageNet-21k dataset or the in-house JFT-300M dataset, ViT approaches or beats state of the art on multiple image recognition benchmarks. In particular, the best model reaches the accuracy of 88.55% on ImageNet, 90.72% on ImageNet-ReaL, 94.55% on CIFAR-100, and 77.63% on the VTAB suite of 19 tasks.

但是，如果在更大规模的数据集(14M-300M images)上训练时，结果就有了改变。我们发现，大规模训练打败了inductive bias。我们的ViT在足够大的数据集上预训练，并迁移到更少数据点的任务上时，得到了非常好的结果。当在公开的ImageNet-21k数据集，或室内JFT-300M数据集上进行预训练，ViT在多个图像识别基准测试中达到或超过了目前最好的结果。特别是，最好的模型在ImageNet上达到了88.55%的准确率，在ImageNet-Real上90.72%，在CIFAR-100上94.55%，在VTAB的19个任务中达到了77.63%的结果。

## 2. Related Work

Transformers were proposed by Vaswani et al. (2017) for machine translation, and have since become the state of the art method in many NLP tasks. Large Transformer-based models are often pre-trained on large corpora and then fine-tuned for the task at hand: BERT (Devlin et al., 2019) uses a denoising self-supervised pre-training task, while the GPT line of work uses language modeling as its pre-training task (Radford et al., 2018; 2019; Brown et al., 2020).

Transformer是在2017年提出用于机器翻译的，从此在很多NLP任务中都成为了目前效果最好的方法。基于Transformer的大型模型通常是在大型语料库上预训练，然后在手头的任务上进行精调：BERT使用的是去噪的自监督预训练任务，而GPT使用语言建模作为其预训练任务。

Naive application of self-attention to images would require that each pixel attends to every other pixel. With quadratic cost in the number of pixels, this does not scale to realistic input sizes. Thus, to apply Transformers in the context of image processing, several approximations have been tried in the past. Parmar et al. (2018) applied the self-attention only in local neighborhoods for each query pixel instead of globally. Such local multi-head dot-product self attention blocks can completely replace convolutions (Hu et al., 2019; Ramachandran et al., 2019; Zhao et al., 2020). In a different line of work, Sparse Transformers (Child et al., 2019) employ scalable approximations to global self-attention in order to be applicable to images. An alternative way to scale attention is to apply it in blocks of varying sizes (Weissenborn et al., 2019), in the extreme case only along individual axes (Ho et al., 2019; Wang et al., 2020a). Many of these specialized attention architectures demonstrate promising results on computer vision tasks, but require complex engineering to be implemented efficiently on hardware accelerators.

将自注意力很简单的应用到图像上，需要每个像素都注意其他像素。其计算代价是像素数量的平方，这不能扩展到真实的输入大小。因此，为将Transformers应用到图像处理中，在过去尝试了几种近似。Parmar等只在每个查询像素的局部邻域应用了自注意力，而没有全局应用。这样局部的多头点积自注意力模块可以完全取代卷积。在不同的工作线上，Sparse Transformers采用了全局自注意力的可扩展近似，以应用到图像中。另一种扩展注意力的方法是，在不同大小的块中应用，在极端情况中，只沿着单个轴。很多这种专用的注意力架构说明在计算机视觉任务中可以得到有希望的结果，但需要复杂的设计，才能在硬件加速器上得到很好的实现。

Most related to ours is the model of Cordonnier et al. (2020), which extracts patches of size 2 × 2 from the input image and applies full self-attention on top. This model is very similar to ViT, but our work goes further to demonstrate that large scale pre-training makes vanilla transformers competitive with (or even better than) state-of-the-art CNNs. Moreover, Cordonnier et al. (2020) use a small patch size of 2 × 2 pixels, which makes the model applicable only to small-resolution images, while we handle medium-resolution images as well.

与我们的工作最相关的，是Cordonnier的模型，从输入图像中提取2x2的块，然后在其之上使用完整的自注意力。这种模型与ViT非常类似，但我们的工作进一步证明了，大规模预训练使经典transformer与目前最好的CNNs不相上下。而且，Cordonnier等使用了很小的2x2图像块，这使模型只能应用到小分辨率的图像中，而我们的模型还可以处理中等分辨率的图像。

There has also been a lot of interest in combining convolutional neural networks (CNNs) with forms of self-attention, e.g. by augmenting feature maps for image classification (Bello et al., 2019) or by further processing the output of a CNN using self-attention, e.g. for object detection (Hu et al., 2018; Carion et al., 2020), video processing (Wang et al., 2018; Sun et al., 2019), image classification (Wu et al., 2020), unsupervised object discovery (Locatello et al., 2020), or unified text-vision tasks (Chen et al., 2020c; Lu et al., 2019; Li et al., 2019).

将CNNs与自注意力结合在一起，也有很多工作，如，扩增特征图进行图像分类，或使用自注意力进一步处理CNNs的输出，如，进行目标检测，视频处理，图像分类，无监督目标发现，或文本视觉联合任务。

Another recent related model is image GPT (iGPT) (Chen et al., 2020a), which applies Transformers to image pixels after reducing image resolution and color space. The model is trained in an unsupervised fashion as a generative model, and the resulting representation can then be fine-tuned or probed linearly for classification performance, achieving a maximal accuracy of 72% on ImageNet.

另一个最近相关的模型是图像GPT，在降低了图像分辨率和色彩空间后，将Transformer应用到图像像素上。这个模型是以无监督的方式训练的，是一个生成式模型，得到的表示可以精调，或进行线性探究，以进行分类，在ImageNet上得到了最大72%的准确率。

Our work adds to the increasing collection of papers that explore image recognition at larger scales than the standard ImageNet dataset. The use of additional data sources allows to achieve state-of-the-art results on standard benchmarks (Mahajan et al., 2018; Touvron et al., 2019; Xie et al., 2020). Moreover, Sun et al. (2017) study how CNN performance scales with dataset size, and Kolesnikov et al. (2020); Djolonga et al. (2020) perform an empirical exploration of CNN transfer learning from large scale datasets such as ImageNet-21k and JFT-300M. We focus on these two latter datasets as well, but train Transformers instead of ResNet-based models used in prior works.

我们的工作也使用了超过标准ImageNet规模的数据集。使用了更多数据，可以在标准基准测试中得到目前最好的结果。而且，Sun等研究了CNN性能与数据集大小的关系，Kolesnikov等、Djolonga等对从大规模数据集上的迁移学习上进行了经验探索。我们也关注ImageNet-21k和JFT-300M这两个数据集，但训练的是Transformers，而不是以前的基于ResNet的模型。

## 3. Method

In model design we follow the original Transformer (Vaswani et al., 2017) as closely as possible. An advantage of this intentionally simple setup is that scalable NLP Transformer architectures – and their efficient implementations – can be used almost out of the box.

在模型设计上，我们尽可能按照原始版Transformer。这种故意的简单设置，一个优势是可扩展的NLP Transformer架构以及其高效的实现，几乎可以开箱即用。

### 3.1 Vision Transformer (ViT)

An overview of the model is depicted in Figure 1. The standard Transformer receives as input a 1D sequence of token embeddings. To handle 2D images, we reshape the image $x ∈ R^{H×W×C}$ into a sequence of flattened 2D patches $x_p ∈ R^{N×(P^2·C)}$, where (H, W) is the resolution of the original image, C is the number of channels, (P, P) is the resolution of each image patch, and $N = HW/P^2$ is the resulting number of patches, which also serves as the effective input sequence length for the Transformer. The Transformer uses constant latent vector size D through all of its layers, so we flatten the patches and map to D dimensions with a trainable linear projection (Eq. 1). We refer to the output of this projection as the patch embeddings.

模型概览如图1所示。标准Transformer的输入为令牌嵌入的1D序列。为处理2D图像，我们改变图像的大小，将$x ∈ R^{H×W×C}$的图像改变为拉直的2D图像块的序列$x_p ∈ R^{N×(P^2·C)}$，其中(H, W)是原始图像的分辨率，C是通道数，(P, P)是每个图像块的分辨率，$N = HW/P^2$是得到的图像块的数量，也是Transformer的有效输入序列长度。Transformer在其所有层中，使用的潜在矢量大小为常数D，所以我们将每个图像块拉直，并使用一个可训练的线性投影来映射到D维。我们称这个投影的输出为块嵌入。

Similar to BERT’s [ class ] token, we prepend a learnable embedding to the sequence of  embedded patches ($z_0^0 = x_{class}$), whose state at the output of the Transformer encoder ($z^0_L$) serves as the image representation y (Eq. 4). Both during pre-training and fine-tuning, a classification head is attached to $z^0_L$. The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time.

与BERT的[ class ]令牌类似，我们给嵌入块的序列预置了一个可学习的嵌入($z_0^0 = x_{class}$)，其在Transformer编码器($z^0_L$)的输出端作为图像的表示y（式4）。在预训练和精调期间，对$z^0_L$加上了一个分类头。分类头是用MLP实现的，在预训练时有一个隐藏层，在精调的时候是一个线性层。

Position embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D position embeddings, since we have not observed significant performance gains from using more advanced 2D-aware position embeddings (Appendix D.3). The resulting sequence of embedding vectors serves as input to the encoder.

位置嵌入加入到块嵌入中，以保持位置信息。我们使用标准的可学习1D位置嵌入，因为我们使用了更高级的2D位置嵌入，并没有显著的性能改进（附录D.3）。得到的嵌入向量序列，是编码器的输入。

The Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multiheaded self-attention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3). Layernorm (LN) is applied before every block, and residual connections after every block (Wang et al., 2019; Baevski & Auli, 2019). The MLP contains two layers with a GELU non-linearity.

Transformer编码器由多头自注意力和MLP模块交替组成。层归一化在每个模块之前都进行了使用，在每个模块之后，都有残差连接。MLP包含2层，使用了GELU非线性。

$$z_0 = [x_{class}; x_p^1 E;...;x_p^N E] + E_{pos}, E∈R^{(P^2·C)×D}, E_{pos}∈R^{(N+1)×D}$$(1)
$$z'_l = MSA(LN(z_{l-1})) + z_{l-1}, l=1...L$$(2)
$$z_l = MLP(LN(z'_l)) + z'_l, l=1...L$$(3)
$$y = LN(z^0_L)$$(4)

**Inductive bias**. We note that Vision Transformer has much less image-specific inductive bias than CNNs. In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer throughout the whole model. In ViT, only MLP layers are local and translationally equivariant, while the self-attention layers are global. The two-dimensional neighborhood structure is used very sparingly: in the beginning of the model by cutting the image into patches and at fine-tuning time for adjusting the position embeddings for images of different resolution (as described below). Other than that, the position embeddings at initialization time carry no information about the 2D positions of the patches and all spatial relations between the patches have to be learned from scratch.

**归纳偏置**。我们注意到，Vision Transformer比CNNs的视觉专用的归纳偏置要少很多。在CNNs中，局部性，二维邻域结构，和平移等变性是内化在每层中的，在整个模型中都存在。在ViT中，只有MLP是局部的，是平移等变的，而自注意力层是全局的。二维邻域结构的应用非常少：在模型的开始，将图像切成块，在精调的时候，为不同分辨率的图像调整位置嵌入（下面进行叙述）。除此以外，在初始化时的位置嵌入并没有图像块的2D位置的任何信息，图像块的所有空间关系需要从头学习。

**Hybrid Architecture**. As an alternative to raw image patches, the input sequence can be formed from feature maps of a CNN (LeCun et al., 1989). In this hybrid model, the patch embedding projection E (Eq. 1) is applied to patches extracted from a CNN feature map. As a special case, the patches can have spatial size 1x1, which means that the input sequence is obtained by simply flattening the spatial dimensions of the feature map and projecting to the Transformer dimension. The classification input embedding and position embeddings are added as described above.

**混合架构**。作为原始图像块的另一个选择，输入序列可以从CNN的特征图中形成。在这个混合模型中，图像块嵌入投影E应用到CNN特征图中提取的块中。作为一种特殊情况，块的空间大小可以是1x1，这意味着输入序列是通过拉值特征图，并投影到Transformer维度上得到的。分类输入嵌入和位置嵌入是按照上面所述的方式加入。

### 3.2. Fine-Tuning and Higher Resolution

Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks. For this, we remove the pre-trained prediction head and attach a zero-initialized D × K feedforward layer, where K is the number of downstream classes. It is often beneficial to fine-tune at higher resolution than pre-training (Touvron et al., 2019; Kolesnikov et al., 2020). When feeding images of higher resolution, we keep the patch size the same, which results in a larger effective sequence length. The Vision Transformer can handle arbitrary sequence lengths (up to memory constraints), however, the pre-trained position embeddings may no longer be meaningful. We therefore perform 2D interpolation of the pre-trained position embeddings, according to their location in the original image. Note that this resolution adjustment and patch extraction are the only points at which an inductive bias about the 2D structure of the images is manually injected into the Vision Transformer.

一般来说，我们在大型数据集上预训练ViT，并精调到更小的下游任务。对此，我们去掉预训练的预测头，并加入一个初始化为0的DxK前向层，其中K是下游任务类别的数量。通常来说，在更高的分辨率下进行精调，比预训练要更好。当送入更高分辨率的图像，我们保持同样的图像块大小，这会得到更大的有效序列长度。Vision Transformer可以处理任意长度的序列，但是，预训练的位置嵌入此时就没有意义了。因此我们对预训练的位置嵌入进行2D插值，根据其在原始图像中的位置。注意，这种分辨率调整和图像块的提取，是仅有的输入到Vision Transformer图像2D结构的归纳偏置。

## 4. Experiments

We evaluate the representation learning capabilities of ResNet, Vision Transformer (ViT), and the hybrid. To understand the data requirements of each model, we pre-train on datasets of varying size and evaluate many benchmark tasks. When considering the computational cost of pre-training the model, ViT performs very favourably, attaining state of the art on most recognition benchmarks at a lower pre-training cost. Lastly, we perform a small experiment using self-supervision, and show that self-supervised ViT holds promise for the future.

我们评估ResNet, Vit和混合架构的表示学习能力。为理解每个模型的数据要求，我们在不同大小的数据集上进行预训练，在很多基准测试任务上进行评估。当考虑预训练模型的计算代价时，ViT的性能非常不错，在多数识别基准测试中，以很低的预训练代价，得到了目前最好的性能。最后，我们使用自监督方式进行了试验，表明自监督ViT也很有希望。

### 4.1 Setup

**Datasets**. To explore model scalability, we use the ILSVRC-2012 ImageNet dataset with 1k classes and 1.3M images (we refer to it as ImageNet in what follows), its superset ImageNet-21k with 21k classes and 14M images (Deng et al., 2009), and JFT (Sun et al., 2017) with 18k classes and 303M high-resolution images. We de-duplicate the pre-training datasets w.r.t. the test sets of the downstream tasks following Kolesnikov et al. (2020). We transfer the models trained on these dataset to several benchmark tasks: ImageNet on the original validation labels and the cleaned-up ReaL labels (Beyer et al., 2020), CIFAR-10/100 (Krizhevsky, 2009), Oxford-IIIT Pets (Parkhi et al., 2012), and Oxford Flowers-102 (Nilsback & Zisserman, 2008). For these datasets, pre-processing follows Kolesnikov et al. (2020).

**数据集**。为探索模型的可扩展性，我们使用ImageNet ILSVRC 2012数据集，有1k类别，1.3M图像（下面我们称之为ImageNet），其超集ImageNet-21k，有21k类别，14M图像，和JFT数据集，有18k类别，303M高分辨率图像。我们对预训练数据集按照下游任务的测试集来去除重复。我们将在这些数据集上训练的模型，迁移到几个基准测试任务中：ImageNet，有原始验证标签和清理过的Real标签，CIFAR-10/100，Oxford-IIIT Pets，和Oxford Flowers-102。对这些数据集，我们按照Kolesnikov等的方法进行预处理。

We also evaluate on the 19-task VTAB classification suite (Zhai et al., 2019b). VTAB evaluates low-data transfer to diverse tasks, using 1 000 training examples per task. The tasks are divided into three groups: Natural – tasks like the above, Pets, CIFAR, etc. Specialized – medical and satellite imagery, and Structured – tasks that require geometric understanding like localization.

我们还在19-任务VTAB分类套装中进行了评估。VTAB评估很多任务的低数据迁移，每个任务使用1000张训练图像。任务分成三组：自然的，与上面类似的任务，Pets，CIFAR等；专用的，医学和卫星成像，和结构化的，需要几何理解的任务，如定位。

**Model Variants**. We base ViT configurations on those used for BERT (Devlin et al., 2019), as summarized in Table 1. The “Base” and “Large” models are directly adopted from BERT and we add the larger “Huge” model. In what follows we use brief notation to indicate the model size and the input patch size: for instance, ViT-L/16 means the “Large” variant with 16×16 input patch size. Note that the Transformer’s sequence length is inversely proportional to the square of the patch size, thus models with smaller patch size are computationally more expensive.

**模型变体**。我们对ViT的配置是基于BERT使用的配置，如表1所示。Base模型和Large模型是直接从BERT中采用的，我们加入了更大的Huge模型。在下面，我们使用简写来指示模型大小和输入批次大小：比如，ViT-L/16意思是，Large模型变体，输入图像块大小为16×16。注意，Transformer序列长度是与图像块大小的平方成反比的，因此图像块更小的模型，计算量更大一些。

For the baseline CNNs, we use ResNet (He et al., 2016), but replace the Batch Normalization layers (Ioffe & Szegedy, 2015) with Group Normalization (Wu & He, 2018), and used standardized convolutions (Qiao et al., 2019). These modifications improve transfer (Kolesnikov et al., 2020), and we denote the modified model “ResNet (BiT)”. For the hybrids, we feed the intermediate feature maps into ViT with patch size of one “pixel”. To experiment with different sequence lengths, we either (i) take the output of stage 4 of a regular ResNet50 or (ii) remove stage 4, place the same number of layers in stage 3 (keeping the total number of layers), and take the output of this extended stage 3. Option (ii) results in a 4x longer sequence length, and a more expensive ViT model.

对于基准CNNs，我们使用ResNet，但将批归一化替换为了组归一化，并使用标准卷积。这些修改改进了迁移性能，我们称修改的模型为ResNet(BiT)。对于混合模型，我们将中间的特征图送入ViT中，块的大小就是一个像素。为试验不同的序列长度，我们(i)以常规ResNet50的第4阶段输出，或(ii)去掉第4阶段，在第3阶段中放上相同数量的层（保持层总量相同），取这个拓展阶段3的输出。选项(ii)得到长了4倍的序列长度，和更贵的ViT模型。

**Training & Fine-tuning**. We train all models, including ResNets, using Adam (Kingma & Ba, 2015) with β1 = 0.9, β2 = 0.999, a batch size of 4096 and apply a high weight decay of 0.1, which we found to be useful for transfer of all models (Appendix D.1 shows that, in contrast to common practices, Adam works slightly better than SGD for ResNets in our setting). We use a linear learning rate warmup and decay, see Appendix B.1 for details. For fine-tuning we use SGD with momentum, batch size 512, for all models, see Appendix B.1.1. For ImageNet results in Table 2, we fine-tuned at higher resolution: 512 for ViT-L/16 and 518 for ViT-H/14, and also used Polyak & Juditsky (1992) averaging with a factor of 0.9999 (Ramachandran et al., 2019; Wang et al., 2020b).

**训练&精调**。我们训练所有模型，包括ResNets，都使用Adam，其参数β1 = 0.9, β2 = 0.999，批大小为4096，权重衰减很高，为0.1，我们发现这在所有模型的迁移中都很好用。我们使用线性学习速率的预热和衰减，见附录B.1。对精调，我们对所有模型都使用带有动量的SGD，批大小为512。对表2中的ImageNet结果，我们在更高的分辨率进行精调：对ViT-L/16为512，对ViT-H/14为518，我们还使用Polyak & Juditsky (1992)平均，其系数为0.9999。

**Metrics**. We report results on downstream datasets either through few-shot or fine-tuning accuracy. Fine-tuning accuracies capture the performance of each model after fine-tuning it on the respective dataset. Few-shot accuracies are obtained by solving a regularized least-squares regression problem that maps the (frozen) representation of a subset of training images to {−1, 1}^K target vectors. This formulation allows us to recover the exact solution in closed form. Though we mainly focus on fine-tuning performance, we sometimes use linear few-shot accuracies for fast on-the-fly evaluation where fine-tuning would be too costly.

**度量**。我们在下游任务给出准确率的结果。精调的准确率，在各自数据集上进行精调后，给出每个模型的性能。小样本准确率是通过求解一个正则化的最小二乘回归问题得到的，将一部分训练图像的冻结表示，映射到{−1, 1}^K个目标向量。这种表示使我们能以闭合的形式发现精确的解。虽然我们主要关注精调的性能，在精调的代价太高时，有时候我们也使用线性小样本准确率，以进行快速即时评估。

### 4.2 Comparison to State of the Art

We first compare our largest models – ViT-H/14 and ViT-L/16 – to state-of-the-art CNNs from the literature. The first comparison point is Big Transfer (BiT) (Kolesnikov et al., 2020), which performs supervised transfer learning with large ResNets. The second is Noisy Student (Xie et al., 2020), which is a large EfficientNet trained using semi-supervised learning on ImageNet and JFT-300M with the labels removed. Currently, Noisy Student is the state of the art on ImageNet and BiT-L on the other datasets reported here. All models were trained on TPUv3 hardware, and we report the number of TPUv3-core-days taken to pre-train each of them, that is, the number of TPU v3 cores (2 per chip) used for training multiplied by the training time in days.

我们首先将我们最大的模型ViT-H/14和ViT-L/16，与目前文献中最好的CNNs进行比较。第一个比较点是Big Transfer(BiT)，用大型ResNets进行有监督迁移学习。第二个是Noisy Student，是一个大型EfficientNet，在去掉了标签的ImageNet和JFT-300M上用半监督学习上训练得到的。目前，Noisy Student是在ImageNet上目前最好的结果，BiT-L是其他数据集上最好的结果。所有的模型都用TPUv3硬件进行训练，我们给出每个模型预训练所耗费的TPUv3-core-days数，即，所用的TPUv3核心数量（每片2个）乘以训练时间天数。

Table 2 shows the results. The smaller ViT-L/16 model pre-trained on JFT-300M outperforms BiT-L (which is pre-trained on the same dataset) on all tasks, while requiring substantially less computational resources to train. The larger model, ViT-H/14, further improves the performance, especially on the more challenging datasets – ImageNet, CIFAR-100, and the VTAB suite. Interestingly, this model still took substantially less compute to pre-train than prior state of the art. However, we note that pre-training efficiency may be affected not only by the architecture choice, but also other parameters, such as training schedule, optimizer, weight decay, etc. We provide a controlled study of performance vs. compute for different architectures in Section 4.4. Finally, the ViT-L/16 model pre-trained on the public ImageNet-21k dataset performs well on most datasets too, while taking fewer resources to pre-train: it could be trained using a standard cloud TPUv3 with 8 cores in approximately 30 days.

表2给出了结果。在JFT-300M上预训练的小一些的ViT-L/16模型，在所有任务上都超过了BiT-L（在相同的数据集上预训练的），而需要的训练资源更少。大一些的模型ViT-H/14，进一步改进了性能，尤其是在更有挑战性的数据集上，包括ImageNet, CIFAR-100, 和VTAB套装。有趣的是，与之前最好的模型相比，这些模型训练所需的资源仍然很少。但是，我们注意到，预训练效率的影响因素，不止包括架构选择，还包括其他参数，比如训练方案，优化器，权重衰减，等。我们在4.4节中对不同架构的性能vs计算量进行了受控的研究。最后，在公开的ImageNet-21k上预训练的ViT-L/16模型，在多数数据集上表现也很好，而所需的训练资源也更少：使用标准的云8核TPUv3，在大约30天内就可以训练出来。

Figure 2 decomposes the VTAB tasks into their respective groups, and compares to previous SOTA methods on this benchmark: BiT, VIVI – a ResNet co-trained on ImageNet and Youtube (Tschannen et al., 2020), and S4L – supervised plus semi-supervised learning on ImageNet (Zhai et al., 2019a). ViT-H/14 outperforms BiT-R152x4, and other methods, on the Natural and Structured tasks. On the Specialized the performance of the top two models is similar.

图2将VTAB任务分解到其不同的组中，与之前的最好方法在这个基准测试上进行了比较：BiT，VIVI，S4L。在Natural和Structured任务上，ViT-H/14超过了BiT-R152x4，和其他方法。在Specialized上，两个最好的模型性能类似。

### 4.3 Pre-Training Data Requirements 预训练数据的要求

The Vision Transformer performs well when pre-trained on a large JFT-300M dataset. With fewer inductive biases for vision than ResNets, how crucial is the dataset size? We perform two series of experiments.

在大型JFT-300M数据集上进行训练时，ViT模型表现很好。其归纳偏置在视觉上比ResNets是要少的，所以数据集的规模究竟有多关键呢？我们进行了两个系列的实验。

First, we pre-train ViT models on datasets of increasing size: ImageNet, ImageNet-21k, and JFT-300M. To boost the performance on the smaller datasets, we optimize three basic regularization parameters – weight decay, dropout, and label smoothing. Figure 3 shows the results after fine-tuning to ImageNet (results on other datasets are shown in Table 5). When pre-trained on the smallest dataset, ImageNet, ViT-Large models underperform compared to ViT-Base models, despite (moderate) regularization. With ImageNet-21k pre-training, their performances are similar. Only with JFT-300M, do we see the full benefit of larger models. Figure 3 also shows the performance region spanned by BiT models of different sizes. The BiT CNNs outperform ViT on ImageNet, but with the larger datasets, ViT overtakes.

第一，我们在不同大小的数据集上预训练了ViT模型：ImageNet, ImageNet-21k, 和JFT-300M。为提升在两个更小数据集上的性能，我们优化了三个基本正则化参数-权重衰减，dropout和标签平滑。图3给出了在ImageNet上精调了之后的结果（在其他数据集上的结果在表5中给出）。当在最小的数据集ImageNet上预训练时，ViT-Large模型比Vit-Base模型效果要差，尽管有一些正则化。在ImageNet-21k上预训练时，其性能是类似的。只有在JFT-300M上预训练时，才能看到更大模型的完全优势。在ImageNet上预训练，BiT CNNs的性能比ViT要好，但在更大的数据集上，ViT效果要更好。

Second, we train our models on random subsets of 9M, 30M, and 90M as well as the full JFT-300M dataset. We do not perform additional regularization on the smaller subsets and use the same hyper-parameters for all settings. This way, we assess the intrinsic model properties, and not the effect of regularization. We do, however, use early-stopping, and report the best validation accuracy achieved during training. To save compute, we report few-shot linear accuracy instead of full fine-tuning accuracy. Figure 4 contains the results. Vision Transformers overfit more than ResNets with comparable computational cost on smaller datasets. For example, ViT-B/32 is slightly faster than ResNet50; it performs much worse on the 9M subset, but better on 90M+ subsets. The same is true for ResNet152x2 and ViT-L/16. This result reinforces the intuition that the convolutional inductive bias is useful for smaller datasets, but for larger ones, learning the relevant patterns directly from data is sufficient, even beneficial.

第二，我们在JFT-300M数据集的随机9M，30M和90M子集，和完全集上训练我们的模型。我们在小子集上，没有进行更多的正则化，对所有设置使用相同的超参数。这样，我们评估了模型的内在性质，而不是正则化的影响。我们使用了早停，给出了训练过程中得到的最佳验证准确率。为节省计算，我们给出小样本线性准确率，而不是完整的精调准确率。图4给出了结果。在更小的数据集上，用类似的计算量，ViT过拟合的比ResNets要多。比如，ViT-B/32比ResNet50要快一些；在9M子集上表现要差很多，但在90M+子集上要好一些。对于ResNet152x2和ViT-L/16是同样的结论。这个结果加强了下面的直觉，即卷积的归纳偏置在更小的数据集上是有用的，但对于更大的数据集，从数据中直接学习相关的模式就足够了，甚至更有优势。

Overall, the few-shot results on ImageNet (Figure 4), as well as the low-data results on VTAB (Table 2) seem promising for very low-data transfer. Further analysis of few-shot properties of ViT is an exciting direction of future work.

总体上，在ImageNet上的小样本结果（图4），和在VTAB上的低数据结果（表2），看起来在低数据迁移上非常有希望。ViT的小样本性质是一个值得研究的方向。

### 4.4 Scaling Study

We perform a controlled scaling study of different models by evaluating transfer performance from JFT-300M. In this setting data size does not bottleneck the models' performances, and we assess performance versus pre-training cost of each model. The model set includes: 7 ResNets, R50x1, R50x2, R101x1, R152x1, R152x2, pre-trained for 7 epochs, plus R152x2 and R200x3 pre-trained for 14 epochs; 6 Vision Transformers, ViT-B/32, B/16, L/32, L/16, pre-trained for 7 epochs, plus L/16 and H/14 pre-trained for 14 epochs; and 5 hybrids, R50+ViT-B/32, B/16, L/32, L/16 pretrained for 7 epochs, plus R50+ViT-L/16 pre-trained for 14 epochs (for hybrids, the number at the end of the model name stands not for the patch size, but for the total dowsampling ratio in the ResNet backbone).

我们进行了不同模型的有限缩放研究，评估从JFT-300M得到的迁移性能。在这个设置中，数据规模并不是模型性能的瓶颈，我们评估每个模型的性能与预训练代价。模型集包括：7个ResNets，R50x1, R50x2, R101x1, R152x1, R152x2, 预训练7轮，加上R152x2和R200x3预训练14轮；6个ViT模型，ViT-B/32, B/16, L/32, L/16, 预训练7轮，加上L/16和H/14预训练14轮；5个混合模型，R50+ViT-B/32, B/16, L/32, L/16，预训练7轮，加上R50+ViT-L/16预训练14轮（对于混合模型，模型名称最后的数量并不代表图像块大小，而是在ResNet骨干中的总计降采样率）。

Figure 5 contains the transfer performance versus total pre-training compute (see Appendix D.4 for details on computational costs). Detailed results per model are provided in Table 6 in the Appendix. A few patterns can be observed. First, Vision Transformers dominate ResNets on the performance/compute trade-off. ViT uses approximately 2 − 4× less compute to attain the same performance (average over 5 datasets). Second, hybrids slightly outperform ViT at small computational budgets, but the difference vanishes for larger models. This result is somewhat surprising, since one might expect convolutional local feature processing to assist ViT at any size. Third, Vision Transformers appear not to saturate within the range tried, motivating future scaling efforts.

图5给出了迁移学习性能vs总计预训练计算量。每个模型的详细结果在表6中给出。可以观察出几个模式。第一，在性能/计算的折中中，ViT比ResNets要好。ViT使用了计算量大约少了2-4x，就得到了相同的性能。第二，混合模型比ViT在很小的计算预算上略好，但对于很大的模型来说，这个差异就消失了。结果是有些意外的，因为计算卷积局部特征会在任何大小上都对ViT有帮助。第三，ViT在尝试的范围内，似乎并没有性能饱和，在未来可以进行更多的缩放实验。

### 4.5 Inspecting Vision Transformer

To begin to understand how the Vision Transformer processes image data, we analyze its internal representations. The first layer of the Vision Transformer linearly projects the flattened patches into a lower-dimensional space (Eq. 1). Figure 7 (left) shows the top principal components of the the learned embedding filters. The components resemble plausible basis functions for a low-dimensional representation of the fine structure within each patch.

为理解ViT是怎样处理图像数据的，我们分析了其内部表示。ViT的第一层，将拉直的图像块线性投影到低维空间（式1）。图7左给出了学习的嵌入滤波器的主要组成部分。对于每个图像块中的精细结构的低维表示来说，这些组成部分很像可行的基函数。

After the projection, a learned position embedding is added to the patch representations. Figure 7 (center) shows that the model learns to encode distance within the image in the similarity of position embeddings, i.e. closer patches tend to have more similar position embeddings. Further, the row-column structure appears; patches in the same row/column have similar embeddings. Finally, a sinusoidal structure is sometimes apparent for larger grids (Appendix D). That the position embeddings learn to represent 2D image topology explains why hand-crafted 2D-aware embedding variants do not yield improvements (Appendix D.3).

在投影后，给图像块表示加入了一个学习得到的位置嵌入。图7中间是模型学习在图像中对距离在位置嵌入中的相似度进行编码，即，更近的图像块的位置嵌入更相似。而且，行列结构也出现了；相同的行/列的图像块，有类似的嵌入。最后，正弦结构对于更大的网格来说是很明显的。位置嵌入学习表示2D图像的拓扑结构，解释了为什么手工设计的2D嵌入变体不会得到改进。

Self-attention allows ViT to integrate information across the entire image even in the lowest layers. We investigate to what degree the network makes use of this capability. Specifically, we compute the average distance in image space across which information is integrated, based on the attention weights (Figure 7, right). This "attention distance" is analogous to receptive field size in CNNs. We find that some heads attend to most of the image already in the lowest layers, showing that the ability to integrate information globally is indeed used by the model. Other attention heads have consistently small attention distances in the low layers. This highly localized attention is less pronounced in hybrid models that apply a ResNet before the Transformer (Figure 7, right), suggesting that it may serve a similar function as early convolutional layers in CNNs. Further, the attention distance increases with network depth. Globally, we find that the model attends to image regions that are semantically relevant for classification (Figure 6).

自注意力使ViT能够在最低层就整合了整个图像的信息。我们研究了网络在何种程度上利用了这种能力。具体的，我们计算了图像空间中整合信息的跨度的平均距离，基于注意力权重（图7右）。这个“注意力距离”与CNNs中的感受野大小是类似的。我们发现，在最低层，一些头就已经注意到图像的大部分了，表明模型确实使用了整合全局信息的能力。其他的注意力头在低层中一直是较小的注意力距离。这种高度局部的注意力，在混合模型中并没有那么明显（在Transformer之前使用了ResNet，图7右），说明其可以起到CNNs中早起卷积层的类似作用。而且，注意力距离随着网络加深而变大。全局上来说，我们发现模型关注的图像的区域，与分类是语义相关的（图6）。

### 4.6 Self-Supervision

Transformers show impressive performance on NLP tasks. However, much of their success stems not only from their excellent scalability but also from large scale self-supervised pre-training (Devlin et al., 2019; Radford et al., 2018). We also perform a preliminary exploration on masked patch prediction for self-supervision, mimicking the masked language modeling task used in BERT. With self-supervised pre-training, our smaller ViT-B/16 model achieves 79.9% accuracy on ImageNet, a significant improvement of 2% to training from scratch, but still 4% behind supervised pre-training. Appendix B.1.2 contains further details. We leave exploration of contrastive pre-training (Chen et al., 2020b; He et al., 2020; Bachman et al., 2019; Henaff et al., 2020) to future work.

Transformers在NLP任务上得到了非常好的性能。但是，其成功不仅是因为其优秀的可扩展性，而且是来自于大规模自监督的预训练。我们还在掩膜图像块预测的自监督训练上进行了初步探索，模仿的是BERT中的掩膜语言建模任务。进行了自监督预训练后，我们较小的ViT-B/16模型在ImageNet上获得了79.9%的准确率，对从头训练来说，改进了显著的2%，对有监督预训练来说还差了4%。未来我们还要探索对比预训练。

## 5. Conclusion

We have explored the direct application of Transformers to image recognition. Unlike prior works using self-attention in computer vision, we do not introduce image-specific inductive biases into the architecture apart from the initial patch extraction step. Instead, we interpret an image as a sequence of patches and process it by a standard Transformer encoder as used in NLP. This simple, yet scalable, strategy works surprisingly well when coupled with pre-training on large datasets. Thus, Vision Transformer matches or exceeds the state of the art on many image classification datasets, whilst being relatively cheap to pre-train.

我们探索了Transformer在图像识别中的直接应用。与之前在计算机视觉中使用自注意力的工作不同，我们没有在架构中引入图像专用的归纳偏置，除了开始的图像块提取步骤。我们将图像解释为图像块序列，用NLP使用的标准Transformer编码器对其进行处理。这种简单但是可扩展的策略，在大型数据集上进行预训练时，效果非常好。因此，ViT在很多图像分类数据集上预目前最好的效果不相上下，甚至有所超越，而其训练代价相对较低。

While these initial results are encouraging, many challenges remain. One is to apply ViT to other computer vision tasks, such as detection and segmentation. Our results, coupled with those in Carion et al. (2020), indicate the promise of this approach. Another challenge is to continue exploring self-supervised pre-training methods. Our initial experiments show improvement from self-supervised pre-training, but there is still large gap between self-supervised and large-scale supervised pretraining. Finally, further scaling of ViT would likely lead to improved performance.

这些初始的效果是令人鼓舞的，但仍存在很多挑战。一个是将ViT应用到其他计算机视觉任务中，比如检测和分割。我们的结果与Carion等的结果一起，指出这种方法是有希望的。另一个挑战，是持续探索自监督预训练方法。我们的初始实验表明，对自监督预训练有改进，但在自监督和大规模有监督预训练之间仍有很大的差距。最后，ViT的进一步放大很可能进一步改进性能。