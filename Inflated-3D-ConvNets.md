# Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset 你往何处去，行为识别？新模型和Kinetics数据集

João Carreira et al.  Deepmind

## Abstract 摘要

The paucity of videos in current action classification datasets (UCF-101 and HMDB-51) has made it difficult to identify good video architectures, as most methods obtain similar performance on existing small-scale benchmarks. This paper re-evaluates state-of-the-art architectures in light of the new Kinetics Human Action Video dataset. Kinetics has two orders of magnitude more data, with 400 human action classes and over 400 clips per class, and is collected from realistic, challenging YouTube videos. We provide an analysis on how current architectures fare on the task of action classification on this dataset and how much performance improves on the smaller benchmark datasets after pre-training on Kinetics.

目前的行为分类数据(UCF-101和HMDB-51)视频量少，这使其很难识别好的视频架构，因为多数方法在现有的小型基准测试上得到的结果类似。本文在新的Kinetics人体行为视频数据集上重新评估了目前最好的架构。Kinetics的数据量多了2个数量级，有400个人体行为类别，每个类别超过400个视频片段，都是从实际的有挑战的Youtube视频中收集的。我们分析了现有的框架在这个数据集上进行行为分类的表现，还有在Kinetics上进行预预训练后，在更小型的数据集上表现的改进如何等。

We also introduce a new Two-Stream Inflated 3D ConvNet (I3D) that is based on 2D ConvNet inflation: filters and pooling kernels of very deep image classification ConvNets are expanded into 3D, making it possible to learn seamless spatio-temporal feature extractors from video while leveraging successful ImageNet architecture designs and even their parameters. We show that, after pre-training on Kinetics, I3D models considerably improve upon the state-of-the-art in action classification, reaching 80.9% on HMDB-51 and 98.0% on UCF-101.

我们还提出了一种新的Two-Stream Inflated 3D ConvNet(I3D)，这是基于2D卷积网络膨胀而成的：极其深图像分类卷积网络的滤波器核和池化核膨胀成3D的，使其能够从视频中不停的学习空域-时域特征提取器，同时利用成功的ImageNet架构设计甚至是其参数。我们展示了，在Kinetics上预训练后，I3D模型在目前最好的行为分类结果上改进了很多，在HMDB-51上达到了80.9%的准确率，在UCF-101上达到了98.0%。

## 1. Introduction 引言

One of the unexpected benefits of the ImageNet challenge has been the discovery that deep architectures trained on the 1000 images of 1000 categories, can be used for other tasks and in other domains. One of the early examples of this was using the fc7 features from a network trained on ImageNet for the PASCAL VOC classification and detection challenge [10, 23]. Furthermore, improvements in the deep architecture, changing from AlexNet to VGG-16, immediately fed through to commensurate improvements in the PASCAL VOC performance [25]. Since then, there have been numerous examples of ImageNet trained architectures warm starting or sufficing entirely for other tasks, e.g. segmentation, depth prediction, pose estimation, action classi-
fication.

ImageNet挑战的一个未曾预料到的收益是，在1000个类别上1000幅图像训练得到的深度架构，发现可以用于其他任务和其他领域中。这样的一个较早的例子是，使用在ImageNet上训练的网络中的fc7特征，用于PASCAL VOC分类和检测挑战[10,23]。而且，深度架构的改进，从AlexNet到VGG-16，也立刻在PASCAL VOC上得到相应的改进[25]。自那以后，出现了很多在ImageNet上训练出来的架构作为预热，或完全适用于其他任务，如分割，深度预测，姿态估计，行为分类。

In the video domain, it is an open question whether training an action classification network on a sufficiently large dataset, will give a similar boost in performance when applied to a different temporal task or dataset. The challenges of building video datasets has meant that most popular benchmarks for action recognition are small, having on the order of 10k videos.

在视频领域，在一个足够大的数据集上训练出来的行为分类网络，应用于不同的时域任务或数据集中时，是不是会得到类似的性能提升，这是一个尚未解决的问题。构建视频数据集的挑战，意味着多数用于行为识别的基准测试都是很小的，都只在10k视频的量级上。

In this paper we aim to provide an answer to this question using the new Kinetics Human Action Video Dataset [16], which is two orders of magnitude larger than previous datasets, HMDB-51 [18] and UCF-101 [29]. Kinetics has 400 human action classes with more than 400 examples for each class, each from a unique YouTube video.

本文中，我们的目标是给出这个问题的答案，使用的是新的Kinetics人体行为视频数据集[16]，这比之前的数据集至少大2个数量级，包括HMDB-51[18]和UCF-101[29]。Kinetics包括400个人体行为类别，每个类别超过400个样本，每一个都是从唯一的Youtube视频中得到的。

Our experimental strategy is to reimplement a number of representative neural network architectures from the literature, and then analyze their transfer behavior by first pretraining each one on Kinetics and then fine-tuning each on HMDB-51 and UCF-101. The results suggest that there is always a boost in performance by pre-training, but the extent of the boost varies significantly with the type of architecture. Based on these findings, we introduce a new model that has the capacity to take advantage of pre-training on Kinetics, and can achieves a high performance. The model termed a “Two-Stream Inflated 3D ConvNets” (I3D), builds upon state-of-the-art image classification architectures, but inflates their filters and pooling kernels (and optionally their parameters) into 3D, leading to very deep, naturally spatio-temporal classifiers. An I3D model based on Inception-v1 [13] obtains performance far exceeding the state-of-the-art, after pre-training on Kinetics.

我们的试验策略是，重新实现一些文献中有代表性的神经网络架构，在Kinetics上预训练每个模型，然后在HMDB-51和UCF-101上进行精调，分析其迁移行为。结果表明，通过预训练，总是可以得到性能提升，但随着架构类型的不同，提升的幅度很不一样。基于这些发现，我们提出了一个新的模型，可以利用在Kinetics上的预训练结果，得到很高的性能。模型名为Two-Stream Inflated 3D ConvNets(I3D)，基于目前最好的图像分类架构构建得到，但将其滤波器核和池化核（以及其参数）膨胀成3D结构，形成了非常深的、很自然的空域-时域分类器。基于Inception-v1[13]的I3D模型，在Kinetics上预训练后，得到的结果，远超现有的最好结果。

In our model comparisons, we did not consider more classic approaches such as bag-of-visual-words representations [6, 19, 22, 33]. However, the Kinetics dataset is publicly available, so others can use it for such comparisons.

在我们的模型对比中，我们没有考虑很经典的方法，如bag-of-visual-words表示[6,19,22,33]。但是，Kinetics是公开可用的，所以其他人也可以使用进行这样的对比。

The next section outlines the set of implemented action classification models. Section 3 gives an overview of the Kinetics dataset. Section 4 reports the performance of models on previous benchmarks and on Kinetics, and section 5 studies how well the features learned on Kinetics transfer to different datasets. The paper concludes with a discussion of the results.

下一节列举了实现的行为分类模型。第3部分给出了Kinetics数据集的概览。第4部分给出了模型在之前的基准测试和在Kinetics上的性能，第5部分研究了在Kinetics上学习到的特征迁移到其他数据集上性能如何。文章以结果的讨论进行总结结束。

Figure 1. A still from ‘Quo Vadis’ (1951). Where is this going? Are these actors about to kiss each other, or have they just done so? More importantly, where is action recognition going? Actions can be ambiguous in individual frames, but the limitations of existing action recognition datasets has meant that the best-performing video architectures do not depart significantly from single-image analysis, where they rely on powerful image classifiers trained on ImageNet. In this paper we demonstrate that video models are best pre-trained on videos and report significant improvements by using spatio-temporal classifiers pre-trained on Kinetics, a freshly collected, large, challenging human action video dataset.

图1. ‘Quo Vadis’(1951)中的一个画面。要向哪里去？这些演员要彼此亲吻，还是已经这样做过了？更重要的是，行为识别要向哪里去？在单帧视频中，行为是不明确的，但现有行为识别数据集的局限意味着，性能最好的视频架构与单帧图像分析的性能相差不远，而这都依赖于在ImageNet上训练出来的图像分类器。本文中，我们证明了，视频模型在视频中预训练出来，才能得到最好的结果；通过在Kinetics上进行预训练，并使用空域-时域分类器，得到了显著的改进；Kinetics是一个最新收集的、大型、非常有挑战的人体行为视频数据集。

## 2. Action Classification Architectures 行为分类架构

While the development of image representation architectures has matured quickly in recent years, there is still no clear front running architecture for video. Some of the major differences in current video architectures are whether the convolutional and layers operators use 2D (image-based) or 3D (video-based) kernels; whether the input to the network is just an RGB video or it also includes pre-computed optical flow; and, in the case of 2D ConvNets, how information is propagated across frames, which can be done either using temporally-recurrent layers such as LSTMs, or feature aggregation over time.

图像表示架构的发展在近年来迅速成熟，但对于视频却仍然没有清晰的前沿架构。目前的视频架构的主要区别是，卷积和层运算符使用的是2D（基于图像的）或3D（基于视频的）核；网络的输入只是RGB视频，或者还包括预先计算的光流；在2D卷积网络的情况下，信息是怎样在各帧之间传播的，这可以是使用时域的循环层如LSTM，或一定时间内累积的特征。

In this paper we compare and study a subset of models that span most of this space. Among 2D ConvNet methods, we consider ConvNets with LSTMs on top [5, 37], and two-stream networks with two different types of stream fusion [8, 27]. We also consider a 3D ConvNet [14, 30]: C3D [31].

本文中，我们比较研究了现有的大部分模型。在2D卷积网络方法中，我们首先考虑使用LSTMs的卷积网络[5,37]，和two-stream网络，两种不同类型的流融合[8,27]。我们还考虑了一种3D卷积网络[14,30]: C3D[31]。

As the main technical contribution, we introduce Two-Stream Inflated 3D ConvNets (I3D). Due to the high-dimensionality of their parameterization and the lack of labeled video data, previous 3D ConvNets have been relatively shallow (up to 8 layers). Here we make the observation that very deep image classification networks, such as Inception [13], VGG-16 [28] and ResNet [12], can be trivially inflated into spatio-temporal feature extractors, and that their pre-trained weights provide a valuable initialization. We also find that a two-stream configuration is still useful.

我们还提出了Two-Stream Inflated 3D ConvNets(I3D)，这是主要的技术贡献。之前的3D卷积网络，由于其参数维度很高，以及缺少标记的视频数据，其深度相对较浅。这里我们观察到，非常深的图像分类网络，如Inception[13], VGG-16[28]和ResNet[12]，可以轻松的膨胀成为空域-时域特征提取器，而其预训练的权重可以作为很好的初始化。我们还发现，two-stream配置也是非常有用的。

A graphical overview of the five types of architectures we evaluate is shown in figure 2 and the specification of their temporal interfaces is given in table 1. 我们评估的五种架构类型的概览如图2所示，其时域接口指标也在表1中给出。

Figure 2. Video architectures considered in this paper. K stands for the total number of frames in a video, whereas N stands for a subset of neighboring frames of the video. a)LSTM b)3D-ConvNet c)Two-Stream d)3D-Fused Two-Stream e)Two-Stream 3D ConvNet

Many of these models (all but C3D) have an Imagenet pre-trained model as a subcomponent. Our experimental strategy assumes a common ImageNet pre-trained image classification architecture as back bone, and for this we chose Inception-v1 with batch normalization [13], and morph it in different ways. The expectation is that with this back bone in common, we will be able to tease apart those changes that benefit action classification the most.

很多这些模型（除了C3D之外所有的）都有ImageNet预训练模型作为子组件。我们的试验策略假设有ImageNet预训练的图像分类架构作为骨干，所以我们选择带有BN的Inception-v1[13]，就用不同的方法使其变形。有了这个作为骨干，我们会可以梳理出这些变化中对行为分类最有用的这部分。

### 2.1. The Old I: ConvNet+LSTM

The high performance of image classification networks makes it appealing to try to reuse them with as minimal change as possible for video. This can be achieved by using them to extract features independently from each frame then pooling their predictions across the whole video [15]. This is in the spirit of bag of words image modeling approaches [19, 22, 33]; but while convenient in practice, it has the issue of entirely ignoring temporal structure (e.g. models can’t potentially distinguish opening from closing a door).

图像分类网络的高性能，使研究者很愿意将其重用在视频任务中，对其做尽量少的改变。可以将其用于单独提取每一帧的特征，然后对整个视频的预测进行池化[15]。这是词袋图像建模方法[19,22,33]的思想；虽然实践上很方便，但完全忽略了时域结构（如模型不能区分开门和关门）。

In theory, a more satisfying approach is to add a recurrent layer to the model [5, 37], such as an LSTM, which can encode state, and capture temporal ordering and long range dependencies. We position an LSTM layer with batch normalization (as proposed by Cooijmans et al. [4]) after the last average pooling layer of Inception-V1, with 512 hidden units. A fully connected layer is added on top for the classifier.

理论上，更令人满意的方法是给模型增加一个循环层[5,37]，比如LSTM，可以对状态编码，捕获时域次序和长程依赖关系。我们在Inception-V1的最后一个平均池化层后，放置一个带BN的LSTM层（正如Cooijmans等[4]提出的），有512个隐藏单元。分类器最后加入了全连接层。

The model is trained using cross-entropy losses on the outputs at all time steps. During testing we consider only the output on the last frame. Input video frames are subsampled by keeping one out of every 5, from an original 25 frames-per-second stream. The full temporal footprint of all models is given in table 1.

这个模型使用所有时间步骤上的输出上的交叉熵损失进行训练。在测试时，我们只考虑最后一帧的输出。输入视频帧进行了采样，从原始的25 FPS码流中每五帧保留了一帧。所有模型的完整时域覆盖如表1所示。

### 2.2. The Old II: 3D ConvNets

3D ConvNets seem like a natural approach to video modeling, and are just like standard convolutional networks, but with spatio-temporal filters. They have been explored several times, previously [14, 30, 31, 32]. They have a very important characteristic: they directly create hierarchical representations of spatio-temporal data. One issue with these models is that they have many more parameters than 2D ConvNets because of the additional kernel dimension, and this makes them harder to train. Also, they seem to preclude the benefits of ImageNet pre-training, and consequently previous work has defined relatively shallow custom architectures and trained them from scratch [14, 15, 30, 31]. Results on benchmarks have shown promise but have not been competitive with state-of-the-art, making this type of models a good candidate for evaluation on our larger dataset.

3D卷积网络似乎是视频建模的一种很自然的方法，就像标准卷积网络一样，但是使用的是空域-时域滤波器。之前有过几次探索[14,30,31,32]。它们有一种非常重要的特征：它们直接创建空域-时域数据的层次化表示。这些模型的一个问题是，它们比2D卷积网络的参数多很多，因为多了一个核维度，这使其很难训练。同时，这类模型还不能利用ImageNet预训练的好处，结果是，之前的工作都是定义的相对较浅的架构，都是从头训练的[14,15,30,31]。基准测试上的结果看起来很有希望，但与目前最好的水平还有距离，这类模型是在我们更大的数据集上进行评估的很好的候选。

For this paper we implemented a small variation of C3D [31], which has 8 convolutional layers, 5 pooling layers and 2 fully connected layers at the top. The inputs to the model are short 16-frame clips with 112 × 112-pixel crops as in the original implementation. Differently from [31] we used batch normalization after all convolutional and fully connected layers. Another difference to the original model is in the first pooling layer, we use a temporal stride of 2 instead of 1, which reduces the memory footprint and allows for bigger batches – this was important for batch normalization (especially after the fully connected layers, where there is no weight tying). Using this stride we were able to train with 15 videos per batch per GPU using standard K40 GPUs.

本文中，我们实现了C3D[31]的一个小变体，包含8个卷积层，5个池化层和在顶部有2个全连接层。这个模型的输入是16帧的视频片段，大小112×112，原始实现也是这个配置。与[31]不同的是，我们在所有BN和全连接层后都使用了BN。另一个不同是在原始模型的第一个池化层上，我们使用了时域步长2，而不是1，这减少了内存占用，使得批可以更大，这对于BN来说很重要（尤其是全连接后的BN，其中没有任何weight tying）。使用这个步长，我们可以在标准K40 GPU上，每个GPU每次批次训练15个视频。

### 2.3. The Old III: Two-Stream Networks

LSTMs on features from the last layers of ConvNets can model high-level variation, but may not be able to capture fine low-level motion which is critical in many cases. It is also expensive to train as it requires unrolling the network through multiple frames for backpropagation-through-time.

ConvNets最后层的特征上的LSTMs可以对高层次变化进行建模，但可能不能捕获细节的低层次运动，在很多情况下这非常重要。这种网络训练起来也很昂贵，因为需要对多帧展开网络，以对时间反向传播。

A different, very practical approach, introduced by Simonyan and Zisserman [27], models short temporal snapshots of videos by averaging the predictions from a single RGB frame and a stack of 10 externally computed optical flow frames, after passing them through two replicas of an ImageNet pre-trained ConvNet. The flow stream has an adapted input convolutional layer with twice as many input channels as flow frames (because flow has two channels, horizontal and vertical), and at test time multiple snapshots are sampled from the video and the action prediction is averaged. This was shown to get very high performance on existing benchmarks, while being very efficient to train and test.

一种不同的，非常使用的方法是由Simonyan and Zisserman [27]提出的，其对短视频片段的建模，是对一个RGB单帧进行预测，和10个另外计算的光流帧进行预测，将其通过两个ImageNet预训练的ConvNets，然后对这些预测进行平均。这个flow stream有一个修正的输入卷积层，其输入通道数为flow frames的2倍（因为flow有2个通道，竖直的和水平的），在测试时，从视频中取样多个片段，然后行为预测得到平均。这在目前的基准测试中得到很好的性能，训练和测试效率也很高。

A recent extension [8] fuses the spatial and flow streams after the last network convolutional layer, showing some improvement on HMDB while requiring less test time augmentation (snapshot sampling). Our implementation follows this paper approximately using Inception-V1. The inputs to the network are 5 consecutive RGB frames sampled 10 frames apart, as well as the corresponding optical flow snippets. The spatial and motion features before the last average pooling layer of Inception-V1 (5 × 7 × 7 feature grids, corresponding to time, x and y dimensions) are passed through a 3 × 3 × 3 3D convolutional layer with 512 output channels, followed by a 3 × 3 × 3 3D max-pooling layer and through a final fully connected layer. The weights of these new layers are initialized with Gaussian noise.

一个最近的扩展版[8]，在最后一个网络卷积层后，融合了空域和flow stream，在HMDB上得到了一些改进，同时在测试时也需要较少的扩充（snapshot sampling）。我们的实现大约也采用这篇文章的方法，使用Inception-V1。网络的输入是5个连续的RGB帧，每10帧取1帧，还有对应的光流片段。Inception-V1最后一个平均池化层之前的时域特征和运动特征通过一个3×3×3的3D卷积层，512个输出通道，然后是一个3×3×3的3D最大池化层，最后是一个全连接层。这些层的权重初始化为高斯噪声。

Both models, the original two-stream and the 3D fused version, are trained end-to-end (including the two-stream averaging process in the original model). 这两个模型，原始的two-stream和3D融合版，都是端到端学习的。

### 2.4. The New: Two-Stream Inflated 3D ConvNets

With this architecture, we show how 3D ConvNets can benefit from ImageNet 2D ConvNet designs and, optionally, from their learned parameters. We also adopt a two-stream configuration here – it will be shown in section 4 that while 3D ConvNets can directly learn about temporal patterns from an RGB stream, their performance can still be greatly improved by including an optical-flow stream.

在这个架构下，我们展示一下我们的3D ConvNets怎样从ImageNet 2D ConvNet设计中获益，以及从其学习到的参数中获益。我们还采用了two-stream配置，这会在第4部分中展示，3D ConvNets可以直接从RGB流中学习时域模式，其性能通过包含一个光流stream，可以得到很大的改善。

**Inflating 2D ConvNets into 3D**. A number of very successful image classification architectures have been developed over the years, in part through painstaking trial and error. Instead of repeating the process for spatio-temporal models we propose to simply convert successful image (2D) classification models into 3D ConvNets. This can be done by starting with a 2D architecture, and inflating all the filters and pooling kernels – endowing them with an additional temporal dimension. Filters are typically square and we just make them cubic – N × N filters become N × N × N.

**将2D ConvNet膨胀成3D的**。一些非常成功的图像分类框架已经提出了好几年了，某种程度上是通过痛苦的试错得到的。我们没有对时空模型重复这个过程，而是提出将成功的2D图像分类网络转换成3DConvNet。我们从一个2D架构开始，对所有的滤波器和池化核进行膨胀，赋予其另外的时间维度。滤波器通常是方形的，我们只是将其变成立体的，N × N的滤波器，变成N × N × N的。

**Bootstrapping 3D filters from 2D Filters**. Besides the architecture, one may also want to bootstrap parameters from the pre-trained ImageNet models. To do this, we observe that an image can be converted into a (boring) video by copying it repeatedly into a video sequence. The 3D models can then be implicitly pre-trained on ImageNet, by satisfying what we call the boring-video fixed point: the pooled activations on a boring video should be the same as on the original single-image input. This can be achieved, thanks to linearity, by repeating the weights of the 2D filters N times along the time dimension, and rescaling them by dividing by N . This ensures that the convolutional filter response is the same. Since the outputs of convolutional layers for boring videos are constant in time, the outputs of pointwise non-linearity layers and average and max-pooling layers are the same as for the 2D case, and hence the overall network response respects the boring-video fixed point. [21] studies other bootstrapping strategies.

**从2D滤波器到3D滤波器**。除了架构，我们还想利用ImageNet预训练模型的参数构造3D ConvNet的参数。一幅图像可以转化成无意义视频，即重复将其复制成视频序列。3D模型也可以在ImageNet上进行隐式的预训练，即满足我们称为无意义视频的固定点：在无意义视频上池化的激活，应当与原始的单幅图像输入一样。这可以将2D滤波器的权重沿着时间维度重复N次，将其除以N，以重新确定其尺度。这确保了卷积滤波器的响应是一样的。因为无意义视频的卷积层的输出在时间轴上是常数，pointwise非线性层、平均池化层、最大池化层和2D的情况是一样的，所以总体网络响应就是无意义视频的定点式的。[21]研究了其他bootstrapping策略。

**Pacing receptive field growth in space, time and network depth**. The boring video fixed-point leaves ample freedom on how to inflate pooling operators along the time dimension and on how to set convolutional/pooling temporal stride – these are the primary factors that shape the size of feature receptive fields. Virtually all image models treat the two spatial dimensions (horizontal and vertical) equally – pooling kernels and strides are the same. This is quite natural and means that features deeper in the networks are equally affected by image locations increasingly far away in both dimensions. A symmetric receptive field is however not necessarily optimal when also considering time – this should depend on frame rate and image dimensions. If it grows too quickly in time relative to space, it may conflate edges from different objects breaking early feature detection, while if it grows too slowly, it may not capture scene dynamics well.

**感受野在空间、时间和网络深度中的变化**。无意义视频定点使我们可以自由的沿着时间维度膨胀池化算子，也可以自由的设置卷积/池化的时间步长，这是形成特征感受野的基本因素。实际上，所有图像模型对待空间两个维度都一样（水平和垂直），池化核和步长都一样。这非常自然，意味着网络中更深的特征受到两个方向上的远距离位置的影响是一样的。考虑时间维度之后，对称的感受野并不一定是最优的，这应当考虑到帧率和图像维度。如果相对于空间，增加很迅速，会将不同目标的边缘混在一起，破坏早期的特征检测，而如果太慢的话，可能无法很好的捕获场景动态。

In Inception-v1, the first convolutional layer has stride 2, then there are four max-pooling layers with stride 2 and a 7 × 7 average-pooling layer preceding the last linear classification layer, besides the max-pooling layers in parallel Inception branches. In our experiments the input videos were processed at 25 frames per second; we found it helpful to not perform temporal pooling in the first two max-pooling layers (by using 1 × 3 × 3 kernels and stride 1 in time), while having symmetric kernels and strides in all other max-pooling layers. The final average pooling layer uses a 2 × 7 × 7 kernel. The overall architecture is shown in fig. 3. We train the model using 64-frame snippets and test using the whole videos, averaging predictions temporally.

在Inception-v1中，第一个卷积层的步长为2，然后有4个步长为2的最大池化层，和一个7×7的平均池化层，最后是一个线性分类层，另外还有并行的最大池化层分支。在我们的试验中，输入视频以25FPS进行处理，我们发现在前两个最大池化层中不进行时域池化会有帮助（使用1×3×3的核，时间步长为1），其他最大池化层的核和步长则是对称的了。最后的平均池化层使用一个2×7×7核。总体架构如图3所示。我们使用64帧片段来训练模型，使用整个视频进行测试，在时域上对预测进行平均。

Figure 3. The Inflated Inception-V1 architecture (left) and its detailed inception submodule (right). The strides of convolution and pooling operators are 1 where not specified, and batch normalization layers, ReLu’s and the softmax at the end are not shown. The theoretical sizes of receptive field sizes for a few layers in the network are provided in the format “time,x,y” – the units are frames and pixels. The predictions are obtained convolutionally in time and averaged.

**Two 3D Streams**. While a 3D ConvNet should be able to learn motion features from RGB inputs directly, it still performs pure feedforward computation, whereas optical flow algorithms are in some sense recurrent (e.g. they perform iterative optimization for the flow fields). Perhaps because of this lack of recurrence, experimentally we still found it valuable to have a two-stream configuration – shown in fig. 2(e) – with one I3D network trained on RGB inputs, and another on flow inputs which carry optimized, smooth flow information. We trained the two networks separately and averaged their predictions at test time.

**双3D流**。一个3D ConvNet可以从RGB输入中直接学习运动特征，但仍然是进行的纯前向计算，而光流算法在某种意义上是循环的（如，它们对流场进行迭代的优化）。可能是由于缺少循环性，试验中我们仍然发现双流配置很有用处，如图2(e)所示，一个I3D网络在RGB输入上进行训练，另一个在流输入上进行训练，承载的是优化的、光滑的流信息。我们独立训练两个网络，在测试时对其预测进行平均。

### 2.5. Implementation Details 实现细节

All models but the C3D-like 3D ConvNet use ImageNet-pretrained Inception-V1 [13] as base network. For all architectures we follow each convolutional layer by a batch normalization [13] layer and a ReLU activation function, except for the last convolutional layers which produce the class scores for each network.

除了C3D类的3D ConvNet之外，所有的模型都使用ImageNet预训练的Inception-V1[13]作为基础网络。对于所有的架构，我们在每个卷积层后都有一个BN层[13]和ReLU激活函数，除了最后一个卷积层，生成的是每个网络的类别分数。

Training on videos used standard SGD with momentum set to 0.9 in all cases, with synchronous parallelization across 32 GPUs for all models except the 3D ConvNets which receive a large number of input frames and hence require more GPUs to form large batches – we used 64 GPUs for these. We trained models on on Kinetics for 110k steps, with a 10x reduction of learning rate when validation loss saturated. We tuned the learning rate hyperparameter on the validation set of Kinetics. Models were trained for up to 5k steps on UCF-101 and HMDB-51 using a similar learning rate adaptation procedure as for Kinetics but using just 16 GPUs. All the models were implemented in TensorFlow [1].

在视频上的训练使用的是标准SGD，动量为0.9，在32个GPU上进行同步并行化（除了输入帧数量非常多，所以需要更多GPU训练大一些的批，这种情况下我们使用64 GPU）。我们在Kinetics训练模型110k步，当验证损失饱和时，我们就将学习率除以10。我们在Kinetics验证集上调整学习速率超参数。在UCF-101和HMDB-51上训练5k次的模型，也使用了类似的学习速率调整过程，但使用了16 GPUs。所有的模型都使用TensorFlow实现的。

Data augmentation is known to be of crucial importance for the performance of deep architectures. During training we used random cropping both spatially – resizing the smaller video side to 256 pixels, then randomly cropping a 224 × 224 patch – and temporally, when picking the starting frame among those early enough to guarantee a desired number of frames. For shorter videos, we looped the video as many times as necessary to satisfy each model’s input interface. We also applied random left-right flipping consistently for each video during training. During test time the models are applied convolutionally over the whole video taking 224 × 224 center crops, and the predictions are averaged. We briefly tried spatially-convolutional testing on the 256 × 256 videos, but did not observe improvement. Better performance could be obtained by also considering left-right flipped videos at test time and by adding additional augmentation, such as photometric, during training. We leave this to future work.

数据扩充对于深度架构的性能来说非常重要。在训练过程中，我们使用空间随机剪切和时间随机剪切，即把较小的视频边变为256像素，然后随机剪切成224×224的块，还有选择视频的初始帧，确保视频有合适数量的帧。对于更短的视频，我们重复视频很多次，以确保满足每个模型的输入条件。我们还在训练过程中进行了视频随机左右翻转。在测试时，对224×224中心剪切块的图像，进行整体的卷积，预测进行了平均。我们还在256×256的视频上进行了空间卷积测试，但没有发现明显改进。在测试时考虑左右翻转的视频，可以得到更好的表现，或在训练时加入更多的扩充，如photometric。我们将这个放在未来工作中。

We computed optical flow with a TV-L1 algorithm [38]. 我们使用TV-L1算法计算光流。

Table 1. Number of parameters and temporal input sizes of the models.

Method | Params | Training Input Frames | Training Temporal Footprint | Testing Input Frames | Testing Temporal Footprint
--- | --- | --- | --- | --- | ---
ConvNet+LSTM | 9M | 25rgb | 5s | 50rgb | 10s
3D-ConvNet | 79M | 16rgb | 0.64s | 240rgb | 9.6s
Two-Stream | 12M | 1rgb+10flow | 0.4s | 25rgb+250flow | 10s
3D-Fused | 39M | 5rgb+50flow | 2s | 25rgb+250flow | 10s
Two-Stream I3D | 25M | 64rgb+64flow | 2.56s | 250rgb+250flow | 10s

## 3. The Kinetics Human Action Video Dataset

The Kinetics dataset is focused on human actions (rather than activities or events). The list of action classes covers: Person Actions (singular), e.g. drawing, drinking, laughing, punching; Person-Person Actions, e.g. hugging, kissing, shaking hands; and, Person-Object Actions, e.g. opening presents, mowing lawn, washing dishes. Some actions are fine grained and require temporal reasoning to distinguish, for example different types of swimming. Other actions require more emphasis on the object to distinguish, for example playing different types of wind instruments.

Kinetics数据集关注人体动作（而不是行为或事件）。动作类别列表包括：单人动作，如画画，饮用，大笑，击打；人与人的动作，如拥抱，接吻，握手；人与物体的动作，如打开礼物，修理草坪，洗盘子。一些动作是细粒度的，需要时域推理以进行辨识，比如不同类型的游泳。其他动作需要更强调目标来区分，比如弹奏不同类型的乐器。

The dataset has 400 human action classes, with 400 or more clips for each class, each from a unique video, for a total of 240k training videos. The clips last around 10s, and there are no untrimmed videos. The test set consists of 100 clips for each class. A full description of the dataset and how it was built is given in [16].

数据集有400个人体动作类别，每个类别超过400个片段，每个都是从唯一的视频中截取的，总计有240k训练视频。视频片段大约10s，没有未修整的视频。测试集每个类别包括100个片段。完整的数据集描述以及怎样构建成的见[16]。

## 4. Experimental Comparison of Architectures

In this section we compare the performance of the five architectures described in section 2 whilst varying the dataset used for training and testing. 本节中，我们比较第2部分中描述的5种架构的表现，在不同的数据集上进行训练和测试。

Table 2 shows the classification accuracy when training and testing on either UCF-101, HMDB-51 or Kinetics. We test on the split 1 test sets of UCF-101 and HMDB-51 and on the held-out test set of Kinetics. There are several note-worthy observations. First, our new I3D models do best in all datasets, with either RGB, flow, or RGB+flow modalities. This is interesting, given its very large number of parameters and that UCF-101 and HMDB-51 are so small, and shows that the benefits of ImageNet pre-training can extend to 3D ConvNets.

表2给出了在UCF-101，HMDB-51或Kinetics上进行训练和测试的分类准确率。我们在UCF-101和HMDB-51的测试集分割上进行测试，在Kinetics的保留测试集上进行测试。有几个值得注意的观察结果。首先，我们新的I3D模型在所有数据集上表现都最好，不论是RGB模式，流模式，或RGB+流模式。这是很有趣的，因为其参数非常多，而UCF-101和HMDB-51数据集又非常小，说明了ImageNet预训练的好处可以扩展到3D ConvNet中。

Second, the performance of all models is far lower on Kinetics than on UCF-101, an indication of the different levels of difficulty of the two datasets. It is however higher than on HMDB-51; this may be in part due to lack of training data in HMDB-51 but also because this dataset was purposefully built to be hard: many clips have different actions in the exact same scene (e.g. “drawing sword” examples are taken from same videos as “sword” and “sword exercise”). Third, the ranking of the different architectures is mostly consistent.

第二，所有模型在Kinetics上的表现远低于在UCF-101上的表现，说明了这两个数据集难度不在一个层次上。但HMDB-51难度也相对较大；可能部分是因为在HMDB-51上缺少训练数据，也因为这个数据集故意构建的这么难：很多片段在相同的场景下都有不同的动作。第三，不同架构的排名基本一致。

Table 2. Architecture comparison: (left) training and testing on split 1 of UCF-101; (middle) training and testing on split 1 of HMDB-51; (right) training and testing on Kinetics. All models are based on ImageNet pre-trained Inception-v1, except 3D-ConvNet, a C3D-like [31] model which has a custom architecture and was trained here from scratch. Note that the Two-Stream architecture numbers on individual RGB and Flow streams can be interpreted as a simple baseline which applies a ConvNet independently on 25 uniformly sampled frames then averages the predictions.

Architecture | UCF-101 RGB | UCF-101 Flow | UCF-101 RGB+Flow | HMDB-51 RGB | HMDB-51 Flow | HMDB-51 RGB+Flow | Kinetics RGB | Kinetics FLow | Kinetics RGB+Flow
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
(a) LSTM | 81.0 | - | - | 36.0 | - | - | 63.6 | - | -
(b) 3D-ConvNet | 51.6 | - | - | 24.3 | - | - | 56.1 | - | -
(c) Two-Stream | 83.6 | 85.6 | 91.2 | 43.2 | 56.3 | 58.3 | 62.2 | 52.4 | 65.6
(d) 3D-Fused | 83.2 | 85.8 | 89.3 | 49.2 | 55.5 | 56.8 | - | - | 67.2
(e) Two-Stream I3D | 84.5 | 90.6 | 93.4 | 49.8 | 61.9 | 66.4 | 71.1 | 63.4 | 74.2

Additionally, two-stream architectures exhibit superior performance on all datasets, but the relative value of RGB and flow differs significantly between Kinetics and the other datasets. The contribution from flow alone, is slightly higher than that of RGB on UCF-101, much higher on HMDB-51, and substantially lower on Kinetics. Visual inspection of the datasets suggests that Kinetics has much more camera motion which may make the job of the motion stream harder. The I3D model seems able to get more out of the flow stream than the other models, however, which can probably be explained by its much longer temporal receptive field (64 frames vs 10 during training) and more integrated temporal feature extraction machinery. While it seems plausible that the RGB stream has more discriminative information – we often struggled with our own eyes to discern actions from flow alone in Kinetics, and this was rarely the case from RGB – there may be opportunities for future research on integrating some form of motion stabilization into these architectures.

另外，双流架构在所有数据集上都表现出了很好的性能，但RGB与光流的相对值在Kinetics与其他数据集上非常不一样。在UCF-101上，光流的贡献比RGB的贡献稍多一些，在HMDB-51上则是多很多，而在Kinetics上则是低很多。数据集的可视化检视说明，Kinetics里的摄像机运动很多，这可能使得运动光流的任务更难一些。I3D模型似乎比其他模型更能从光流中得到更多的信息，但是，模型的时域感受野更长（测试时64帧，训练时10帧），时域特征提取积分也很多，这可能可以解释。RGB流似乎有更具区分性的信息，但我们在Kinetics中要努力用眼睛才能辨别动作和光流本身，但在RGB的情况下则不是这样，未来或许会有研究，在这些架构中整合进入一些运动稳定的形式。

We also evaluated the value of training models in Kinetics starting from ImageNet-pretrained weights versus from scratch – the results are shown in table 3. It can be seen that ImageNet pre-training still helps in all cases and this is slightly more noticeable for the RGB streams, as would be expected.

我们还在Kinetics上评估了模型训练的价值，比较了ImageNet预训练的权重和从头训练的权重，结果如表3所示。可以看出，在所有情况下，ImageNet预训练的权重仍然效果比较好，RGB流更显著一些。

Table 3. Performance training and testing on Kinetics with and without ImageNet pretraining. Numbers in brackets () are the Top-5 accuracy, all others are Top-1.

Architecture | Kinetics RGB | Kinetics Flow | Kinetics RGB+Flow | ImageNet then Kinetics RGB | Flow | RGB+Flow
--- | --- | --- | --- | --- | --- | ---
(a)LSTM | 53.9 | - | - | 63.3 | - | -
(b)3D-ConvNet | 56.1 | - | - | - | - | -
(c)Two-Stream | 57.9 | 49.6 | 62.8 | 62.2 | 52.4 | 65.6
(d)3D-Fused | - | - | 62.7 | - | - | 67.2
(e)Two-Stream I3D | 68.4(88.0) | 61.5(83.4) | 71.6(90.0) | 71.1(89.3) | 63.4(84.9) | 74.2(91.3)

## 5. Experimental Evaluation of Features 试验评估的特征

In this section we investigate the generalizability of the networks trained on Kinetics. We consider two measures of this: first, we freeze the network weights and use the network to produce features for the (unseen) videos of the UCF-101/HMDB-51 datasets. We then train multi-way soft-max classifiers for the classes of UCF-101/HMDB-51 (using their training data), and evaluate on their test sets; Second, we fine-tune each network for the UCF-101/HMDB-51 classes (using the UCF-101/HMDB-51 training data), and again evaluate on the UCF-101/HMDB-51 test sets.

本节中，我们研究了在Kinetics上训练的网络的泛化能力。我们考虑两种度量：第一，我们冻结权重，然后用网络对UCF-101/HMDB-51数据集中未见过的视频进行提取特征。我们然后对UCF-101/HMDB-51中的类别训练多路softmax分类器（使用其训练数据），在其测试集上进行评估；第二，我们对UCF-101/HMDB-51的类别精调网络（使用UCF-101/HMDB-51训练数据），然后再次在UCF-101/HMDB-51测试集上进行评估。

We also examine how important it is to pre-train on ImageNet+Kinetics instead of just Kinetics. 我们还比较了在ImageNet+Kinetics上进行预训练和只在Kinetics上进行训练的区别。

The results are given in table 4. The clear outcome is that all architectures benefit from pre-training on the additional video data of Kinetics, but some benefit significantly more than the others – notably the I3D-ConvNet and 3D-ConvNet (although the latter starting from a much lower base). Training just the last layers of the models after pretraining in Kinetics (Fixed) also leads to much better performance than directly training on UCF-101 and HMDB-51 for I3D models.

结果如表4所示。明显的结果是，所有模型在Kinetics上更多的视频数据上进行预训练后性能都变得更好了，但一些模型好的程度明显比其他的好，特别是I3D-ConvNet和3D-ConvNet（虽然后者是从很低的基准上开始的）。在Kinetics上进行预训练后，只训练模型的最后几层，也比直接在UCF-101和HMDB-51上训练I3D模型，带来了更好的性能。

One explanation for the significant better transferability of features of I3D models is their high temporal resolution – they are trained on 64-frame video snippets at 25 frames per second and process all video frames at test time, which makes it possible for them to capture fine-grained temporal structure of actions. Stated differently, methods with sparser video inputs may benefit less from training on this large video dataset because, from their perspective, videos do not differ as much from the images in ImageNet. The difference over the C3D-like model can be explained by our I3D models being much deeper, while having much fewer parameters, by leveraging an ImageNet warm-start, by being trained on 4× longer videos, and by operating on 2×higher spatial resolution videos.

I3D有明显更好的迁移特征，其一个解释是时域分辨率很高，是以25fps的64帧视频片段进行训练的，在测试时处理所有的帧，这使得模型可以捕获动作的细粒度时域结构。换句话说，更稀疏的视频输入的方法，从这个大型视频数据集的训练中受益的会更少，因为从这些算法的角度来说，视频与ImageNet中的图像的差距没那么大。C3D类的模型的区别，可以说是I3D更深，而且参数更少，利用ImageNet进行预热，在4x长的视频中进行训练，在2x高空间分辨率视频上进行运算。

The performance of the two-stream models is surprisingly good even when trained from scratch (without ImageNet or Kinetics), mainly due to the accuracy of the flow stream, which seems much less prone to overfitting (not shown). Kinetics pretraining helps significantly more than ImageNet.

双流模型的性能，即使是从头训练，也是异常的好（不在ImageNet或Kinetics上预训练），主要是由于光流stream的准确率，似乎很不容易过拟合。Kinetics预训练比ImageNet预训练效果更好。

Table 4. Performance on the UCF-101 and HMDB-51 test sets (split 1 of both) for architectures starting with / without ImageNet pretrained weights. Original: train on UCF-101 or HMDB-51; Fixed: features from Kinetics, with the last layer trained on UCF-101 or HMDB-51; Full-FT: Kinetics pre-training with end-to-end fine-tuning on UCF-101 or HMDB-51.

Architecture | UCF-101 Original | UCF-101 Fixed | UCF-101 Full-FT | HMDB-51 Original | HMDB-51 Fixed | HMDB-51 Full-FT
--- | --- | --- | --- | --- | --- | ---
(a)LSTM | 81.0/54.2 | 88.1/82.6 | 91.0/86.8 | 36.0/18.3 | 50.8/47.1 | 53.4/49.7
(b)3D-ConvNet | -/51.6 | -/76.0 | -/79.9 | -/24.3 | -/47.0 | -/49.4
(c)Two-Stream | 91.2/83.6 | 93.9/93.3 | 94.2/93.8 | 58.3/47.1 | 66.6/65.9 | 66.6/64.3
(d)3D-Fused | 89.3/69.5 | 94.3/89.8 | 94.2/91.5 | 56.8/37.3 | 69.9/64.6 | 71.0/66.5
(e)Two-Stream I3D | 93.4/88.8 | 97.7/97.4 | 98.0/97.6 | 66.4/62.2 | 79.7/78.6 | 81.2/81.3

### 5.1. Comparison with the State-of-the-Art

We show a comparison of the performance of I3D models and previous state-of-the-art methods in table 5, on UCF-101 and HMDB-51. We include results when pretraining on the Kinetics dataset (with and without ImageNet pre-training). The conv1 filters of the trained models are shown in fig. 4.

我们在表5中给出I3D模型与之前最好的方法在UCF-101和HMDB-51上的性能比较，还包括了在Kinetics数据集上进行预训练的结果（在ImageNet上进行了预训练或不进行）。训练好的模型的conv1滤波器如图4所示。

Many methods get similar results, but the best performing method on these datasets is currently the one by Feichtenhofer and colleagues [7], which uses ResNet-50 models on RGB and optical flow streams, and gets 94.6% on UCF-101 and 70.3% on HMDB-51 when combined with the dense trajectories model [33]. We benchmarked our methods using the mean accuracy over the three standard train/test splits. Either of our RGB-I3D or RGB-Flow models alone, when pre-trained on Kinetics, outperforms all previous published performance by any model or model combinations. Our combined two-stream architecture widens the advantage over previous models considerably, bringing overall performance to 98.0 on UCF-101 and 80.9 on HMDB-51, which correspond to 63% and 35% misclassification reductions, respectively compared to the best previous model [7].

很多方法得到了类似的结果，但在这些数据集上目前表现最好的方法是Feichtenhofer等[7]提出的方法，使用的是ResNet-50模型在RGB和光流stream上，在与[33]的dense trajectories模型综合在一起时，在UCF-101上得到了94.6%，在HMDB-51上得到了70.3%。我们对我们的方法在三个标准训练/测试分割上使用平均准确率进行了基准测试。在Kinetics上进行预训练后，我们的单个RGB-I3D或RGB-Flow模型，超过了之前所有发表的模型或模型组合。我们的综合双流架构显著增加了与之前模型的差距，在UCF-101上的总体性能达到了98.0%，在HMDB-51上达到了80.9%，与之前最好的模型相比，分别对应着63%和35%的误分类降低[7]。

The difference between Kinetics pre-trained I3D models and prior 3D ConvNets (C3D) is even larger, although C3D is trained on more videos, 1M examples from Sports-1M plus an internal dataset, and even when ensembled and combined with IDT. This may be explainable by the better quality of Kinetics but also because of I3D simply being a better architecture.

Kinetics预训练的I3D模型，和之前的3D ConvNet(C3D)模型之间的差距甚至更大，虽然C3D在更多的视频上进行了训练，Sports-1M上的1M样本外加内部数据集，甚至与IDT结合形成集成模型。这可以通过Kinetics是一个更好的数据集解释，本身I3D也是一个更好的架构。

Figure 4. All 64 conv1 filters of each Inflated 3D ConvNet after training on Kinetics (the filter dimensions are 7 × 7 × 7, and the 7 time dimensions are shown left-to-right across the figure). The sequence on top shows the flow network filters, the one in the middle shows filters from the RGB I3D network, and the bottom row shows the original Inception-v1 filters. Note that the I3D filters possess rich temporal structure. Curiously the filters of the flow network are closer to the original ImageNet-trained Inception-v1 filters, while the filters in the RGB I3D network are no longer recognizable. Best seen on the computer, in colour and zoomed in.

Table 5. Comparison with state-of-the-art on the UCF-101 and HMDB-51 datasets, averaged over three splits. First set of rows contains results of models trained without labeled external data.

Model | UCF-101 | HMDB-51
--- | --- | ---
Two-Stream [27] | 88.0 | 59.4
IDT [33] | 86.4 | 61.7
Dynamic Image Networks + IDT [2] | 89.1 | 65.2
TDD + IDT [34] | 91.5 | 65.9
Two-Stream Fusion + IDT [8] | 93.5 | 69.2
Temporal Segment Networks [35] | 94.2 | 69.4
ST-ResNet + IDT [7] | 94.6 | 70.3
Deep Networks [15], Sports 1M pre-training | 65.2 | -
C3D one network [31], Sports 1M pre-training | 82.3 | -
C3D ensemble [31], Sports 1M pre-training | 85.2 | -
C3D ensemble + IDT [31], Sports 1M pre-training | 90.1 | -
RGB-I3D, Imagenet+Kinetics pre-training | 95.6 | 74.8
Flow-I3D, Imagenet+Kinetics pre-training | 96.7 | 77.1
Two-Stream I3D, Imagenet+Kinetics pre-training | 98.0 | 80.7
RGB-I3D, Kinetics pre-training | 95.1 | 74.3
Flow-I3D, Kinetics pre-training | 96.5 | 77.3
Two-Stream I3D, Kinetics pre-training | 97.8 | 80.9

## 6. Discussion 讨论

We return to the question posed in the introduction, “is there a benefit in transfer learning from videos?”. It is evident that there is a considerable benefit in pre-training on (the large video dataset) Kinetics, just as there has been such benefits in pre-training ConvNets on ImageNet for so many tasks. This demonstrates transfer learning from one dataset (Kinetics) to another dataset (UCF-101/HMDB-51) for a similar task (albeit for different action classes). However, it still remains to be seen if there is a benefit in using Kinetics pre-training for other video tasks such as semantic video segmentation, video object detection, or optical flow computation. We plan to make publicly available I3D models trained on the official Kinetics dataset’s release to facilitate research in this area.

我们回到在引言中提出的问题，“视频中的迁移学习是否有益处？” 很明显在大型视频数据集Kinetics上进行预训练是有相当的好处的，就像在ImageNet上预训练ConvNet在这么多任务上都有好处一样。这证明了从一个数据集(Kinetics)到另一个数据集(UCF-101/HMDB-51)在类似的任务上进行迁移学习是可行的。但是，使用Kinetics预训练对其他视频任务，如视频语义分割，视频目标检测，或光流计算是否有好处，仍然有待验证。我们计划公开在Kinetics数据集上训练的I3D模型，已帮助这个领域的研究。

Of course, we did not perform a comprehensive exploration of architectures – for example we have not employed action tubes [11, 17] or attention mechanisms [20] to focus in on the human actors. Recent works have proposed imaginative methods for determining the spatial and temporal extent (detection) of actors within the two-stream architectures, by incorporating linked object detections in time [24, 26]. The relationship between space and time is a mysterious one. Several very creative papers have recently gone out of the box in attempts to capture this relationship, for example by learning frame ranking functions for action classes and using these as a representation [9], by making analogies between actions and transformations [36], or by creating 2D visual snapshots of frame sequences [2] – this idea is related to the classic motion history work of [3]. It would be of great value to also include these models in our comparison but we could not, due to lack of time and space.

当然，我们没有全面综合的探索所有架构，比如，我们没有使用动作tubes[11,17]或注意力机制[20]以关注人体动作者。最近的工作提出了imaginative方法以确定动作者的空间和时间范围，使用的也是双流架构，将不同时间上的目标检测结果连接起来[24,26]。时间与空间的关系是很神秘的。最近有几篇很有创造力的文章，试图发现这种关系，比如对行为类别学习帧排序函数，将其作为一种表示[9]；在行为和变换之间做类比[36]；或生成帧序列的2D视觉快照[2]，这种思想与经典的运动历史工作[3]有一定关系。如果能将这些模型一并进行比较肯定非常有意义，但由于缺少时间和空间，我们没有这样做。