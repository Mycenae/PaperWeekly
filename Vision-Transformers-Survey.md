# Transformers in Vision: A Survey

Salman Khan et. al.

## 0. Abstract

Astounding results from Transformer models on natural language tasks have intrigued the vision community to study their application to computer vision problems. Among their salient benefits, Transformers enable modeling long dependencies between input sequence elements and support parallel processing of sequence as compared to recurrent networks e.g., Long short-term memory (LSTM). Different from convolutional networks, Transformers require minimal inductive biases for their design and are naturally suited as set-functions. Furthermore, the straightforward design of Transformers allows processing multiple modalities (e.g., images, videos, text and speech) using similar processing blocks and demonstrates excellent scalability to very large capacity networks and huge datasets. These strengths have led to exciting progress on a number of vision tasks using Transformer networks. This survey aims to provide a comprehensive overview of the Transformer models in the computer vision discipline. We start with an introduction to fundamental concepts behind the success of Transformers i.e., self-attention, large-scale pre-training, and bidirectional feature encoding. We then cover extensive applications of transformers in vision including popular recognition tasks (e.g., image classification, object detection, action recognition, and segmentation), generative modeling, multi-modal tasks (e.g., visual-question answering, visual reasoning, and visual grounding), video processing (e.g., activity recognition, video forecasting), low-level vision (e.g., image super-resolution, image enhancement, and colorization) and 3D analysis (e.g., point cloud classification and segmentation). We compare the respective advantages and limitations of popular techniques both in terms of architectural design and their experimental value. Finally, we provide an analysis on open research directions and possible future works. We hope this effort will ignite further interest in the community to solve current challenges towards the application of transformer models in computer vision.

Transformer模型在NLP上得到了令人震惊的好效果，这促使视觉团体研究其在计算机视觉中的应用。Transformer明显的优势是，对输入序列元素的长程依赖关系进行了建模，与循环网络如LSTM相比，支持序列的并行处理。与卷积网络不通的是，Transformer在设计上需要的改动很少，很自然的适合集合函数。而且，Transformer的直接设计允许处理多种模态（如，图像，视频，文本和语音），使用的是类似的处理模块，对非常大容量的网络和大型数据集有非常好的扩展性。这些优势使得采用Transformer在视觉任务的处理上有了非常多的进展。本综述的目的是概览Transformer模型在计算机视觉中的应用。我们以Transformer成功背后的基本概念开始，即，自注意力，大规模预训练，双向特征编码。然后我们覆盖了视觉中Transformer的多项应用，包括流行的识别任务（如分类，目标检测，行为识别，和分割），生成式建模，多模态任务（如，视觉问题回答，视觉推理，和视觉grounding），视频处理（如，行为识别，视频预测），低层视觉（如，图像超分辨，图像增强和上色）和3D分析（如，点云分类和分割）。我们比较了流行技术的各自优势和局限，包括架构设计和试验结果。最后，我们给出了研究方向的分析。我们希望本文会点燃未来研究的兴趣。

**Index Terms**  Self-attention, transformers, bidirectional encoders, deep neural networks, convolutional networks, self-supervision.

## 1. Introduction

Transformer models [1] have recently demonstrated exemplary performance on a broad range of language tasks e.g., text classification, machine translation [2] and question answering. Among these models, the most popular ones include BERT (Bidirectional Encoder Representations from Transformers) [3], GPT (Generative Pre-trained Transformer) v1-3 [4]–[6], RoBERTa (Robustly Optimized BERT Pre-training) [7] and T5 (Text-to-Text Transfer Transformer) [8]. The profound impact of Transformer models has become more clear with their scalability to very large capacity models [9], [10]. For example, the BERT-large [3] model with 340 million parameters was significantly outperformed by the GPT-3 [6] model with 175 billion parameters while the latest mixture-of-experts Switch transformer [10] scales up to a whopping 1.6 trillion parameters!

Transformer模型最近在很多语言任务中得到了非常好的性能，如，文本分类，机器翻译和问题回答。在这些模型中，最流行的包括BERT，GPT v1-v3，RoBERTa和T5.Transformer模型的深远影响随着扩展到非常大的模型，也变得越来越明显。比如，3.4亿参数的BERT模型的性能，被1750亿参数的GPT-3模型明显超过了，而最近的专家混合Switch transformer扩展到了1.6万亿参数。

The breakthroughs from Transformer networks in Natural Language Processing (NLP) domain has sparked great interest in the computer vision community to adapt these models for vision and multi-modal learning tasks (Fig. 1). However, visual data follows a typical structure (e.g., spatial and temporal coherence), thus demanding novel network designs and training schemes. As a result, Transformer models and their variants have been successfully used for image recognition [11], [12], object detection [13], [14], segmentation [15], image super-resolution [16], video understanding [17], [18], image generation [19], text-image synthesis [20] and visual question answering [21], [22], among several other use cases [23]–[26]. This survey aims to cover such recent and exciting efforts in the computer vision domain, providing a comprehensive reference to interested readers.

Transformer网络在NLP领域中的突破，使计算机视觉研究者很感兴趣，于是将这种模型修改以用于视觉和多模态学习任务（图1）。但是，视觉数据有一种典型的结构（如，空间和时间的连续性），因此需要新的网络设计和训练方案。结果是，Transformer模型及其变体成功的用于了图像识别，目标检测，分割，图像超分辨，视频理解，图像生成，文本图像合成和视觉问答，以及其他几种情况中。本文的目的是回顾计算机视觉领域中的这些工作。

Transformer architectures are based on a self-attention mechanism that learns the relationships between elements of a sequence. As opposed to recurrent networks that process sequence elements recursively and can only attend to short-term context, transformer architectures can attend to complete sequences thereby learning long-range relationships and can be easily parallelized. An important feature of these models is their scalability to very high-complexity models and large-scale datasets. Since Transformers assume minimal prior knowledge about the structure of the problem as compared to their convolutional and recurrent counterparts in deep learning [27]–[29], they are typically pre-trained using pretext tasks on large-scale (unlabelled) datasets [1], [3]. Such a pre-training avoids costly manual annotations, thereby encoding highly expressive and generalizable representations that model rich relationships between the entities present in a given dataset. The learned representations are then fine-tuned on the downstream tasks in a supervised manner to obtain favorable results.

Transformer架构是基于自注意力机制的，学习了序列元素之间的关系。循环网络迭代的处理序列元素，只能关注到短程的上下文，Transformer架构可以关注到整个序列，因此学习了长程关系，可以很容易的并行化。这些模型的一个重要特征，是其可扩展到非常复杂模型和大型数据集的能力。与深度学习中的卷积网络和循环网络相比，Transformer对问题结构的先验知识有最小的假设，它们通常使用pretext任务在大规模未标注数据集上进行预训练。这样一种预训练避免了昂贵的手工标注，编码了非常具有表达力和可泛化的表示，对给定数据集中的实体的丰富关系进行了建模。学习到的表示然后在下游任务中，以有监督的方式进行精调，得到较好的结果。

This paper provides a holistic overview of the transformer models developed for computer vision applications. We develop a taxonomy of the network design space and highlight the major strengths and shortcomings of the existing methods. Other literature reviews mainly focus on the NLP domain [30], [31] or cover generic attention-based approaches [30], [32]. By focusing on the newly emerging area of visual transformers, we comprehensively organize the recent approaches according to the intrinsic features of self-attention and the investigated task. We first provide an introduction to the salient concepts underlying Transformer networks and then elaborate on the specifics of recent vision transformers. Where ever possible, we draw parallels between the Transformers used in the NLP domain [1] and the ones developed for vision problems to flash major novelties and interesting domain-specific insights. Recent approaches show that convolution operations can be fully replaced with attention-based transformer modules and have also been used jointly in a single design to encourage symbiosis between the two complementary set of operations. This survey finally details open research questions with an outlook towards the possible future work.

本文给出了Transformer模型在计算机视觉应用的完全概览。我们对网络设计进行了分类，强调了现有方法的主要优势和问题。其他的综述文章主要关注的是NLP领域，或关注的是基于注意力机制的通用方法。我们聚焦在新出现的视觉Transformers，我们对最近出现的方法进行了综合组织，根据的是其自注意力的内在特征，和研究的任务。我们首先给出了Transformer网络下的主要概念，列举出了最近视觉Transformer的主要类型。只要可行，我们就也给出在NLP领域中的Transformer，和视觉问题中的进行比较，发现其主要创新点，和感兴趣的领域相关思考。最近的方法表明，卷积运算可以完全被基于注意力的Transformer模块所替代，也在一些中共同使用，以鼓励两种互补运算的共生。本文最后详述了开放的研究问题，展望了未来的可能工作。

## 2. Foudations

There exist two key ideas that have contributed towards the development of transformer models. (a) The first one is self-attention, which allows capturing ‘long-term’ information and dependencies between sequence elements as compared to conventional recurrent models that find it challenging to encode such relationships. (b) The second key idea is that of pre-training on a large (un)labelled corpus in a (un)supervised manner, and subsequently fine-tuning to the target task with a small labeled dataset [3], [7], [33]. Below, we provide a brief tutorial on these two ideas (Sec. 2.2 and 2.1), along with a summary of seminal Transformer networks (Sec. 2.3 and 2.4) where these ideas have been applied. This background will help us better understand the forthcoming Transformer based models used in the computer vision domain (Sec. 3).

Transformer模型的提出，有两个关键思想贡献最大。(a)第一个是自注意力，与传统的循环模型相比，允许捕获序列元素间的长程信息和依赖关系，而传统模型则很难编码这种关系；(b)第二个关键思想是，在大型无标注语料上，以无监督的方式进行预训练，然后到目标任务上用小型标注数据集进行精调。下面，我们给出这两个思想的简要教程，也总结了一些基础Transformer网络，应用了这些思想。这个背景会帮助我们更好的理解用于计算机视觉领域中的基于Transformer模型。

### 2.1 Self-Attention

Given a sequence of items, self-attention mechanism estimates the relevance of one item to other items (e.g., which words are likely to come together in a sentence). The self-attention mechanism is an integral component of Transformers, which explicitly models the interactions between all entities of a sequence for structured prediction tasks. Basically, a self-attention layer updates each component of a sequence by aggregating global information from the complete input sequence.

给定一个元素序列，自注意力机制估计一个元素与其他元素的关联性（如，哪些词语更可能组成一个语句）。自注意力机制是Transformer不可或缺的组成部分，显式的对一个序列中的所有实体之间的相互作用进行了建模，应用于结构化的预测任务。基本上，自注意力层通过从整个输入序列中聚积全局信息，来更新每个部分。

Let us denote a sequence of n entities ($x_1, x_2, ..., x_n$) by $X ∈ R^{n×d}$, where d is the embedding dimension to represent each entity. The goal of self-attention is to capture the interaction amongst all n entities by encoding each entity in terms of the global contextual information. This is done by defining three learnable weight matrices to transform Queries ($W^Q ∈R^{n×d_q}$), Keys ($W^K ∈R^{n×d_k}$) and Values ($W^V ∈R^{n×d_v}$). The input sequence X is first projected onto these weight matrices to get $Q = XW^Q, K = XW^K$ and $V = XW^V$. The output $X ∈ R^{n×d_v}$ of the self attention layer is then given by,

我们将n个实体的序列($x_1, x_2, ..., x_n$)表示为$X ∈ R^{n×d}$，其中d是表示每个实体的嵌入维度。自注意力的目标是捕获所有n个实体间的相互作用，方法是对每个实体的全局上下文信息进行编码。通过定义三个可学习的权重矩阵来实现，即Queries ($W^Q ∈R^{n×d_q}$), Keys ($W^K ∈R^{n×d_k}$)和Values ($W^V ∈R^{n×d_v}$)。输入序列X首先投影到这些权重矩阵上，得到$Q = XW^Q, K = XW^K$ and $V = XW^V$。自注意力曾的输出$X ∈ R^{n×d_v}$由下式给出：

$$Z = softmax(\frac {QK^T} {\sqrt{d_q}})V$$

For a given entity in the sequence, the self-attention basically computes the dot-product of the query with all keys, which is then normalized using softmax operator to get the attention scores. Each entity then becomes the weighted sum of all entities in the sequence, where weights are given by the attention scores (Fig. 2 and Fig. 3, top row-left block).

对于序列中的给定实体，自注意力基本上计算的就是query与所有keys的点乘，然后使用softmax算子进行归一化，以得到注意力值。每个实体然后就变成了序列中所有实体的加权和，其中权重是由注意力分数给定的。

**Masked Self-Attention**: The standard self-attention layer attends to all entities. For the Transformer model [1] which is trained to predict the next entity of the sequence, the self-attention blocks used in the decoder are masked to prevent attending to the subsequent future entities. This is simply done by an element-wise multiplication operation with a mask $M ∈ R^{n×n}$, where M is an upper-triangular matrix. The masked self-attention is defined by,

掩模自注意力：标准的自注意力层会注意到所有实体。对于训练用于预测序列下一个实体的Transformer模型，解码器中所用的自注意力模块被掩模，以防止注意到后续的未来实体。这通过与掩模$M ∈ R^{n×n}$逐元素的相乘运算完成，其中M是一个上三角矩阵。掩模的自注意定义为

$$softmax(\frac {QK^T} {\sqrt{d_q}} ◦ M)$$

where ◦ denotes Hadamard product. Basically, while predicting an entity in the sequence, the attention scores of the future entities are set to zero in masked self-attention. 其中◦表示Hadamard积。基本上，在预测序列中的一个实体时，未来实体的注意力分数在掩模自注意力中是设为0的。

**Multi-Head Attention**: In order to encapsulate multiple complex relationships amongst different elements in the sequence, the multi-head attention comprises multiple self-attention blocks (h = 8 in the original Transformer model [1]). Each block has its own set of learnable weight matrices {$W^{Q_i}, W^{K_i}, W^{V_i}$}, where i = 0 · · · (h−1). For an input X, the output of the h self-attention blocks in multi-head attention is then concatenated into a single matrix $[Z_0, Z_1, · · · Z_{h−1}] ∈ R^{n×h·d_v}$ and projected onto a weight matrix $W ∈ R^{h·d_v×d}$ (Fig. 3, top row).

**多头注意力**：为封装序列中不同元素的多个复杂关系，多头注意力由多个自注意力模块组成（在原始Transformer模型中h=8）。每个模块都有其自己的可学习权重矩阵集{$W^{Q_i}, W^{K_i}, W^{V_i}$}，其中i = 0 · · · (h−1)。对于输入X，多头注意力中的h个自注意力模块的输出，然后拼接成单个矩阵$[Z_0, Z_1, · · · Z_{h−1}] ∈ R^{n×h·d_v}$，投影到一个权重矩阵$W ∈ R^{h·d_v×d}$上。

The main difference of self-attention with convolution operation is that the weights are dynamically calculated instead of static weights (that stay the same for any input) as in the case of convolution. Further, self-attention is invariant to permutations and changes in the number of input points. As a result, it can easily operate on irregular inputs as opposed to standard convolution that requires grid structure.

自注意力与卷积运算的主要区别是，权重是动态计算得到的，而不是静态权重（对于任意输入都是一样的），卷积就是静态权重的情况。而且，自注意力对于排列组合和输入点的数量变化来说是不变的。结果是，可以很容易的在不规则输入上进行运算，而标准卷积需要网格的输入结构。

### 2.2 (Un)Supervised Pre-training

Self-attention based Transformer models generally operate in a two-stage training mechanism. First, pre-training is performed on a large-scale dataset (and sometimes a combination of several available datasets [22], [35]) in either a supervised [11] or an unsupervised manner [3], [36], [37]. Later, the pre-trained weights are adapted to the downstream tasks using small-mid scale datasets. Examples of downstream tasks include image classification [38], object detection [13], zero-shot learning [20], question-answering [10] and action recognition [18]. The effectiveness of pretraining for large-scale Transformers has been advocated in both the language and vision domains. For example, Vision Transformer model (ViT-L) [11] experiences an absolute 13% drop in accuracy on ImageNet test set when trained only on ImageNet train set as compared to the case when pretrained on JFT dataset [39] with 300 million images.

基于自注意力的Transformer模型一般采用两阶段的训练机制。首先，在大型数据集上（有时候是几个可用数据集的组合）以有监督或无监督的方式进行预训练。然后，预训练的权重在后续任务中，使用小规模或中规模的数据集进行精调。下游任务的例子包括，图像分类，目标检测，zero-shot学习，问题回答和行为识别。大规模Transformer预训练的有效性，在语言和视觉领域中都得到了验证。比如，ViT-L只在ImageNet训练集上进行训练，与在3亿图像的JFT数据集上进行预训练相比，在ImageNet测试集上的准确率下降了13%。

Since acquiring manual labels at a massive scale is cumbersome, self-supervised learning has been very effectively used in the pre-training stage. The self-supervision based pre-training stage training has played a crucial role in unleashing the scalability and generalization of Transformer networks, enabling training even above a trillion parameter networks (e.g., the latest Switch Transformer [10] from Google). An extensive survey on SSL can be found in [40], [41]. As nicely summarized by Y. LeCun [42], the basic idea of SSL is to fill in the blanks, i.e., try to predict the occluded data in images, future or past frames in temporal video sequences or predict a pretext task e.g., the amount of rotation applied to inputs, the permutation applied to image patches or the color of a gray-scale image. Another effective way to impose self-supervised constraints is via contrastive learning. In this case, nuisance transformations are used to create two types of modified versions of the same image i.e., without changing the underlying class semantics (e.g., image stylizing, cropping) and with semantic changes (e.g., replacing an object with another in the same scene, or changing the class with minor adversarial changes to the image). Subsequently, the model is trained to be invariant to the nuisance transformations and emphasize on modeling minor changes that can alter semantic labels.

由于大规模获得人工标签是很困难的，自监督学习在预训练阶段得到了有效使用。基于自监督的预训练，在揭示Transformer网络的可扩展性和泛化性上，有很关键的作用，使训练超过10亿参数的网络成为可能（如，Google最新的Switch Transformer）。SSL的综述见[40,41]。Y. LeCun [42]进行了很好的总结，SSL的基本思想是填空，即预测图像的遮挡数据，视频序列中的未来帧或过去帧，或预测一个pretext任务，如，对输入的旋转角度，对图像块的排列，或灰度图像的色彩。另一种进行自监督约束的有效方式是，通过对比学习。在这种情况下，用几种变换来创建一幅图像的几种版本，即，不变换类别语义（如，图像风格化，剪切），变换语义（如，在同样场景中将一个目标替换成另一个，或用很小的对抗变化来改变类别）。然后，模型进行训练，以对这些变换获得不变性，以及强调对变换语义标签的很小的变化进行建模。

Self-supervised learning provides a promising learning paradigm since it enables learning from a vast amount of readily available non-annotated data. In the SSL based pretraining stage, a model is trained to learn a meaningful representation of the underlying data by solving a pretext task. The pseudo-labels for the pretext task are automatically generated (without requiring any expensive manual annotations) based on data attributes and task definition. Therefore, the pretext task definition is a critical choice in SSL. We can broadly categorize existing SSL methods based upon their pretext tasks into (a) generative approaches which synthesize images or videos (given conditional inputs), (b) context-based methods which exploit the relationships between image patches or video frames, and (c) cross-modal methods which leverage from multiple data modalities. Examples of generative approaches include conditional generation tasks such as masked image modeling [35] and image colorization [43], image super-resolution [44], image in-painting [45], and GANs based methods [46], [47]. The context-based pretext methods solve problems such as a jigsaw puzzle on image patches [48]–[50], masked object classification [22], predict geometric transformation such as rotation [38], [51], or verify temporal sequence of video frames [52]–[54]. Cross-modal pretext methods verify the correspondence of two input modalities e.g., text & image [55], audio & video [56], [57] or RGB & flow [58].

自监督学习给出一个有希望的学习范式，因为是从大量可用的无标注数据中进行学习。在基于SSL的预训练阶段，通过求解一个pretext任务，训练一个模型以学习数据的有意义表示。pretext任务的伪标签是基于数据属性和任务定义，自动生成的（不需要昂贵的手工标注）。因此，pretext任务定义是SSL中的一个关键选择。我们可以将现有的SSL方法，基于其pretext任务分类成(a)生成式方法，合成图像或视频（给定条件输入）；(b)基于上下文的方法，探索图像块或视频帧之间的关系，(c)跨模态的方法，利用多数据模态。生成式方法的例子包括条件生成任务，如掩模图像建模，图像上色，图像超分辨，图像修补，和基于GANs的方法。基于上下文的pretext方法求解的问题包括，图像块的拼图，掩模目标分类，预测几何变换如旋转，验证视频帧的时序。跨模态的pretext方法验证两个输入模态的对应性，如，文字和图像，音频和视频，RGB和flow。

### 2.3 Transformer Model

The architecture of the Transformer model proposed in [1] is shown in Fig. 3. It has an encoder-decoder structure. The encoder (middle row) consists of six identical blocks (i.e., N=6 in Fig. 3), with each block having two sub-layers: a multi-head self-attention network, and a simple position-wise fully connected feed-forward network. Residual connections [59] alongside layer normalization [60] are employed after each block as in Fig. 3. Note that, different from regular convolutional networks where feature aggregation and feature transformation are simultaneously performed (e.g., with a convolution layer followed by a non-linearity), these two steps are decoupled in the Transformer model i.e., self-attention layer only performs aggregation while the feed-forward layer performs transformation. Similar to the encoder, the decoder (bottom row) in the Transformer model comprises six identical blocks. Each decoder block has three sub-layers, first two (multi-head self-attention, and feed-forward) are similar to the encoder, while the third sublayer performs multi-head attention on the outputs of the corresponding encoder block, as shown in Fig. 3.

[1]中提出的Transformer模型的架构如图3所示，是一种编码器-解码器结构。编码器（中间行）包含6个相同的模块（即N=6），每个模块都有两个子层：一个多头自注意力网络，和一个简单的逐位置全连接前向网络。在每个模块后都采用了残差连接与层归一化。注意，常规卷积网络中，特征聚积和特征变换是同时进行的（如，卷积层后带有一个非线性处理），与之不同的是，这两个步骤在Transformer模型中是分开的，即，自注意力层只进行聚积，而前向层进行变换。与编码器类似，Transformer模型中的解码器（下行）也有6个相同的模块组成。每个解码器模块有三个子层，前两个（多头自注意力，和前向传播）与编码器类似，而第三个子层在对应的编码器模块上进行多头注意力运算，如图3所示。

The original Transformer model in [1] was trained for the Machine Translation task. The input to the encoder is a sequence of words (sentence) in one language. Positional encodings are added to the input sequence to capture the relative position of each word in the sequence. Positional encodings have the same dimensions as the input d = 512, and can be learned or pre-defined e.g., by sine or cosine functions. Being an auto-regressive model, the decoder of the Transformer [1] uses previous predictions to output the next word in the sequence. The decoder, therefore, takes inputs from the encoder as well as the previous outputs to predict the next word of the sentence in the translated language. To facilitate residual connections the output dimensions of all layers are kept the same i.e., d = 512. The dimensions of query, key and value weight matrices in multi-head attention are set to dq = 64, dk = 64, dv = 64.

[1]中的原始Transformer模型训练用于机器翻译任务。编码器的输入是一种语言的词语序列（语句）。位置编码加入到输入序列中，以捕获序列中每个词语的相对位置。位置编码与输入相比有相同的维度，d=512，可以学习或预先指定，如用sin或cos函数。Transformer的解码器是一个自回归模型，使用之前的预测来输出序列中的下一个词语。解码器的输入因此是编码器的输出，和之前的输出，在翻译的语言中预测语句的下一个词语。为使用残差连接，所有层的输出维度都是一样的，即d=512。多头注意力中，Query，key和value权重矩阵的维度设置为dq = 64, dk = 64, dv = 64。

### 2.4 Bidirectional Representations 双向表示

The training strategy of the original Transformer model [1] could only attend to the context on the left of a given word in the sentence. This is limiting, since for most language tasks, contextual information from both left and right sides is important. Bidirectional Encoder Representations from Transformers (BERT) [3] proposed to jointly encode the right and left context of a word in a sentence, thus improving the learned feature representations for textual data in an unsupervised manner. To enable bidirectional training, [3] basically introduced two pretext tasks: Masked Language Model and Next Sentence Prediction. The model pre-trained on these pretext tasks in an unsupervised manner was then fine-tuned for the downstream task. For this purpose, task-specific additional output module is appended to the pre-trained model, and the full model is fine-tuned end-to-end.

原始Transformer模型的训练策略，只能注意到语句中给定词语的左边的上下文。这是一种局限，因为多数语言任务中，左边和右边的上下文信息都是重要的。BERT就是用于语句中一个词语左右上下文的同时编码，因此改进了学习到的特征表示，用于文本任务，以无监督的方式进行。为进行双向训练，BERT引入了两个pretext任务：掩模语言模型和下一个语句预测。在这些pretext任务上以无监督的方式预训练的模型，然后进行精调用于下游任务。为此，特定任务的额外输出模块接到预训练模型后，整个模型进行端到端的精调。

The network architecture of the base BERT [3] model is based upon the original Transformer model proposed in [1] and is similar to the GPT [4]. The main architectural difference compared to [1] is that the BERT model only uses the Transformer encoder (similar to the middle row, Fig. 3) while the GPT [4] only uses the Transformer decoder (similar to the bottom row, Fig. 3). The core contribution of BERT [3] is the pretext task definition, which enables bidirectional feature encoding in an unsupervised manner. To this end, BERT [3] proposed two strategies: (1) Masked Language Model (MLM) - A fixed percentage (15%) of words in a sentence are randomly masked and the model is trained to predict these masked words using cross-entropy loss. In predicting the masked words, the model learns to incorporate the bidirectional context. (2) Next Sentence Prediction (NSP) - Given a pair of sentences, the model predicts a binary label i.e., whether the pair is valid from the original document or not. The training data for this can easily be generated from any monolingual text corpus. A pair of sentences A and B is formed, such that B is the actual sentence (next to A) 50% of the time, and B is a random sentence for other 50% of the time. NSP enables the model to capture sentence-to-sentence relationships which are crucial in many language modeling tasks such as Question Answering and Natural Language Inference.

基础BERT模型的网络架构是基于原始Transformer[1]模型的，与GPT类似。与[1]相比主要的架构差异是，BERT模型只用了Transformer的编码器（与图3中的中间行类似），而GPT只使用了Transformer的解码器（与图3中的下行类似）。BERT的核心贡献是pretext任务的定义，可以以无监督的方式进行双向特征编码。为此，BERT提出了两个策略：(1)掩模语言模型MLM，一个语句中固定比例(15%)的词语进行随机掩模，模型进行训练以预测掩模的词语，使用的是交叉熵损失。在预测掩模词语中，模型学习利用双向上下文；(2)下一个词语预测NSP，给定一对语句，模型预测一个二值标签，即这一对对于原始文档是否是有效的。训练数据可以很容易生成。语句对A和B生成后，50%的情况下，B是A相邻的实际语句，另50%是随机语句。NSP使模型可以捕获语句到语句的关系，在很多语言建模中是非常关键的，比如问题回答和自然语言推理。

## 3 Transformers & Self-Attention in Vision

We provide an overview of main themes followed in Transformers designed for vision applications in Fig. 4. Existing frameworks generally apply global or local attention, leverage CNN representations or utilize matrix factorization to enhance design efficiency and use vectorized attention models. We explain these research directions below in the form of task-specific groups of approaches.

图4给出了用Transformer设计的视觉应用的概览。现有的框架一般使用全局注意力或局部注意力，利用CNN表示或利用矩阵分解来增强设计效率，使用矢量化的注意力模型。我们下面解释了这些研究方向。

### 3.1 Transformers for Image Recognition

Convolution operation is the work-horse of the conventional deep neural networks used in computer vision and it brought breakthroughs such as solving complex image recognition tasks on high dimensional datasets like ImageNet [61]. However, convolution also comes with its shortcomings e.g., it operates on a fixed-sized window thus unable to capture long-range dependencies such as arbitrary relations between pixels in both spatial and time domains in a given video. Furthermore, convolution filter weights remain fixed after training so the operation cannot adapt dynamically to any variation to the input. In this section, we review methods that alleviate the above-mentioned issues in conventional deep neural networks by using Self-attention operations and Transformer networks (a specific form of self-attention). There are two main design approaches to self-attention. (a) Global self-attention which is not restricted by the size of input features e.g., [62] introduces a layer inspired from non-local means that applies attention to the whole feature map while [63] reduces the computational complexity of non-local operation [62] by designing sparse attention maps. (b) Local self-attention tries to model relation within a given neighborhood e.g., [64] proposed to restrict the attention within a specific window around a given pixel position to reduce the computational overhead. Similarly, [62] further improved local self-attention such that it can dynamically adapt its weight aggregation to variations in the input data/features.

卷积运算是传统深度神经网络在计算机视觉中的工作主力，带来了在高维数据集如ImageNet上求解复杂图像识别任务的突破。但是，卷积也有其缺点，如，在固定大小的窗口上进行运算，因此不能捕获长程依赖关系，如在给定的视频中，时间和空间域上像素的任意关系。而且，卷积滤波器的权重在训练后就保持固定，所以这种运算不能对输入的变化进行动态自适应。在本节中，我们回顾了缓解上述这些在传统深度神经网络中存在的问题的方法，使用的是自注意力运算和Transformer网络（一种特定形式的自注意力）。自注意力有两种主要的设计方法。(a)全局自注意力，不受输入特征大小的限制，如[62]提出了一种受到非局部方法启发得到的层，对整个特征图应用注意力，[63]设计了稀疏注意力图，降低了非局部运算的计算复杂度；(b) 局部自注意力试图对给定的邻域内的关系进行建模，如[64]提出将注意力限制在特定窗口中，以降低计算代价。类似的，[62]进一步改进了局部自注意力，这样可以使权重聚积对输入数据/特征的变化进行动态自适应。

Recently, global self-attention has been successfully applied by using NLP Transformer encoder directly on image patches [11], removing the need for handcrafted network design. Transformer is data-hungry in nature e.g., a large-scale dataset like ImageNet is not enough to train Vision Transformer from scratch so [12] proposes to distill knowledge from a CNN teacher to a student Vision Transformer which allowed Transformer training on only ImageNet without any additional data. Here, we describe key insights from different methods based on local/global self-attention including Transformers specifically designed to solve the image recognition task.

最近，通过将NLP Transformer直接应用到图像块中，全局自注意力已经成功的应用到图像中，去掉了手工设计网络的需要。Transformer本质上是非常渴求数据的，如，像ImageNet这样的大型数据集不足以从头训练Vision Transformer了，所以[12]提出从CNN teacher蒸馏知识到一个student Vision Transformer，使Transformer可以只在ImageNet上训练得到，不需要任何额外数据。这里，我们描述不同方法的洞见，都是基于局部/全局自注意力的，包括特意为解决图像识别任务的Transformers。

#### 3.1.1 Non-Local Neural Networks

This approach is inspired by non-local means operation [65] which was mainly designed for image denoising. This operation modifies a given pixel in a patch with a weighted sum of other pixel values in an image. However, instead of considering a fixed-sized window around a pixel, it selects distant pixels to contribute to the filter response based on the similarity between the patches. By design, the non-local operation models long-range dependencies in the image space. Motivated by this, Wang et al. [66] proposed a differentiable non-local operation for deep neural networks to capture long-range dependencies both in both space and time in a feed-forward fashion. Given a feature map, their proposed operator [66] computes the response at a position as a weighted sum of the features at all positions in the feature map. This way, the non-local operation is able to capture interactions between any two positions in the feature map regardless of the distance between them. Videos classification is an example of a task where long-range interactions between pixels exist both in space and time. Equipped with the capability to model long-range interactions, [66] demonstrated the superiority of non-local deep neural networks for more accurate video classification on Kinetics dataset [67].

#### 3.1.2 Criss-Cross Attention

#### 3.1.3 Stand-Alone Self-Attention

#### 3.1.4 Local Relation Networks

#### 3.1.5 Attention Augmented Convolutional Networks

#### 3.1.6 Vectorized Self-Attention

#### 3.1.7 Vision Transformer

#### 3.1.8 Data-Efficient Image Transformers

#### 3.1.9 CLIP: Contrastive Language–Image Pre-training

### 3.2 Transformers for Object Detection

Similar to image classification, Transformer models are applied to a set of image features obtained from a backbone CNN model to predict precise object bounding boxes and their corresponding class labels. Below, the first approach [13] tackles the detection problem, for the first time, using Transformer networks and the second approach [14] mainly extends [13] to a multi-scale architecture and focuses on improving computational efficiency.

与图像分类类似，Transformer模型是应用到CNN骨干模型得到的图像特征上，来预测精确的边界框及其对应的图像类别标签。下面，[13]第一次使用Transformer网络来解决检测问题，[14]将[13]拓展到了多尺度架构，聚焦在改进计算效率上。

#### 3.2.1 Detection Transformer - DETR

In order to apply Transformer model, DETR [13] treats object detection as a set prediction problem and proposes a set loss function. This means that given a set of image features, predict the set of object bounding boxes. The first contribution (the Transformer model) enables the prediction of a set of objects (in a single shot) and allows modeling their relationships. The second contribution (the set loss) allows bipartite matching between predictions and ground-truth boxes. The main advantage of DETR is that it removes the dependence on hand-crafted modules and operations, such as the RPN (region proposal network) and NMS (non-maximal suppression) commonly used in object detection [83]–[87]. In this manner, the dependence on prior knowledge and careful engineering design is relaxed for complex structured tasks like object detection.

为使用Transformer模型，DETR将目标检测视为一个集合预测问题，提出一个集合损失函数。这意味着，给定图像特征集合，预测目标边界框集合。Transformer模型可以一次性预测目标集合，并对其关系进行建模。集合损失可以在预测和真值框之间进行bipartite匹配。DETR的主要贡献是，去除了对手工模块和运算的依赖，比如RPN和NMS，这都是在目标检测中常用的。以这种方式，对先验知识和工程设计的依赖，在目标检测这种复杂结构的任务中就没那么高了。

Given spatial feature maps from the CNN backbone, the encoder first flattens the spatial dimensions into a single dimension, as illustrated in Fig. 9. This gives a sequence of features d × n, where d is the feature dimension and n = h×w with h, w being the height and width of the spatial feature maps. These features are then encoded and decoded using multi-head self-attention modules as proposed in [1]. The main difference in the decoding stage is that all boxes are predicted in parallel while [1] uses an RNN to predict sequence elements one by one. Since the encoder and decoder are permutation invariant, learned positional encodings are used as the object queries by the decoder to generate different boxes. Note that the spatial structure in a CNN detector (e.g., Faster R-CNN) automatically encodes the positional information. DETR obtains performance comparable to the popular Faster R-CNN model [83] which is an impressive feat given its simple design. The DETR has also been extended to interesting applications in other domains, e.g., Cell-DETR [88] extends it for instance segmentation of biological cells. A dedicated attention branch is added to obtain instance-wise segmentations in addition box predictions that are enhanced with a CNN decoder to generate accurate instance masks.

从CNN骨干中给定空间特征图后，编码器首先将空间维度进行拉平，成为单个维度，如图9所示。这给出了一个特征序列d × n，其中d是特征维度，n = h×w，h和w是空间特征图的高度和宽度。这些特征然后用[1]中提出的多头自注意力模块进行编码和解码。解码阶段的主要差异是，所有框都是并行预测的，而[1]使用RNN来一个一个预测序列元素。由于编码器和解码器对排列组合是不变的，学习到的位置编码由解码器用作目标queries，以生成不同的框。注意CNN检测器（如Faster R-CNN）中的空间结构自动编码了位置信息。DETR得到了与Faster R-CNN类似的性能，与其简单设计相比，这是非常不错的。DETR已经拓展到了其他领域中的有趣应用，如，Cell-DETR将其拓展到生物学细胞中的实例分割，加入了一个精巧的注意力分支，在框预测以外，还可以得到逐个实例的分割，这个分支由CNN解码器进行加强，可以生成精确的实例掩模。

#### 3.2.2 Deformable - DETR

### 3.3 Transformers for Segmentation

A dense prediction task like image segmentation into semantic labels and object instances requires modeling rich interactions between pixels. Here, we explain an axial self-attention operation [91] that seeks to reduce the complexity of self-attention and a cross-modal approach [15] that can segment regions corresponding to a given language expression.

图像分割这样的密集预测任务，需要对像素之间的丰富相互作用进行建模。这里，我们解释了一个轴向的自注意力运算，寻求降低自注意力和跨模态方法[15]的复杂度，这个方法可以根据给定的语言表示，分割对应的区域。

#### 3.3.1 Axial-Attention for Panoptic Segmentation

#### 3.3.2 CMSA: Cross-Modal Self-Attention

### 3.4 Transformers for Image Generation

Image generation tasks are interesting from the perspective of generative modeling and because the representations learned in an unsupervised manner can later be used for down-stream tasks. Here, we summarize different Transformer-based architectures for image generation tasks [97]–[100]. We also cover a structured generation task where scene objects are populated given a room layout [23].

图像生成任务，从生成式建模角度来看是很有趣的，因为以无监督方式学习到的表示，可以用于后续的任务。这里，我们总结了用于图像生成任务的不同的基于Transformer的架构。我们还覆盖了一个结构化的生成任务，其中场景目标在给定的房间设置后进行放置。

#### 3.4.1 Image Transformer

#### 3.4.2 Image GPT

#### 3.4.3 High-Resolution Image Synthesis

#### 3.4.4 TransGAN: Transformers based GAN model

#### 3.4.5 SceneFormer

### 3.5 Transformers for Text-to-Image Synthesis

### 3.6 Transformers for Low-level Vision

Transformer models have also been proposed for low-level vision tasks including image super-resolution, denoising, deraining, and colorization. Specifically, Transformer network for super-resolution [16] uses attention mechanisms to search relevant textures from reference images and transfer them to low-resolution images to generate super-resolved outputs. Similarly, the work of [19] shows how to exploit the potential of pre-training and transfer learning with a shared Transformer based backbone to address multiple image restoration tasks (e.g., denoising, deraining, and super-resolution) with dedicated task-heads. The colorization transformer [24] proposes a progressive design for image colorization to achieve high-resolution outputs. Next, we provide details of these aforementioned image restoration Transformer models.

#### 3.6.1 Transformers for Super-Resolution

#### 3.6.2 Transformers for Image Processing Tasks

#### 3.6.3 Colorization Transformer

### 3.7 Transformers for Multi-Modal Tasks

#### 3.7.1 ViLBERT: Vision and Language BERT

#### 3.7.2 LXMERT

#### 3.7.3 VisualBERT

#### 3.7.4 VL-BERT

#### 3.7.5 Unicoder-VL

#### 3.7.6 Unified VLP

#### 3.7.7 UNITER

#### 3.7.8 Oscar: Object-Semantics Aligned Pre-Training

#### 3.7.9 Vokenization

#### 3.7.10 Vision-and-Language Navigation

### 3.8 Video Understanding

#### 3.8.1 VideoBERT: Joint Video and Language Modeling

#### 3.8.2 Masked Transformer

#### 3.8.3 Parameter Efficient Multi-Modal Transformers

#### 3.8.4 Video Action Transformer

#### 3.8.5 Video Transformer Network

#### 3.8.6 Video Instance Segmentation Transformer

#### 3.8.7 Skeleton-Based Action Recognition

### 3.9 Transformers in Low-shot Learning

#### 3.9.1 Cross-Transformer

#### 3.9.2 FEAT: Few-Shot Embedding Adaptation

### 3.10 Transformers for Clustering

### 3.11 Transformers for 3D Analysis

#### 3.11.1 Point Transformer

#### 3.11.2 Point-Cloud Transformer

#### 3.11.3 Pose and Mesh Reconstruction

## 4 Open Problems & Future Directions

Despite excellent performance from Transformer models and their interesting salient features (Table 1), there exist several challenges associated with their applicability to practical settings (Table 2). The most important bottlenecks include requirement for large-amounts of training data and associated high computational costs. There have also been some challenges to visualize and interpret Transformer models. In this section, we provide an overview of these challenges, mention some of the recent efforts to address those limitations and highlight the open research questions.

尽管Transformer模型有很好的性能，及其有趣的明显的特征（表1），将其应用到实际设置中时，有几个挑战（表2）。最重要的瓶颈包括，需要大量的训练数据，以及相关的高计算代价。Transformer模型的可视化与解释，也有一些挑战。本节中，我们提供了这些挑战的概览，以及解决这些限制的一些努力，强调了开放的研究问题。

### 4.1 High Computational Cost

As discussed in Sec. 1, Transformer models have high parametric complexity. This results in high training and inference cost, both in terms of computational time and resources required for processing. As an example, the BERT [3] basic model (with 109 million parameters) took around 1.89 peta-flop days for training, while the latest GPT3 [6] model (175 billion parameters) took around 3640 peta-flop days for training (a staggering ∼1925x increase). This comes with a huge price tag, e.g., according to one estimate [180], GPT3 training might have cost OpenAI around 4.6 million USD. Additionally, these large-scale models require aggressive compression techniques (e.g., distillation) to make their inference feasible for real-world settings.

就像在第一节讨论的，Transformer模型的参数复杂度很高。这带来了很高的训练和推理代价，包括计算时间和资源。举例来说，BERT基础模型需要1.89 peta-flop days进行训练，而最新的GPT3模型需要大约3640 peta-flop days进行训练（增加了大约1925倍）。这需要很大的代价，如，根据一种估计，GPT3训练大约耗费了OpenAI大约4.6 million美元。另外，这些大规模模型需要很激进的压缩技术（如，蒸馏），以使其推理对实际设置适用。

In the language domain, recent works focus on reducing the high complexity of Transformer models (basically arising from the self-attention mechanism [1] where a token’s representation is updated by considering all tokens from the previous layer). For example, [161], [185] explore selective or sparse attention to previous layer tokens which updating each next layer token. Linformer [33] reduces complexity of standard self-attention operation from O(n2) to O(n) (both in time and memory requirements). The main idea is to show that a low-rank matrix is sufficient to model the self-attention mechanism. The Reformer model [186] employed locally-sensitive hashing (LSH) to minimize the complexity of self-attention from O(n2) to O(nlog(n)). In similar pursuit, the recent Lambda Networks propose to model context as a linear function which helps reduce complexity of self-attention [187].

在语言领域，最近的工作聚焦在降低Transformer模型的高复杂度（基本上是从[1]的自注意力机制中发展出来的，其中一个token的表示是通过考虑之前层的所有tokens来更新的）。比如，[161,185]探索了选择性或稀疏注意力，用之前层的tokens更新每个下一层的token。Linformer[33]使标准自注意力运算的复杂度从O(n2)降低到了O(n)，包括时间复杂度和空间复杂度。其主要思想是，一个低秩矩阵足以对自注意力机制进行建模。Reformer模型[186]采用局部敏感哈希来最小化自注意力的复杂度，从O(n2)到O(nlog(n))。类似的，最近的Lambda网络提出将上下文建模为一个线性函数，帮助降低了自注意力的复杂度。

Vyas et al. [188] developed an efficient cluster attention approach to deal with large input sequences that approximates the original self-attention. They propose a cluster attention approach that groups queries into clusters and then computes attention between cluster centers (instead of attention between all the queries that leads to quadratic complexity). The main idea is that the queries close in the Euclidean space should have similar attention distributions. With a fixed number of clusters, this intuition helps reduce the quadratic complexity to linear complexity of O(nc) with respect to the input sequence length n (where c is the number of clusters). We refer readers to [31] for a nice literature survey on efficient Transformers in NLP.

Vyas等[188]提出了一种高效的聚类注意力方法，处理大型数据序列，对原始自注意力进行近似。他们提出了一个聚类注意力方法，将queries分组为聚类，然后计算聚类中心之间的注意力（而不是在所有queries之间的注意力，这样会带来平方的复杂度）。其主要思想是，在欧式空间中接近的queries应当有接近的注意力分布。在固定数量的聚类下，这个直觉帮助将平方复杂度降低到了线性复杂度O(nc)，其中n为输入序列的长度，c是聚类的数量。我们推荐读者参考[31]来综述NLPs中的高效Transformer。

Similar to the NLP domain, computer vision models also suffer from the high computational cost of Transformer models. For example, image generators that are based on sequence-based Transformers (e.g., iGPT) have a high compute cost limiting their applicability to high-resolution inputs. In future, it is interesting to explore how such models can be extended to high-dimensional cases e.g., using a multi-scale transformer design with a somewhat local context modeling. By inducing inductive biases based on our understanding of the visual learning tasks (e.g., patial relationships in the local neighbourhood), the high computational cost can be reduced. Similarly, using sparse attention maps modeled with low-rank factorization in the matrices can also help towards reducing the computational cost [160].

与NLP领域类似，计算机视觉模型中的Transformer模型计算代价也非常高。比如，图像生成器是基于序列Transformer的，其计算代价巨大，不能应用到高分辨率的输入中。在未来，探索模型怎样拓展到高维情况下，是很有趣的，比如，使用多尺度transformer设计，但是在某种局部上下文建模下。基于我们对视觉学习任务的理解（如，局部邻域中的部分关系），通过引入inductive biases，可以降低计算代价。类似的，使用稀疏注意力图，采用低秩分解进行建模，也可以降低计算代价。

### 4.2 Large Data Requirements

Since Transformer architectures do not inherently encode inductive biases (prior knowledge) to deal with visual data, they typically require large amounts of training data during pre-training to figure out the underlying modality-specific rules. For example, a CNN has inbuilt translation invariance, weight sharing, and partial scale invariance due to pooling operations or multi-scale processing blocks. However, a Transformer network needs to figure out these image-specific properties on its own by looking at a large number of examples. Similarly, relationships between video frames need to be discovered automatically by the self-attention mechanism by looking at a large database of video sequences. This results in longer training times, a significant increase in computational requirements, and large datasets for processing. For example, the ViT [11] model requires hundreds of millions of image examples to obtain a decent performance on the ImageNet benchmark dataset. The question of learning a Transformer in a data-efficient manner is an open research problem and recent works report encouraging steps towards its resolution. For example, DeiT [12] uses a distillation approach to achieve data efficiency while T2T (Tokens-to-Token) ViT [189] models local structure by combining spatially close tokens together, thus leading to competitive performance when trained only on ImageNet from scratch (without pre-training).

由于Transformer架构并没有内在的编码inductive biases（先验知识），以处理视觉数据，他们一般在预训练的时候需要大量训练数据，以指出潜在的与模态相关的规则。比如，CNN有内在的平移不变性，权重共享，和由池化运算或多尺度处理模块带来的部分尺度不变性。但是，Transformer网络需要独自处理这些与图像相关的属性，方法就是观察大量样本。类似的，视频帧之间的关系需要通过自注意力机制自动发现，方法是观察大型视频序列数据集。这带来了更长的训练时间，计算需求的显著增加，和要处理的大型数据集。比如，ViT[11]模型需要数亿幅图像样本来在ImageNet基准数据集上得到不错的性能。从数据中更高效的学习Transformer的问题，是一个开放的研究问题，最近的工作给出了很有希望的解决方法。比如，DeiT使用蒸馏方法来得到数据高效性，而T2T ViT通过将空间上接近的tokens结合到一起，来对局部结构进行建模，在从ImageNet上从头训练时，得到了很不错的性能。

### 4.3 Vision Tailored Transformer Designs

We note that most of the existing works focused on vision tasks tend to directly apply Transformer models on computer vision problems. These include architectures designed for image recognition [11], video understanding [17] and especially multi-modal processing [133]. Although the initial results from these simple applications are quite encouraging and motivate us to look further into the strengths of self-attention and self-supervised learning, current architectures may still remain better tailored for language problems (with a sequence structure) and need further intuitions to make them more efficient for visual inputs. For example, vector attention from [75] is a nice work in this direction which attempts to specifically tailor self-attention operation for visual inputs via learning channel-wise attentions. Similarly, [190] uses a Jigsaw puzzle based self-supervision loss as a parallel branch in the Transformers to improve person re-identification. A recent work [189] rearranges the spatially close tokens to better model relationships in spatially proximal locations. One may argue that the architectures like Transformer models should remain generic to be directly applicable across domains, we notice that the high computational and time cost for pre-training such models demands novel design strategies to make their training more affordable on vision problems.

我们注意到，多数已有的聚焦在视觉任务的工作中，一般直接将Transformer模型应用到计算机视觉问题中。这包括设计用于图像识别，视觉理解和尤其是多模态处理的架构。虽然这些应用的初始结果是非常鼓舞人心的，激励我们进一步观察自注意力和自监督学习的潜力，目前的架构仍然是为语言问题设计的（带有序列架构），需要更多的直觉来使其在视觉输入的情况下更有效。比如，[75]中的矢量注意力是这个方向的一个很好工作，试图为视觉输入特意定制自注意力运算，通过学习逐通道的注意力。类似的，[190]使用基于拼图的自监督损失作为并行分支，来改进行人重识别。最近的工作[189]重新安排了空间上接近的tokens以更好的对空间上接近的位置进行关系建模。可以认为，像Transformer这样的架构应当更通用，以直接跨领域应用，我们注意到，预训练这种模型的很高的计算和时间代价，需要新的设计在训练上更高效。

### 4.4 Interpretability of Transformers

Given the strong performance of Transformer architectures, it is interesting and critical to interpret their decisions, e.g., by visualizing relevant regions in an image for a given classification decision. The main challenge is that the attention originating in each layer, gets inter-mixed in the subsequent layers in a complex manner, making it difficult to visualize the relative contribution of input tokens towards final predictions. This is an open problem, however, some recent works [191]–[193] target enhanced interpretability of Transformers and report encouraging results. Attention rollout and attention flow methods were proposed in [192] to estimate the accurate attentions. However, this method functions in an ad-hoc manner and makes simplistic assumptions e.g., input tokens are linearly combined using attention weights across the layers. Chefer et al. [193] note that the attention scores obtained directly via the self-attention process (encoding relationships between tokens) or reassignments in [192] do not provide an optimal solution. As an alternative, they propose to assign and propagate relevancy scores in the Transformer network such that the sum of relevancy is constant throughout the network. Their design can handle both the positive and negative attributions experienced in the self-attention layer. The proposed framework has an added advantage of being able to provide class-specific visualizations. Despite these seminal works, visualizing and interpreting Transformers is an unsolved problem and methods are needed to obtain spatially precise activation-specific visualizations. Further progress in this direction can help in better understanding the Transformer models, diagnosing any erroneous behaviors and biases in the decision process. It can also help us design novel architectures that can help us avoid any biases.

由于Transformer架构性能非常好，解释其决策就非常有趣和关键了，如，对给定的分类决策，可视化图像中相关的区域。主要的挑战是，在每一层中的注意力，在后续层中以很复杂的方式进行了混杂，使输入tokens对最终预测的相对贡献难以可视化。这是一个开放的问题，但是，一些最近的工作的目标是增强Transformers的可解释性，给出了非常好的结果。[192]中提出的注意力rollout和注意力flow，可以估计准确的注意力。但是，这种方法以一种ad-hoc的方式起作用，做出了简单的假设，如，输入tokens是使用注意力权重在层中线性组合的。Chefer等注意到，[192]中通过自注意力过程（在tokens之间编码关系）获得的注意力分数或重新指定，并不会给出最有解。他们提出在Transformer网络中指定并传播相关性分数，这样相关性的和在整个网络中是一个常数。其设计可以处理在自注意力层中的正和负属性。提出的框架有一个额外的优势，可以给出类别相关的可视化。这个方向的更多进展可以更好的理解Transformer模型，诊断在决策过程中的任意错误行为和偏置。这也可以帮助我们设计新的架构，帮助我们避免任意偏置。

### 4.5 Hardware Efficient Designs

Large-scale Transformer networks can have intensive power and computation requirements, hindering their deployment on edge devices and resource-constrained environments such as internet-of-things (IoT) platforms. Some recent efforts have been reported to compress and accelerate NLP models on embedded systems such as FPGAs [194]. Li et al. [194] used an enhanced block-circulant matrix-based representation to compress NLP models and proposed a new Field Programmable Gate Array (FPGA) architecture design to efficiently manage resources for high throughput and low latency. They could achieve 27x, 3x and 81x improvements in performance (throughput measured in FPS), reduced power consumption, and energy efficiency relative a CPU for RoBERTa model [7]. Towards this goal, [195] proposed to design Hardware-Aware Transformers (HAT) using neural architecture search strategies [196]–[198]. Specifically, a SuperTransformer model is first trained for performance approximation which can estimate a model’s performance without fully training it. This model comprises the largest possible model in the search space while sharing weights between common parts. Eventually, an evolutionary search is performed considering the hardware latency constraints to find a suitable SubTransformer model for a target hardware platform (e.g., IoT device, GPU, CPU). However, such hardware efficient designs are currently lacking for the vision Transformers to enable their seamless deployment in resource-constrained devices. Further, the search cost of the evolutionary algorithms remains significant with the associated impact of CO2 emissions on the environment.

大规模Transformer网络能力巨大，也需要大量计算，这使其不能在边缘设备和资源受限的环境上部署，比如IoT平台。一些最近的工作在嵌入式系统如FPGA中压缩和加速NLP模型。Li等使用了enhanced block-circulant matrix-based表示来压缩NLP模型，提出了一种新的FPGA架构设计，来高效的管理高通量低延迟的资源。他们可以对RoBERTa模型得到性能上27x，3x和81x的改进，降低能耗，能量效率改进。为达到此目标，[195]提出设计Hardware-Aware Transformers，使用的是NAS策略。具体的，首先训练一个SuperTransformer模型，以估计性能，估计在未充分训练下的模型性能。模型包括搜索空间中最大可能的模型，在通用部件中共享权重。最终，进行烟花搜索，为目标硬件平台找到一个合适的SubTransformer模型（如，IoT设备，GPU，CPU）。但是，这样的硬件高效设计，目前缺少视觉Transformer来进行无缝部署。而且，演化算法的搜索代价还是很大。

### 4.6 Leveraging Rich Multi-modal Annotations

In cases, where training data is available with dense labels in multiple domains (e.g., language and vision [17], [199]), an interesting question to consider is whether the pre-training process leveraging rich labels on a small dataset speedup its learning. This question has been explored in Virtex [200], a model that seeks to learn strong visual representations using dense textual annotations (e.g., image captions). Since, the captions encode information about objects present in an image, their relationships, actions and attributes, they can provide better supervision to learn more generalizable and transferable representations. Particularly, they showed that a model trained with a visual backbone followed by a bidirectional language model (forward and backward Transformers) [3] to predict captions, can learn strong features on MS-COCO dataset in an unsupervised manner. When these features are transferred to the ImageNet model, they perform better or equally-well compared to the unsupervised/supervised features learned directly on the ImageNet dataset. Since, Transformer models can process multiple modalities in a unified architecture, it will be interesting to explore how densely annotated datasets can reduce the data requirement of Transformers and if dense-annotations allow transferring well to novel unseen conditions in one particular modality at inference.

在一些情况下，训练数据在多个领域中的密集标签是可用的（如，语言和视觉），要考虑的一个有趣问题是，利用小型数据集上的丰富标签，能否使预训练过程加速其学习。Virtex[200]探索了这个问题，这个模型用密集文本标注来学习强视觉表示。因为标题中有图像中存在的目标的信息，及其相互关系，行为和属性，他们可以给出更好的监督，以学习泛化性更好，可迁移的表示。特别是，他们表示，视觉骨干网络和双向语言网络一起训练出的模型，来预测标题，可以在MS-COCO上以无监督的方式学到很强的特征。当这些特征迁移到ImageNet模型中，与直接在ImageNet数据集中学习的无监督/有监督特征比，性能更好，或一样好。因为，Transformer模型可以在统一的框架中处理多模态数据，探索密集标注的数据怎样降低Transformers的数据需求，密集标注是否可以使得更好的迁移到未曾见过的条件，这是非常有趣的。

## 5 Conclusion

Attention has played a key role in delivering efficient and accurate computer vision systems, while simultaneously providing insights into the function of deep neural networks. This survey reviews the self-attention approaches and specifically focuses on the Transformer and bidirectional encoding architectures that are built on the principle of self-attention. We first cover fundamental concepts pertaining to self-attention architectures and later provide an in-depth analysis of competing approaches for a broad range of computer vision applications. Specifically, we include state of the art self-attention models for image recognition, object detection, semantic and instance segmentation, video analysis and classification, visual question answering, visual commonsense reasoning, image captioning, vision-language navigation, clustering, few-shot learning, and 3D data analysis. We systematically highlight the key strengths and limitations of the existing methods and particularly elaborate on the important future research directions. With its specific focus on computer vision tasks, this survey provides a unique view of the recent progress in self-attention and Transformer-based methods. We hope this effort will drive further interest in the vision community to leverage the potential of Transformer models and improve on their current limitations e.g., reducing their carbon footprint.

注意力在高效准确计算机视觉系统中扮演了关键的角色，同时为DNN的作用提供了洞见。本综述回顾了自注意力方法，特别关注了构建在自注意力基础上的Transformer和双向编码架构。我们首先介绍了自注意力架构的基本概念，然后深度分析了很多计算机视觉应用中的方法。具体的，我们包含了图像识别，目标检测，语义分割和实例分割，视频分析和分类，视觉问答，视觉常识推理，图像标题，视觉语言导航，聚类，少样本学习和3D数据分析的最新的自注意力模型。我们系统的强调了现有方法的优势和局限，特别是详述了重要的未来研究方向。本文重点关注了计算机视觉任务，给出了自注意力和Transformer方法的最近进展。我们希望本文会激发更多的兴趣，利用Transformer模型并改进其局限，如，降低碳消耗。