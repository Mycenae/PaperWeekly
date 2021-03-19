# Attention Is All You Need

Ashish Vaswani et. al. Google Brain

## 0. Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

主要的序列转换模型是基于复杂的循环或卷积神经网络的，包含一个编码器和解码器。性能最好的模型还将编码器与解码器通过一种注意力机制连接起来。我们提出一种新的网络架构，Transformer，只基于注意力机制，完全不需要循环和卷积。在两个机器翻译任务上的实验表明，这些模型质量非常好，并行性更好，需要更少的时间去训练。我们的模型在WMT 2014英语到德语的翻译任务中，获得了28.4 BLEU，改进了目前最好的结果2 BLEU，包括集成模型。在WMT 2014英语到德语的翻译任务中，我们的模型确立了一个新的目前最好BLEU分数41.8，在8个GPUs上训练了3.5天，这是其他文献的最好模型训练的时间的一小部分。我们证明了Transformer对其他任务泛化的很好，将其成功的应用到了英文组别分析中，包括大量训练数据和有限训练数据两种情况。

## 1. Introduction

Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].

RNN, LSTM和GRN，目前已经是序列建模和转换问题的最好方法，比如语言建模和机器翻译。很多工作都在推动循环语言模型和编码器-解码器架构的发展。

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht−1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

循环模型沿着输入和输出序列的符号位置分解计算。将位置与计算时间中的步骤对齐，它们生成了隐藏状态ht的序列，是之前的隐藏状态ht-1和t位置的输入的函数。内在的序列本质使得训练样本中不能并行，这对于更长的序列长度变得更加关键，因为内存限制了样本之间的batching。最近的工作在计算效率中获得了显著的改进，通过的是分解技巧[21]和条件计算[32]，同时后者还改进了模型性能。但是，序列计算的基本局限还在。

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.

注意力机制在各种任务中已经成为了序列建模和转换模型的有机组成部分，可以对输入或输出序列中不论其距离的依赖关系的建模。在非常多的情况中，这种注意力机制是与循环网络共同使用的。

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

本文中，我们提出了Transformer，这种模型架构中没有循环结构，完全依赖于注意力机制，在输入和输出之间得到全局的依赖关系。Transformer可以进行更多的并行，在8个P100 GPUs上训练了12个小时后，在翻译质量上就可以达到目前最好。

## 2. Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

降低序列计算的目标，也是Extended Neural GPU [16], ByteNet [18] 和 ConvS2S [9]的基础，这些都利用CNN作为基础部件，对所有输入和输出位置并行计算隐藏表示。在这些模型中，从任意两个输入或输出位置的信号中计算其相关，需要的计算量，随着位置的距离增加而增加，对于ConvS2S来说是线性的，对于ByteNet来说是对数方式的。这使得学习远距离位置之间的依赖关系非常困难。在Transformer中，这降低到了常数次运算，其代价是有效分辨率下降了，因为对注意力加权的位置进行平均了，对这一效果，我们用3.2节中的多头注意力来进行应对。

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].

自注意力，有时候也称为内注意力，是一种与单个序列不同位置相关的注意力机制，以计算序列的表示。自注意力已经在很多任务中成功的应用，包括阅读理解，抽象总结，文本推演和学习任务无关的语句表示。

End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].

端到端的存储网络是基于循环注意力机制的，而不是序列对齐的循环结构，在单语言问题回答和语言建模任务中表现良好。

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

据我们所知，Transformer是完全依赖自注意力来计算输入输出表示，而不用序列对齐的RNNs或卷积，的第一个转换模型。在下面的小节中，我们会描述Transformer，推动自注意力，讨论其对其他模型的优势。

## 3. Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35]. Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.

多数有竞争力的神经序列转换模型，都是编码器-解码器结构的。这里，编码器将输入的符号表示序列(x1, ..., xn)映射到一个连续表示序列z = (z1, ..., zn)。给定z，解码器生成输出的符号序列(y1, ..., ym)，一次一个元素。在每个步骤中，模型是自回归的，将之前生成的符号作为额外的输入，同时生成下一个输出。

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

Transformer也是这个整体架构的，编码器和解码器都使用了堆叠的自注意力，和逐点的全连接层，分别如图1的左右两边所示。

### 3.1 Encoder and Decoder Stacks

**Encoder**: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512.

编码器：编码器是由N=6个相同的层叠加起来的。每层有2个子层。第一个是一个多头自注意力机制，第二个是一个简单的逐位置的全连接前向网络。我们在这两个子层中采用了一个残差连接，层后都有层的归一化。即，没个子层的输出是LayerNorm(x+Sublayer(x))，其中Sublayer(x)是子层本身所实现的函数。为方便这些残差连接，模型所有的子层，以及嵌入层，产生的输出的维度都是dmodel = 512。

**Decoder**: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

解码器：解码器也是由N=6个完全一样的层堆叠成的。除了每个编码层中的两个子层，解码器插入了第三个子层，对编码器堆叠的输出进行多头注意力运算。与编码器类似，我们对每个子层采用残差连接，外加层的归一化。我们还修改了解码器层中的自注意力子层，防止位置注意到后续的位置。这种掩膜，与输出嵌入偏移了一个位置的事实相结合，确保了位于位置i的预测，只依赖于位置前于i的已知输出。

### 3.2 Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

注意力函数，可以描述为，将query和key-value的集合映射到输出中，其中query，key，value和输出，都是向量。输出是通过values的加权和来计算得到的，其中对每个value指定的权重，是通过query与对应的key的compatibility函数计算得到的。

#### 3.2.1 Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values.

我们称我们特别的注意力为“缩放的点积注意力”（图2）。输入包括queries和keys，维度为dk，values的维度为dv。我们计算query与所有keys的点积，每个都除以$\sqrt{d_k}$，应用一个softmax函数，以得到values上的权重。

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V. We compute the matrix of outputs as:

实践中，我们在queries的集合中同时计算注意力函数，这些queries打包成了一个矩阵Q。这些keys和values也打包到一起，成为矩阵K和V。我们计算矩阵的输出为

$$Attention(Q,K,V) = softmax (\frac {QK^T} {\sqrt{d_k}}) V$$(1)

The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac {1}{\sqrt{d_k}}$. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

两种最常用的注意力函数为加性注意力，和点乘注意力。点乘注意力与我们的算法相同，除了缩放系数$\frac {1}{\sqrt{d_k}}$。加性注意力使用一个带有单层隐藏层的前向网络来计算compatibility函数。这两者在理论复杂度上比较类似，但点乘注意力在实践中更快更节省空间，因为可以高度优化的矩阵相乘代码来实现。

While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of dk [3]. We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $\frac {1}{\sqrt{d_k}}$.

对于dk值较小时，两种机制的性能类似，但对于dk值较大时，在不缩放时，加性注意力的性能会超过点乘注意力。我们推测，对于dk值较大的情况，点乘的值变得很大，使softmax函数到了梯度很小的区域。为应对这种效果，我们将点乘的结果乘以因子$\frac {1}{\sqrt{d_k}}$。

#### 3.2.2 Multi-Head Attention

Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.

对于dmodel维的keys, values, queries，我们没有进行单个注意力函数计算，我们发现，将queries, keys, values，用不同的学习到的线性投影，分别线性投影h次成为dk, dk, dv维，效果更好。在每个这些投影版的queries, keys和values上，我们然后并行进行注意力函数的计算，得到dv维的输出values。这些值拼接在一起，再一次进行投影，得到最终的values，如图2所示。

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

多头注意力使模型可以同时注意到不同位置不同表示子空间的信息。在单注意力头的情况下，平均的效果使得这种情况是不可能的。

$$MultiHead(Q, k, V) = Concat(head1, ..., headh) W^O, where headi = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

Where the projections are parameter matrices $W_i^Q ∈ R^{d_{model}×d_k}, W_i^K ∈ R^{d_{model}×d_k}, W_i^V ∈ R^{d_{model}×d_v}$ and $W^O ∈ R^{d_{hd_v×model}}$.

In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

本文中，我们采用h=8个并行的注意力层，或头。对于每个头，我们都采用dk=dv=dmodel/h=64。由每个头的维度下降了，总计的计算代价，与满维度的单头注意力是类似的。

#### 3.2.3 Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways: Transformer以三种不同的方式使用多头注意力：

- In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9]. 在编码器-解码器注意力层中，queries来自前一个解码器层，存储keys和values来自编码器的输出。这使解码器中的每个位置注意到输入序列中的所有位置。这模仿了seq2seq模型的典型编码器-解码器注意力机制。

- The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder. 编码器带有自注意力层。在自注意力层中，所有的keys, values和queries都来自相同的地方，在这种情况下，是编码器中前一层的输出。编码器中的每个位置，都会注意到编码器中前一层的所有位置。

- Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections. See Figure 2. 类似的，解码器中的自注意力层，允许解码器中的每个位置，都注意到包含本位置及其以前的解码器的所有位置。我们需要防止左边的信息流入解码器中，以保持自回归的性质。我们在缩放点乘注意力中实现这个，是通过在softmax的输入中，把对应到非法连接的所有值都掩膜掉。见图2。

### 3.3 Position-wise Feed-Forward Networks

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between. 除了注意力的子层，编码器和解码器的每层中，都还包含了一个全连接的前向网络，独立的相同的应用到每个位置。这由其中带有ReLU激活的两个线性变换组成。

$$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$$(2)

While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality dff = 2048. 线性变换在不同的位置都是一样的，但它们在不同的层中使用不同的参数。另一种描述这个的方式是，两个核大小为1的卷积。输入和输出的维度是dmodel=512，内层的维度为dff=2048。

### 3.4 Embeddings and Softmax

Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension dmodel. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]. In the embedding layers, we multiply those weights by $\sqrt{dmodel}$.

与其他序列转换模型类似，我们使用学习到的嵌入将输入tokens和输出tokens转换到dmodel维度的向量。我们还使用通常学习到的线性变换和softmax函数，来将解码器输出转换到预测的下一个token概率。在我们的模型中，我们在嵌入层和pre-softmx线性变换中共享相同的矩阵，与[30]类似。在嵌入层中，我们将这些权重乘以$\sqrt{dmodel}$。

### 3.5 Positional Encoding

Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [9].

由于我们的模型不包含循环和卷积结构，为使模型使用序列的顺序，我们必须将序列中的tokens的相对或绝对位置注入进去。为此，我们向输入嵌入中，在编码器和解码器堆叠的下面，加入了位置编码。位置编码与嵌入一样都有dmodel的维度，这样两者可以相加。有很多位置编码的选择，学习得到的和固定的[9]。

In this work, we use sine and cosine functions of different frequencies: 本文中，我们使用不同频率的sine和cosine函数：

$$PE_{(pos,2i)} = sin(pos/10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}})$$

where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, PEpos+k can be represented as a linear function of PEpos.

其中pos是位置，i是维度。即，位置编码的每个维度都对应一个正弦曲线。波长形成了从2π到10000 · 2π的一个几何数列。我们选择这个函数，是因为，我们假设这可以使模型很容易学到并被由相对位置注意，因为对于任意的固定偏移k，PEpos+k可以表示为PEpos的线性函数。

We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

我们还试验了学习得到的位置嵌入[9]，发现这两者得到了几乎一样的结果（见表3）。我们选择了正弦函数版，因为这使模型可以外插到更长的序列长度，比在训练时遇到的更长。

## 4. Why Self-Attention

In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations (x1, ..., xn) to another sequence of equal length (z1, ..., zn), with xi, zi ∈ Rd, such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.

本节中，我们比较了自注意力层与循环层、卷积层的各个方面，它们都用于将变长序列的符号表示(x1, ..., xn)映射到另一个等长的序列(z1, ..., zn)，其中xi, zi ∈ Rd，比如在一个典型的序列转换编码器或解码器的隐藏层中。激发我们使用自注意力，我们考虑三个想要的东西。

One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

一个是每层的总计计算复杂度。另一个是可并行的计算量，用需要的序列运算的最少数量度量。

The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

第三个是网络中长程依赖关系的路径长度。在很多序列转换任务中，学习长程依赖关系是非常关键的挑战。影响学习这种依赖关系能力的一个关键因素是，前向和反向信号在网络中需要穿越的路径长度。在输入和输出序列的任意位置组合之间的这些路径越短，学习长程依赖关系就越容易。因此我们还比较，网络中不同类型层里，在任意两个输入和输出位置之间，的最大路径长度。

As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [38] and byte-pair [31] representations. To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size r in the input sequence centered around the respective output position. This would increase the maximum path length to O(n/r). We plan to investigate this approach further in future work.

如表1所示，一个自注意力层连接所有位置，用到的序列执行运算是常数，而一个循环层需要O(n)序列运算。以运算复杂度而论，当序列长度比表示的维度d要小时，自注意力层比循环层更快，这在机器翻译中，目前最好的模型通常都是这样的，比如word-piece [38]和byte-pair[31]表示。为改进非常长的序列的计算性能，自注意力可以局限在只考虑输入序列中，以相应的输出位置为中心的大小为r的邻域中。这可以增加最大路径长度到O(n/r)。我们计划在未来的工作中研究这种方法。

A single convolutional layer with kernel width k < n does not connect all pairs of input and output positions. Doing so requires a stack of O(n/k) convolutional layers in the case of contiguous kernels, or O(logk(n)) in the case of dilated convolutions [18], increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more expensive than recurrent layers, by a factor of k. Separable convolutions [6], however, decrease the complexity considerably, to O(k · n · d + n · d^2). Even with k = n, however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model.

单个卷积层，核宽度k < n，并不会将所有输入输出位置对连接到一起。在contiguous核的情况下，这样需要堆叠O(n/k)个卷积层，或在dilated卷积的情况下，需要O(logk(n))个卷积层，在网络中的任意两个位置之间的最长路径距离变长了。卷积层一般比循环层更贵一些，多乘以了一个因子k。可分离卷积显著降低了复杂度，到O(k · n · d + n · d^2)。即使在k=n的情况下，可分离卷积的复杂度是等于一个自注意力层和一个逐点的前向层的，也就是我们所采取的方法。

As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.

作为额外的好处，自注意力会得到解释性更好的模型。我们检查了我们模型中的注意力分布，在附录中进行了展示和讨论。单个注意力头很明显学习在不同的任务中得到了还不错的效果，很多似乎表现出了与句子句法和语义结构的行为。

## 5. Training

This section describes the training regime for our models. 本节描述了我们模型的训练方案。

### 5.1 Training Data and Batching

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared source-target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38]. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

我们在标准WMT 2014英文-德文数据集上进行了训练，数据集包含大约450万语句对。语句是用byte-pair编码方式进行编码的，其共享源目标词汇为大约37000 tokens。对于英文-法文，我们使用更大的WMT 2014 英文-法文数据集，包含3600万语句，将tokens分割成32000 word-piece词汇。语句通过大致的序列长度对批量进行了处理。每个训练batch，包含的语句对集合，包含大约25000源tokens和25000目标tokens。

### 5.2 Hardware and Schedule

We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).

我们在一个有8个NVidia P100 GPUs的机器中训练了我们的模型。我们的基础模型使用的超参数在整篇文章中都使用，每个训练步骤耗时大约0.4s。我们训练的基础模型总计10万步，耗时12小时。我们的大模型（见表3的底部行），每步的时间是1.0s。大模型训练了30万步，大约3.5天。

### 5.3 Optimizer

We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ϵ = 10^−9. We varied the learning rate over the course of training, according to the formula:

我们使用Adam优化器，β1 = 0.9, β2 = 0.98, ϵ = 10^−9。我们根据下式在训练过程中变化学习速率

$$lrate = d^{−0.5}_{model} · min(step_num^{−0.5}, step_num · warmup_steps^{−1.5})$$(3)

This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup_steps = 4000.

这对应着，在开始的warmup_steps训练步骤中线性增大训练步长，然后逐步减少训练步长。我们使用warmup_steps = 4000。

### 5.4 Regularization

We employ three types of regularization during training: 我们在训练中采用三种正则化类型：

**Residual Dropout**. We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop = 0.1. 我们对每个子层的输出，在加入到子层输入并归一化之前，采用dropout。此外，我们对编码器和解码器堆叠中的嵌入的和和位置编码采用dropout。对于基础模型，我们采用的Pdrop=0.1。

**Label Smoothing**. During training, we employed label smoothing of value ϵls = 0.1 [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score. 在训练中，我们采用了标签平滑，值为ϵls = 0.1。这伤害了perplexity，因为模型学习的结果是更加不确定，但改进了准确率和BLEU分数。

## 6. Results

### 6.1 Machine Translation

On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

在WMT 2014英文到德文的翻译任务中，大transformer模型（表2）超过了之前最好的模型（包括集成模型）2.0 BLEU，确立了一个新的目前最好的BLEU分数28.4。这个模型的配置列在表3的最下面一行。训练在8个P100 GPUs上耗时3.5天。即使是我们的基础模型，也超过了所有之前发表的模型和集成模型，但训练代价只是竞争模型的一小部分。

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate Pdrop = 0.1, instead of 0.3.

在WMT 2014英文到法文翻译任务中，我们的大模型得到的BLEU分数为41.0，超过了所有之前发表的单模型，训练代价是之前最好的模型的1/4还少。为英文到法文训练的Transformer大模型，其dropout率为Pdrop=0.1，而不是0.3。

For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty α = 0.6 [38]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length + 50, but terminate early when possible [38].

对于基础模型，我们使用将最后5个checkpoints进行平均的单个模型，这是每个10分钟间隔保存的。对于大模型，我们对最后20个checkpoints进行平均。我们使用beam搜索，beam大小为4，长度惩罚α = 0.6。这些超参数是在开发集上经过试验之后选择的。我们在推理时设置最大输出长度为输入长度+50，但在可能的时候早停。

Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU.

表2总结了我们的结果，将我们的翻译质量和训练代价与其他模型架构进行了比较。我们估计了用于训练一个模型的浮点数运算数量，将训练时间和GPU数量，以及每个GPU的单精度浮点能力进行了相乘。

### 6.2 Model Variations

To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3.

为评估Transformer的不同部分的重要性，我们以不同方式变化了基础模型，在英文到德文翻译的开发集上度量了性能变化。我们使用了一样的beam搜索，但并没有进行checkpoint平均。结果如表3所示。

In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.

在表3的行A中，我们变化了注意力头的数量，和注意力key和value的维度，使得计算代价一样，如3.2.2中所述。单头注意力性能比最好的设置下降了0.9 BLEU，头如果太多的话，性能也会下降。

In Table 3 rows (B), we observe that reducing the attention key size dk hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical results to the base model.

在表3的行B中，我们观察到，降低注意力key的大小dk会伤害模型质量。这说明，确定compatibility不是容易的，比点乘更复杂的compatibility函数会有好处。我们在行C和行D中进一步观察到，更大的模型是更好的，dropout对于防止过拟合是非常有用的。在行E中，我们将正弦位置编码替换为学习得到的位置编码，结果与基础模型基本一样。

### 6.3 English Constituency Parsing

To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes [37].

为评估Transformer是否可以泛化到其他任务，我们在英文constituency解析中进行试验。这个任务提出了具体的挑战：输出是受到很强的结构约束影响的，比输入明显要长。而且，RNN seq-to-seq模型在小数据范畴内没有得到目前最好的结果。

We trained a 4-layer transformer with dmodel = 1024 on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences. We also trained it in a semi-supervised setting, using the larger high-confidence and BerkleyParser corpora from with approximately 17M sentences [37]. We used a vocabulary of 16K tokens for the WSJ only setting and a vocabulary of 32K tokens for the semi-supervised setting.

我们在WSJ Penn Treebank上训练了一个4层的transformer，dmodel=1024，大概有40K个训练语句。我们以半监督的设置来进行训练，使用更大的高置信度和BerkleyParser语料，有大约17M语句。我们使用的词汇库对于WSJ为16K tokens，对半监督设置为32K tokens。

We performed only a small number of experiments to select the dropout, both attention and residual (section 5.4), learning rates and beam size on the Section 22 development set, all other parameters remained unchanged from the English-to-German base translation model. During inference, we increased the maximum output length to input length + 300. We used a beam size of 21 and α = 0.3 for both WSJ only and the semi-supervised setting.

我们在选择dropout，注意力和残差，学习速率和beam大小上只进行了少数几次试验（5.4节），在22节的开发集上，其他参数与英文到德文的基础翻译模型保持一致。在推理中，我们增加最大输出长度到输入长度+300。我们对WSJ和半监督的设置都是使用的beam大小为21，α = 0.3。

Our results in Table 4 show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar [8].

我们在表4中的结果表明，尽管缺少与任务具体的精调，我们的模型表现还是非常的好，比之前的模型都得到了更好的结果，除了Recurrent Neural Network Grammar [8]。

In contrast to RNN sequence-to-sequence models [37], the Transformer outperforms the Berkeley-Parser [29] even when training only on the WSJ training set of 40K sentences.

与RNN seq-to-seq模型相比，Transformer即使只在WSJ训练集的40K个语句中训练，也超过了Berkeley-Parser。

## 7. Conclusion

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

本文中，我们提出了Transformer，第一个完全基于注意力的序列转换模型，将目前在编码器-解码器架构中最常使用的循环层，替换为了多头自注意力。

For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

对于翻译任务，Transformer比基于循环层或卷积层的架构，可以明显更快的训练。在WMT 2014 英文到德文和英文到法文翻译任务中，我们得到了目前最好的结果。在前一个任务中，我们的最好模型甚至超过了之前的集成模型。

We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.

我们对未来的基于注意力的模型感到很激动，计划将其用于其他任务。我们计划去拓展Transformer到输入输出不止是文本的情况，研究局部，受限的注意力机制，以高效的处理大型输入输出，如图像，音频和视频。让生成更加不那么序列化，是我们的另一个研究目标。

The code we used to train and evaluate our models is available at https://github.com/tensorflow/tensor2tensor. 模型已开源。