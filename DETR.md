# End-to-End Object Detection with Transformers

Nicolas Carion et. al. Facebook AI

## 0. Abstract

We present a new method that views object detection as a direct set prediction problem. Our approach streamlines the detection pipeline, effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task. The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bi-partite matching, and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. The new model is conceptually simple and does not require a specialized library, unlike many other modern detectors. DETR demonstrates accuracy and run-time performance on par with the well-established and highly-optimized Faster R-CNN baseline on the challenging COCO object detection dataset. Moreover, DETR can be easily generalized to produce panoptic segmentation in a unified manner. We show that it significantly outperforms competitive baselines. Training code and pretrained models are available at https://github.com/facebookresearch/detr.

我们提出了一种新方法，将目标检测看作是一个直接的集合预测问题。我们的方法使检测的流程更加流线型，有效的去除了很多手工设计的部件，如非最大抑制过程，或锚框的生成，直接将我们的关于任务的先验知识集成到网络中。新框架的主要组成部分，我们称之为检测Transformer或DETR，是一种基于集合的全局损失，通过bi-partite匹配来强迫唯一的预测，以及一个transformer编码器-解码器架构。给定一个固定的学习的目标查询的集合，DETR对目标和全局图像上下文的关系进行推理，以直接并行输出最终的预测集合。新模型在概念上非常简单，不需要专用的库，与很多其他的现代检测器不同。DETR在COCO目标检测数据集上，证明了其准确率与运行时间性能与现在广泛使用高度优化的Faster R-CNN基准类似。而且，DETR可以很容易的泛化生成全景分割模型，方式非常一致。我们证明了，其显著超过了其他基准。训练代码和预训练模型已开源。

## 1 Introduction

The goal of object detection is to predict a set of bounding boxes and category labels for each object of interest. Modern detectors address this set prediction task in an indirect way, by defining surrogate regression and classification problems on a large set of proposals [37,5], anchors [23], or window centers [53,46]. Their performances are significantly influenced by postprocessing steps to collapse near-duplicate predictions, by the design of the anchor sets and by the heuristics that assign target boxes to anchors [52]. To simplify these pipelines, we propose a direct set prediction approach to bypass the surrogate tasks. This end-to-end philosophy has led to significant advances in complex structured prediction tasks such as machine translation or speech recognition, but not yet in object detection: previous attempts [43,16,4,39] either add other forms of prior knowledge, or have not proven to be competitive with strong baselines on challenging benchmarks. This paper aims to bridge this gap.

目标检测的目标是，对每个感兴趣的目标，预测边界框和类别标签的一个集合。现代检测器是以一种间接的方式来处理这种集合预测任务，在候选、锚框或窗口中心的大型集合上定义了代理回归和分类问题。其性能受到后处理步骤的显著影响，将接近重合的预测重叠为一个，还受到锚框的设计的影响，还受到指定目标框为锚框的直观推理所影响。为简化这些流程，我们提出了一种直接的集合预测方法，以避开这些代理任务。这种端到端的哲学带来了复杂结构预测任务的显著进步，比如机器翻译或语音识别，但在目标检测中还没有成果：之前的尝试要么加入了其他形式的先验信息，或还没有证明与很强的基准有类似的性能。本文的目标是弥补这个差距。

We streamline the training pipeline by viewing object detection as a direct set prediction problem. We adopt an encoder-decoder architecture based on transformers [47], a popular architecture for sequence prediction. The self-attention mechanisms of transformers, which explicitly model all pairwise interactions between elements in a sequence, make these architectures particularly suitable for specific constraints of set prediction such as removing duplicate predictions.

我们通过将目标检测视为一个直接的集合预测问题，将训练的流程变得更加流线型。我们采用了一种基于transformers的编码器-解码器架构，这是一种序列预测的流行架构。Transformers的自注意力机制，显式的对一个序列中的所有元素的成对相互作用进行建模，使这些架构特别适合于集合预测的具体约束，比如去除重叠的预测。

Our DEtection TRansformer (DETR, see Figure 1) predicts all objects at once, and is trained end-to-end with a set loss function which performs bipartite matching between predicted and ground-truth objects. DETR simplifies the detection pipeline by dropping multiple hand-designed components that encode prior knowledge, like spatial anchors or non-maximal suppression. Unlike most existing detection methods, DETR doesn’t require any customized layers, and thus can be reproduced easily in any framework that contains standard CNN and transformer classes.

我们的DETR一次性的预测所有目标，而且使用一个集合损失函数进行了端到端的训练，这个损失函数进行的是预测目标和真值目标之间的bipartite匹配。DETR简化了预测流程，抛弃了多个手工设计的编码了先验知识的部件，如空间锚框，或非最大抑制。与多数现有的检测方法不同，DETR不需要任何定制的层，所以可以在任何框架中很容易的复现，只要包含标准的CNN和transformer类。

Compared to most previous work on direct set prediction, the main features of DETR are the conjunction of the bipartite matching loss and transformers with (non-autoregressive) parallel decoding [29,12,10,8]. In contrast, previous work focused on autoregressive decoding with RNNs [43,41,30,36,42]. Our matching loss function uniquely assigns a prediction to a ground truth object, and is invariant to a permutation of predicted objects, so we can emit them in parallel.

与多数之前的直接集合预测工作相比，DETR的主要特征是将bipartite匹配损失和平行解码的transformer的结合。比较之下，之前的工作聚焦在用RNNs的自动回归编码。我们的匹配损失函数对一个真值目标指定一个唯一的预测，对预测的目标的排列组合也是不变的，所以我们可以将其并列发射。

We evaluate DETR on one of the most popular object detection datasets, COCO [24], against a very competitive Faster R-CNN baseline [37]. Faster R-CNN has undergone many design iterations and its performance was greatly improved since the original publication. Our experiments show that our new model achieves comparable performances. More precisely, DETR demonstrates significantly better performance on large objects, a result likely enabled by the non-local computations of the transformer. It obtains, however, lower performances on small objects. We expect that future work will improve this aspect in the same way the development of FPN [22] did for Faster R-CNN.

我们在最流行的一个目标检测数据集COCO上评估了DETR，与Faster R-CNN基准进行了比较。Faster R-CNN已经经历了很多设计的迭代，其性能自动原始发表的文章起已经改进了很多。我们的试验表明，新模型获得了类似的性能。更精确的，DETR在大型目标上得到了更好的性能，这一结果很可能是transformer的非局部计算的结果。但是，在小目标上得到了略差的性能。我们期待未来的工作会改进这个方面，就像FPN改进了Faster R-CNN一样。

Training settings for DETR differ from standard object detectors in multiple ways. The new model requires extra-long training schedule and benefits from auxiliary decoding losses in the transformer. We thoroughly explore what components are crucial for the demonstrated performance.

DETR的训练设置与标准的目标检测器相比，在几个方面都不太一样。新模型需要非常长的训练过程，而且transformer中的辅助解码损失也使得新模型更好。我们彻底的探索了到底是哪些部件对得到的性能是最关键的。

The design ethos of DETR easily extend to more complex tasks. In our experiments, we show that a simple segmentation head trained on top of a pretrained DETR outperfoms competitive baselines on Panoptic Segmentation [19], a challenging pixel-level recognition task that has recently gained popularity.

DETR的设计可以很容易的拓展到更复杂的任务。在我们的试验中，我们表明一个简单的分割头，与预训练的DETR一起，就可以超过很好的全景分割基准，这是一个很有挑战的像素级识别任务，最近才流行起来。

## 2 Related work

Our work build on prior work in several domains: bipartite matching losses for set prediction, encoder-decoder architectures based on the transformer, parallel decoding, and object detection methods.

我们的工作是在之前几个领域的工作上构建起来的：集合预测的bipartite匹配损失，基于transformer的编码器-解码器架构，并行解码，和目标检测方法。

### 2.1 Set Prediction

There is no canonical deep learning model to directly predict sets. The basic set prediction task is multilabel classification (see e.g., [40,33] for references in the context of computer vision) for which the baseline approach, one-vs-rest, does not apply to problems such as detection where there is an underlying structure between elements (i.e., near-identical boxes). The first difficulty in these tasks is to avoid near-duplicates. Most current detectors use postprocessings such as non-maximal suppression to address this issue, but direct set prediction are postprocessing-free. They need global inference schemes that model interactions between all predicted elements to avoid redundancy. For constant-size set prediction, dense fully connected networks [9] are sufficient but costly. A general approach is to use auto-regressive sequence models such as recurrent neural networks [48]. In all cases, the loss function should be invariant by a permutation of the predictions. The usual solution is to design a loss based on the Hungarian algorithm [20], to find a bipartite matching between ground-truth and prediction. This enforces permutation-invariance, and guarantees that each target element has a unique match. We follow the bipartite matching loss approach. In contrast to most prior work however, we step away from autoregressive models and use transformers with parallel decoding, which we describe below.

没有经典的深度学习模型直接进行集合预测。基础集合预测任务是多标签分类，其基准的一对多方法，对检测这样的任务无法应用，因为其中有一些潜在的元素之间的结构（即，接近一致的框）。这些任务中的第一个困难是，避免接近重复的检测。多数目前的检测器采用了后处理来解决这个问题，比如非极大抑制，但是直接集合预测是不需要后处理的。他们需要全局推理方案，对所有预测元素之间的相互作用进行建模，以避免冗余。对常数大小的集合预测，密集的全连接网络是充分的，但是太过于昂贵。一种通用方法是，使用自回归序列模型，比如RNN。在所有情况中，损失函数应当对于预测的组合顺序是不变的。通常的解决方案是，设计一个基于Hungarian算法的损失，以找到真值和预测间的bipartite匹配。这就强加了排列组合的不变性，并保证了每个目标元素都有一个唯一的匹配。我们也采用了bipartite匹配损失的方法。与多数之前的工作形成对比的是，我们并没有使用自回归模型，而使用了transformers，并带有平行解码，我们在下面会进行描述。

### 2.2 Transformers and Parallel Decoding

Transformers were introduced by Vaswani et al. [47] as a new attention-based building block for machine translation. Attention mechanisms [2] are neural network layers that aggregate information from the entire input sequence. Transformers introduced self-attention layers, which, similarly to Non-Local Neural Networks [49], scan through each element of a sequence and update it by aggregating information from the whole sequence. One of the main advantages of attention-based models is their global computations and perfect memory, which makes them more suitable than RNNs on long sequences. Transformers are now replacing RNNs in many problems in natural language processing, speech processing and computer vision [8,27,45,34,31].

Transformers由Vaswani等[47]提出，作为机器翻译的一种新的基于注意力机制的组件。注意力机制是一种神经网络层，从整个输入序列中聚积信息。Transformers引入了自注意力层，这与非局部神经网络类似，扫描一个序列中的每个元素，通过从整个序列中聚积信息，从而进行更新。基于注意力的模型的一个主要优势，是其全局计算和完美的记忆，这使其在长序列中比RNNs更合适。Transformers在很多问题中都在替换RNNs，包括NLP，语音处理和计算机视觉。

Transformers were first used in auto-regressive models, following early sequence-to-sequence models [44], generating output tokens one by one. However, the prohibitive inference cost (proportional to output length, and hard to batch) lead to the development of parallel sequence generation, in the domains of audio [29], machine translation [12,10], word representation learning [8], and more recently speech recognition [6]. We also combine transformers and parallel decoding for their suitable trade-off between computational cost and the ability to perform the global computations required for set prediction.

Transformers首先用于自回归模型中，根据早期的序列到序列模型，逐一生成输出tokens。但是，高昂的推理代价（与输出长度成正比，很难成批进行）带来了并行序列生成的提出，应用到了音频处理，机器翻译，词表示学习，和最近的语音识别中。我们也将transformers和并行解码结合起来，在计算代价和其进行全局计算的能力进行合适的折中。

### 2.3 Object detection

Most modern object detection methods make predictions relative to some initial guesses. Two-stage detectors [37,5] predict boxes w.r.t. proposals, whereas single-stage methods make predictions w.r.t. anchors [23] or a grid of possible object centers [53,46]. Recent work [52] demonstrate that the final performance of these systems heavily depends on the exact way these initial guesses are set. In our model we are able to remove this hand-crafted process and streamline the detection process by directly predicting the set of detections with absolute box prediction w.r.t. the input image rather than an anchor.

多数现代目标检测方法都是相对于一些初始猜测进行预测。两阶段检测器相对于建议来预测框，而单阶段方法相对于锚框（或可能的目标中心的网格）来进行预测。最近的工作证明了，这些系统的最终性能，都严重依赖于这些初始猜测。在我们的模型中，我们可以去除这些手工设计的过程，使得检测过程更加流线型，方法是直接预测检测集合，用的是相对于原始图像的绝对框预测，而不是相对于锚框。

**Set-based loss**. Several object detectors [9,25,35] used the bipartite matching loss. However, in these early deep learning models, the relation between different prediction was modeled with convolutional or fully-connected layers only and a hand-designed NMS post-processing can improve their performance. More recent detectors [37,23,53] use non-unique assignment rules between ground truth and predictions together with an NMS.

**基于集合的损失**。几种目标检测器都使用了bipartite匹配损失，但是，在这些早期的深度学习模型中，不同的预测之间的关系都是用卷积或全连接层建模的，手工设计的NMS后处理可以改进其性能。最近的检测器在真值和预测之间使用非唯一指定规则和NMS。

Learnable NMS methods [16,4] and relation networks [17] explicitly model relations between different predictions with attention. Using direct set losses, they do not require any post-processing steps. However, these methods employ additional hand-crafted context features like proposal box coordinates to model relations between detections efficiently, while we look for solutions that reduce the prior knowledge encoded in the model.

可学习的NMS方法和关系网络，在不同的使用注意力的预测之间显式的建模关系。使用直接的集合损失，他们不需要任何后处理步骤。但是，这些方法采用额外的手工设计的上下文特征，如建议的框坐标之间，来在检测之间高效的建模关系，而我们寻找的解决方案，则要减少在模型中编码的先验知识。

**Recurrent detectors**. Closest to our approach are end-to-end set predictions for object detection [43] and instance segmentation [41,30,36,42]. Similarly to us, they use bipartite-matching losses with encoder-decoder architectures based on CNN activations to directly produce a set of bounding boxes. These approaches, however, were only evaluated on small datasets and not against modern baselines. In particular, they are based on autoregressive models (more precisely RNNs), so they do not leverage the recent transformers with parallel decoding.

**循环检测器**。与我们的方法最接近的是，用于目标检测和实例分割的端到端的集合预测。与我们的类似，他们使用的是bipartite匹配损失，和基于CNN激活的编码器-解码器架构，以直接产生边界框的集合。这些方法，都是在小型数据集上评估的，也没有与现代基准进行比较。特别是，他们是基于自回归模型的（更具体的来说，是RNNs），所以他们并没有使用最近的带有并行编码的transformers。

## 3 The DETR model

Two ingredients are essential for direct set predictions in detection: (1) a set prediction loss that forces unique matching between predicted and ground truth boxes; (2) an architecture that predicts (in a single pass) a set of objects and models their relation. We describe our architecture in detail in Figure 2.

对于检测中的直接集合预测，有两个组成部分非常关键：(1)集合预测损失，迫使预测框和真值框有唯一的匹配；(2)一次性预测目标集合并对其关系进行建模的一个架构。我们在图2中给出了架构的细节。

### 3.1 Object detection set prediction loss

DETR infers a fixed-size set of N predictions, in a single pass through the decoder, where N is set to be significantly larger than the typical number of objects in an image. One of the main difficulties of training is to score predicted objects (class, position, size) with respect to the ground truth. Our loss produces an optimal bipartite matching between predicted and ground truth objects, and then optimize object-specific (bounding box) losses.

DETR推理得到固定大小的N个预测集合，通过解码器在一次推理中得到，其中N的数值比图像中目标的典型数量要大很多。训练的一个主要困难是，得到预测目标相对于真值的分数（类别，位置，大小）。我们的损失，在预测和真值目标之间产生一个最优bipartite匹配，然后优化具体目标（边界框）损失。

Let us denote by y the ground truth set of objects, and $\hat y = \{ \hat y_i \} _{i=1}^N$ the set of N predictions. Assuming N is larger than the number of objects in the image, we consider y also as a set of size N padded with ∅ (no object). To find a bipartite matching between these two sets we search for a permutation of N elements σ ∈ Σ_N with the lowest cost:

我们用y表示真值目标集，用$\hat y = \{ \hat y_i \} _{i=1}^N$表示N个预测的集合。假设N比图像中的目标数量要大，我们认为y中还包含数个∅（无目标），集合元素数量也为N。为在这两个集合中找到一种bipartite匹配，我们搜索N个元素的排列组合σ ∈ Σ_N，找到最低代价的：

$$\hat σ = argmin_{σ ∈ Σ_N} \sum_i^N L_{match} (y_i, \hat y_{σ(i)})$$(1)

where $L_{match} (y_i, \hat y_{σ(i)})$ is a pair-wise matching cost between ground truth $y_i$ and a prediction with index σ(i). This optimal assignment is computed efficiently with the Hungarian algorithm, following prior work (e.g. [43]).

其中$L_{match} (y_i, \hat y_{σ(i)})$是一个在真值$y_i$和带有索引σ(i)的一个预测之间的成对匹配代价。这种最优指定可以使用Hungarian算法进行高效的计算。

The matching cost takes into account both the class prediction and the similarity of predicted and ground truth boxes. Each element i of the ground truth set can be seen as a $y_i = (c_i, b_i)$ where $c_i$ is the target class label (which may be ∅) and $b_i ∈ [0,1]^4$ is a vector that defines ground truth box center coordinates and its height and width relative to the image size. For the prediction with index σ(i) we define probability of class $c_i$ as $\hat p_{σ(i)} (c_i)$ and the predicted box as $\hat b_{σ(i)}$. With these notations we define $L_{match} (y_i, \hat y_{σ(i)})$ as $-1_{c_i \neq ∅} \hat p_{σ(i)} (c_i) + 1_{c_i \neq ∅} L_{box} (b_i, \hat b_{σ(i)})$.

匹配代价考虑了类别预测和预测框和真值框之间的相似度。真值集合的每个元素i可以视为$y_i = (c_i, b_i)$，其中$c_i$是目标类别标签（可能是∅），$b_i ∈ [0,1]^4$是一个向量，定义了相对于图像大小的真值框的中心坐标，及其高度和宽度。对于索引为σ(i)的预测，我们定义了类别$c_i$的概率为$\hat p_{σ(i)} (c_i)$，预测的框为$\hat b_{σ(i)}$。有了这些概念，我们定义$L_{match} (y_i, \hat y_{σ(i)})$为$-1_{c_i \neq ∅} \hat p_{σ(i)} (c_i) + 1_{c_i \neq ∅} L_{box} (b_i, \hat b_{σ(i)})$。

This procedure of finding matching plays the same role as the heuristic assignment rules used to match proposal [37] or anchors [22] to ground truth objects in modern detectors. The main difference is that we need to find one-to-one matching for direct set prediction without duplicates.

寻找匹配的这个过程，与启发式的指定规则扮演相同的角色，用于将建议或锚框与真值目标进行匹配，与现代检测器中一样。主要的区别是，我们需要对直接集合预测找到一对一的匹配，不能有重复。

The second step is to compute the loss function, the Hungarian loss for all pairs matched in the previous step. We define the loss similarly to the losses of common object detectors, i.e. a linear combination of a negative log-likelihood for class prediction and a box loss defined later:

第二个步骤是计算损失函数，即在之前步骤中匹配上的所有对的Hungarian损失。我们定义的损失与常见的目标检测器类似，即类别预测和边界框损失的负log似然的组合：

$$L_{Hungarian} (y, \hat y) = \sum_{i=1}^N [-log \hat p_{σ(i)} (c_i) + 1_{c_i \neq ∅} L_{box} (b_i, \hat b_{σ(i)})]$$(2)

where $\hat σ$ is the optimal assignment computed in the first step (1). In practice, we down-weight the log-probability term when c_i = ∅ by a factor 10 to account for class imbalance. This is analogous to how Faster R-CNN training procedure balances positive/negative proposals by subsampling [37]. Notice that the matching cost between an object and ∅ doesn’t depend on the prediction, which means that in that case the cost is a constant. In the matching cost we use probabilities $\hat p_{σ(i)} (c_i)$ instead of log-probabilities. This makes the class prediction term commensurable to L_box(·, ·) (described below), and we observed better empirical performances.

其中$\hat σ$是步骤(1)中计算得到的最佳指定。在实践中，我们在c_i = ∅时，降低log概率项的权重10倍，以考虑类别不均衡的问题。这与Faster R-CNN训练过程中正负建议下采样的均衡过程是类似的。注意，目标与∅的匹配代价并不依赖于预测，这意味着在那种情况下，代价是一个常数。在匹配代价中，我们使用概率$\hat p_{σ(i)} (c_i)$，代替了log概率。这使得类别预测项与L_box(·, ·)可以相比，我们也得到了更好的经验性能。

**Bounding box loss**. The second part of the matching cost and the Hungarian loss is L_box(·) that scores the bounding boxes. Unlike many detectors that do box predictions as a ∆ w.r.t. some initial guesses, we make box predictions directly. While such approach simplify the implementation it poses an issue with relative scaling of the loss. The most commonly-used l1 loss will have different scales for small and large boxes even if their relative errors are similar. To mitigate this issue we use a linear combination of the l1 loss and the generalized IoU loss [38] $L_{iou}(·,·)$ that is scale-invariant. Overall, our box loss is $L_{box}(b_i, \hat b_{σ(i)})$ defined as $λ_{iou} L_{iou} (b_i, \hat b_{σ(i)}) + λ_{L1} ||b_i − \hat b_{σ(i)}||_1$ where $λ_{iou}, λ_{L1}$ ∈ R are hyperparameters.
These two losses are normalized by the number of objects inside the batch.

**边界框损失**。匹配代价和Hungarian损失的第二部分是L_box(·)，对边界框进行打分。很多检测预测框的过程是相对于一些初始猜测进行一些偏移，本文与之不同，我们直接进行边界框预测。这种方法简化了实现，但对损失函数相对缩放提出了问题。最常使用的l1损失对于大的框和小的框有不同的尺度，即使其相对误差是类似的。为缓解这个问题，我们使用l1损失和泛化IoU损失的线性组合，$L_{iou}(·,·)$是尺度不变的。总体上，我们的边界框损失$L_{box}(b_i, \hat b_{σ(i)})$，定义为$λ_{iou} L_{iou} (b_i, \hat b_{σ(i)}) + λ_{L1} ||b_i − \hat b_{σ(i)}||_1$，其中$λ_{iou}, λ_{L1}$ ∈ R是超参数。这两种损失都由批次内的目标数量进行归一化。

### 3.2 DETR architecture

The overall DETR architecture is surprisingly simple and depicted in Figure 2. It contains three main components, which we describe below: a CNN backbone to extract a compact feature representation, an encoder-decoder transformer, and a simple feed forward network (FFN) that makes the final detection prediction.

DETR的总体架构非常简单，如图2所示。包含三个主要部分，如下所述：一个CNN骨干网络提取紧凑的特征表示，一个编码器-解码器transformer，和一个简单的前向网络FFN，进行最终的检测预测。

Unlike many modern detectors, DETR can be implemented in any deep learning framework that provides a common CNN backbone and a transformer architecture implementation with just a few hundred lines. Inference code for DETR can be implemented in less than 50 lines in PyTorch [32]. We hope that the simplicity of our method will attract new researchers to the detection community.

与很多现代检测器不同，DETR可以在任意深度学习框架中实现，只要框架提供了常见的CNN骨干和transformer架构的实现，只要几百行代码就可以了。DETR的推理代码在PyTorch中可以用少于50行代码实现。我们希望方法的简单性会吸引更多的研究来研究检测。

**Backbone**. Starting from the initial image $x_{img} ∈ R^{3×H_0×W_0}$ (with 3 color channels), a conventional CNN backbone generates a lower-resolution activation map $f ∈ R^{C×H×W}$. Typical values we use are C = 2048 and H, W =  H0/32, W0/32.

**骨干**。从初始图像$x_{img} ∈ R^{3×H_0×W_0}$开始（三种色彩），传统CNN骨干生成一种更低分辨率的激活图$f ∈ R^{C×H×W}$。典型值我们使用C=2048，H, W =  H0/32, W0/32。

**Transformer encoder**. First, a 1x1 convolution reduces the channel dimension of the high-level activation map f from C to a smaller dimension d. creating a new feature map $z_0 ∈ R^{d×H×W}$. The encoder expects a sequence as input, hence we collapse the spatial dimensions of z0 into one dimension, resulting in a d×HW feature map. Each encoder layer has a standard architecture and consists of a multi-head self-attention module and a feed forward network (FFN). Since the transformer architecture is permutation-invariant, we supplement it with fixed positional encodings [31,3] that are added to the input of each attention layer. We defer to the supplementary material the detailed definition of the architecture, which follows the one described in [47].

**Transformer编码器**。首先，1x1卷积降低了高层激活图f的通道维度，从C到更小的维度d，生成了一个新的特征图$z_0 ∈ R^{d×H×W}$。编码器以一个序列作为输入，这里我们将z0的空间维度拉平到一维，得到了一个d×HW的特征图。每个编码器层有一个标准架构，由多头自注意力模块和一个前向网络FFN组成。由于transformer架构是对排列组合不变的，我们附上了固定的位置编码，加到了每个注意力层的输入中。我们在附录中给出了架构的详细定义，与[47]中的类似。

**Transformer decoder**. The decoder follows the standard architecture of the transformer, transforming N embeddings of size d using multi-headed self- and encoder-decoder attention mechanisms. The difference with the original transformer is that our model decodes the N objects in parallel at each decoder layer, while Vaswani et al. [47] use an autoregressive model that predicts the output sequence one element at a time. We refer the reader unfamiliar with the concepts to the supplementary material. Since the decoder is also permutation-invariant, the N input embeddings must be different to produce different results. These input embeddings are learnt positional encodings that we refer to as object queries, and similarly to the encoder, we add them to the input of each attention layer. The N object queries are transformed into an output embedding by the decoder. They are then independently decoded into box coordinates and class labels by a feed forward network (described in the next subsection), resulting N final predictions. Using self- and encoder-decoder attention over these embeddings, the model globally reasons about all objects together using pair-wise relations between them, while being able to use the whole image as context.

**Transformer解码器**。解码器是transformer的标准架构，将大小为d的N个嵌入，使用多头自注意力和编码器-解码器注意力机制进行transform。与原始transformer之前的差异是，我们的模型在每个解码器层中对N个目标进行并行解码，而Vaswani等[47]使用了一个自回归模型来预测输入序列，一次预测一个元素。我们推荐对这个概念不熟悉的读者参考附录材料。由于解码器也是对排列不变的，所以N个输入嵌入需要是不一样的，来生成不同的结果。这些输入嵌入是学习得到的位置编码，我们称之为目标queries，与编码器类似，我们将之加入到每个注意力层的输入中。N个目标queries由解码器变换到输出嵌入。它们由一个前向网络（下一节介绍）独立的解码成框的坐标和类别标签，得到N个最终的预测结果。在这些嵌入上使用自注意力和编码器-解码器注意力，模型会使用所有目标之间成对的关系，在全局上对所有目标进行推理，而使用整幅图作为上下文。

**Prediction feed-forward networks (FFNs)**. The final prediction is computed by a 3-layer perceptron with ReLU activation function and hidden dimension d, and a linear projection layer. The FFN predicts the normalized center coordinates, height and width of the box w.r.t. the input image, and the linear layer predicts the class label using a softmax function. Since we predict a fixed-size set of N bounding boxes, where N is usually much larger than the actual number of objects of interest in an image, an additional special class label ∅ is used to represent that no object is detected within a slot. This class plays a similar role to the “background” class in the standard object detection approaches.

**预测前向网络FFN**。最终的预测是由一个带有ReLU激活函数的3层感知机，隐藏层维度为d，和一个线性投影层计算的。FFN预测的是相对输入图像的，归一化的中心坐标，和框的高度和宽度，线性层使用softmax函数来预测类别标签。由于我们预测的是固定数量的N个边界框的集合，其中N通常比图像中实际的感兴趣目标数量要大的多，所以使用的额外的特殊类别标签∅来表示在一个slot中没有检测到任何物体。这个类别与标准目标检测方法中的背景类别的作用类似。

**Auxiliary decoding losses**. We found helpful to use auxiliary losses [1] in decoder during training, especially to help the model output the correct number of objects of each class. We add prediction FFNs and Hungarian loss after each decoder layer. All predictions FFNs share their parameters. We use an additional shared layer-norm to normalize the input to the prediction FFNs from different decoder layers.

**辅助解码器损失**。对解码器使用辅助损失，我们发现在训练过程中是有帮助的，尤其是对模型输出每个类别中正确数量的目标有帮助。我们在每个解码层后加上了预测FFNs和Hungarian损失。所有预测FFNs共享参数。我们使用一个额外的共享的层范数来对预测FFNs的输入进行归一化。

## 4. Experiments

We show that DETR achieves competitive results compared to Faster R-CNN in quantitative evaluation on COCO. Then, we provide a detailed ablation study of the architecture and loss, with insights and qualitative results. Finally, to show that DETR is a versatile and extensible model, we present results on panoptic segmentation, training only a small extension on a fixed DETR model. We provide code and pretrained models to reproduce our experiments at https://github.com/facebookresearch/detr.

我们展示DETR在COCO的量化评估中获得与Faster R-CNN类似的结果。然后，我们对架构和损失函数进行了详细的剥离分析，包括洞见和量化结果。最后，为展示DETR是一个多样的可扩展的模型，我们给出了在全景分割中的结果，在一个固定的DETR模型中训练了一个小的拓展。代码和预训练模型已经开源。

**Dataset**. We perform experiments on COCO 2017 detection and panoptic segmentation datasets [24,18], containing 118k training images and 5k validation images. Each image is annotated with bounding boxes and panoptic segmentation. There are 7 instances per image on average, up to 63 instances in a single image in training set, ranging from small to large on the same images. If not specified, we report AP as bbox AP, the integral metric over multiple thresholds. For comparison with Faster R-CNN we report validation AP at the last training epoch, for ablations we report median over validation results from the last 10 epochs.

**数据集**。我们在COCO 2017检测和全景分割数据集上进行试验，包含118k训练图像，5k验证图像。每幅图像都用边界框和全景分割进行了标注。训练集中平均每幅图像有7个实例，最多63个实例。如果没有指定，我们说的AP是bbox AP，多个阈值上的积分度量。为与Faster R-CNN进行比较，我们在最后的训练epoch上给出验证AP，对于分离实验，我们给出验证集上最后10个epochs的中值。

**Technical details**. We train DETR with AdamW [26] setting the initial transformer’s learning rate to 10^−4, the backbone’s to 10^−5, and weight decay to 10^−4. All transformer weights are initialized with Xavier init [11], and the backbone is with ImageNet-pretrained ResNet model [15] from torchvision with frozen batchnorm layers. We report results with two different backbones: a ResNet-50 and a ResNet-101. The corresponding models are called respectively DETR and DETR-R101. Following [21], we also increase the feature resolution by adding a dilation to the last stage of the backbone and removing a stride from the first convolution of this stage. The corresponding models are called respectively DETR-DC5 and DETR-DC5-R101 (dilated C5 stage). This modification increases the resolution by a factor of two, thus improving performance for small objects, at the cost of a 16x higher cost in the self-attentions of the encoder, leading to an overall 2x increase in computational cost. A full comparison of FLOPs of these models and Faster R-CNN is given in Table 1.

**技术细节**。我们用AdamW训练DETR，transformer的初始学习速率为10^-4，骨干的初始学习速率为10^-5，权重衰减为10^-4。所有transformer权重用Xavier init[11]进行初始化，骨干用torchvision中ImageNet预训练的ResNet模型，带有frozen batchnorm层。我们给出了两种不同的骨干网络的结果：一个ResNet-50，一个ResNet-101。对应的模型分别称为DETR和DETR-R101。按照[21]，我们还增加了特征分辨率，对骨干的最后阶段增加了一个dilation，对这个阶段的第一个卷积，去掉了步长。对应的模型我们分别称之为DETR-DC5和DETR-DC5-R101。这种修正将分辨率增加到2倍，因此改进了对小目标的性能，代价是在编码器的自注意力中的代价高了16倍，带来了总体上2倍的计算代价。表1比较了这些模型和Faster R-CNN的FLOPs。

We use scale augmentation, resizing the input images such that the shortest side is at least 480 and at most 800 pixels while the longest at most 1333 [50]. To help learning global relationships through the self-attention of the encoder, we also apply random crop augmentations during training, improving the performance by approximately 1 AP. Specifically, a train image is cropped with probability 0.5 to a random rectangular patch which is then resized again to 800-1333. The transformer is trained with default dropout of 0.1. At inference time, some slots predict empty class. To optimize for AP, we override the prediction of these slots with the second highest scoring class, using the corresponding confidence. This improves AP by 2 points compared to filtering out empty slots. Other training hyperparameters can be found in section A.4. For our ablation experiments we use training schedule of 300 epochs with a learning rate drop by a factor of 10 after 200 epochs, where a single epoch is a pass over all training images once. Training the baseline model for 300 epochs on 16 V100 GPUs takes 3 days, with 4 images per GPU (hence a total batch size of 64). For the longer schedule used to compare with Faster R-CNN we train for 500 epochs with learning rate drop after 400 epochs. This schedule adds 1.5 AP compared to the shorter schedule.

我们使用了尺度扩充，将输入图像大小进行变换，这样其短边至少480像素，最大800像素，长边最长1333像素。为通过编码器的自注意力来帮助学习全局关系，我们还在训练过程中使用随机剪切扩充，大约改进了1 AP的性能。具体的，训练图像以0.5的概率，剪切为一个随机矩形块，然后再改变到800-1333的大小。Transformer用默认的dropout 0.1进行训练。在推理时，一些slots预测到了空的类。为优化AP，我们用第二高分数的类别覆盖了这些slots的预测，使用对应的置信度。与滤除掉空slots相比，这有了2点的改进。其他训练超参数可以在A.4节中找到。对分离实验，我们使用的训练方案为300 epochs，学习率在200 epochs后除以10，其中单个epoch是将所有训练图像训练一次的过程。训练基准模型300 epochs在16个V100 GPUs上耗时3天，每个GPU 4幅图像（总计batch size为64）。由于训练方案变长了，我们比较的Faster R-CNN训练了500轮，学习速率在400轮后降低。这种方案与较短的方案相比，提高了1.5 AP。

### 4.1 Comparison with Faster R-CNN

Transformers are typically trained with Adam or Adagrad optimizers with very long training schedules and dropout, and this is true for DETR as well. Faster R-CNN, however, is trained with SGD with minimal data augmentation and we are not aware of successful applications of Adam or dropout. Despite these differences we attempt to make a Faster R-CNN baseline stronger. To align it with DETR, we add generalized IoU [38] to the box loss, the same random crop augmentation and long training known to improve results [13]. Results are presented in Table 1. In the top section we show Faster R-CNN results from Detectron2 Model Zoo [50] for models trained with the 3x schedule. In the middle section we show results (with a “+”) for the same models but trained with the 9x schedule (109 epochs) and the described enhancements, which in total adds 1-2 AP. In the last section of Table 1 we show the results for multiple DETR models. To be comparable in the number of parameters we choose a model with 6 transformer and 6 decoder layers of width 256 with 8 attention heads. Like Faster R-CNN with FPN this model has 41.3M parameters, out of which 23.5M are in ResNet-50, and 17.8M are in the transformer. Even though both Faster R-CNN and DETR are still likely to further improve with longer training, we can conclude that DETR can be competitive with Faster R-CNN with the same number of parameters, achieving 42 AP on the COCO val subset. The way DETR achieves this is by improving APL (+7.8), however note that the model is still lagging behind in APS (-5.5). DETR-DC5 with the same number of parameters and similar FLOP count has higher AP, but is still significantly behind in APS too. Faster R-CNN and DETR with ResNet-101 backbone show comparable results as well.

### 4.2 Ablations

Attention mechanisms in the transformer decoder are the key components which model relations between feature representations of different detections. In our ablation analysis, we explore how other components of our architecture and loss influence the final performance. For the study we choose ResNet-50-based DETR model with 6 encoder, 6 decoder layers and width 256. The model has 41.3M parameters, achieves 40.6 and 42.0 AP on short and long schedules respectively, and runs at 28 FPS, similarly to Faster R-CNN-FPN with the same backbone.

Transformer解码器中的注意力机制是非常关键的部件，对不同检测的特征表示的关系进行建模。在我们的分离分析中，我们探索了我们架构和损失函数的其他部件怎样影响最终的性能。研究中，我们选择了基于ResNet-50的DETR模型，有6个编码器，6个解码器，宽度256。模型有41.3M参数，在短期和长期的训练方案中分别获得了40.6和42.0 AP，运行速度28 FPS，与相同骨干的Faster R-CNN-FPN类似。

**Number of encoder layers**. We evaluate the importance of global image-level self-attention by changing the number of encoder layers (Table 2). Without encoder layers, overall AP drops by 3.9 points, with a more significant drop of 6.0 AP on large objects. We hypothesize that, by using global scene reasoning, the encoder is important for disentangling objects. In Figure 3, we visualize the attention maps of the last encoder layer of a trained model, focusing on a few points in the image. The encoder seems to separate instances already, which likely simplifies object extraction and localization for the decoder.

**编码器层的数量**。我们通过改变编码器层的数量，评估全局图像层次的自注意力的重要性（表2）。在没有编码器层的情况下，总体AP下降了3.9点，大目标下降的更多，达到了6.0 AP。我们假设，使用了全局场景推理，编码器对于区分目标是很重要的。在图3中，我们对训练好的模型的最后一个编码器层的注意力图进行了可视化，聚焦在图像中的几个点上。编码器似乎已经分离出了实例，这很可能简化了解码器的目标提取和定位。

**Number of decoder layers**. We apply auxiliary losses after each decoding layer (see Section 3.2), hence, the prediction FFNs are trained by design to predict objects out of the outputs of every decoder layer. We analyze the importance of each decoder layer by evaluating the objects that would be predicted at each stage of the decoding (Fig. 4). Both AP and AP50 improve after every layer, totalling into a very significant +8.2/9.5 AP improvement between the first and the last layer. With its set-based loss, DETR does not need NMS by design. To verify this we run a standard NMS procedure with default parameters [50] for the outputs after each decoder. NMS improves performance for the predictions from the first decoder. This can be explained by the fact that a single decoding layer of the transformer is not able to compute any cross-correlations between the output elements, and thus it is prone to making multiple predictions for the same object. In the second and subsequent layers, the self-attention mechanism over the activations allows the model to inhibit duplicate predictions. We observe that the improvement brought by NMS diminishes as depth increases. At the last layers, we observe a small loss in AP as NMS incorrectly removes true positive predictions.

**解码器层的数量**。我们在每个解码器层后，使用了辅助损失函数（见3.2节），因此，预测FFNs在设计上就训练了，来从每个解码器层的输出中预测得到目标。我们分析了每个解码器层的重要性，评估了目标解码的每个阶段预测得到的情况（图4）。AP和AP50每多一层都有所提升，第一层和最后一层的差距达到了+8.2/9.5 AP。在基于集合的损失的基础上，DETR在设计上就不需要NMS。为验证，我们对每个解码器层的输出，用默认参数运行了标准NMS过程。NMS改进了第一个解码器的预测的结果。这是因为，一个解码器层不能计算得到输出元素的任意相互关系，因为对每个目标容易做出多个预测。在第二层和后面的层中，自注意力机制使模型可以避免重复的预测。我们观察到NMS带来的改进随着层深度的增加逐渐消失。在最后的层中，我们观察到，在使用NMS后，AP有小幅度下降，因为错误的去掉了正确的预测。

Similarly to visualizing encoder attention, we visualize decoder attentions in Fig. 6, coloring attention maps for each predicted object in different colors. We observe that decoder attention is fairly local, meaning that it mostly attends to object extremities such as heads or legs. We hypothesise that after the encoder has separated instances via global attention, the decoder only needs to attend to the extremities to extract the class and object boundaries.

与编码器注意力的可视化类似，我们在图6中可视化了解码器的注意力，对每个预测目标以不同的颜色显示了注意力图。我们观察到，解码器注意力是很局部的，意味着其通常关注的是目标的极限点，比如头或腿。我们假设，在编码器通过全局注意力分离了实例后，解码器只需要关注极限点来提取出类别和目标的边界。

**Importance of FFN**. FFN inside tranformers can be seen as 1 × 1 convolutional layers, making encoder similar to attention augmented convolutional networks [3]. We attempt to remove it completely leaving only attention in the transformer layers. By reducing the number of network parameters from 41.3M to 28.7M, leaving only 10.8M in the transformer, performance drops by 2.3 AP, we thus conclude that FFN are important for achieving good results.

**FFN的重要性**。Transformer中的FFN可以视为1 × 1的卷积层，使编码器与注意力扩充的卷积网络类似[3]。我们倾向于完全移除之，在transformer中只留下注意力。我们将网络参数从41.3M降低到了28.7M，只留下了transformer中的10.8M参数，性能下降了2.3 AP，我们因此得到结论，FFN对于得到很好的结果是很重要的。

**Importance of positional encodings**. There are two kinds of positional encodings in our model: spatial positional encodings and output positional encodings (object queries). We experiment with various combinations of fixed and learned encodings, results can be found in table 3. Output positional encodings are required and cannot be removed, so we experiment with either passing them once at decoder input or adding to queries at every decoder attention layer. In the first experiment we completely remove spatial positional encodings and pass output positional encodings at input and, interestingly, the model still achieves more than 32 AP, losing 7.8 AP to the baseline. Then, we pass fixed sine spatial positional encodings and the output encodings at input once, as in the original transformer [47], and find that this leads to 1.4 AP drop compared to passing the positional encodings directly in attention. Learned spatial encodings passed to the attentions give similar results. Surprisingly, we find that not passing any spatial encodings in the encoder only leads to a minor AP drop of 1.3 AP. When we pass the encodings to the attentions, they are shared across all layers, and the output encodings (object queries) are always learned.

**位置编码的重要性**。在我们的模型中有两种位置编码：空间位置编码和输出位置编码（目标queries）。我们用固定编码和学习编码的各种组合来进行实验，结果如表3所示。输出位置编码是必要的，不能去掉，所以我们要么一次性在解码器输入中传入，或在每个解码器注意力层中加上queries。在第一个试验中，我们完移除了空间位置编码，在输入端传入输出位置编码，有趣的是，模型仍然获得了32 AP，比基准降低了7.8 AP。然后，我们传入固定的sine空间位置编码，输出编码是一次性输入的，和原始transformer是一样的，我们发现这样性能下降了1.4 AP。学习位置编码传入注意力给出了类似的结果。令人惊奇的是，我们发现，对编码器不传入任何空间编码只下降了1.3 AP。当我们将编码传入到注意力中，他们在所有层中都共享的，输出编码（目标queries）一直是学习得到的。

Given these ablations, we conclude that transformer components: the global self-attention in encoder, FFN, multiple decoder layers, and positional encodings, all significantly contribute to the final object detection performance.

有了分离实验，我们得出transformer部件的结论：编码器中的全局自注意力，FFN，多个解码器层，和位置编码，都对最终的目标检测性能有显著影响。

**Loss ablations**. To evaluate the importance of different components of the matching cost and the loss, we train several models turning them on and off. There are three components to the loss: classification loss, ℓ1 bounding box distance loss, and GIoU [38] loss. The classification loss is essential for training and cannot be turned off, so we train a model without bounding box distance loss, and a model without the GIoU loss, and compare with baseline, trained with all three losses. Results are presented in table 4. GIoU loss on its own accounts for most of the model performance, losing only 0.7 AP to the baseline with combined losses. Using ℓ1 without GIoU shows poor results. We only studied simple ablations of different losses (using the same weighting every time), but other means of combining them may achieve different results.

**损失分离**。为评估匹配代价和损失的各个不同部分的重要性，我们进行了不同的选择，训练了几个模型。损失函数有三个组成部分：分类损失，ℓ1边界框距离损失，GIoU损失。分类损失对于训练是必备的，因此不能取消，所以我们训练了一个模型，其中没有边界框距离损失，和一个没有GIoU损失的模型，和带有三个部分的基准进行比较。结果如表4所示。GIoU损失是模型性能的支柱，与基准相比，只损失了0.7 AP。不使用GIoU得到了很差的结果。我们只简单研究了不同损失的组合（每次使用相同的加权），但其他组合方法可能得到不同的结果。

### 4.3 Analysis

**Decoder output slot analysis**. In Fig. 7 we visualize the boxes predicted by different slots for all images in COCO 2017 val set. DETR learns different specialization for each query slot. We observe that each slot has several modes of operation focusing on different areas and box sizes. In particular, all slots have the mode for predicting image-wide boxes (visible as the red dots aligned in the middle of the plot). We hypothesize that this is related to the distribution of objects in COCO.

**解码器输出slot分析**。图7中，我们将不同slots对COCO 2017验证集上所有图像的预测框进行了可视化。DETR对不同的query slot学习了不同的专门化。我们观察到，每个slot都有不同的运算模式，聚焦在不同的区域和框大小上。特别是，所有的slots都有预测图像范围内的框的模式（图中间的红色点）。我们假设，这是与COCO中目标的大小分布有关的。

**Generalization to unseen numbers of instances**. Some classes in COCO are not well represented with many instances of the same class in the same image. For example, there is no image with more than 13 giraffes in the training set. We create a synthetic image to verify the generalization ability of DETR (see Figure 5). Our model is able to find all 24 giraffes on the image which is clearly out of distribution. This experiment confirms that there is no strong class-specialization in each object query.

**泛化到未曾见过的实例数量**。COCO中的一些类别，在同样的图像中没有被同样类别中的很多实例很好的表示。比如，在训练集中，没有哪个图像超过了13个长颈鹿。我们生成了一种合成图像，来验证DETR的泛化能力（见图5）。我们的模型可以找到图中的所有24个长颈鹿，这是在已有数据分布之外的。这个实验证明了，在每个目标query中，没有很强的类别专门化。

### 4.4 DETR for panoptic segmentation

Panoptic segmentation [19] has recently attracted a lot of attention from the computer vision community. Similarly to the extension of Faster R-CNN [37] to Mask R-CNN [14], DETR can be naturally extended by adding a mask head on top of the decoder outputs. In this section we demonstrate that such a head can be used to produce panoptic segmentation [19] by treating stuff and thing classes in a unified way. We perform our experiments on the panoptic annotations of the COCO dataset that has 53 stuff categories in addition to 80 things categories.

全景分割[19]最近吸引了很多注意力。与Faster R-CNN到Mask R-CNN的拓展类似，DETR可以很自然进行拓展，在解码器输出上增加一个掩模头。本节中，我们证明了，这样一个头可以用于生成全景分割，将stuff和thing类别以统一的方式进行对待。我们在COCO数据集上进行全景分割实验，有53个stuff类别，和80个things的类别。

We train DETR to predict boxes around both stuff and things classes on COCO, using the same recipe. Predicting boxes is required for the training to be possible, since the Hungarian matching is computed using distances between boxes. We also add a mask head which predicts a binary mask for each of the predicted boxes, see Figure 8. It takes as input the output of transformer decoder for each object and computes multi-head (with M heads) attention scores of this embedding over the output of the encoder, generating M attention heatmaps per object in a small resolution. To make the final prediction and increase the resolution, an FPN-like architecture is used. We describe the architecture in more details in the supplement. The final resolution of the masks has stride 4 and each mask is supervised independently using the DICE/F-1 loss [28] and Focal loss [23].

我们训练DETR来在COCO的stuff和things的类别中预测框，使用同样的方案。要使训练可行，需要预测框，由于Hungarian匹配的计算是使用框之间的距离。我们还加上了一个掩模头，对每个预测的框预测一个二值掩模，如图8所示。它的输入是transformer解码器对每个目标的输出，计算这个嵌入到编码器输出的多头注意力分数（有M个头），对每个目标以低分辨率生成M个注意力热力图。为进行最终预测，增加分辨率，使用了一个类似于FPN的架构。我们在附录中详述这个架构。掩模的最终分辨率步长为4，每个掩模都使用DICE/F-1损失和Focal损失来独立进行监督。

The mask head can be trained either jointly, or in a two steps process, where we train DETR for boxes only, then freeze all the weights and train only the mask head for 25 epochs. Experimentally, these two approaches give similar results, we report results using the latter method since it results in a shorter total wall-clock time training.

掩模头可以联合进行训练，或以两步的过程进行训练，在两步过程中，我们只对框训练DETR，然后冻结这些权重，只训练掩模头25 epoch。试验上，这两个过程给出了类似的结果，我们使用后面的方法，因为其训练所需时间比较短。

To predict the final panoptic segmentation we simply use an argmax over the mask scores at each pixel, and assign the corresponding categories to the resulting masks. This procedure guarantees that the final masks have no overlaps and, therefore, DETR does not require a heuristic [19] that is often used to align different masks.

为预测最终的全景分割，我们简单的在每个像素中对掩模分数使用一个argmax，对结果掩模指定对应的类别。这个过程确保了，最终掩模不会重叠，因此，DETR不需要heuristics来对齐不同的掩模。

**Training details**. We train DETR, DETR-DC5 and DETR-R101 models following the recipe for bounding box detection to predict boxes around stuff and things classes in COCO dataset. The new mask head is trained for 25 epochs (see supplementary for details). During inference we first filter out the detection with a confidence below 85%, then compute the per-pixel argmax to determine in which mask each pixel belongs. We then collapse different mask predictions of the same stuff category in one, and filter the empty ones (less than 4 pixels).

**训练细节**。我们训练DETR，DETR-DC5和DETR-R101模型的方案，是边界框检测以预测COCO数据集中的stuff和things类别的框。新的掩模头训练25 epochs。在推理过程中，我们首先滤除置信度低于85%的检测，然后计算逐像素的argmax，以确定每个像素属于哪个掩模。我们然后将同样的stuff类别的不同掩模预测合并成一个，然后滤掉空的（少于4个像素）。

**Main results**. Qualitative results are shown in Figure 9. In table 5 we compare our unified panoptic segmenation approach with several established methods that treat things and stuff differently. We report the Panoptic Quality (PQ) and the break-down on things (PQth) and stuff (PQst). We also report the mask AP (computed on the things classes), before any panoptic post-treatment (in our case, before taking the pixel-wise argmax). We show that DETR outperforms published results on COCO-val 2017, as well as our strong PanopticFPN baseline (trained with same data-augmentation as DETR, for fair comparison). The result break-down shows that DETR is especially dominant on stuff classes, and we hypothesize that the global reasoning allowed by the encoder attention is the key element to this result. For things class, despite a severe deficit of up to 8 mAP compared to the baselines on the mask AP computation, DETR obtains competitive PQth. We also evaluated our method on the test set of the COCO dataset, and obtained 46 PQ. We hope that our approach will inspire the exploration of fully unified models for panoptic segmentation in future work.

**主要结果**。定性结果如图9所示。表5中，我们比较了我们的统一全景分割方法与其他几种不同的方法，它们将stuff和things区别对待。我们给出全景质量(PQ)，并区别对待things (PQth)和stuff (PQst)。我们还给出掩模AP（在things类别上计算得到），在任何全景后处理之前（在我们的情况下，是在进行逐像素的argmax之前）。我们证明了，DETR超过了在COCO 2017 val上发表过的结果，以及我们的很强的全景FPN基准（是用相同的数据扩增进行训练的，以进行公平的比较）。结果的分析表明，DETR在stuff类别中尤其给力，我们假设编码器注意力的全局推理是这个结果的关键元素。对于things类别，在与基准的比较中，其掩模AP的计算上最严重有8 mAP的下降，但DETR得到了很有竞争力的PQth。我们还在COCO数据集的测试集上评估了我们的方法，得到了46 PQ。我们希望我们的方法会启发全景分割的完全统一模型。

## 5. Conclusion

We presented DETR, a new design for object detection systems based on transformers and bipartite matching loss for direct set prediction. The approach achieves comparable results to an optimized Faster R-CNN baseline on the challenging COCO dataset. DETR is straightforward to implement and has a flexible architecture that is easily extensible to panoptic segmentation, with competitive results. In addition, it achieves significantly better performance on large objects than Faster R-CNN, likely thanks to the processing of global information performed by the self-attention.

我们提出了DETR，这是一种基于transformer和bipartite匹配损失进行直接集合预测的目标检测新设计。在COCO数据集上，这种方法的结果与Faster R-CNN不相上下。DETR实现起来很容易，架构灵活，可以很容易的拓展到全景分割，得到很好的结果。另外，在大目标上的结果，明显比Faster R-CNN要好，这很可能是因为自注意力的全局信息处理的结果。

This new design for detectors also comes with new challenges, in particular regarding training, optimization and performances on small objects. Current detectors required several years of improvements to cope with similar issues, and we expect future work to successfully address them for DETR.

这种新设计也带来了新的挑战，特别是训练，优化和小目标上的性能。目前的检测器需要几年的改进来处理类似的问题，我们期待未来的工作可以很好的处理。