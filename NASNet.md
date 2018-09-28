# Learning Transferable Architectures for Scalable Image Recognition

Barret Zoph et al. Google Brain

## Abstract 摘要

Developing neural network image classification models often requires significant architecture engineering. In this paper, we study a method to learn the model architectures directly on the dataset of interest. As this approach is expensive when the dataset is large, we propose to search for an architectural building block on a small dataset and then transfer the block to a larger dataset. The key contribution of this work is the design of a new search space (which we call the “NASNet search space”) which enables transferability. In our experiments, we search for the best convolutional layer (or “cell”) on the CIFAR-10 dataset and then apply this cell to the ImageNet dataset by stacking together more copies of this cell, each with their own parameters to design a convolutional architecture, which we name a “NASNet architecture”. We also introduce a new regularization technique called ScheduledDropPath that significantly improves generalization in the NASNet models. On CIFAR-10 itself, a NASNet found by our method achieves 2.4% error rate, which is state-of-the-art. Although the cell is not searched for directly on ImageNet, a NASNet constructed from the best cell achieves, among the published works, state-of-the-art accuracy of 82.7% top-1 and 96.2% top-5 on ImageNet. Our model is 1.2% better in top-1 accuracy than the best human-invented architectures while having 9 billion fewer FLOPS – a reduction of 28% in computational demand from the previous state-of-the-art model. When evaluated at different levels of computational cost, accuracies of NASNets exceed those of the state-of-the-art human-designed models. For instance, a small version of NASNet also achieves 74% top-1 accuracy, which is 3.1% better than equivalently-sized, state-of-the-art models for mobile platforms. Finally, the image features learned from image classification are generically useful and can be transferred to other computer vision problems. On the task of object detection, the learned features by NASNet used with the Faster-RCNN framework surpass state-of-the-art by 4.0% achieving 43.1% mAP on the COCO dataset.

开发神经网络图像分类模型经常需要进行重要的架构设计。在本文中，我们研究了一种方法直接在数据集上学习模型架构。当数据集规模很大时，这个方法非常昂贵，我们提出先在小数据集上搜索架构模块，然后再将模块转移到更大的数据集上。本文的主要贡献是设计了一个新的搜索空间，我们称之为NASNet搜索空间，这个空间中可以实现转移操作。在我们的试验中，我们在CIFAR-10数据集上搜索最好的卷积层（或cell），然后将这个cell应用到ImageNet数据集上，其方法是将这个cell的多个副本堆叠起来，每个cell都有自己的参数，这样设计出的卷积架构我们称之为NASNet架构。我们还提出了一种新的正则化方法，称之为ScheduledDropPath，可以极大的改善NASNet模型的泛化能力。在CIFAR-10上，我们的方法找到的NASNet得到的错误率为2.4%，是目前最好的。尽管这个cell不是直接在ImageNet上搜索到的，从目前最好的cell存档里构建出的NASNet，得到了目前最好的82.7%的top-1准确率和96.2%的top-5准确率。我们模型的top-1准确率比人创造的架构的最好结果高1.2%，但计算量少了90亿FLOPS，也就是比以前最好表现的模型少了28%的计算量。。当在不同层次的计算量上评估时，NASNet的准确率超过了人类设计模型的最好结果。比如，很小版本的NASNet也得到了74%的top-1准确率，这比为移动平台设计的类似规模的最好模型准确度高出3.1%。最后，图像分类中学到的图像特征在其他一般情况也很有用，可以转移用在其他计算机视觉问题中。在目标检测的任务中，NASNet学习到的特征与Faster-RCNN框架一起使用，超过目前最优成绩4.0%，在COCO数据集上达到了43.1%的mAP。

## 1. Introduction 简介

Developing neural network image classification models often requires significant architecture engineering. Starting from the seminal work of [32] on using convolutional architectures [17, 34] for ImageNet [11] classification, successive advancements through architecture engineering have achieved impressive results [53, 59, 20, 60, 58, 68].

开发神经网络图像分类模型通常需要设计重要的网络架构。[32]在ImageNet[11]分类任务中使用了卷积架构[17,34]，从他的种子工作开始，后面再架构设计上有很多进展，取得了令人印象深刻的结果[53, 59, 20, 60, 58, 68]。

In this paper, we study a new paradigm of designing convolutional architectures and describe a scalable method to optimize convolutional architectures on a dataset of interest, for instance the ImageNet classification dataset. Our approach is inspired by the recently proposed Neural Architecture Search (NAS) framework [71], which uses a reinforcement learning search method to optimize architecture configurations. Applying NAS, or any other search methods, directly to a large dataset, such as the ImageNet dataset, is however computationally expensive. We therefore propose to search for a good architecture on a proxy dataset, for example the smaller CIFAR-10 dataset, and then transfer the learned architecture to ImageNet. We achieve this transferrability by designing a search space (which we call “the NASNet search space”) so that the complexity of the architecture is independent of the depth of the network and the size of input images. More concretely, all convolutional networks in our search space are composed of convolutional layers (or “cells”) with identical structure but different weights. Searching for the best convolutional architectures is therefore reduced to searching for the best cell structure. Searching for the best cell structure has two main benefits: it is much faster than searching for an entire network architecture and the cell itself is more likely to generalize to other problems. In our experiments, this approach significantly accelerates the search for the best architectures using CIFAR-10 by a factor of 7× and learns architectures that successfully transfer to ImageNet.

在本文中，我们研究了一种新的设计卷积架构的范式，描述了一种可在感兴趣的数据集上进行量变的优化卷积架构的方法，比如ImageNet分类数据集上。我们的方法是受最近提出来的NAS框架[71]启发提出的，其中使用了强化学习搜索方法来优化架构配置。直接在大型数据集（如ImageNet）上使用NAS或其他任何搜索方法，计算量非常之大。所以我们提出先在一个代理数据集，如较小的CIFAR-10数据集上搜索一个好的架构，然后将学到的架构迁移到ImageNet上。我们设计了一个搜索空间得到这种可迁移性，空间称之为NASNet搜索空间，这样架构的复杂度与网络的深度和输入图像的尺寸的无关。更具体的，我们搜索空间中的所有卷积网络都是由同样结构但不同权值的卷积层（或称cell）组成的。搜索最佳的卷积结构也就成了搜索最佳cell结构的问题。搜索最佳cell结构有两个主要的优点：首先是比搜索整个网络架构快的多，然后cell本身对其他问题的泛化能力好。在我们的试验中，这种方法使用CIFAR-10数据集明显加速了架构搜索（7倍），学到的结构成功迁移到了ImageNet上。

Our main result is that the best architecture found on CIFAR-10, called NASNet, achieves state-of-the-art accuracy when transferred to ImageNet classification without much modification. On ImageNet, NASNet achieves, among the published works, state-of-the-art accuracy of 82.7% top-1 and 96.2% top-5. This result amounts to a 1.2% improvement in top-1 accuracy than the best human-invented architectures while having 9 billion fewer FLOPS. On CIFAR-10 itself, NASNet achieves 2.4% error rate, which is also state-of-the-art.

我们得到的主要结果是在CIFAR-10上搜索到的最佳结构，称为NASNet，当迁移到ImageNet分类任务中后，没有经过多少修正，就取得了目前最好的结果。在ImageNet上，NASNet取得了top-1准确率82.7%，top-5准确率96.2%，这是目前发表的最好的结果。这个结果的top-1准确率比人类设计架构的最好结果高出1.2%，同时计算量少了90亿FLOPS。在CIFAR-10上，NASNet的错误率为2.4%，也是目前最好的结果。

Additionally, by simply varying the number of the convolutional cells and number of filters in the convolutional cells, we can create different versions of NASNets with different computational demands. Thanks to this property of the cells, we can generate a family of models that achieve accuracies superior to all human-invented models at equivalent or smaller computational budgets [60, 29]. Notably, the smallest version of NASNet achieves 74.0% top-1 accuracy on ImageNet, which is 3.1% better than previously engineered architectures targeted towards mobile and embedded vision tasks [24, 70].

另外，通过调整卷积cell的数目和卷积cell中滤波器的数目，我们就可以创造出不同版本的NASNet，其计算需求不同。多亏cell的这个性质，我们可以生成一族模型，与人设计的同等级别计算量的模型比，可以取得更好的准确度[60,29]。值的注意的是，最小版本的NASNet在ImageNet上取得了74.0%的top-1准确度，比之前为移动和嵌入式视觉任务设计的架构的结果要高3.1%[24,70]。

Finally, we show that the image features learned by NASNets are generically useful and transfer to other computer vision problems. In our experiments, the features learned by NASNets from ImageNet classification can be combined with the Faster-RCNN framework [47] to achieve state-of-the-art on COCO object detection task for both the largest as well as mobile-optimized models. Our largest NASNet model achieves 43.1% mAP, which is 4% better than previous state-of-the-art.

最后，我们说明，NASNet学习到的图像特征，是可以迁移到其他计算机视觉任务中使用的。在我们的试验中，NASNet从ImageNet分类任务学习到的特征可以和Faster-RCNN框架结合，在COCO目标检测任务中得到最好的结果，这在最大的模型和为移动应用优化的模型中都是最好的结果。我们最大的NASNet模型得到了43.1%的mAP结果，比之前的最好结果高出4.3%。

## 2. Related Work 相关工作

The proposed method is related to previous work in hyperparameter optimization [44, 4, 5, 54, 55, 6, 40] – especially recent approaches in designing architectures such as Neural Fabrics [48], DiffRNN [41], MetaQNN [3] and DeepArchitect [43]. A more flexible class of methods for designing architecture is evolutionary algorithms [65, 16, 57, 30, 46, 42, 67], yet they have not had as much success at large scale. Xie and Yuille [67] also transferred learned architectures from CIFAR-10 to ImageNet but performance of these models (top-1 accuracy 72.1%) are notably below previous state-of-the-art (Table 2).

我们提出的方法与之前工作中的超参数优化有关[44, 4, 5, 54, 55, 6, 40]，尤其是最近设计架构的方法如Neural Fabrics [48], DiffRNN [41], MetaQNN [3] and DeepArchitect [43]。更加灵活的一类设计架构的方法是演化算法[65, 16, 57, 30, 46, 42, 67]，但他们还没有在大尺度上获得成功。Xie and Yuille[67]也将从CIFAR-10学习到的架构迁移到了ImageNet，但这些模型的性能（top-1准确度72.1%）明显低于之前的最好结果（见表2）。

The concept of having one neural network interact with a second neural network to aid the learning process, or learning to learn or meta-learning [23, 49] has attracted much attention in recent years [1, 62, 14, 19, 35, 45, 15]. Most of these approaches have not been scaled to large problems like ImageNet. An exception is the recent work focused on learning an optimizer for ImageNet classification that achieved notable improvements [64].

使一个神经网络与另一个神经网络互动来帮助学习过程，这个概念也就是学习怎样学习，或称元学习[23,49]，在近年来得到了很多关注[1, 62, 14, 19, 35, 45, 15]。这些方法没有在大规模问题如ImageNet中进行相应的研究。有一个例外是最近[64]聚焦ImageNet分类优化器学习，取得了引人注意的结果。

The design of our search space took much inspiration from LSTMs [22], and Neural Architecture Search Cell [71]. The modular structure of the convolutional cell is also related to previous methods on ImageNet such as VGG [53], Inception [59, 60, 58], ResNet/ResNext [20, 68], and Xception/MobileNet [9, 24].

我们的搜索空间的设计从LSTM[22]和神经架构搜索单元[71]中得到了很多启发。卷积cell的模块化的结构与ImageNet上之前的方法有关，如VGG [53], Inception [59, 60, 58], ResNet/ResNext [20, 68], and Xception/MobileNet [9, 24]。

## 3. Method 方法

Our work makes use of search methods to find good convolutional architectures on a dataset of interest. The main search method we use in this work is the Neural Architecture Search (NAS) framework proposed by [71]. In NAS, a controller recurrent neural network (RNN) samples child networks with different architectures. The child networks are trained to convergence to obtain some accuracy on a held-out validation set. The resulting accuracies are used to update the controller so that the controller will generate better architectures over time. The controller weights are updated with policy gradient (see Figure 1).

我们的工作使用了搜索方法在感兴趣的数据集上寻找好的卷积结构。本文中我们使用的主要的搜索方法是[71]提出的神经结构搜索NAS框架。在NAS中，一个控制器RNN对不同结构的子网络进行取样。子网络经过训练收敛，在验证集上得到一定的准确率。得到的准确率结果用来对控制器进行更新，这样控制器会生成更好的架构。控制器的权重用策略梯度进行更新（见图1）。

Figure 1. Overview of Neural Architecture Search [71]. A controller RNN predicts architecture A from a search space with probability p. A child network with architecture A is trained to convergence achieving accuracy R. Scale the gradients of p by R to update the RNN controller.

图1 NAS概览[71]。一个控制RNN从搜索空间中以概率p预测结构A；结构A的子网络训练收敛，得到准确率R；用R以概率p控制梯度的尺度来更新控制器。

The main contribution of this work is the design of a novel search space, such that the best architecture found on the CIFAR-10 dataset would scale to larger, higher-resolution image datasets across a range of computational settings. We name this search space the NASNet search space as it gives rise to NASNet, the best architecture found in our experiments. One inspiration for the NASNet search space is the realization that architecture engineering with CNNs often identifies repeated motifs consisting of combinations of convolutional filter banks, nonlinearities and a prudent selection of connections to achieve state-of-the-art results (such as the repeated modules present in the Inception and ResNet models [59, 20, 60, 58]). These observations suggest that it may be possible for the controller RNN to predict a generic convolutional cell expressed in terms of these motifs. This cell can then be stacked in series to handle inputs of arbitrary spatial dimensions and filter depth.

本文的主要贡献是设计了一个新的搜索空间，这样在CIFAR-10数据集上搜索到的最好的结构，可以在很大范围的计算设置中，按比例放大到更大、分辨率更高的图像数据集上。我们将这个搜索空间命名为NASNet搜索空间，因为从中产生了NASNet，这是我们在试验中找到的最好的架构。我们设计NASNet搜索空间是受到下面事实的启发：即CNN的结构设计通常是重复的主题，包括卷积滤波器组，非线性处理的组合，还有谨慎选择的连接，这样就可以得到最佳的结果（比如在Inception和ResNet模型中的重复模块[59, 20, 60, 58]）。这些观察说明，可以使控制器RNN预测一个一般性的卷积cell，这个cell是由这些主题表达出来的。这个cell就可以堆叠起来，处理任意大小和滤波器深度的输入。

In our approach, the overall architectures of the convolutional nets are manually predetermined. They are composed of convolutional cells repeated many times where each convolutional cell has the same architecture, but different weights. To easily build scalable architectures for images of any size, we need two types of convolutional cells to serve two main functions when taking in a feature map as input: (1) convolutional cells that return a feature map of the same dimension, and (2) convolutional cells that return a feature map where the feature map height and width is reduced by a factor of two. We name the first type and second type of convolutional cells Normal Cell and Reduction Cell respectively. For the Reduction Cell, we make the initial operation applied to the cell’s inputs have a stride of two to reduce the height and width. All of our operations that we consider for building our convolutional cells have an option of striding.

在我们的方法中，卷积网络的总体结构是手动预先确定的。它们由很多个卷积cell重复堆叠组成，每个卷积cell都有相同的结构，但不同的权重。为对任意大小的图像轻松构建可按比例缩放的架构，在以特征图为输入时，我们需要两种卷积cell来提供两种主要的功能：(1)返回同样大小特征图的卷积cell，(2)返回特征图宽高大小都减半的卷积cell。我们称两种卷积cell分别为Normal Cell和Reduction Cell。对于Reduction Cell，我们使作用于输入的最初操作的步长为2，以减少高度和宽度。我们用来构建卷积cell的所有操作都有步长的选项。

Figure 2 shows our placement of Normal and Reduction Cells for CIFAR-10 and ImageNet. Note on ImageNet we have more Reduction Cells, since the incoming image size is 299x299 compared to 32x32 for CIFAR. The Reduction and Normal Cell could have the same architecture, but we empirically found it beneficial to learn two separate architectures. We use a common heuristic to double the number of filters in the output whenever the spatial activation size is reduced in order to maintain roughly constant hidden state dimension [32, 53]. Importantly, much like Inception and ResNet models [59, 20, 60, 58], we consider the number of motif repetitions N and the number of initial convolutional filters as free parameters that we tailor to the scale of an image classification problem.

图2所示的是在CIFAR-10和ImageNet数据集上Normal Cell和Reduction Cell的布置。注意在ImageNet上我们有更多的Reduction Cell，因为输入图像大小是299×299，在CIFAR-10上为32×32。Reduction和Normal Cell可以有同样的结构，但我们的经验显示两种不同的结构会更好。我们使用普通的启发式方法，在空间激活尺寸减小时，滤波器数量加倍，以保持隐藏状态维数大致不变[32,53]。重要的是，与Inception和ResNet模型[59, 20, 60, 58]很像，我们将模块重复次数N和初始卷积滤波器数量作为自由参数，可以改变以适应图像分类问题的比例。

Figure 2. Scalable architectures for image classification consist of two repeated motifs termed Normal Cell and Reduction Cell. This diagram highlights the model architecture for CIFAR-10 and ImageNet. The choice for the number of times the Normal Cells that gets stacked between reduction cells, N, can vary in our experiments.

图2 可按比例调整的图像分类网络结构，包括两种重复的模块，称为Normal Cell和Reduction Cell。这个图高亮显示了CIFAR-10和ImageNet的模型架构。Normal Cell在Reduction Cell之间重复叠加，叠加的次数N，可以在试验中变化。

What varies in the convolutional nets is the structures of the Normal and Reduction Cells, which are searched by the controller RNN. The structures of the cells can be searched within a search space defined as follows (see Appendix, Figure 7 for schematic). In our search space, each cell receives as input two initial hidden states $h_i$ and $h_{i−1}$ which are the outputs of two cells in previous two lower layers or the input image. The controller RNN recursively predicts the rest of the structure of the convolutional cell, given these two initial hidden states (Figure 3). The predictions of the controller for each cell are grouped into B blocks, where each block has 5 prediction steps made by 5 distinct softmax classifiers corresponding to discrete choices of the elements of a block:

在卷积网络中可以变化的是Normal Cell和Reduction Cell的结构，这是由控制器RNN搜索得到的。cell结构可以在如下定义的搜索空间中搜索得到（见附录，图7的原理图）。在我们的搜索空间中，每个cell接收两个初始隐藏状态$h_i$ and $h_{i−1}$作为输入，是前两层的两个cell的输出，或输入图像。在给定这两个初始隐藏状态的情况下（图3），控制器RNN递归的预测卷积cell的剩下的结构。控制器对每个cell的预测都被分组成B个块，每个块有5个预测步骤，也就是5个不同的softmax分类器，对应一个块的元素的具体选择：

- Step 1. Select a hidden state from $h_i$, $h_{i−1}$ or from the set of hidden states created in previous blocks.
- Step 2. Select a second hidden state from the same options as in Step 1.
- Step 3. Select an operation to apply to the hidden state selected in Step 1.
- Step 4. Select an operation to apply to the hidden state selected in Step 2.
- Step 5. Select a method to combine the outputs of Step 3 and 4 to create a new hidden state.

Figure 3. Controller model architecture for recursively constructing one block of a convolutional cell. Each block requires selecting 5 discrete parameters, each of which corresponds to the output of a softmax layer. Example constructed block shown on right. A convolutional cell contains B blocks, hence the controller contains 5B softmax layers for predicting the architecture of a convolutional cell. In our experiments, the number of blocks B is 5.

图3 控制器模型结构示意图，可以递归的构建一个卷积的cell的模块。每个模块需要选择5个具体参数，每个都对应着一个softmax层的输出。构建模块的例子如右图所示。一个卷积cell包括B个模块，所以控制器包含5B个softmax层来预测卷积cell的结构。在我们的试验中，模块的数量B设定为5。

The algorithm appends the newly-created hidden state to the set of existing hidden states as a potential input in subsequent blocks. The controller RNN repeats the above 5 prediction steps B times corresponding to the B blocks in a convolutional cell. In our experiments, selecting B = 5 provides good results, although we have not exhaustively searched this space due to computational limitations.

算法将新生成的隐藏状态追加到已有的隐藏状态集合中，作为后续模块的潜在输入。控制器RNN重复上面5个预测步骤B次，对应着卷积cell的B个模块。在我们的试验中，选择B=5可以得到好的结果，但由于计算量限制，我们没有穷尽搜索这个空间。

In steps 3 and 4, the controller RNN selects an operation to apply to the hidden states. We collected the following set of operations based on their prevalence in the CNN literature: 在第3和第4步中，控制器RNN选出操作应用在隐藏状态中。我们根据CNN文献中的广泛程度，收集了下面的操作集合：

Operation | Operation
--- | ---
• identity | • 1x3 then 3x1 convolution
• 1x7 then 7x1 convolution | • 3x3 dilated convolution
• 3x3 average pooling | • 3x3 max pooling
• 5x5 max pooling | • 7x7 max pooling
• 1x1 convolution | • 3x3 convolution
• 3x3 depthwise-separable conv | • 5x5 depthwise-seperable conv
• 7x7 depthwise-separable conv | 

In step 5 the controller RNN selects a method to combine the two hidden states, either (1) element-wise addition between two hidden states or (2) concatenation between two hidden states along the filter dimension. Finally, all of the unused hidden states generated in the convolutional cell are concatenated together in depth to provide the final cell output.

在第5步中，控制器RNN选择一个方法来将两个隐藏状态结合在一起，有两种选项，1)两个隐藏状态逐元素相加，2)两个隐藏状态沿着滤波器维度拼接起来。最后，所有未使用的卷积cell中产生的隐藏状态拼接在一起，来产生最后的cell的输出。

To allow the controller RNN to predict both Normal Cell and Reduction Cell, we simply make the controller have 2 × 5B predictions in total, where the first 5B predictions are for the Normal Cell and the second 5B predictions are for the Reduction Cell.

为使控制器RNN可以预测Normal Cell和Reduction Cell两种，我们直接使控制器进行2×5B次预测，前面5B个预测是进行Normal Cell预测的，后面5B个预测是进行Reduction Cell预测的。

Finally, our work makes use of the reinforcement learning proposal in NAS [71]; however, it is also possible to use random search to search for architectures in the NASNet search space. In random search, instead of sampling the decisions from the softmax classifiers in the controller RNN, we can sample the decisions from the uniform distribution. In our experiments, we find that random search is slightly worse than reinforcement learning on the CIFAR-10 dataset. Although there is value in using reinforcement learning, the gap is smaller than what is found in the original work of[71]. This result suggests that 1)the NASNet search space is well-constructed such that random search can perform reasonably well and 2) random search is a difficult baseline to beat. We will compare reinforcement learning against random search in Section 4.4.

最后，我们的工作使用了NAS[71]中建议的强化学习；但是，使用随机搜索在NASNet搜索空间中搜索结构也是可能的。在随机搜索中，我们可以从均匀分布的决策中进行取样，而不是从控制器RNN中的softmax分类器对决策取样。在我们的试验中，我们发现在CIFAR-10数据集上随机搜索比强化学习略差一些。虽然使用强化学习是有价值的，但与[71]原始工作中相比差距更小一些。这个结果说明，1)NASNet搜索空间构建的很好，随机搜索可以得到很好的效果；2)随机搜索是一种很好的搜索策略。我们在4.4节中比较强化学习与随机搜索。

## 4. Experiments and Results 试验和结果

In this section, we describe our experiments with the method described above to learn convolutional cells. In summary, all architecture searches are performed using the CIFAR-10 classification task [31]. The controller RNN was trained using Proximal Policy Optimization (PPO) [51] by employing a global workqueue system for generating a pool of child networks controlled by the RNN. In our experiments, the pool of workers in the workqueue consisted of 500 GPUs.

在本节中，我们用上述方法进行了试验，进行卷积cell的学习。概括来说，所有的结构搜索都用CIFAR-10分类任务[31]进行。控制器RNN用Proximal Policy Optimization(PPO)[51]进行训练，其中采用了一种全局工作队列系统，生成由RNN控制的子网络池。在我们的试验中，工作队列中的worker池包含500个GPUs。

The result of this search process over 4 days yields several candidate convolutional cells. We note that this search procedure is almost 7× faster than previous approaches [71] that took 28 days. (In particular, we note that previous architecture search [71] used 800 GPUs for 28 days resulting in 22,400 GPU-hours. The method in this paper uses 500 GPUs across 4 days resulting in 2,000 GPU-hours. The former effort used Nvidia K40 GPUs, whereas the current efforts used faster NVidia P100s. Discounting the fact that the we use faster hardware, we estimate that the current procedure is roughly about 7× more efficient.) Additionally, we demonstrate below that the resulting architecture is superior in accuracy.

这个4天的搜索过程的结果是，产生了几个候选的卷积cell。我们注意到，这个搜索过程比之前的方法[71]（28天）快了7倍。（特别是，之前的架构搜索[71]用了800个GPU进行了28天，也就是22400 GPU小时，我们的方法用了500个GPU搜索了4天，也就是2000 GPU小时；之前的工作用的是NVidia K40 GPUs，我们的工作用的是更快的NVidia P100；打个折扣，我们估计我们的效率要高7倍）另外，我们得到的结构准确率更高。

Figure 4 shows a diagram of the top performing Normal Cell and Reduction Cell. Note the prevalence of separable convolutions and the number of branches compared with competing architectures [53, 59, 20, 60, 58]. Subsequent experiments focus on this convolutional cell architecture, although we examine the efficacy of other top-ranked convolutional cells in ImageNet experiments (described in Appendix B) and report their results as well. We call the three networks constructed from the best three searches NASNet-A, NASNet-B and NASNet-C.

图4所示的是性能最好的Normal Cell和Reduction Cell。注意可分离卷积的广泛应用，和分支数目与其他架构的对比[53, 59, 20, 60, 58]。后面的试验聚焦在这种卷积cell的结构，但我们也试验了其他在ImageNet试验中性能最好的卷积cell的功效，报告了其结果。我们称得到的三个最好结果的网络为NASNet-A，NASNet-B，NASNet-C。

Figure 4. Architecture of the best convolutional cells (NASNet-A) with B = 5 blocks identified with CIFAR-10 . The input (white) is the hidden state from previous activations (or input image). The output (pink) is the result of a concatenation operation across all resulting branches. Each convolutional cell is the result of B blocks. A single block is corresponds to two primitive operations (yellow) and a combination operation (green). Note that colors correspond to operations in Figure 3.

图4 最好的卷积cell的结构，B=5，在CIFAR-10数据集上得到。输入（白色）是前面激活的隐藏状态（或输入图像）。输出（粉色）是所有分支结果的拼接。每个卷积cell包括B个模块。一个模块对应着两个简单操作（黄色）和一个组合操作（绿色）。这里的颜色与图3中的颜色是对应的。

We demonstrate the utility of the convolutional cells by employing this learned architecture on CIFAR-10 and a family of ImageNet classification tasks. The latter family of tasks is explored across a few orders of magnitude in computational budget. After having learned the convolutional cells, several hyper-parameters may be explored to build a final network for a given task: (1) the number of cell repeats N and (2) the number of filters in the initial convolutional cell. After selecting the number of initial filters, we use a common heuristic to double the number of filters whenever the stride is 2. Finally, we define a simple notation, e.g., 4 @ 64, to indicate these two parameters in all networks, where 4 and 64 indicate the number of cell repeats and the number of filters in the penultimate layer of the network, respectively.

我们将学到的结构在CIFAR-10数据集和一系列ImageNet分类任务中应用，以检验卷积cell的功用。ImageNet上的一系列分类任务计算量各不相同，相差较大。在学到卷积cell后，需要研究一下几个超参数来为给定的任务建立最后的网络：(1)cell重复的数量N，(2)最初的卷积cell中的滤波器数量。选择了开始的滤波器的数量后，我们使用常见的启发式方法，在步长为2的地方之后，滤波器数量就加倍。最后，我们定义一个简单的记号，比如，4@64，表示所有网络中的这两个参数，这里4就是cell重复的次数，64是网络倒数第二层的滤波器数量。

For complete details of of the architecture learning algorithm and the controller system, please refer to Appendix A. Importantly, when training NASNets, we discovered ScheduledDropPath, a modified version of DropPath [33], to be an effective regularization method for NASNet. In DropPath [33], each path in the cell is stochastically dropped with some fixed probability during training. In our modified version, ScheduledDropPath, each path in the cell is dropped out with a probability that is linearly increased over the course of training. We find that DropPath does not work well for NASNets, while ScheduledDropPath significantly improves the final performance of NASNets in both CIFAR and ImageNet experiments.

学习算法和控制器系统结构的全部细节详见附录A。重要的是，在训练NASNet的时候，我们发现了ScheduledDropPath，这是DropPath[33]的修正版，可以有效的对NASNet进行正则化。在DropPath[33]中，cell中的每个path在训练的时候根据固定概率随机drop。在我们的修正版中，也就是ScheduledDropPath中，cell中的每个path dropout的概率随着训练过程的进行而线性增加。我们发现DropPath不太适用于NASNet，而ScheduledDropPath显著改进了NASNet在CIFAR和ImageNet试验中的最终性能。

### 4.1. Results on CIFAR-10 Image Classification 在CIFAR-10上的图像分类结果

For the task of image classification with CIFAR-10, we set N = 4 or 6 (Figure 2). The test accuracies of the best architectures are reported in Table 1 along with other state-of-the-art models. As can be seen from the Table, a large NASNet-A model with cutout data augmentation [12] achieves a state-of-the-art error rate of 2.40% (averaged across 5 runs), which is slightly better than the previous best record of 2.56% by [12]. The best single run from our model achieves 2.19% error rate.

对于CIFAR-10上的图像分类任务，我们设N=4或6（图2）。表1是最好结构的测试准确率，还有其他目前最好的模型的结果。从表1中可以看出，大型NASNet-A模型在数据扩充[12]的情况下得到了最好的错误率结果2.40%（在5次运行中平均），这比之前的最好结果2.56%[12]略好了一点点。我们模型单次运行的最好结果为2.19%错误率。

Table 1. Performance of Neural Architecture Search and other state-of-the-art models on CIFAR-10. All results for NASNet are the mean accuracy across 5 runs.

表1 NASNet和其他目前最好的模型在CIFAR-10数据集上的性能。NASNet的所有结果都是5次运行的均值准确率。

model | depth | params | error rate (%)
--- | --- | --- | ---
DenseNet (L = 40,k = 12) [26] | 40 | 1.0M | 5.24
DenseNet(L = 100,k = 12) [26] | 100 | 7.0M | 4.10
DenseNet (L = 100,k = 24) [26] | 100 | 27.2M | 3.74
DenseNet-BC (L = 100,k = 40) [26] | 190 | 25.6M | 3.46
Shake-Shake 26 2x32d [18] | 26 | 2.9M | 3.55
Shake-Shake 26 2x96d [18] | 26 | 26.2M | 2.86
Shake-Shake 26 2x96d + cutout [12] | 26 | 26.2M | 2.56
NAS v3 [71] | 39 | 7.1M | 4.47
NAS v3 [71] | 39 | 37.4M | 3.65
NASNet-A (6 @ 768) | - | 3.3M | 3.41
NASNet-A (6 @ 768) + cutout | - | 3.3M | 2.65
NASNet-A (7 @ 2304) | - | 27.6M | 2.97
NASNet-A (7 @ 2304) + cutout | - | 27.6M | 2.40
NASNet-B (4 @ 1152) | - | 2.6M | 3.73
NASNet-C (4 @ 640) | - | 3.1M | 3.59

### 4.2. Results on ImageNet Image Classification 在ImageNet上的图像分类结果

We performed several sets of experiments on ImageNet with the best convolutional cells learned from CIFAR-10. We emphasize that we merely transfer the architectures from CIFAR-10 but train all ImageNet models weights from scratch.

我们将在CIFAR-10上学到的最好的卷积cell在ImageNet上进行了几个试验。我们强调，我们只是将从CIFAR-10中学到的结构进行了迁移，但在ImageNet上从头开始训练所有的模型参数。

Results are summarized in Table 2 and 3 and Figure 5. In the first set of experiments, we train several image classification systems operating on 299x299 or 331x331 resolution images with different experiments scaled in computational demand to create models that are roughly on par in computational cost with Inception-v2 [29], Inception-v3 [60] and PolyNet [69]. We show that this family of models achieve state-of-the-art performance with fewer floating point operations and parameters than comparable architectures. Second, we demonstrate that by adjusting the scale of the model we can achieve state-of-the-art performance at smaller computational budgets, exceeding streamlined CNNs hand-designed for this operating regime [24, 70].

在表2、表3及图5中总结了所有结果。在第一个系列试验中，我们训练了几个图像分类系统，输入图像大小为299×299或331×331分辨率，根据计算需求按比例进行缩放，生成的模型与Inception-v2 [29], Inception-v3 [60] and PolyNet [69]在计算量上水平类似。我们的试验显示，与类似模型相比，这一族模型在更少的浮点数运算和较少的参数情况下取得了目前最好的结果。第二，我们证明，通过调整模型尺度，可以在更小的计算量上得到最好的结果，超过了那些手动设计的流水线型的CNN[24,70]。

Note we do not have residual connections between convolutional cells as the models learn skip connections on their own. We empirically found manually inserting residual connections between cells to not help performance. Our training setup on ImageNet is similar to [60], but please see Appendix A for details.

注意我们在卷积cell之间没有残差连接，因为模型自己学习跳过连接。我们从经验中得到，在cell之间手工插入残差连接对性能没有帮助。我们在ImageNet上的训练设置与[60]相似，详见附录A。

Table 2 shows that the convolutional cells discovered with CIFAR-10 generalize well to ImageNet problems. In particular, each model based on the convolutional cells exceeds the predictive performance of the corresponding hand-designed model. Importantly, the largest model achieves a new state-of-the-art performance for ImageNet (82.7%) based on single, non-ensembled predictions, surpassing previous best published result by ∼1.2% [8]. Among the unpublished works, our model is on par with the best reported result of 82.7% [25], while having significantly fewer floating point operations. Figure 5 shows a complete summary of our results in comparison with other published results. Note the family of models based on convolutional cells provides an envelope over a broad class of human-invented architectures.

表2所示的是由CIFAR-10发现的卷积cell在ImageNet问题上泛化的很好。特别是，基于卷积cell的每个模型都超过了相应的手工设计的模型。重要的是，最大的模型在ImageNet上取得了目前最好的结果，82.7%，而这是基于单个模型的，非集成模型的，超过以前最好的结果1.2%[8]。在未发表的工作中，我们的模型与之前最好的结果[25]的82.7%类似，但浮点数运算明显少了很多。图5所示的是我们的结果与其他发表的结果的全面对比。注意基于卷积cell的模型族超过了很多人工设计的架构。

Table 2. Performance of architecture search and other published state-of-the-art models on ImageNet classification. Mult-Adds indicate the number of composite multiply-accumulate operations for a single image. Note that the composite multiple-accumulate operations are calculated for the image size reported in the table. Model size for [25] calculated from open-source implementation.

表2 结构搜索和其他发表的目前最好的模型在ImageNet分类中的表现。Mult-Adds表示一个图像上的符合乘法-累加运算数目。注意复合乘法-累加运算是在表格中的图像大小上计算的。[25]的模型大小是从开源实现中计算得到的。

Model | image size | parameters | Mult-Adds | Top 1 Acc. (%) | Top 5 Acc. (%)
--- | --- | --- | --- | --- | ---
Inception V2 [29] | 224×224 | 11.2M | 1.94B | 74.8 | 92.2
NASNet-A (5 @ 1538) | 299×299 | 10.9M | 2.35B | 78.6 | 94.2
Inception V3 [60] | 299×299 | 23.8M | 5.72B | 78.8 | 94.4
Xception [9] | 299×299 | 22.8M | 8.38B | 79.0 | 94.5
Inception ResNet V2 [58] | 299×299 | 55.8M | 13.2B | 80.1 | 95.1
NASNet-A (7 @ 1920) | 299×299 | 22.6M | 4.93B | 80.8 | 95.3
ResNeXt-101 (64 x 4d) [68] | 320×320 | 83.6M | 31.5B | 80.9 | 95.6
PolyNet [69] | 331×331 | 92M | 34.7B | 81.3 | 95.8
DPN-131 [8] | 320×320 | 79.5M | 32.0B | 81.5 | 95.8
SENet [25] | 320×320 | 145.8M | 42.3B | 82.7 | 96.2
NASNet-A (6 @ 4032) | 331×331 | 88.9M | 23.8B | 82.7 | 96.2

Finally, we test how well the best convolutional cells may perform in a resource-constrained setting, e.g., mobile devices (Table 3). In these settings, the number of floating point operations is severely constrained and predictive performance must be weighed against latency requirements on a device with limited computational resources. MobileNet [24] and ShuffleNet [70] provide state-of-the-art results obtaining 70.6% and 70.9% accuracy, respectively on 224x224 images using ∼550M multliply-add operations. An architecture constructed from the best convolutional cells achieves superior predictive performance (74.0% accuracy) surpassing previous models but with comparable computational demand. In summary, we find that the learned convolutional cells are flexible across model scales achieving state-of-the-art performance across almost 2 orders of magnitude in computational budget.

最后，我们测试了最好的卷积cell在一个资源受限的设置中表现如何，也就是在移动设备上的表现（见表3）。在这些设置中，浮点数运算的数目严重受限，预测性能必须权衡设备有限的计算资源。MobileNet [24] and ShuffleNet [70]是目前最好的结果，准确率为70.6%和70.9%，这个结果是在在224×224的图像使用了大约550M乘-加操作得到的。从最好的卷积cell构建出的结构得到了更好的预测结果，即74.0%的准确率，但计算量大致相同。总结来说，我们发现学习到的卷积cell在各种计算需求下都可以取得最佳表现。

Table 3. Performance on ImageNet classification on a subset of models operating in a constrained computational setting, i.e., < 1.5B multiply-accumulate operations per image. All models use 224x224 images. † indicates top-1 accuracy not reported in [59] but from open-source implementation.

表3 部分模型在受限的计算环境中的ImageNet分类表现，即每个图像的乘法-累加运算小于1.5B，所有的模型都使用224×224图像。

Model | parameters | Mult-Adds | Top 1 Acc. (%) | Top 5 Acc. (%)
--- | --- | --- | --- | ---
Inception V1 [59] | 6.6M | 1,448M | 69.8 | 89.9
MobileNet-224 [24] | 4.2M | 569M | 70.6 | 89.5
ShuffleNet (2x) [70] | ∼ 5M | 524M | 70.9 | 89.8
NASNet-A (4 @ 1056) | 5.3M | 564M | 74.0 | 91.6
NASNet-B (4 @ 1536) | 5.3M | 488M | 72.8 | 91.3
NASNet-C (3 @ 960) | 4.9M | 558M | 72.5 | 91.0

Figure 5. Accuracy versus computational demand (left) and number of parameters (right) across top performing published CNN architectures on ImageNet 2012 ILSVRC challenge prediction task. Computational demand is measured in the number of floating-point multiply-add operations to process a single image. Black circles indicate previously published results and red squares highlight our proposed models.

图5 准确度与计算量（左）、参数数量（右）图，在ImageNet-2012 ILSVRC 挑战赛分类任务中性能最好的模型的对比。计算量用处理一幅图像的浮点数乘法-加法操作数量来衡量。黑色圆圈表示以前发表的结果，红色方块是我们提出的模型。

### 4.3. Improved features for object detection 改进特征的目标检测

Image classification networks provide generic image features that may be transferred to other computer vision problems [13]. One of the most important problems is the spatial localization of objects within an image. To further validate the performance of the family of NASNet-A networks, we test whether object detection systems derived from NASNet-A lead to improvements in object detection [28].

图像分类网络提供了一般性的图像特征，可能迁移到其他计算机视觉问题中[13]。其中最重要的一个问题是图像中目标的空间位置。为进一步验证NASNet-A族模型，我们对从NASNet-A中推导出的目标检测系统是否在目标检测中有所改进做了试验[28]。

To address this question, we plug in the family of NASNet-A networks pretrained on ImageNet into the Faster-RCNN object detection pipeline [47] using an opensource software platform [28]. We retrain the resulting object detection pipeline on the combined COCO training plus validation dataset excluding 8,000 mini-validation images. We perform single model evaluation using 300-500 RPN proposals per image. In other words, we only pass a single image through a single network. We evaluate the model on the COCO mini-val [28] and test-dev dataset and report the mean average precision (mAP) as computed with the standard COCO metric library [38]. We perform a simple search over learning rate schedules to identify the best possible model. Finally, we examine the behavior of two object detection systems employing the best performing NASNet-A image featurization (NASNet-A, 6 @ 4032) as well as the image featurization geared towards mobile platforms (NASNet-A, 4 @ 1056).

为解决这个问题，我们将在ImageNet上预训练的NASNet-A网络插入到Faster-RCNN目标检测管道中[47]，使用的是开源软件平台[28]。我们在COCO训练和验证数据集上（除去8000个mini-validation图像）重新训练得到的目标检测管道。我们进行单模型评估时，每个图像使用300-500 RPN proposals。也就是说，我们只将一幅图像送入单个网络。我们在COCO mini-val[28]和test-dev数据集上进行评估，用标准COCO衡量库计算得到mAP。我们对学习速率进行简单搜索，得到最佳模型。最后，我们检查了两个目标检测系统的，它们分别使用最佳表现的NASNet-A图像特征化(NASNet-A, 6 @ 4032)和移动平台上的最佳模型(NASNet-A, 4 @ 1056)。

For the mobile-optimized network, our resulting system achieves a mAP of 29.6% – exceeding previous mobile-optimized networks that employ Faster-RCNN by over 5.0% (Table 4). For the best NASNet network, our resulting network operating on images of the same spatial resolution (800 × 800) achieves mAP = 40.7%, exceeding equivalent object detection systems based off lesser performing image featurization (i.e. Inception-ResNet-v2) by 4.0% [28, 52] (see Appendix for example detections on images and side-by-side comparisons). Finally, increasing the spatial resolution of the input image results in the best reported, single model result for object detection of 43.1%, surpassing the best previous best by over 4.0%[37]. (A primary advance in the best reported object detection system is the introduction of a novel loss [37].  Pairing this loss with NASNet-A image featurization may lead to even further performance gains. Additionally, performance gains are achievable through ensembling multiple inferences across multiple model instances and image crops (e.g., [28]).) These results provide further evidence that NASNet provides superior, generic image features that may be transferred across other computer vision tasks. Figure 10 and Figure 11 in Appendix C show four examples of object detection results produced by NASNet-A with the Faster-RCNN framework.

对于移动优化的网络，我们的系统得到了mAP 29.6%的结果，超过了以前移动优化的Faster-RCNN算法5.0%（见表4）。对于最好的NASNet网络，我们的网络在同样分辨率的图像(800×800)得到mAP为40.7%，超过了基于Inception-ResNet-v2的等价目标检测系统4.0%[28,52]（详见附录，有图像样本检测和对比）。最后，增加输入图像的空间分辨率会得到目前最好的单模型目标检测结果43.1%，超过了之前最好的结果4.0%[37]。（最主要的进步是由于引入了一个新的损失函数[37]，将这个损失函数与NASNet-A图像特征化结合，可能会得到进一步的性能提升。另外，还可以通过集成多个算法推理和剪切块推理[28]也可以得到性能提升。）这些结果说明NASNet给出了更好的一般性的图像特征，可以在计算机视觉任务中迁移使用。附录C中的图10和11给出了4个目标检测的例子，就是由NASNet-A和Faster-RCNN框架得到的。

### 4.4. Efficiency of architecture search methods 结构搜索方法的效率

Though what search method to use is not the focus of the paper, an open question is how effective is the reinforcement learning search method. In this section, we study the effectiveness of reinforcement learning for architecture search on the CIFAR-10 image classification problem and compare it to brute-force random search (considered to be a very strong baseline for black-box optimization [5]) given an equivalent amount of computational resources.

虽然使用什么搜索方法不是本文关注的重点，但强化学习搜索方法的效率仍然是一个待研究的问题，我们在CIFAR-10图像分类问题中研究了结构搜索中的强化学习的效率，将其与暴力随机搜索进行比较（对于黑盒优化问题，这是一种很强的基准），其计算资源都是相同的。

Figure 6 shows the performance of reinforcement learning (RL) and random search (RS) as more model architectures are sampled. Note that the best model identified with RL is significantly better than the best model found by RS by over 1% as measured by on CIFAR-10. Additionally, RL finds an entire range of models that are of superior quality to random search. We observe this in the mean performance of the top-5 and top-25 models identified in RL versus RS. We take these results to indicate that although RS may provide a viable search strategy, RL finds better architectures in the NASNet search space.

图6所示的在模型学习了很多样本后，强化学习RL和随机搜索RS的表现。注意标记出来最好的RL模型比RS的有明显改善，在CIFAR-10上的测量高出了1%还要多。另外，RL找到的很大范围内的模型都比随机搜索要好。我们在top-5和top-25模型的平均性能都观察到了这种现象。我们认为这些结果说明，虽然RS是一种可行的搜索策略，但RL总是能在NASNet搜索空间中找到更好的结构。

## 5. Conclusion 结论

In this work, we demonstrate how to learn scalable, convolutional cells from data that transfer to multiple image classification tasks. The learned architecture is quite flexible as it may be scaled in terms of computational cost and parameters to easily address a variety of problems. In all cases, the accuracy of the resulting model exceeds all human-designed models – ranging from models designed for mobile applications to computationally-heavy models designed to achieve the most accurate results.

在本文中，我们展示了怎样从数据中学习可按比例变化的卷积cell，还可以迁移到多个图像分类任务中去。学到的架构非常灵活，因为可以根据计算量和参数量调整，很容易的解决很多问题。在所有情况下，得到的模型的准确率都超过了人类设计的模型，从为移动应用设计的模型，到计算量很大的模型（以取得最佳准确率为目的的设计）。

The key insight in our approach is to design a search space that decouples the complexity of an architecture from the depth of a network. This resulting search space permits identifying good architectures on a small dataset (i.e., CIFAR-10) and transferring the learned architecture to image classifications across a range of data and computational scales.

我们方法的关键思想是设计一个搜索空间，使结构的复杂度和网络深度不相关。得到的搜索空间可以在小型数据集（如CIFAR-10）上找到好的结构，将学习到的结构迁移到不同数据规模和计算量的图像分类任务中去。

The resulting architectures approach or exceed state-of-the-art performance in both CIFAR-10 and ImageNet datasets with less computational demand than human-designed architectures [60, 29, 69]. The ImageNet results are particularly important because many state-of-the-art computer vision problems (e.g., object detection [28], face detection [50], image localization [63]) derive image features or architectures from ImageNet classification models. For instance, we find that image features obtained from ImageNet used in combination with the Faster-RCNN framework achieves state-of-the-art object detection results. Finally, we demonstrate that we can use the resulting learned architecture to perform ImageNet classification with reduced computational budgets that outperform streamlined architectures targeted to mobile and embedded platforms [24, 70].

我们得到的结构在CIFAR-10和ImageNet数据集上都接近或超过了目前最好的结果，而计算量比人设计的结构要少的多[60, 29, 69]。ImageNet上的结果尤其重要，因为很多最好的计算机视觉问题（如目标检测，人脸检测和图像定位）都从ImageNet分类模型中导出图像特征或结构。比如，我们发现从ImageNet上得到的图像特征与Faster-RCNN框架结合使用可以取得目前最好的目标检测结果。最后，我们说明我们可以使用学习得到的结构以很少的计算量进行ImageNet分类，性能超过了面向移动和嵌入式平台设计的流水线结构。

## Reference 参考文献

## Appendix 附录

### A. Experimental Details 试验细节

#### A.1. Dataset for Architecture Search 搜索结构的数据集

The CIFAR-10 dataset [31] consists of 60,000 32x32 RGB images across 10 classes (50,000 train and 10,000 test images). We partition a random subset of 5,000 images from the training set to use as a validation set for the controller RNN. All images are whitened and then undergone several data augmentation steps: we randomly crop 32x32 patches from upsampled images of size 40x40 and apply random horizontal flips. This data augmentation procedure is common among related work.

CIFAR-10数据集[31]包括6万幅32×32的RGB图像，分为10类（5万幅训练用图，1万幅测试用图）。我们从5万幅训练图像中随机分割出一个子集作为验证集，用于控制器RNN的训练。所有图像都经过白化处理，然后经过几个数据扩充步骤：我们从上采样的图像(40×40)中随机剪切32×32图像块，然后进行随机水平翻转。这种数据扩充过程在相关工作中非常常见。

#### A.2. Controller architecture 控制器结构

The controller RNN is a one-layer LSTM [22] with 100 hidden units at each layer and 2 × 5B softmax predictions for the two convolutional cells (where B is typically 5) associated with each architecture decision. Each of the 10B predictions of the controller RNN is associated with a probability. The joint probability of a child network is the product of all probabilities at these 10B softmaxes. This joint probability is used to compute the gradient for the controller RNN. The gradient is scaled by the validation accuracy of the child network to update the controller RNN such that the controller assigns low probabilities for bad child networks and high probabilities for good child networks.

控制器RNN是一个一层LSTM[22]网络，每层有100个隐藏节点，对两个卷积cell来说有2×5B个softmax预测，这里B通常为5。控制器RNN的每10B个预测都有一个概率。一个子网络的联合概率是这10B的softmax的概率乘积。这个联合概率用来计算控制器RNN的梯度。梯度根据子网络在验证集上的正确率来控制幅度，更新控制器RNN后，控制器给好的子网络赋高的概率，给差的子网络赋低的概率。

Unlike [71], who used the REINFORCE rule [66] to update the controller, we employ Proximal Policy Optimization (PPO) [51] with learning rate 0.00035 because training with PPO is faster and more stable. To encourage exploration we also use an entropy penalty with a weight of 0.00001. In our implementation, the baseline function is an exponential moving average of previous rewards with a weight of 0.95. The weights of the controller are initialized uniformly between -0.1 and 0.1.

[71]使用了强化规则[66]来更新控制器，我们采用Proximal Policy Optimization(PPO)[51]，学习率0.0035，因为用PPO训练更快速也更稳定。为鼓励探索，我们还使用了熵惩罚函数，权值为0.0001。在我们的实现中，基准函数是以前的回报的指数滑动平均，权重0.95。控制器的权重初始化为-0.1到0.1的均匀分布随机数。

#### A.3. Training of the Controller 控制器的训练

For distributed training, we use a workqueue system where all the samples generated from the controller RNN are added to a global workqueue. A free “child” worker in a distributed worker pool asks the controller for new work from the global workqueue. Once the training of the child network is complete, the accuracy on a held-out validation set is computed and reported to the controller RNN. In our experiments we use a child worker pool size of 450, which means there are 450 networks being trained on 450 GPUs concurrently at any time. Upon receiving enough child model training results, the controller RNN will perform a gradient update on its weights using PPO and then sample another batch of architectures that go into the global workqueue. This process continues until a predetermined number of architectures have been sampled. In our experiments, this predetermined number of architectures is 20,000 which means the search process is terminated after 20,000 child models have been trained. Additionally, we update the controller RNN with minibatches of 20 architectures. Once the search is over, the top 250 architectures are then chosen to train until convergence on CIFAR-10 to determine the very best architecture.

对于分布式训练，我们使用工作队列系统，其中从控制器RNN生成的所有样本都加入到一个全局工作队列中。分布式worker池中的自由子worker从全局工作队列中问控制器要新的工作。一旦子网络训练完成，就开始计算验证集上的准确率，然后报告给控制器RNN。在我们的试验中我们使用的子worker池的大小为450，这意味着任意时候都可以有450个网络在450个GPU上同时训练。接收到足够多的子模型训练结果后，控制器RNN会用PPO对权值进行梯度更新，然后对在全局工作队列中的另一批结构进行取样。这个过程一直继续下去，直到达到预先确定的结构取样数量。在我们的试验中，预先确定的结构数目是20000，也就是说在训练了20000个子模型之后，搜索过程就停止了。另外，我们用20个结构的minibatch来更新控制器RNN。一旦搜索结束，选择最高的250个结构在CIFAR-10上进行训练到收敛，来确定最佳的结构。

#### A.4. Details of architecture search space 结构搜索空间的细节

We performed preliminary experiments to identify a flexible, expressive search space for neural architectures that learn effectively. Generally, our strategy for preliminary experiments involved small-scale explorations to identify how to run large-scale architecture search.

我们进行预备试验来识别灵活，有表达力的搜索空间，可以更高效的学习。一般的，预备试验的策略是小规模的探索，以识别怎样进行大规模结构搜索。

- All convolutions employ ReLU nonlinearity. Experiments with ELU nonlinearity [10] showed minimal benefit.
- 所有的卷积都用ReLU非线性处理。用ELU非线性处理的试验[10]显示好处很少。
- To ensure that the shapes always match in convolutional cells, 1x1 convolutions are inserted as necessary.
- 为确保卷积cell中的形状匹配，需要的时候加入一些1×1卷积。
- Unlike [24], all depthwise separable convolution do not employ Batch Normalization and/or a ReLU between the depthwise and pointwise operations.
- 与[24]不同，所有depthwise separable卷积都不用批归一化，depthwise操作和pointwise操作间也没有ReLU处理。
- All convolutions followed an ordering of ReLU, convolution operation and Batch Normalization following[21].
- 所有卷积都遵照ReLU、卷积操作、BN的顺序，与[21]中相同。
- Whenever a separable convolution is selected as an operation by the model architecture, the separable convolution is applied twice to the hidden state. We found this empirically to improve overall performance.
- 不论什么时候separable卷积被模型结构选做一个操作，separable卷积应用在隐藏状态上两边。我们的经验表明，这可以改进总体的性能。

#### A.5. Training with ScheduledDropPath 用ScheduledDropPath进行训练

We performed several experiments with various stochastic regularization methods. Naively applying dropout [56] across convolutional filters degraded performance. However, we discovered a new technique called ScheduledDropPath, a modified version of DropPath [33], that works well in regularizing NASNets. In DropPath, we stochastically drop out each path (i.e., edge with a yellow box in Figure 4) in the cell with some fixed probability. This is similar to [27] and [69] where they dropout full parts of their model during training and then at test time scale the path by the probability of keeping that path during training. Interestingly we also found that DropPath alone does not help NASNet training much, but DropPath with linearly increasing the probability of dropping out a path over the course of training significantly improves the final performance for both CIFAR and ImageNet experiments. We name this method ScheduledDropPath.

我们用多种随机正则化方法进行了几个试验。在卷积滤波器之间简单的应用dropout[56]会使性能下降。但是，我们发现一种新技术称为ScheduledDropPath，是DropPath[33]的改进版，对NASNets起到了很好的正则化作用。在DropPath中，我们以某固定概率随机丢弃cell中的每条路径（即，图4中的带黄色方框的边）。这与[27]和[69]相似，其中在训练时会丢弃掉模型的一些部分，在测试时以一定概率保持路径。有意思的是，我们还发现，单独的DropPath对NASNet训练没什么帮助，但在训练过程中以线性增加的概率dropout一个路径可以明显改进在CIFAR和ImageNet上的最终性能。我们称这种方法为ScheduledDropPath。

#### A.6. Training of CIFAR models 在CIFAR上训练模型

All of our CIFAR models use a single period cosine decay as in [39, 18]. All models use the momentum optimizer with momentum rate set to 0.9. All models also use L2 weight decay. Each architecture is trained for a fixed 20 epochs on CIFAR-10 during the architecture search process. Additionally, we found it beneficial to use the cosine learning rate decay during the 20 epochs the CIFAR models were trained as this helped to further differentiate good architectures. We also found that having the CIFAR models use a small N = 2 during the architecture search process allowed for models to train quite quickly, while still finding cells that work well once more were stacked.

所有的CIFAR模型都像[39,18]一样使用周期性的cosine衰减。所有模型都使用动量优化器，动量系数为0.9。所有的模型也使用L2权值衰减。在结构搜索过程中，每个结构都在CIFAR-10上固定进行20个epoch的训练。另外，我们发现，在训练CIFAR模型的20个epoch中，使用cosine学习速率衰减也有好处，因为进一步区分开了好的架构。我们还发现，让CIFAR模型使用较小的N=2会使模型训练的很快，而叠加起来之后工作效果也不错。

#### A.7. Training of ImageNet models 在ImageNet上训练模型

We use ImageNet 2012 ILSVRC challenge data for large scale image classification. The dataset consists of ∼ 1.2M images labeled across 1,000 classes [11]. Overall our training and testing procedures are almost identical to [60]. ImageNet models are trained and evaluated on 299x299 or 331x331 images using the same data augmentation procedures as described previously [60]. We use distributed synchronous SGD to train the ImageNet model with 50 workers (and 3 backup workers) each with a Tesla K40 GPU [7]. We use RMSProp with a decay of 0.9 and epsilon of 1.0. Evaluations are calculated using with a running average of parameters over time with a decay rate of 0.9999. We use label smoothing with a value of 0.1 for all ImageNet models as done in [60]. Additionally, all models use an auxiliary classifier located at 2/3 of the way up the network. The loss of the auxiliary classifier is weighted by 0.4 as done in [60]. We empirically found our network to be insensitive to the number of parameters associated with this auxiliary classifier along with the weight associated with the loss. All models also use L2 regularization. The learning rate decay scheme is the exponential decay scheme used in [60]. Dropout is applied to the final softmax matrix with probability 0.5.

我们使用ImageNet ILSVRC-2012数据来进行大规模图像分类。数据集包括120万幅图，1000个类别[11]。总体来说，我们的训练和测试过程与[60]几乎一样。ImageNet模型的训练和评估都是在299×299或331×331的图像上进行的，使用了前述的同样的数据扩充过程。我们使用分布式同步SGD训练ImageNet模型，50个worker，3个备份worker，每个worker都是一个Tesla K40 GPU[7]。我们使用RMSProp优化方法，衰减为0.9，epsilon为1.0。评估计算是使用随时间的参数滑动平均，衰减0.9999，我们使用标签平滑技术。另外，所有模型都使用辅助分类器，位置在网络的2/3处。辅助分类器的损失函数权重为0.4。我们的经验指出，我们的网络对辅助分类器的参数数量不敏感，也对损失函数的权重不敏感。所有模型都使用L2正则化。学习速率衰减方案是[60]中的指数衰减方案。最终softmax矩阵使用了dropout，概率0.5。

### B. Additional Experiments 其他试验

We now present two additional cells that performed well on CIFAR and ImageNet. The search spaces used for these cells are slightly different than what was used for NASNet-A. For the NASNet-B model in Figure 8 we do not concatenate all of the unused hidden states generated in the convolutional cell. Instead all of the hidden states created within the convolutional cell, even if they are currently used, are fed into the next layer. Note that B = 4 and there are 4 hidden states as input to the cell as these numbers must match for this cell to be valid. We also allow addition followed by layer normalization [2] or instance normalization [61] to be predicted as two of the combination operations within the cell, along with addition or concatenation.

我们现在提出两种另外的cell，在CIFAR和ImageNet上表现良好。这些cell的搜索空间与NASNet-A的略有不同。对于图8中的NASNet-B我们没有拼接那些没用过的隐藏状态。所有在卷积cell中生成的隐藏状态，即使是那些正在使用的，全都feed进下一层。注意B=4，所以有4个隐藏状态作为cell输入，数量需要和有效cell匹配。===

For NASNet-C (Figure 9), we concatenate all of the unused hidden states generated in the convolutional cell like in NASNet-A, but now we allow the prediction of addition followed by layer normalization or instance normalization like in NASNet-B.

### C. Example object detection results

Finally, we will present examples of object detection results on the COCO dataset in Figure 10 and Figure 11. As can be seen from the figures, NASNet-A featurization works well with Faster-RCNN and gives accurate localization of objects.