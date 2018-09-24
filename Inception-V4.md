# Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning 残差连接对学习的影响

Christian Szegedy et al.  Google Inc.

## Abstract 摘要

Very deep convolutional networks have been central to the largest advances in image recognition performance in recent years. One example is the Inception architecture that has been shown to achieve very good performance at relatively low computational cost. Recently, the introduction of residual connections in conjunction with a more traditional architecture has yielded state-of-the-art performance in the 2015 ILSVRC challenge; its performance was similar to the latest generation Inception-v3 network. This raises the question of whether there are any benefit in combining the Inception architecture with residual connections. Here we give clear empirical evidence that training with residual connections accelerates the training of Inception networks significantly. There is also some evidence of residual Inception networks outperforming similarly expensive Inception networks without residual connections by a thin margin. We also present several new streamlined architectures for both residual and non-residual Inception networks. These variations improve the single-frame recognition performance on the ILSVRC 2012 classification task significantly. We further demonstrate how proper activation scaling stabilizes the training of very wide residual Inception networks. With an ensemble of three residual and one Inception-v4, we achieve 3.08% top-5 error on the test set of the ImageNet classification (CLS) challenge.

非常深卷积网络在近年来在图像识别中的大幅进展中起着中心作用。一个例子就是Inception结构已经证明在相对较低的计算代价下可以取得很好的效果。最近，残差连接与传统架构的融合在ILSVRC-2015中取得了最好结果；其表现与最新的Inception-v3架构类似。这就提出了一个问题，Inception结构与残差结构的融合是不是有好处。这里我们给出清楚的经验证据，残差连接的训练，大大加速了Inception网络的训练。也有一些证据说明，残差Inception结构网络比单纯的Inception结构网络性能好一点点。我们还提出几种新的流水结构，既有残差结构，也有非残差结构的Inception网络。这些区别明显改进了在ILSVRC-2012分类任务中的单帧识别性能。我们进一步证明适当的激活幅度控制稳定了非常宽的残差Inception网络。集成了三个残差网络和一个Inception-v4，我们在ImageNet分类挑战的测试集上得到了3.08% top-5错误率。

## 1. Introduction 引言

Since the 2012 ImageNet competition [11] winning entry by Krizhevsky et al [8], their network “AlexNet” has been successfully applied to a larger variety of computer vision tasks, for example to object-detection [4], segmentation [10], human pose estimation [17], video classification[7], object tracking[18], and super resolution[3]. These examples are but a few of all the applications to which deep convolutional networks have been very successfully applied ever since.

自从2012年ImageNet竞赛中Krizhevsky et al [8]获胜，他们的网络AlexNet已经成功应用于很大范围的计算机视觉任务，例如目标检测，目标分割，人体姿态检测，视频分类，目标跟踪和超分辨率。这些例子只是深度卷积网络成功应用的一小部分例子。

In this work we study the combination of the two most recent ideas: Residual connections introduced by He et al. in [5] and the latest revised version of the Inception architecture [15]. In [5], it is argued that residual connections are of inherent importance for training very deep architectures. Since Inception networks tend to be very deep, it is natural to replace the filter concatenation stage of the Inception architecture with residual connections. This would allow Inception to reap all the benefits of the residual approach while retaining its computational efficiency.

在本文中，我们研究了最近的两种思想的融合：He et al. in [5]提出的残差连接，和Inception结构的最新修正版本[15]。在[5]中，残差连接是训练非常深结构网络的内在基础。因为Inception网络就非常深，所以将Inception结构的滤波器拼接阶段替换为残差连接就非常自然。这使Inception可以得到残差方法的全部好处，而且还保持计算上的高效率。

Besides a straightforward integration, we have also studied whether Inception itself can be made more efficient by making it deeper and wider. For that purpose, we designed a new version named Inception-v4 which has a more uniform simplified architecture and more inception modules than Inception-v3. Historically, Inception-v3 had inherited a lot of the baggage of the earlier incarnations. The technical constraints chiefly came from the need for partitioning the model for distributed training using DistBelief[2]. Now, after migrating our training setup to TensorFlow [1] these constraints have been lifted, which allowed us to simplify the architecture significantly. The details of that simplified architecture are described in Section 3.

除了简单的综合，我们还研究了Inception结构自己是否可以更深更宽，从而效率更高。为了这个目的，我们设计了新版本，称为Inception-v4，其结构更加一致简化，比Inception-v3拥有更多的Inception模块。Inception-v3一直继承了earlier incarnation的包袱。这个技术约束主要是由于需要切分模型使用DistBelief进行分布式训练。现在我们将训练放到了TensorFlow中，这些约束就没有了，这使我们可以大幅简化模型架构。我们在第3节中描述简化版的架构。

In this report, we will compare the two pure Inception variants, Inception-v3 and v4, with similarly expensive hybrid Inception-ResNet versions. Admittedly, those models were picked in a somewhat ad hoc manner with the main constraint being that the parameters and computational complexity of the models should be somewhat similar to the cost of the non-residual models. In fact we have tested bigger and wider Inception-ResNet variants and they performed very similarly on the ImageNet classification challenge [11] dataset.

本文我们比较了两种纯粹的Inception变体，v3和v4，以及复杂度类似的Inception-ResNet版本。无可否认，这些模型都是随意选的，主要约束是参数数量和计算复杂度应当与非残差模型比较类似。实际上，我们测试过更大更宽的Inception-ResNet变体，它们的表现在ImageNet分类挑战数据集上非常类似。

The last experiment reported here is an evaluation of an ensemble of all the best performing models presented here. As it was apparent that both Inception-v4 and Inception-ResNet-v2 performed similarly well, exceeding state-of-the art single frame performance on the ImageNet validation dataset, we wanted to see how a combination of those pushes the state of the art on this well studied dataset. Surprisingly, we found that gains on the single-frame performance do not translate into similarly large gains on ensembled performance. Nonetheless, it still allows us to report 3.1% top-5 error on the validation set with four models ensembled setting a new state of the art, to our best knowledge.

本文的最后一个实验是所有最佳模型集成后的评估。很明显，Inception-v4和Inception-ResNet-v2都性能很好，超过了最好的在ImageNet验证数据集上的单框架性能，我们希望看到这些综合起来能将最佳性能提升到什么程度。令人惊讶的是，我们发现在单框架下表现的提升并没有带到集成方法上。尽管如此，集成了4个模型，还是在验证集上得到了3.1%的top-5错误率，这也是目前所知的最佳成绩。

In the last section, we study some of the classification failures and conclude that the ensemble still has not reached the label noise of the annotations on this dataset and there is still room for improvement for the predictions.

在最后一节，我们研究了一些分类的错误情况，得出结论集成模型还是没有达到数据集上标注的标签噪声，预测结果还有改进的空间。

## 2. Related Work 相关工作

Convolutional networks have become popular in large scale image recognition tasks after Krizhevsky et al. [8]. Some of the next important milestones were Network-in-network [9] by Lin et al., VGGNet [12] by Simonyan et al. and GoogLeNet (Inception-v1) [14] by Szegedy et al. Residual connection were introduced by He et al. in [5] in which they give convincing theoretical and practical evidence for the advantages of utilizing additive merging of signals both for image recognition, and especially for object detection. The authors argue that residual connections are inherently necessary for training very deep convolutional models. Our findings do not seem to support this view, at least for image recognition. However it might require more measurement points with deeper architectures to understand the true extent of beneficial aspects offered by residual connections. In the experimental section we demonstrate that it is not very difficult to train competitive very deep networks without utilizing residual connections. However the use of residual connections seems to improve the training speed greatly, which is alone a great argument for their use. The Inception deep convolutional architecture was introduced in [14] and was called GoogLeNet or Inception-v1 in our exposition. Later the Inception architecture was refined in various ways, first by the introduction of batch normalization [6] (Inception-v2) by Ioffe et al. Later the architecture was improved by additional factorization ideas in the third iteration [15] which will be referred to as Inception-v3 in this report.

卷积网络在Krizhevsky et al. [8]之后在大规模图像识别任务中变得流行起来。其他一些重要的里程碑包括Network-in-network [9] by Lin et al., VGGNet [12] by Simonyan et al. and GoogLeNet (Inception-v1) [14] by Szegedy et al。残差连接由He et al. in [5]提出，他给出了很有说服力的理论证据和实践证据，利用加性信号融合对于图像识别，尤其是目标检测非常有好处。作者论证说，残差连接对于非常深度卷积网络的是非常必要的。我们的发现不太支持这种观点，至少在图像识别领域是。但可能需要更多更深的结构的试验来理解残差连接的真正益处。在我们的试验部分，我们说明了没有残差连接，也可以很容易的训练更深的网络。但残差连接似乎可以明显加速训练过程，这是其应用的一个好处。Inception深度卷积架构在[14]中提出，被称为GoogLeNet或Inception-v1。然后Inception架构以多种方式进行了改进，首先由Ioffe et al引入了批归一化[6]，也称为Inception-v2，然后[15]中加入了卷积分解的思想，我们在本文中称之为Inception-v3。

## 3. Architectural Choices 架构选择

### 3.1. Pure Inception blocks 纯Inception模块

Our older Inception models used to be trained in a partitioned manner, where each replica was partitioned into a multiple sub-networks in order to be able to fit the whole model in memory. However, the Inception architecture is highly tunable, meaning that there are a lot of possible changes to the number of filters in the various layers that do not affect the quality of the fully trained network. In order to optimize the training speed, we used to tune the layer sizes carefully in order to balance the computation between the various model sub-networks. In contrast, with the introduction of TensorFlow our most recent models can be trained without partitioning the replicas. This is enabled in part by recent optimizations of memory used by backpropagation, achieved by carefully considering what tensors are needed for gradient computation and structuring the computation to reduce the number of such tensors. Historically, we have been relatively conservative about changing the architectural choices and restricted our experiments to varying isolated network components while keeping the rest of the network stable. Not simplifying earlier choices resulted in networks that looked more complicated that they needed to be. In our newer experiments, for Inception-v4 we decided to shed this unnecessary baggage and made uniform choices for the Inception blocks for each grid size. Plase refer to Figure 9 for the large scale structure of the Inception-v4 network and Figures 3, 4, 5, 6, 7 and 8 for the detailed structure of its components. All the convolutions not marked with “V” in the figures are same-padded meaning that their output grid matches the size of their input. Convolutions marked with “V” are valid padded, meaning that input patch of each unit is fully contained in the previous layer and the grid size of the output activation map is reduced accordingly.

我们更老的Inception模型过去训练时是一种分割的方式，其中每个复制都被分割进一个数个子网络组成的系统中，以能将整个模型都放入内存中。但是，Inception架构是高度可调节的，意思是在各个不同的层中可以使用不同数目的滤波器，这些变化不会影响充分训练的网络的质量。为优化训练速度，我们过去把层的大小仔细调节，为的是平衡各个子网络的计算量。而随着TensorFlow的引入，我们最新的模型可以不进行副本分割而训练。这是由于最新的反向传播使用的内存得到了优化，才可以有这样的功能；通过仔细选择什么张量需要梯度计算，调整计算的结构以减少这样的张量的数量。过去，我们对修改模型架构的选择相对连续，限制我们的试验在不同的孤立的网络模块，而保持网络剩下的部分稳定。没有简化此前的选择，结果使网络看起来比它们需要的更复杂。在我们更新的试验中，对于Inception-v4我们决定摆脱这个不需要的包袱，对每个网格尺寸选择一样的Inception模块。参见图9的大规模结构的Inception-v4网络，参见图3,4,5,6,7,8的各模块详细结构。图中没有V印记的所有卷积都是一样的填充方式，其输出网格与输入大小匹配。有V印记的卷积是有效填充的，意思是每个单元的输入块是全包括在前面一层里的，输出激活图的网格尺寸也相应的减小了。

### 3.2. Residual Inception Blocks 残差Inception模块

For the residual versions of the Inception networks, we use cheaper Inception blocks than the original Inception. Each Inception block is followed by filter-expansion layer (1 × 1 convolution without activation) which is used for scaling up the dimensionality of the filter bank before the addition to match the depth of the input. This is needed to compensate for the dimensionality reduction induced by the Inception block.

对于残差版的Inception网络，我们使用比原始Inception网络更便宜的Inception模块。每个Inception模块后面跟着滤波器扩展层（1×1的卷积，但没有激活），其作用是在匹配输入的深度进行加法前按比例扩大滤波器组的维数。这是为了补偿Inception模块带来的维数下降效果。

We tried several versions of the residual version of Inception. Only two of them are detailed here. The first one “Inception-ResNet-v1” roughly the computational cost of Inception-v3, while “Inception-ResNet-v2” matches the raw cost of the newly introduced Inception-v4 network. See Figure 15 for the large scale structure of both varianets. (However, the step time of Inception-v4 proved to be significantly slower in practice, probably due to the larger number of layers.)

我们尝试过几个版本的残差Inception网络。这里详述其中两个。第一个Inception-ResNet-v1与Inception-v3的计算量差不多，而Inception-ResNet-v2与新提出的Inception-v4网络计算量类似。图15为两种变体的大致结构。但，Inception-v4的步骤时间在实际中明显更少，可能是因为层数太多。

Another small technical difference between our residual and non-residual Inception variants is that in the case of Inception-ResNet, we used batch-normalization only on top of the traditional layers, but not on top of the summations. It is reasonable to expect that a thorough use of batch-normalization should be advantageous, but we wanted to keep each model replica trainable on a single GPU. It turned out that the memory footprint of layers with large activation size was consuming disproportionate amount of GPU-memory. By omitting the batch-normalization on top of those layers, we were able to increase the overall number of Inception blocks substantially. We hope that with better utilization of computing resources, making this trade-off will become unecessary.

残差Inception网络与非残差Inception网络的另一个小的技术区别是在Inception-ResNet的情况下，我们只在传统层上面使用批归一化，在求和层的上面就不使用了。如果全都使用批归一化，会有好处，但我们想在每个GPU上每个模型的复制都是可训练的。结果显示，激活规模大的层消耗的GPU内存很大。通过在这些层上忽略BN，我们可以增加总计的Inception模块。我们希望更好的利用计算资源，这点折中没有必要。

### 3.3. Scaling of the Residuals 残差尺度的控制

Also we found that if the number of filters exceeded 1000, the residual variants started to exhibit instabilities and the network has just “died” early in the training, meaning that the last layer before the average pooling started to produce only zeros after a few tens of thousands of iterations. This could not be prevented, neither by lowering the learning rate, nor by adding an extra batch-normalization to this layer.

我们还发现如果滤波器数量超过1000，那么残差变体网络就会表现出不稳定性，网络会在训练过程中很早就“死去”，意思是平均池化之前的最后一层在几千上万次迭代后只输出零值了。这不可避免，不论是降低学习速率，或对这一层增加额外的批归一化处理。

We found that scaling down the residuals before adding them to the previous layer activation seemed to stabilize the training. In general we picked some scaling factors between 0.1 and 0.3 to scale the residuals before their being added to the accumulated layer activations (cf. Figure 20).

我们发现在将残差连接加入前一层之前，按比例缩小残差的尺寸，，似乎可以使训练稳定下来。总体来说，我们选择一些尺度因子(0.1到0.3之间)在将其加入到累积层激活时，缩小其尺寸（参见图20）。

A similar instability was observed by He et al. in [5] in the case of very deep residual networks and they suggested a two-phase training where the first “warm-up” phase is done with very low learning rate, followed by a second phase with high learning rata. We found that if the number of filters is very high, then even a very low (0.00001) learning rate is not sufficient to cope with the instabilities and the training with high learning rate had a chance to destroy its effects. We found it much more reliable to just scale the residuals.

在He et al.[5]中的非常深度残差网络，观察到了类似的不稳定现象，他们建议训练分为两阶段，第一阶段为预热阶段，学习速率很低，第二阶段是高学习速率阶段。我们发现如果滤波器数量非常多，那么即使非常低的学习速率(0.00001)也不能应对不稳定性，并且高学习速率的训练可能会毁掉这种效果。我们发现控制残差的尺度还是更可靠一些。

Even where the scaling was not strictly necessary, it never seemed to harm the final accuracy, but it helped to stabilize the training.

即使在控制尺度不是那么必须的地方，也没有对最后的准确率造成损害，但确实对稳定训练有帮助。

## 4. Training Methodology 训练方法

We have trained our networks with stochastic gradient utilizing the TensorFlow [1] distributed machine learning system using 20 replicas running each on a NVidia Kepler GPU. Our earlier experiments used momentum [13] with a decay of 0.9, while our best models were achieved using RMSProp [16] with decay of 0.9 and $\epsilon$ = 1.0. We used a learning rate of 0.045, decayed every two epochs using an exponential rate of 0.94. Model evaluations are performed using a running average of the parameters computed over time.

我们用随机梯度下降法训练网络，使用TensorFlow分布式机器学习系统，有20个副本每个都在一个NVIDIA Kepler GPU上运行。我们较早的试验使用了动量法，衰减为0.9，我们最好的模型是使用RMSProp训练得到的，衰减为0.9，$\epsilon$为1.0。我们使用的学习速率为0.045，每两个epoch衰减，衰减的速率为0.94。模型评估用的是参数量的运行平均。

## 5. Experimental Results 试验结果

First we observe the top-1 and top-5 validation-error evolution of the four variants during training. After the experiment was conducted, we have found that our continuous evaluation was conducted on a subset of the validation set which omitted about 1700 blacklisted entities due to poor bounding boxes. It turned out that the omission should have been only performed for the CLSLOC benchmark, but yields somewhat incomparable (more optimistic) numbers when compared to other reports including some earlier reports by our team. The difference is about 0.3% for top-1 error and about 0.15% for the top-5 error. However, since the differences are consistent, we think the comparison between the curves is a fair one.

首先我们看4个变体的top-1和top-5验证错误率在训练过程中的演化情况。在试验进行后，我们发现评估是在验证集的子集上进行的，因为忽略了1700个黑名单上的实体，这是因为不好的边界框造成的。结果发现，这种忽略应当只在CLSLOC基准测试上进行，但当与其他报告包括我们团队的前面一些报告相比时产生了不可比（更乐观的）数目。其差异大约是0.3%的top-1错误率，0.15%的top-5错误率。但是，由于差异是连续的，我们认为曲线间的比较是公平的比较。

On the other hand, we have rerun our multi-crop and ensemble results on the complete validation set consisting of 50000 images. Also the final ensemble result was also performed on the test set and sent to the ILSVRC test server for validation to verify that our tuning did not result in an over-fitting. We would like to stress that this final validation was done only once and we have submitted our results only twice in the last year: once for the BN-Inception paper and later during the ILSVR-2015 CLSLOC competition, so we believe that the test set numbers constitute a true estimate of the generalization capabilities of our model.

另一方面，我们在包含50000幅图的完整验证集上重新运行了我们的多剪切块和集成结果。最后的集成结果也在测试集上进行了运行，送入了ILSVRC测试服务器进行验证，核实我们的精调没有得到过拟合的结果。我们要强调，这个最终的验证只进行了一次，我们将结果在去年只提交了两次：一次是BN-Inception论文，后面再ILSVR-2015 CLSLOC竞赛上，所以我们相信测试集数目确实构成了一次对模型泛化能力的真实估计。

Finally, we present some comparisons, between various versions of Inception and Inception-ResNet. The models Inception-v3 and Inception-v4 are deep convolutional networks not utilizing residual connections while Inception-ResNet-v1 and Inception-ResNet-v2 are Inception style networks that utilize residual connections instead of filter concatenation.

最后，我们对各种Inception和Inception-ResNet模型进行了各种比较。Inception-v3模型和Inception-v4模型是没有使用残差连接的深度卷积网络，而Inception-ResNet-v1和Inception-ResNet-v2是使用了残差连接的Inception类网络，没有使用滤波器拼接。

Table 2 shows the single-model, single crop top-1 and top-5 error of the various architectures on the validation set.表2是单模型，单剪切块的top-1和top-5错误率，是多个模型架构在验证集上的结果。

Table 3 shows the performance of the various models with a small number of crops: 10 crops for ResNet as was reported in [5], for the Inception variants, we have used the 12 crops evaluation as as described in [14]. 表3所示的是各种模型在较小数量的剪切块上的表现：[5]中是ResNet在10剪切块上的结果，对于Inception变体，我们使用了12个剪切块评估，如[14]中提到的方法一样。

Table 4 shows the single model performance of the various models using. For residual network the dense evaluation result is reported from [5]. For the inception networks, the 144 crops strategy was used as described in [14]. 表4所示的是各种模型的单模型表现。对于残差网络，[5]中是稠密评估结果。对于Inception网络，使用了[14]中叙述的144剪切块策略。

Table 5 compares ensemble results. For the pure residual network the 6 models dense evaluation result is reported from [5]. For the inception networks 4 models were ensembled using the 144 crops strategy as described in [14]. 表5比较了集成算法结果。对于纯残差网络，[5]中是6模型稠密评估的结果。对于Inception网络，集成了4个模型，使用了[14]中的144剪切块策略。

## 6. Conclusions 结论

We have presented three new network architectures in detail: 我们提出了三种新的网络结构：

- Inception-ResNet-v1: a hybrid Inception version that has a similar computational cost to Inception-v3 from [15].
- Inception-ResNet-v1：混合Inception模型，计算代价与Inception-v3类似；
- Inception-ResNet-v2: a costlier hybrid Inception version with significantly improved recognition performance.
- Inception-ResNet-v2：混合Inception模型，计算代价较大，但识别性能提升明显；
- Inception-v4: a pure Inception variant without residual connections with roughly the same recognition performance as Inception-ResNet-v2.
- Inception-v4：纯Inception变体，没有残差连接，与Inception-ResNet-v2识别率表现差不多。

We studied how the introduction of residual connections leads to dramatically improved training speed for the Inception architecture. Also our latest models (with and without residual connections) outperform all our previous networks, just by virtue of the increased model size.

我们研究了残差连接的引入极大改进了Inception架构网络的训练速度。最新的模型（带有或不带有残差连接）比以前所有的网络都要好，其原因只是由于模型规模变大了。

## References