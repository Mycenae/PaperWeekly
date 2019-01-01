# YOLOv3: An Incremental Improvement

Joseph Redmon, Ali Farhadi University of Washington

## Abstract 摘要

We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that’s pretty swell. It’s a little bigger than last time but more accurate. It’s still fast though, don’t worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 $AP_{50}$ in 51 ms on a Titan X, compared to 57.5 $AP_{50}$ in 198 ms by RetinaNet, similar performance but 3.8× faster. As always, all the code is online at https://pjreddie.com/yolo/.

我们提出YOLO的一些升级。我们修改了一些小的设计，使其更好。我们还训练了这个新的网络，效果很好。比上一个模型更大一些，但也更精确，同时还是最快的。再320×320的分辨率上，YOLOv3的运行速度为22ms(45.5FPS)，准确度28.2mAP，和SSD一样精确，但速度快了三倍。当我们采用老的0.5IOU mAP检测度量标准，YOLOv3是非常好的。在Titan X上，$AP_{50}$为57.9，速度51ms(19.6FPS)，与RetinaNet相比，$AP_{50}$为57.5，速度为198ms(5.1FPS)，性能接近，但速度快了3.8倍。像以往一样，所有的代码都在线可用。

## 1. Introduction 引言

Sometimes you just kinda phone it in for a year, you know? I didn’t do a whole lot of research this year. Spent a lot of time on Twitter. Played around with GANs a little. I had a little momentum left over from last year [12] [1]; I managed to make some improvements to YOLO. But, honestly, nothing like super interesting, just a bunch of small changes that make it better. I also helped out with other people’s research a little.

我们对YOLO进行了一些改进，并没有什么令人非常感兴趣的东西，得到了更好的一版YOLOv3。

Actually, that’s what brings us here today. We have a camera-ready deadline [4] and we need to cite some of the random updates I made to YOLO but we don’t have a source. So get ready for a TECH REPORT!

The great thing about tech reports is that they don’t need intros, y’all know why we’re here. So the end of this introduction will signpost for the rest of the paper. First we’ll tell you what the deal is with YOLOv3. Then we’ll tell you how we do. We’ll also tell you about some things we tried that didn’t work. Finally we’ll contemplate what this all means.

我们首先介绍一下YOLOv3的要点，然后是怎样实现的，我们还会告诉你一些尝试过的东西，但是没有起作用的，最后我们对这些改进的意义进行了思考。

Figure 1. We adapt this figure from the Focal Loss paper [9]. YOLOv3 runs significantly faster than other detection methods with comparable performance. Times from either an M40 or Titan X, they are basically the same GPU.

图1. 我们用了Focal Loss文章[9]中的图。YOLOv3在相近的性能下，比其他检测方法运行的都要快的多。运行时间是M40或Titan X上的，其算力基本相同。

Method | mAP | time
--- | --- | ---
[B] SSD321 | 28.0 | 61
[C] DSSD321 | 28.0 | 85
[D] R-FCN | 29.9 | 85
[E] SSD513 | 31.2 | 125
[F] DSSD513 | 33.2 | 156
[G] FPN FRCN | 36.2 | 172
RetinaNet-50-500 | 32.5 | 73
RetinaNet-101-500 | 34.4 | 90
RetinaNet-101-800 | 37.8 | 198
YOLOv3-320 | 28.2 | 22
YOLOv3-416 | 31.0 | 29
YOLOv3-608 | 33.0 | 51

## 2. The Deal

So here’s the deal with YOLOv3: We mostly took good ideas from other people. We also trained a new classifier network that’s better than the other ones. We’ll just take you through the whole system from scratch so you can understand it all.

下面是YOLOv3的要点，我们主要从别人那里得到了这些好主意。我们还训练了一个新的分类器网络，比其他人的都要好。我们会带你看看整个系统的细节，所以可以彻底理解它。

### 2.1. Bounding Box Prediction 边界框预测

Following YOLO9000 our system predicts bounding boxes using dimension clusters as anchor boxes [15]. The network predicts 4 coordinates for each bounding box, $t_x$, $t_y$, $t_w$, $t_h$. If the cell is offset from the top left corner of the image by ($c_x$, $c_y$) and the bounding box prior has width and height $p_w$, $p_h$, then the predictions correspond to:

与YOLO9000一样，我们的系统使用维度聚类来预测边界框作为锚框[15]。网络对每个边界框预测4个坐标，$t_x$, $t_y$, $t_w$, $t_h$。如果一个单元相对图像左上角的偏移是($c_x$, $c_y$)，边界框的先验宽度和高度是$p_w$, $p_h$，那么预测就对应着：

$$b_x = σ(t_x) + c_x$$
$$b_y = σ(t_y) + c_y$$
$$b_w = p_w e^{t_w}$$
$$b_h = p_h e^{t_h}$$

Figure 2. Bounding boxes with dimension priors and location prediction. We predict the width and height of the box as offsets from cluster centroids. We predict the center coordinates of the box relative to the location of filter application using a sigmoid function. This figure blatantly self-plagiarized from [15].

图2. 边界框的先验维度和位置预测。我们以聚类重心的偏移来预测框的宽度和高度。我们预测框的中心坐标相对于滤波器位置，使用了sigmoid函数。这个图从[15]中引用而来。

During training we use sum of squared error loss. If the ground truth for some coordinate prediction is $\hat t_*$ our gradient is the ground truth value (computed from the ground truth box) minus our prediction: $\hat t_* - t_*$. This ground truth value can be easily computed by inverting the equations above.

训练中我们使用平方误差损失函数。如果坐标预测的真值为$\hat t_*$，我们的梯度就是真值（从真值框中计算得到）减去我们的预测：$\hat t_* - t_*$。这个真值可以轻松的计算出来，即将上述公式反转即可。

YOLOv3 predicts an objectness score for each bounding box using logistic regression. This should be 1 if the bounding box prior overlaps a ground truth object by more than any other bounding box prior. If the bounding box prior is not the best but does overlap a ground truth object by more than some threshold we ignore the prediction, following [17]. We use the threshold of .5. Unlike [17] our system only assigns one bounding box prior for each ground truth object. If a bounding box prior is not assigned to a ground truth object it incurs no loss for coordinate or class predictions, only objectness.

YOLOv3使用logistic回归为每个边界框预测objectness分数。如果边界框先验与真值目标重叠度比其他任何边界框先验要高，就赋值为1。如果边界框先验不是最佳重叠，但确实与真值目标重叠度超过了一定阈值，那么我们就忽略这个预测，这是[17]中的思想。我们使用的阈值为0.5。与[17]不同的是，我们的系统只为每个真值目标指定一个边界框先验。如果边界框先验没有指定给一个真值目标，那么它不会带来任何坐标损失或类别预测损失，只有objectness。

### 2.2. Class Prediction 类别预测

Each box predicts the classes the bounding box may contain using multilabel classification. We do not use a softmax as we have found it is unnecessary for good performance, instead we simply use independent logistic classifiers. During training we use binary cross-entropy loss for the class predictions.

每个框预测边界框可能包含的目标类别，使用的是多标签分类。我们不使用softmax，因为我们发现不需要它来带来好性能，我们只是使用独立的logistic分类器。在训练的时候，我们使用二值交叉熵损失进行类别预测。

This formulation helps when we move to more complex domains like the Open Images Dataset [7]. In this dataset there are many overlapping labels (i.e. Woman and Person). Using a softmax imposes the assumption that each box has exactly one class which is often not the case. A multilabel approach better models the data.

当我们在更复杂的领域工作时，如Open Images Dataset[7]，这种设定有所帮助。在这个数据集里，有很多重叠的标签（如，女人和人）。使用softmax时，默认假设每个框只有一个类别，但经常不是这样的。多标签方法更适合这种数据建模。

### 2.3. Predictions Across Scales 跨尺度预测

YOLOv3 predicts boxes at 3 different scales. Our system extracts features from those scales using a similar concept to feature pyramid networks [8]. From our base feature extractor we add several convolutional layers. The last of these predicts a 3-d tensor encoding bounding box, objectness, and class predictions. In our experiments with COCO [10] we predict 3 boxes at each scale so the tensor is N×N×[3∗(4+1+80)] for the 4 bounding box offsets, 1 objectness prediction, and 80 class predictions.

YOLOv3在三个不同的尺度上预测边界框。我们的系统在这些尺度中使用与Feature Pyramid Networks[8]类似的概念提取特征。从我们的基础特征提取器中，我们增加了几个卷积层。这些卷积层的最后一个预测了一个3d张量，包括边界框，objectness和类别预测。我们在COCO[10]的试验中，在每个尺度上预测3个框，所以这个张量的维数为N×N×[3∗(4+1+80)]，其中4是边界框的偏移，1是objectness预测值，80是COCO的类别数目。

Next we take the feature map from 2 layers previous and upsample it by 2×. We also take a feature map from earlier in the network and merge it with our upsampled features using concatenation. This method allows us to get more meaningful semantic information from the upsampled features and finer-grained information from the earlier feature map. We then add a few more convolutional layers to process this combined feature map, and eventually predict a similar tensor, although now twice the size.

下一步，我们取前2层的特征图，将其作2倍的上采样。我们还从网络更前面的地方取了一个特征图，与我们上采样的特征拼接融合。这种方法是我们能从上采样的特征中得到更有意义的语义信息，从更前面的特征图中得到细粒度信息。我们然后增加了一些卷积层，来处理这些合并的特征图，最终预测到一个类似的张量，其特征已经是两倍大小。

We perform the same design one more time to predict boxes for the final scale. Thus our predictions for the 3rd scale benefit from all the prior computation as well as fine-grained features from early on in the network.

我们将同样的设计思想再次应用一次，对最终的尺度预测边界框。所以我们在第3种尺度上的预测受益于之前的计算，以及网络前面的细粒度特征。

We still use k-means clustering to determine our bounding box priors. We just sort of chose 9 clusters and 3 scales arbitrarily and then divide up the clusters evenly across scales. On the COCO dataset the 9 clusters were: (10×13),(16×30),(33×23),(30×61),(62×45),(59×119),(116 × 90),(156 × 198),(373 × 326).

我们仍然使用k均值聚类来确定边界框先验。我们只是任意的选择了9个聚类和3个尺度，然后在这个尺度上平均分配聚类数目。在COCO数据集上，这9个聚类为：(10×13),(16×30),(33×23),(30×61),(62×45),(59×119),(116 × 90),(156 × 198),(373 × 326)。

### 2.4. Feature Extractor 特征提取器

We use a new network for performing feature extraction. Our new network is a hybrid approach between the network used in YOLOv2, Darknet-19, and that newfangled residual network stuff. Our network uses successive 3×3 and 1×1 convolutional layers but now has some shortcut connections as well and is significantly larger. It has 53 convolutional layers so we call it.... wait for it..... Darknet-53!

我们使用了新的网络来进行特征提取。我们的新网络是两种网络的混合体，分别是YOLOv2中使用的网络Darknet-19，和新的residual网络。我们的网络使用连续的3×3和1×1卷积层，但现在也有一些捷径连接，规模也明显增大了，有53个卷积层，所以我们称之为Darknet-53。

This new network is much more powerful than Darknet-19 but still more efficient than ResNet-101 or ResNet-152. Here are some ImageNet results:

这个新网络比Darknet-19强力很多，但仍然比ResNet-101或ResNet-152更有效率。下面是一些在ImageNet数据集上的结果：

Table 2. Comparison of backbones. Accuracy, billions of operations, billion floating point operations per second, and FPS for various networks.

表2. 骨干网络的对比。各种网络的准确度，多少billion次运算，每秒多少billion次浮点运算，和FPS。

Backbone | Top-1 | Top-5 | Bn Ops | BFLOP/s | FPS
--- | --- | --- | --- | --- | ---
Darknet-19 [15] | 74.1 | 91.8 | 7.29 | 1246 | 171
ResNet-101[5] | 77.1 | 93.7 | 19.7 | 1039 | 53
ResNet-152 [5] | 77.6 | 93.8 | 29.4 | 1090 | 37
Darknet-53 | 77.2 | 93.8 | 18.7 | 1457 | 78

Each network is trained with identical settings and tested at 256×256, single crop accuracy. Run times are measured on a Titan X at 256 × 256. Thus Darknet-53 performs on par with state-of-the-art classifiers but with fewer floating point operations and more speed. Darknet-53 is better than ResNet-101 and 1.5× faster. Darknet-53 has similar performance to ResNet-152 and is 2× faster.

每个网络都用相同的设置进行训练，并在256×256的单剪切块上进行测试。运行参数是在Titan X上的256×256输入上衡量的。所以Darknet-53与目前最好的分类器的性能是不相上下的，但是浮点运算量更少，速度更快。Darknet-53比ResNet-101要好，速度快1.5倍。Darknet-53与ResNet-152性能类似，速度快2倍。

Darknet-53 also achieves the highest measured floating point operations per second. This means the network structure better utilizes the GPU, making it more efficient to evaluate and thus faster. That’s mostly because ResNets have just way too many layers and aren’t very efficient.

Darknet-53的每秒浮点数运算量是最高的。这意味着网络结构可以更好的利用GPU，运算更有效率，所以更快。这主要是因为ResNet的层数太多，效率并不是很高。

### 2.5. Training 训练

We still train on full images with no hard negative mining or any of that stuff. We use multi-scale training, lots of data augmentation, batch normalization, all the standard stuff. We use the Darknet neural network framework for training and testing [14].

我们还是训练整幅图像，没有使用难分样本挖掘或其他什么技术。我们使用多尺度训练，很多的数据扩充，批归一化，所有标准的技术都有。我们使用Darknet神经网络框架来进行训练和测试[14]。

## 3. How We Do 我们是怎么做的

YOLOv3 is pretty good! See table 3. In terms of COCOs weird average mean AP metric it is on par with the SSD variants but is 3× faster. It is still quite a bit behind other models like RetinaNet in this metric though.

YOLOv3效果非常好，详见表3。在COCO数据集的奇怪的mAP衡量标准上，与SSD的变体算法是不相上下的，但速度快了3倍。但在这种衡量标准下，与一些模型如RetinaNet的效果还是有不小的差距的。

Table 3. I’m seriously just stealing all these tables from [9] they take soooo long to make from scratch. Ok, YOLOv3 is doing alright. Keep in mind that RetinaNet has like 3.8× longer to process an image. YOLOv3 is much better than SSD variants and comparable to state-of-the-art models on the $AP_{50}$ metric.

表3. 我们从[9]中借鉴了此表格。YOLOv3的效果还是可以的。要注意到，RetinaNet的速度慢了3.8倍。YOLOv3比SSD的变体算法要好太多，与目前最好的算法在$AP_{50}$的衡量标准下是可以比较的。

| | backbone | AP | $AP_{50}$ | $AP_{75}$ | $AP_S$ | $AP_M$ | $AP_L$
--- | --- | --- | --- | --- | --- | --- | --- 
Two-stage methods | | | | | | | 
Faster R-CNN+++ [5] | ResNet-101-C4 | 34.9 | 55.7 | 37.4 | 15.6 | 38.7 | 50.9
Faster R-CNN w FPN [8] | ResNet-101-FPN | 36.2 | 59.1 | 39.0 | 18.2 | 39.0 | 48.2
Faster R-CNN by G-RMI [6] | Inception-ResNet-v2 [21] | 34.7 | 55.5 | 36.7 | 13.5 | 38.1 | 52.0
Faster R-CNN w TDM [20] | Inception-ResNet-v2-TDM | 36.8 | 57.7 | 39.2 | 16.2 | 39.8 | 52.1
One-stage methods | | | | | | |
YOLOv2 [15] | DarkNet-19 [15] | 21.6 | 44.0 | 19.2 | 5.0 | 22.4 | 35.5
SSD513 [11, 3] | ResNet-101-SSD | 31.2 | 50.4 | 33.3 | 10.2 | 34.5 | 49.8
DSSD513 [3] | ResNet-101-DSSD | 33.2 | 53.3 | 35.2 | 13.0 | 35.4 | 51.1
RetinaNet [9] | ResNet-101-FPN | 39.1 | 59.1 | 42.3 | 21.8 | 42.7 | 50.2
RetinaNet [9] | ResNeXt-101-FPN | 40.8 | 61.1 | 44.1 | 24.1 | 44.2 | 51.2
YOLOv3 608 × 608 | Darknet-53 | 33.0 | 57.9 | 34.4 | 18.3 | 35.4 | 41.9

However, when we look at the “old” detection metric of mAP at IOU= .5 (or $AP_{50}$ in the chart) YOLOv3 is very strong. It is almost on par with RetinaNet and far above the SSD variants. This indicates that YOLOv3 is a very strong detector that excels at producing decent boxes for objects. However, performance drops significantly as the IOU threshold increases indicating YOLOv3 struggles to get the boxes perfectly aligned with the object.

但是，当我们用旧的检测衡量标准即IOU=0.5时的mAP（即表中的$AP_{50}$）时，YOLOv3是很强的，几乎与RetinaNet一样，比SSD变体算法要强的多。这意味着，YOLOv3是一种非常强的检测器，在生成目标框时超过了其他算法。但是，当IOU阈值增加时，性能下降很快，说明YOLOv3在将边界框与目标完全对齐上还有困难。

In the past YOLO struggled with small objects. However, now we see a reversal in that trend. With the new multi-scale predictions we see YOLOv3 has relatively high $AP_S$ performance. However, it has comparatively worse performance on medium and larger size objects. More investigation is needed to get to the bottom of this.

在过去，YOLO在小目标上性能不佳。但是，现在我们看到了相反的局面。采用了新的多尺度预测方法，我们看到YOLOv3的$AP_S$表现相对很高。但在中型目标和大型目标上表现相对差一些。需要研究一下其底层原因。

When we plot accuracy vs speed on the $AP_{50}$ metric (see figure 3) we see YOLOv3 has significant benefits over other detection systems. Namely, it’s faster and better.

图3是$AP_{50}$衡量标准下的准确率vs速度图，我们看到YOLOv3比其他检测系统有明显优势，即，更快更好。

Figure 3. Again adapted from the [9], this time displaying speed/accuracy tradeoff on the mAP at .5 IOU metric. You can tell YOLOv3 is good because it’s very high and far to the left. Can you cite your own paper? Guess who’s going to try, this guy → [16]. Oh, I forgot, we also fix a data loading bug in YOLOv2, that helped by like 2 mAP. Just sneaking this in here to not throw off layout.

Method | mAP-50 | time
--- | --- | ---
[B] SSD321 | 45.4 | 61
[C] DSSD321 | 46.1 | 85
[D] R-FCN | 51.9 | 85
[E] SSD513 | 50.4 | 125
[F] DSSD513 | 53.3 | 156
[G] FPN FRCN | 59.1 | 172
RetinaNet-50-500 | 50.9 | 73 
RetinaNet-101-500 | 53.1 | 90
RetinaNet-101-800 | 57.5 | 198
YOLOv3-320 | 51.5 | 22
YOLOv3-416 | 55.3 | 29
YOLOv3-608 | 57.9 | 51

## 4. Things We Tried That Didn’t Work 我们尝试了但没有起作用的工作

We tried lots of stuff while we were working on YOLOv3. A lot of it didn’t work. Here’s the stuff we can remember.

我们在开发YOLOv3的时候，尝试了很多东西，一些没有起作用，下面是我们还能记起来的东西。

**Anchor box x,y offset predictions**. We tried using the normal anchor box prediction mechanism where you predict the x,y offset as a multiple of the box width or height using a linear activation. We found this formulation decreased model stability and didn’t work very well.

**锚框x,y偏移预测**。我们尝试了使用正常的锚框预测方法，也就是用线性激活将x,y偏移预测为框宽或高的乘数。我们发现这种方法降低模型稳定性，效果不太好。

**Linear x,y predictions instead of logistic**. We tried using a linear activation to directly predict the x,y offset instead of the logistic activation. This led to a couple point drop in mAP.

**线性x,y预测，而不是logistic预测**。我们尝试使用线性预测来直接预测x,y偏移，而不是使用logistic激活。这带来了mAP的一些降低。

**Focal loss**. We tried using focal loss. It dropped our mAP about 2 points. YOLOv3 may already be robust to the problem focal loss is trying to solve because it has separate objectness predictions and conditional class predictions. Thus for most examples there is no loss from the class predictions? Or something? We aren’t totally sure.

**Focal损失函数**。我们从尝试了focal损失函数，这使mAP降低了2个百分点。Focal损失函数尝试解决一些问题，YOLOv3对这些问题已经很稳健了，因为其objectness预测和条件类别预测是分离的。所以对于多数样本，没有类别预测的损失？我们不是太确定。

**Dual IOU thresholds and truth assignment**. Faster R-CNN uses two IOU thresholds during training. If a prediction overlaps the ground truth by .7 it is as a positive example, by [.3−.7] it is ignored, less than .3 for all ground truth objects it is a negative example. We tried a similar strategy but couldn’t get good results.

**双IOU阈值和真值指定**。Faster R-CNN在训练时使用两个IOU阈值。如果预测与真值重叠度大于0.7那么就是正样本，在0.3和0.7之间，就忽略之，对所有的真值目标重叠度都小于0.3的时候是负样本。我们尝试了类似的策略，但是没有得到好结果。

We quite like our current formulation, it seems to be at a local optima at least. It is possible that some of these techniques could eventually produce good results, perhaps they just need some tuning to stabilize the training.

我们很喜欢目前的这种构思，至少是在一个不错的局部极小值点。有可能一些技术可能最后会产生很好的结果，可能只是需要更多的精调来使训练稳定下来。

## 5. What This All Means

YOLOv3 is a good detector. It’s fast, it’s accurate. It’s not as great on the COCO average AP between .5 and .95 IOU metric. But it’s very good on the old detection metric of .5 IOU.

YOLOv3是一个好的检测器，速度快，准确率高。在IOU在0.5和0.95之间时，其COCO平均AP没那么好，但是在老的检测衡量标准即IOU 0.5时是非常好的。

Why did we switch metrics anyway? The original COCO paper just has this cryptic sentence: “A full discussion of evaluation metrics will be added once the evaluation server is complete”. Russakovsky et al report that that humans have a hard time distinguishing an IOU of .3 from .5! “Training humans to visually inspect a bounding box with IOU of 0.3 and distinguish it from one with IOU 0.5 is surprisingly difficult.” [18] If humans have a hard time telling the difference, how much does it matter?

我们为什么切换衡量标准？原始COCO论文中原话很神秘：“一旦评估服务器准备完毕，就完整的讨论一下评估衡量标准”。Russakovsky等报告称，人类在IOU为0.3到0.5的时候，很难区分。“训练人类视觉上检查IOU为0.3边界框，分辨其与IOU为0.5的边界框异常困难。”[18] 如果人类都这么难分辨，那么又有什么关系呢？

But maybe a better question is: “What are we going to do with these detectors now that we have them?” A lot of the people doing this research are at Google and Facebook. I guess at least we know the technology is in good hands and definitely won’t be used to harvest your personal information and sell it to.... wait, you’re saying that’s exactly what it will be used for?? Oh.

但是下面可能是一个更好的问题：“现在我们有了这些检测器，我们应当怎么去用呢？”很多人在Google和Facebook做这些研究，我猜至少我们知道技术应当在好人之手，绝对不能用于收割个人信息去出售，但是这些技术确实可以用于这些用途。

Well the other people heavily funding vision research are the military and they’ve never done anything horrible like killing lots of people with new technology oh wait..... (The author is funded by the Office of Naval Research and Google.)

另外对视觉研究投资很多的是军方，他们现在还没有做出任何恐怖的事，比如用新技术去杀死很多人（作者是由海军研究办公室和Google资助的）。

I have a lot of hope that most of the people using computer vision are just doing happy, good stuff with it, like counting the number of zebras in a national park [13], or tracking their cat as it wanders around their house [19]. But computer vision is already being put to questionable use and as researchers we have a responsibility to at least consider the harm our work might be doing and think of ways to mitigate it. We owe the world that much.

我很希望多数人使用计算机视觉去做开心的好事，比如在国家公园里数数有多少斑马[13]，或者当他们的猫在家里晃悠的时候进行跟踪[19]。但是计算机视觉的一些用处已经让人怀疑，作为研究者，我们有责任考虑我们的工作可能造成什么危害，设法去减缓这种危害。我们亏欠这个世界很多。

In closing, do not @me. (Because I finally quit Twitter).

## Rebuttal 抗辩

We would like to thank the Reddit commenters, labmates, emailers, and passing shouts in the hallway for their lovely, heartfelt words. If you, like me, are reviewing for ICCV then we know you probably have 37 other papers you could be reading that you’ll invariably put off until the last week and then have some legend in the field email you about how you really should finish those reviews execept it won’t entirely be clear what they’re saying and maybe they’re from the future? Anyway, this paper won’t have become what it will in time be without all the work your past selves will have done also in the past but only a little bit further forward, not like all the way until now forward. And if you tweeted about it I wouldn’t know. Just sayin.

我们要感谢Reddit评论者，实验室同事，发邮件者的互动。如果你像我一样正在为ICCV评审论文，那么你可能有37篇其他的文章可以读，你可能一直放到最后一个星期，然后怎么可能完成这些评审呢，论文说的也不明白，可能作者们是来自未来的？blah blah blah

Reviewer #2 AKA Dan Grossman (lol blinding who does that) insists that I point out here that our graphs have not one but two non-zero origins. You’re absolutely right Dan, that’s because it looks way better than admitting to ourselves that we’re all just here battling over 2-3% mAP. But here are the requested graphs. I threw in one with FPS too because we look just like super good when we plot on FPS.

评审者#2坚持认为我们的图有两个非零原点，而不是一个。这绝对没错，是因为这使得2-3%的mAP提升显得非常好看。这里是需要的两张图，一张是以FPS为轴的。

Figure 4. Zero-axis charts are probably more intellectually honest... and we can still screw with the variables to make ourselves look good!

Reviewer #4 AKA JudasAdventus on Reddit writes “Entertaining read but the arguments against the MSCOCO metrics seem a bit weak”. Well, I always knew you would be the one to turn on me Judas. You know how when you work on a project and it only comes out alright so you have to figure out some way to justify how what you did actually was pretty cool? I was basically trying to do that and I lashed out at the COCO metrics a little bit. But now that I’ve staked out this hill I may as well die on it.

评审者#4在Reddit上写到，“读起来很有趣，但是针对COCO度量标准的论点似乎有点弱”。确实没错，我们的结果只能说还行，所以我们在COCO度量标准上做文章，以使我们的工作显得更好。

See here’s the thing, mAP is already sort of broken so an update to it should may be address some of the issues with it or at least justify why the updated version is better in some way. And that’s the big thing I took issue with was the lack of justification. For PASCAL VOC, the IOU threshold was “set deliberately low to account for inaccuracies in bounding boxes in the ground truth data”[2]. Does COCO have better labelling than VOC? This is definitely possible since COCO has segmentation masks maybe the labels are more trustworthy and thus we aren’t as worried about inaccuracy. But again, my problem was the lack of justification.

所以是这样的，mAP已经有些不好用了，所以需要升级一下，解决部分问题，至少说明升级版本在哪方面更好一些。对于PASCAL VOC，IOU阈值“故意设置的很低，以对真值数据中的边界框不准确负责”[2]。COCO的标注比VOC好吗？这绝对可能，因为COCO有分割掩模，可能其标注更值得信赖，所以我们不那么担心不准确。但是，我的问题是缺少正当的理由。

The COCO metric emphasizes better bounding boxes but that emphasis must mean it de-emphasizes something else, in this case classification accuracy. Is there a good reason to think that more precise bounding boxes are more important than better classification? A miss-classified example is much more obvious than a bounding box that is slightly shifted.

COCO度量标准强调更好的边界框，但这种强调一定意味着忽略了其他什么，在这种情况下就是分类准确度。有什么好的理由认为，更精确的边界框比更好的分类更重要吗？错误分类的样本比略微错位的边界框更加明显。

mAP is already screwed up because all that matters is per-class rank ordering. For example, if your test set only has these two images then according to mAP two detectors that produce these results are JUST AS GOOD:

mAP已经不太好用了，因为重要的是每类排序。比如，如果你的测试集只有2幅图像，那么根据mAP两个检测器的这样的结果是一样好的：

Figure 5. These two hypothetical detectors are perfect according to mAP over these two images. They are both perfect. Totally equal.

Now this is OBVIOUSLY an over-exaggeration of the problems with mAP but I guess my newly retconned point is that there are such obvious discrepancies between what people in the “real world” would care about and our current metrics that I think if we’re going to come up with new metrics we should focus on these discrepancies. Also, like, it’s already mean average precision, what do we even call the COCO metric, average mean average precision?

现在这是很明显的夸大mAP问题的例子，但是我猜blah blah blah。

Here’s a proposal, what people actually care about is given an image and a detector, how well will the detector find and classify objects in the image. What about getting rid of the per-class AP and just doing a global average precision? Or doing an AP calculation per-image and averaging over that?

所以下面是一个提议，人们真正在意的是，给定一幅图像和一个检测器，检测器在图像中找到和分类目标的好坏程度。那么不要计算每类的AP，只计算全局的平均精度(AP)如何？或者每幅图像计算一个AP，然后所有图像做出平均？

Boxes are stupid anyway though, I’m probably a true believer in masks except I can’t get YOLO to learn them.

边界框已经很愚蠢了，我是一个掩模的忠实信奉者，但是现在还没法让YOLO去学习掩模。