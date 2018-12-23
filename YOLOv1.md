# You Only Look Once: Unified, Real-time Object Detection

Joseph Redmon et al. University of Washington, Allen Institute for AI, Facebook AI Research

## Abstract 摘要

We present YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.

我们提出了一种目标检测的新方法，YOLO。目标检测的前人工作将分类器的目的重新定义为执行检测。而我们用一种回归问题的框架来进行目标检测，在空间上将边界框分离并与相应的类别概率相关联。单独一个神经网络就可以直接从整幅图像一步预测边界框和类别概率。由于整个检测过程是单独一个网络，所以可以直接对检测性能进行端到端的优化。

Our unified architecture is extremely fast. Our base YOLO model processes images in real-time at 45 frames per second. A smaller version of the network, Fast YOLO, processes an astounding 155 frames per second while still achieving double the mAP of other real-time detectors. Compared to state-of-the-art detection systems, YOLO makes more localization errors but is less likely to predict false positives on background. Finally, YOLO learns very general representations of objects. It outperforms other detection methods, including DPM and R-CNN, when generalizing from natural images to other domains like artwork.

我们的统一框架是非常快的。我们的基础YOLO模型以45FPS的速度实时处理图像。网络更小的一个版本，快速YOLO，可以以155FPS的速度处理，而且得到的mAP是其他实时检测器的2倍。与目前最好的检测系统相比，YOLO的定位错误更多一些，但虚警预测更少。最后，YOLO学习到了目标非常一般的表示。YOLO超过了其他检测模型，包括DPM和R-CNN，可以很好的从自然图像泛化到其他领域。

## 1. Introduction 简介

Humans glance at an image and instantly know what objects are in the image, where they are, and how they interact. The human visual system is fast and accurate, allowing us to perform complex tasks like driving with little conscious thought. Fast, accurate algorithms for object detection would allow computers to drive cars without specialized sensors, enable assistive devices to convey real-time scene information to human users, and unlock the potential for general purpose, responsive robotic systems.

人类看一眼图像，立刻就可以知道图像中有什么目标，在哪里，目标怎样互动。人类视觉系统快速准确，使我们可以执行复杂的任务，如很少有意识的思考就可以进行驾驶。快速准确的目标检测算法可以使计算机驾驶汽车，而无需专门的传感器，还可以使辅助设备将实时场景信息传递给人类用户，从而使一般性目的的可以响应的机器人系统成为可能。

Current detection systems repurpose classifiers to perform detection. To detect an object, these systems take a classifier for that object and evaluate it at various locations and scales in a test image. Systems like deformable parts models (DPM) use a sliding window approach where the classifier is run at evenly spaced locations over the entire image [10].

现有的检测系统都是将分类器的目标转化为检测。为检测一个目标，这些系统用分类器处理目标，在测试图像的各种位置和尺度下进行目标评估。像DPM这样的模型使用滑窗方法，在整个图像所有均匀分布的位置上使用分类器进行处理[10]。

More recent approaches like R-CNN use region proposal methods to first generate potential bounding boxes in an image and then run a classifier on these proposed boxes. After classification, post-processing is used to refine the bounding boxes, eliminate duplicate detections, and rescore the boxes based on other objects in the scene [13]. These complex pipelines are slow and hard to optimize because each individual component must be trained separately.

最近的一些方法如R-CNN使用了区域候选方法作为第一步，在图像中生成潜在的边界框，然后在这些候选框中进行分类。分类后，进行提炼边界框的后处理，去除重复的检测结果，在场景中基于其他目标对框进行重新评分[13]。这些复杂的过程速度很慢，很难优化，这是因为每个部件都需要进行分开进行训练。

We reframe object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities. Using our system, you only look once (YOLO) at an image to predict what objects are present and where they are.

我们重新将目标检测构建成一个单独的回归问题的框架，从图像像素直接得到边界框的坐标和类别概率。使用我们的系统，只需要观察一次图像就可以知道有什么目标，其位置在哪里。

YOLO is refreshingly simple: see Figure 1. A single convolutional network simultaneously predicts multiple bounding boxes and class probabilities for those boxes. YOLO trains on full images and directly optimizes detection performance. This unified model has several benefits over traditional methods of object detection.

YOLO结构非常简单，见图1所示。单独一个卷积网络同时预测多个边界框和这些框的类别概率。YOLO在整图上进行训练，直接优化检测性能。这个统一的模型与传统目标检测方法相比，有以下几个好处。

Figure 1: The YOLO Detection System. Processing images with YOLO is simple and straightforward. Our system (1) resizes the input image to 448 × 448, (2) runs a single convolutional network on the image, and (3) thresholds the resulting detections by the model’s confidence.

图1：YOLO检测系统。用YOLO处理图像简单直接，我们的系统(1)将图像大小变换为448×448，(2)在图像上运行一个卷积网络，(3)根据模型的置信度对得到的检测结果进行阈值处理。

First, YOLO is extremely fast. Since we frame detection as a regression problem we don’t need a complex pipeline. We simply run our neural network on a new image at test time to predict detections. Our base network runs at 45 frames per second with no batch processing on a Titan X GPU and a fast version runs at more than 150 fps. This means we can process streaming video in real-time with less than 25 milliseconds of latency. Furthermore, YOLO achieves more than twice the mean average precision of other real-time systems. For a demo of our system running in real-time on a webcam please see our project webpage: http://pjreddie.com/yolo/.

第一，YOLO速度非常快。我们使用回归问题的框架进行检测，所以不需要复杂的过程。测试时，我们只需在新图像上运行我们的神经网络，来预测检测结果。我们的基础网络在Titan X GPU上没有批处理的情况下可以以45FPS的速度运行，快速版本的网络可以达到150FPS。这意为着我们可以实时处理流媒体视频，延迟少于25ms。进一步，YOLO的mAP是其他实时系统的2倍以上。以下项目网页是我们的系统在网络摄像头上实时运行的一个例子：http://pjreddie.com/yolo/。

Second, YOLO reasons globally about the image when making predictions. Unlike sliding window and region proposal-based techniques, YOLO sees the entire image during training and test time so it implicitly encodes contextual information about classes as well as their appearance. Fast R-CNN, a top detection method [14], mistakes background patches in an image for objects because it can’t see the larger context. YOLO makes less than half the number of background errors compared to Fast R-CNN.

第二，YOLO对图像进行全局推理来进行预测。与滑窗法和基于候选区域的方法不同，YOLO在训练和测试时，处理的是整个图像，所以其隐含的将目标类别的外形和上下文信息都编码进去了。Fast R-CNN是现在最好的检测方法[14]，它会将背景块误以为是目标，因为它不会看到更大的上下文信息。YOLO与Fast R-CNN相比，背景错误率可以减少一半。

Third, YOLO learns generalizable representations of objects. When trained on natural images and tested on artwork, YOLO outperforms top detection methods like DPM and R-CNN by a wide margin. Since YOLO is highly generalizable it is less likely to break down when applied to new domains or unexpected inputs.

第三，YOLO学习的是目标可以泛化的表示。如果用自然图像训练，而对艺术品进行测试，YOLO的表现远远超过DPM和R-CNN。因为YOLO是高度可泛化的，当应用于新的领域或意外的输入时，失败的可能性更小一些。

YOLO still lags behind state-of-the-art detection systems in accuracy. While it can quickly identify objects in images it struggles to precisely localize some objects, especially small ones. We examine these tradeoffs further in our experiments.

YOLO在准确度上与最好的检测系统相比仍然有差距。虽然可以迅速的识别出图像中的目标，但在准确的定位一些目标上还有困难，尤其是小目标。我们在实验中检查这些折中情况。

All of our training and testing code is open source. A variety of pretrained models are also available to download.

我们所有的训练和测试代码都是开源的。很多预训练模型都可以下载。

## 2. Unified Detection 统一的检测

We unify the separate components of object detection into a single neural network. Our network uses features from the entire image to predict each bounding box. It also predicts all bounding boxes across all classes for an image simultaneously. This means our network reasons globally about the full image and all the objects in the image. The YOLO design enables end-to-end training and real-time speeds while maintaining high average precision.

我们将目标检测的分离组件统一成单一的神经网络。我们的网络使用整个图像的特征来预测每个边界框，也是对一幅图像同时预测所有类别的所有边界框。这意味着我们的网络对整个图像和图像中的所有目标进行全局推理。YOLO的设计使端到端的训练和实时性的速度成为可能，同时保持高的检测average precision。

Our system divides the input image into an S × S grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.

我们的系统将输入图像分解成S×S的格子。如果一个目标的中央在一个格子单元中，那么这个格子单元就负责检测那个目标。

Each grid cell predicts B bounding boxes and confidence scores for those boxes. These confidence scores reflect how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts. Formally we define confidence as $Pr(Object) ∗ IOU^{truth}_{pred}$. If no object exists in that cell, the confidence scores should be zero. Otherwise we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth.

每个格子单元预测B个边界框和这些框的置信度分数。这些置信度分数反映了模型认为这个框里含有目标的信心的强弱，还有模型认为预测的框的精确度。我们定义这个置信度为$Pr(Object) ∗ IOU^{truth}_{pred}$。如果单元中没有目标，置信度分数为0；如果有目标，我们希望置信度分数为预测框和真值框的交并比(IOU)。

Each bounding box consists of 5 predictions: x, y, w, h, and confidence. The (x, y) coordinates represent the center of the box relative to the bounds of the grid cell. The width and height are predicted relative to the whole image. Finally the confidence prediction represents the IOU between the predicted box and any ground truth box.

每个边界框包括5个预测值：x, y, w, h和置信度。(x,y)坐标代表框的中央相对于格子单元的边界的位置。宽和高是相对于整个图像的预测值。最后置信度预测代表预测框和任意真值框的交并比。

Each grid cell also predicts C conditional class probabilities, $Pr(Class_i |Object)$. These probabilities are conditioned on the grid cell containing an object. We only predict one set of class probabilities per grid cell, regardless of the number of boxes B.

每个格子单元还预测C个类别概率，$Pr(Class_i |Object)$。这些概率是格子单元包含一个目标的概率。我们只预测每个格子单元包含类概率的集合，而不论框的数量B是多少。

At test time we multiply the conditional class probabilities and the individual box confidence predictions, 在测试时，我们将条件类别概率与单个框的置信度预测相乘

$$Pr(Class_i |Object) ∗ Pr(Object) ∗ IOU_{truth}^{pred} = Pr(Class_i ) ∗ IOU_{truth}^{pred}$$(1)

which gives us class-specific confidence scores for each box. These scores encode both the probability of that class appearing in the box and how well the predicted box fits the object.

这给了我们每个框分类别的置信度分数。这些分数既包含某类别出现在框中的概率，也包括预测的这个框与目标的重合程度。

For evaluating YOLO on PASCAL VOC, we use S = 7, B = 2. PASCAL VOC has 20 labelled classes so C = 20. Our final prediction is a 7 × 7 × 30 tensor.

在PASCAL VOC上评估YOLO的话，我们使用S=7，B=2，PASCAL VOC包含20类标签，所以C=20。我们的最终预测是一个7×7×30的张量。

Figure 2: The Model. Our system models detection as a regression problem. It divides the image into an S × S grid and for each grid cell predicts B bounding boxes, confidence for those boxes, and C class probabilities. These predictions are encoded as an S × S × (B ∗ 5 + C) tensor.

图2：模型。我们的系统将检测作为一个回归问题进行建模。系统将图像分割成S×S的格子，对每个格子单元预测出B个边界框，这些框的置信度，以及C个类别概率。这些预测编码成为S×S×(B*5+C)的张量。

### 2.1. Network Design 网络设计

We implement this model as a convolutional neural network and evaluate it on the PASCAL VOC detection dataset [9]. The initial convolutional layers of the network extract features from the image while the fully connected layers predict the output probabilities and coordinates.

我们用卷积神经网络实现这个模型，并在PASCAL VOC检测数据集上进行评估[9]。网络开始的卷积层提取图像特征，全连接层预测输出概率和坐标。

Our network architecture is inspired by the GoogLeNet model for image classification [34]. Our network has 24 convolutional layers followed by 2 fully connected layers. Instead of the inception modules used by GoogLeNet, we simply use 1 × 1 reduction layers followed by 3 × 3 convolutional layers, similar to Lin et al [22]. The full network is shown in Figure 3.

我们的网络架构受到用于分类的GoogLeNet模型[34]启发。我们的网络有24个卷积层，然后是2个全连接层。我们没有采用GoogLeNet的inception模块，而是仅仅使用1×1的退化层随后跟着3×3的卷积层，与Lin等人的[22]类似。网络整体如图3所示。

Figure 3: The Architecture. Our detection network has 24 convolutional layers followed by 2 fully connected layers. Alternating 1 × 1 convolutional layers reduce the features space from preceding layers. We pretrain the convolutional layers on the ImageNet classification task at half the resolution (224 × 224 input image) and then double the resolution for detection.

图3：网络架构。我们的检测网络有24个卷积层，随后是2个全连接层。中间叠加的1×1卷积层减少了前面层的特征空间。我们在ImageNet分类任务中预训练卷积层，输入图像分辨率为一半(224×224输入图像)，在检测时分辨率加倍。

We also train a fast version of YOLO designed to push the boundaries of fast object detection. Fast YOLO uses a neural network with fewer convolutional layers (9 instead of 24) and fewer filters in those layers. Other than the size of the network, all training and testing parameters are the same between YOLO and Fast YOLO.

我们还训练了一个快速版YOLO，达到快速目标检测的极限。快速YOLO模型使用较少的卷积层（9层而不是24层），在这些层中使用更少的滤波器。除了网络的规模，YOLO和快速YOLO的所有训练和测试参数都是一样的。

The final output of our network is the 7 × 7 × 30 tensor of predictions. 网络的最终输出是7×7×30的预测张量。

### 2.2. Training 训练

We pretrain our convolutional layers on the ImageNet 1000-class competition dataset [30]. For pretraining we use the first 20 convolutional layers from Figure 3 followed by a average-pooling layer and a fully connected layer. We train this network for approximately a week and achieve a single crop top-5 accuracy of 88% on the ImageNet 2012 validation set, comparable to the GoogLeNet models in Caffe’s Model Zoo [24]. We use the Darknet framework for all training and inference [26].

我们在1000类的ImageNet数据集[30]上预训练卷积层。对于预训练，我们使用图3中前20层卷积层，随后是平均池化层和一个全连接层。我们训练这个网络大约花了一个星期，在ImageNet2012验证集上取得了单剪切块top-5准确率88%的结果，与Caffe的模型库中的GoogLeNet可以类比[24]。我们在所有的训练和推理中使用Darknet框架[26]。

We then convert the model to perform detection. Ren et al. show that adding both convolutional and connected layers to pretrained networks can improve performance [29]. Following their example, we add four convolutional layers and two fully connected layers with randomly initialized weights. Detection often requires fine-grained visual information so we increase the input resolution of the network from 224 × 224 to 448 × 448.

我们然后将模型进行转化，来进行检测。Ren等人证明，为预训练网络增加卷积层和全连接层可以改进表现[29]。遵循他们的例子，我们增加了4个卷积层和2个全连接层，并将权值进行随机初始化。检测经常需要细粒度视觉信息，所以我们将网络输入分辨率从224×224增加到448×448。

Our final layer predicts both class probabilities and bounding box coordinates. We normalize the bounding box width and height by the image width and height so that they fall between 0 and 1. We parametrize the bounding box x and y coordinates to be offsets of a particular grid cell location so they are also bounded between 0 and 1.

我们最后的层对类别概率和边界框坐标同时进行预测。我们用图像宽度和高度对边界框的宽度和高度进行归一化，所以其结果在0,1之间。我们将边界框的x,y坐标作为一个特定格子单元位置的便宜，所以其值最后也落在0和1之间。

We use a linear activation function for the final layer and all other layers use the following leaky rectified linear activation:我们在最后的层中使用线性激活函数，其他层中使用下面的leaky ReLU激活函数：

$$φ(x)=x, if x > 0; 0.1x, otherwise$$(2)

We optimize for sum-squared error in the output of our model. We use sum-squared error because it is easy to optimize, however it does not perfectly align with our goal of maximizing average precision. It weights localization error equally with classification error which may not be ideal. Also, in every image many grid cells do not contain any object. This pushes the “confidence” scores of those cells towards zero, often overpowering the gradient from cells that do contain objects. This can lead to model instability, causing training to diverge early on.

我们优化模型输出的平方和误差。我们使用平方和误差因为容易优化，但是它与我们最大化平均准确率的目标并不完全一致。定位误差与分类误差的权重一样，这可能不是最理想的情况。同样的，在每幅图像中，很多格子单元并不包含任何目标。这些格子的置信度分数趋向于0，经常会使包含目标的单元的梯度过大。这会导致模型的不稳定性，使训练很早就发散。

To remedy this, we increase the loss from bounding box coordinate predictions and decrease the loss from confidence predictions for boxes that don’t contain objects. We use two parameters, $λ_{coord}$ and $λ_{noobj}$ to accomplish this. We set $λ_{coord} = 5$ and $λ_{noobj} = 0.5$.

为修复这个问题，我们增加边界框坐标预测的误差权重，降低不包含目标的边界框的置信度预测误差的权重。我们使用两个参数$λ_{coord}$和$λ_{noobj}$来达到目标，我们设$λ_{coord} = 5$及$λ_{noobj} = 0.5$。

Sum-squared error also equally weights errors in large boxes and small boxes. Our error metric should reflect that small deviations in large boxes matter less than in small boxes. To partially address this we predict the square root of the bounding box width and height instead of the width and height directly.

平方和误差对于大边界框和小边界框的权重也是一样的。我们的误差衡量标准应当反应出，大的边界框中的小偏移没有小边界框中的小偏移重要。为部分处理这个问题，我们预测的是边界框的宽度和高度的平方根，而不是直接预测宽度和高度。

YOLO predicts multiple bounding boxes per grid cell. At training time we only want one bounding box predictor to be responsible for each object. We assign one predictor to be “responsible” for predicting an object based on which prediction has the highest current IOU with the ground truth. This leads to specialization between the bounding box predictors. Each predictor gets better at predicting certain sizes, aspect ratios, or classes of object, improving overall recall.

YOLO在每个格子单元中预测多个边界框。在训练时，我们只希望每个目标有一个边界框预测器就好了。我们依据哪个预测与ground truth的IOU最大，来确定负责预测这个目标的预测器。这会带来边界框预测器的专业化。每个预测器都会更加擅长预测特定尺寸、宽高比，或特定目标类别，提高整体召回率。

During training we optimize the following, multi-part loss function: 在训练过程中，我们优化如下的多部分损失函数：

$$$$(3)

where $1^{obj}_i$ denotes if object appears in cell i and $1^{obj}_{ij}$ denotes that the *j*th bounding box predictor in cell i is “responsible” for that prediction.

其中$1^{obj}_i$代表目标在格子i中出现，$1^{obj}_{ij}$代表格子i中的第j个边界框预测器负责预测。

Note that the loss function only penalizes classification error if an object is present in that grid cell (hence the conditional class probability discussed earlier). It also only penalizes bounding box coordinate error if that predictor is “responsible” for the ground truth box (i.e. has the highest IOU of any predictor in that grid cell).

注意损失函数只对格子单元中包含目标的情况（即后面讨论的条件类别概率）惩罚分类错误。它也只对负责ground truth边界框的预测器（即那个格子单元中IOU最高的预测器）的坐标误差进行惩罚。

We train the network for about 135 epochs on the training and validation data sets from PASCAL VOC 2007 and 2012. When testing on 2012 we also include the VOC 2007 test data for training. Throughout training we use a batch size of 64, a momentum of 0.9 and a decay of 0.0005.

我们在PASCAL VOC 2007和2012的训练和验证集上对网络进行了135轮训练。在2012上进行测试的时候，我们也将VOC 2007的测试数据用于训练。整个训练过程中，我们使用的batch size为64，动量0.9，权重衰减为0.0005。

Our learning rate schedule is as follows: For the first epochs we slowly raise the learning rate from $10_{−3}$ to $10_{−2}$. If we start at a high learning rate our model often diverges due to unstable gradients. We continue training with $10_{−2}$ for 75 epochs, then $10_{−3}$ for 30 epochs, and finally $10_{−4}$ for 30 epochs.

我们的学习速率方案是这样的：对于首轮训练，我们缓慢的将学习速率从$10_{−3}$增加到$10_{−2}$。如果我们从高学习速率开始，那么模型会由于不稳定的梯度而发散。我们继续用$10_{−2}$训练了75轮，然后用$10_{−3}$训练了30轮，最后用$10_{−4}$训练了30轮。

To avoid overfitting we use dropout and extensive data augmentation. A dropout layer with rate = .5 after the first connected layer prevents co-adaptation between layers [18]. For data augmentation we introduce random scaling and translations of up to 20% of the original image size. We also randomly adjust the exposure and saturation of the image by up to a factor of 1.5 in the HSV color space.

为避免过拟合，我们使用了dropout和广泛的数据增广。第一个全连接层后为dropout层，参数为rate=0.5，这防止了层间的互相适应[18]。对于数据增广，我们引入了随机的尺度变换和平移，幅度达到了原图像尺寸的20%。我们还随机调整了图像的曝光度和饱和度，幅度最高为HSV色彩空间的1.5倍。

### 2.3. Inference 推理

Just like in training, predicting detections for a test image only requires one network evaluation. On PASCAL VOC the network predicts 98 bounding boxes per image and class probabilities for each box. YOLO is extremely fast at test time since it only requires a single network evaluation, unlike classifier-based methods.

就像在训练中一样，为一幅测试图像预测检测只需要一次网络计算。在PASCAL VOC数据集上，网络在每幅图像中平均预测98个边界框，每个框都包含类别概率。YOLO在测试时速度极快，因为只需要单次网络计算，这和基于分类的方法是不同的。

The grid design enforces spatial diversity in the bounding box predictions. Often it is clear which grid cell an object falls in to and the network only predicts one box for each object. However, some large objects or objects near the border of multiple cells can be well localized by multiple cells. Non-maximal suppression can be used to fix these multiple detections. While not critical to performance as it is for R-CNN or DPM, non-maximal suppression adds 2-3% in mAP.

格子的设计使得边界框预测在空域中存在多样性。一个目标落入哪个格子单元经常是明确的，网络对每个目标只预测一个框。但是，一些较大的目标，或在多个单元的边缘的目标，可能会由多个单元很好的定位。非最大抑制可以用于解决这些多重检测的问题。非最大抑制对于R-CNN或DPM不是很重要，但在YOLO中，可以增加2-3%的mAP。

2.4. Limitations of YOLO YOLO的限制

YOLO imposes strong spatial constraints on bounding box predictions since each grid cell only predicts two boxes and can only have one class. This spatial constraint limits the number of nearby objects that our model can predict. Our model struggles with small objects that appear in groups, such as flocks of birds.

YOLO在边界框预测上加上了很强的空域限制，因为每个格子单元只预测两个边界框，且只能有一个类别。这个空域限制约束了模型可以预测的附近目标的数量。我们的模型在小目标成群出现时预测结果不太好，比如鸟群的情况。

Since our model learns to predict bounding boxes from data, it struggles to generalize to objects in new or unusual aspect ratios or configurations. Our model also uses relatively coarse features for predicting bounding boxes since our architecture has multiple downsampling layers from the input image.

因为我们的模型从数据中学习预测边界框，在出现新的或异常的目标纵横比，或目标配置时，这种情况的泛化能力不是太好。我们的模型也使用相对粗的特征来预测边界框，因为我们的模型有多个下采样层处理输入图像。

Finally, while we train on a loss function that approximates detection performance, our loss function treats errors the same in small bounding boxes versus large bounding boxes. A small error in a large box is generally benign but a small error in a small box has a much greater effect on IOU. Our main source of error is incorrect localizations.

最后，当我们训练采用的损失函数接近于检测性能时，损失函数对待小的边界框和大的边界框是一样的。大的边界框中的小误差一般是无关紧要的，但小边界框中的小误差对IOU的影响会更大。我们的主要错误来源是不正确的定位。

## 3. Comparison to Other Detection Systems 与其他检测系统的对比

Object detection is a core problem in computer vision. Detection pipelines generally start by extracting a set of robust features from input images (Haar [25], SIFT [23], HOG [4], convolutional features [6]). Then, classifiers [36, 21, 13, 10] or localizers [1, 32] are used to identify objects in the feature space. These classifiers or localizers are run either in sliding window fashion over the whole image or on some subset of regions in the image [35, 15, 39]. We compare the YOLO detection system to several top detection frameworks, highlighting key similarities and differences.

目标检测是计算机视觉中的核心问题。检测的过程一般从提取输入图像的稳健特征(Haar[25], SIFT[23], HOG[4], 卷积特征[6])开始。然后，分类器[36, 21, 13, 10]或定位器[1, 32]在特征空间中用于识别目标。这些分类器或定位器或者在整个图像上以滑窗的方式运行，或者在图像的区域子集上运行[35, 15, 39]。我们将YOLO检测系统与几个顶级检测框架进行比较，分析出关键的相似点和不同点。

**Deformable parts models**. Deformable parts models (DPM) use a sliding window approach to object detection [10]. DPM uses a disjoint pipeline to extract static features, classify regions, predict bounding boxes for high scoring regions, etc. Our system replaces all of these disparate parts with a single convolutional neural network. The network performs feature extraction, bounding box prediction, non-maximal suppression, and contextual reasoning all concurrently. Instead of static features, the network trains the features in-line and optimizes them for the detection task. Our unified architecture leads to a faster, more accurate model than DPM.

**可变部件模型**。DPM使用滑窗方法进行目标检测[10]。DPM使用分离的过程来提取静态特征、对区域分类、预测高分值的区域的边界框，等等。我们的系统将所有这些不相关的部分替换为一个卷积神经网络。这个网络同时执行特征提取、边界框预测、非最大抑制和上下文推理所有这些任务。网络提取的不是静态特征，而是在线提取特征，并为检测任务进行优化。我们的统一框架比DPM模型更加快速、更加准确。

**R-CNN**. R-CNN and its variants use region proposals instead of sliding windows to find objects in images. Selective Search [35] generates potential bounding boxes, a convolutional network extracts features, an SVM scores the boxes, a linear model adjusts the bounding boxes, and non-max suppression eliminates duplicate detections. Each stage of this complex pipeline must be precisely tuned independently and the resulting system is very slow, taking more than 40 seconds per image at test time [14].

**R-CNN**。R-CNN及其变体使用区域候选而不是滑窗法在图像中寻找目标。Selective Search[35]生成潜在的边界框，卷积网络提取特征，SVM对边界框进行评分，线性模型调整边界框，非最大抑制去除重复的检测。这个复杂过程的每个阶段都必须独立的精确调整，得到的系统速度很慢，在测试时40多秒才能处理一幅图像。

YOLO shares some similarities with R-CNN. Each grid cell proposes potential bounding boxes and scores those boxes using convolutional features. However, our system puts spatial constraints on the grid cell proposals which helps mitigate multiple detections of the same object. Our system also proposes far fewer bounding boxes, only 98 per image compared to about 2000 from Selective Search. Finally, our system combines these individual components into a single, jointly optimized model.

YOLO与R-CNN有一些类似点。每个格子单元提出可能的边界框，使用卷积网络对这些框进行评分。但是，我们的系统在格子单元的推荐中加上了限制，这减缓了同一目标的多重检测问题。我们的系统得出的候选边界框少的多，每幅图像只有98个，而Selective Search有约2000个。最后，我们的系统将所有这些单独的组建结合成了一个共同优化的模型。

**Other Fast Detectors**. Fast and Faster R-CNN focus on speeding up the R-CNN framework by sharing computation and using neural networks to propose regions instead of Selective Search [14] [28]. While they offer speed and accuracy improvements over R-CNN, both still fall short of real-time performance.

**其他快速检测器**。Fast R-CNN和Faster R-CNN聚焦在加速R-CNN模型上，方法是共享计算和使用神经网络来做区域候选，而不是Selective Search[14,28]。它们改进了R-CNN的速度和准确度，但仍然远未达到实时的效果。

Many research efforts focus on speeding up the DPM pipeline [31] [38] [5]. They speed up HOG computation, use cascades, and push computation to GPUs. However, only 30Hz DPM [31] actually runs in real-time.

很多研究努力聚焦在加速DPM识别过程[31,38,5]。它们加速HOG计算，使用级联，使用GPU计算。但是，只有30Hz DPM[31]真正可以实时运行。

Instead of trying to optimize individual components of a large detection pipeline, YOLO throws out the pipeline entirely and is fast by design.

YOLO没有去优化大型检测过程的单个组件，而是抛弃了整个过程，设计了快速的系统。

Detectors for single classes like faces or people can be highly optimized since they have to deal with much less variation [37]. YOLO is a general purpose detector that learns to detect a variety of objects simultaneously.

单个类别的检测器，如人脸或人，可以高度优化，因为需要处理的变化很少[37]。YOLO是一种通用意义上的检测器，可以同时学习检测大量不同目标。

**Deep MultiBox**. Unlike R-CNN, Szegedy et al. train a convolutional neural network to predict regions of interest [8] instead of using Selective Search. MultiBox can also perform single object detection by replacing the confidence prediction with a single class prediction. However, Multi-Box cannot perform general object detection and is still just a piece in a larger detection pipeline, requiring further image patch classification. Both YOLO and MultiBox use a convolutional network to predict bounding boxes in an image but YOLO is a complete detection system.

**Deep MultiBox**。与R-CNN不同，Szegedy等人训练了一个卷积神经网络来预测roi[8]，所以没有使用Selective Search。MultiBox也可以将置信度预测替换为单类预测，从而进行单目标预测。但MultiBox不能进行通用目标检测，而且仍然只是更大型检测过程的一部分，需要进一步的图像块分类。YOLO和MultiBox都使用了一种卷积网络来预测图像中的边界框，但YOLO是一个完整的检测系统。

**OverFeat**. Sermanet et al. train a convolutional neural network to perform localization and adapt that localizer to perform detection [32]. OverFeat efficiently performs sliding window detection but it is still a disjoint system. OverFeat optimizes for localization, not detection performance. Like DPM, the localizer only sees local information when making a prediction. OverFeat cannot reason about global context and thus requires significant post-processing to produce coherent detections.

**OverFeat**。Sermanet等人训练了一种卷积神经网络来进行定位，并使定位器能够执行检测任务[32]。OverFeat有效的执行了滑窗检测，但仍然是一个分离系统。OverFeat对定位进行优化，但没有对检测性能进行优化。与DPM类似，在预测时，定位器只能看到局部信息。OverFeat不能对全局上下文进行推理，所以需要很多后续处理来生成连续的检测。

**MultiGrasp**. Our work is similar in design to work on grasp detection by Redmon et al [27]. Our grid approach to bounding box prediction is based on the MultiGrasp system for regression to grasps. However, grasp detection is a much simpler task than object detection. MultiGrasp only needs to predict a single graspable region for an image containing one object. It doesn’t have to estimate the size, location, or boundaries of the object or predict it’s class, only find a region suitable for grasping. YOLO predicts both bounding boxes and class probabilities for multiple objects of multiple classes in an image.

**MultiGrasp**。我们的工作在设计上与Redmon等人的grasp检测[27]类似。我们预测边界框的网格方法就是基于MultiGrasp系统的回归grasp的。但是，与目标检测相比，grasp检测是一个非常简单的任务。MultiGrasp只需要对包含一个目标的图像预测单个graspable区域。不需要估计目标的大小、位置或边缘，或预测其类别，只要找到适合grasp的区域。YOLO在一幅图像中，同时预测多个目标多个类别的边界框和类别概率。

## 4. Experiments 实验

First we compare YOLO with other real-time detection systems on PASCAL VOC 2007. To understand the differences between YOLO and R-CNN variants we explore the errors on VOC 2007 made by YOLO and Fast R-CNN, one of the highest performing versions of R-CNN [14]. Based on the different error profiles we show that YOLO can be used to rescore Fast R-CNN detections and reduce the errors from background false positives, giving a significant performance boost. We also present VOC 2012 results and compare mAP to current state-of-the-art methods. Finally, we show that YOLO generalizes to new domains better than other detectors on two artwork datasets.

首先我们将YOLO与其他在PASCAL VOC 2007上的实时检测系统进行比较。为理解YOLO和R-CNN变体的差别，我们研究了YOLO和Fast R-CNN在VOC 2007上的误差，而这是R-CNN[14]表现最好的一个版本。基于不同的误差表现，我们证明YOLO可以用于rescore Fast R-CNN的检测，降低背景虚警错误，得到明显的性能提升。我们还给出了在VOC 2012上的结果，与现有的最好的方法进行了mAP比较。最后，我们在两个艺术品数据集上证明YOLO的泛化能力强于其他检测器。

### 4.1. Comparison to Other Real-Time Systems 与其他实时系统的对比

Many research efforts in object detection focus on making standard detection pipelines fast[5] [38] [31] [14] [17] [28]. However, only Sadeghi et al. actually produce a detection system that runs in real-time (30 frames per second or better) [31]. We compare YOLO to their GPU implementation of DPM which runs either at 30Hz or 100Hz. While the other efforts don’t reach the real-time milestone we also compare their relative mAP and speed to examine the accuracy-performance tradeoffs available in object detection systems.

很多目标检测的研究努力聚焦在使标准的检测过程更快[5,38,31,14,17,28]。但是，只有Sadeghi等人真正得到了能实时运行的检测系统（30FPS甚至更好）[31]。我们将YOLO与其DPM的GPU实现进行了比较，其运行在30Hz或100Hz上。虽然其他方法没有达到实时的标准，我们仍然比较了其mAP和速度，来研究目标检测系统的准确度-性能折中。

Fast YOLO is the fastest object detection method on PASCAL; as far as we know, it is the fastest extant object detector. With 52.7% mAP, it is more than twice as accurate as prior work on real-time detection. YOLO pushes mAP to 63.4% while still maintaining real-time performance.

快速YOLO是PASCAL上最快的目标检测方法；据我们所知，这是现存最快的目标检测器。其mAP为52.7%，比之前的实时检测结果准确度高2倍。YOLO在实时检测的同时，将mAP提升到了63.4%。

We also train YOLO using VGG-16. This model is more accurate but also significantly slower than YOLO. It is useful for comparison to other detection systems that rely on VGG-16 but since it is slower than real-time the rest of the paper focuses on our faster models.

我们还用VGG-16训练了YOLO。这个模型更加准确，但速度明显比YOLO要慢。这对于与其他基于VGG-16的系统进行比较是有用的，但由于速度比实时系统慢很多，所以本文剩下部分主要集中在我们更快速的模型。

Fastest DPM effectively speeds up DPM without sacrificing much mAP but it still misses real-time performance by a factor of 2 [38]. It also is limited by DPM’s relatively low accuracy on detection compared to neural network approaches.

最快的DPM模型没有损失mAP，而且有效的加速了DPM模型，但距离实时的检测还只有一半的性能[38]。而且模型还受限于DPM相对低的准确度，没有达到神经网络方法的mAP。

R-CNN minus R replaces Selective Search with static bounding box proposals [20]. While it is much faster than R-CNN, it still falls short of real-time and takes a significant accuracy hit from not having good proposals.

R-CNN minus R将selective search替换为静态边界框候选[20]。虽然比R-CNN快很多，但仍然没有达到实时的速度，而且由于没有很好的候选，准确率下降了很多。

Fast R-CNN speeds up the classification stage of R-CNN but it still relies on selective search which can take around 2 seconds per image to generate bounding box proposals. Thus it has high mAP but at 0.5 fps it is still far from realtime.

Fast R-CNN加速了R-CNN的分类阶段，但仍然依赖于selective search方法，每幅图像在这一步需要2秒来生成边界框候选。所以可以提升mAP，但0.5fps仍然离实时很远。

The recent Faster R-CNN replaces selective search with a neural network to propose bounding boxes, similar to Szegedy et al. [8] In our tests, their most accurate model achieves 7 fps while a smaller, less accurate one runs at 18 fps. The VGG-16 version of Faster R-CNN is 10 mAP higher but is also 6 times slower than YOLO. The Zeiler-Fergus Faster R-CNN is only 2.5 times slower than YOLO but is also less accurate.

最近的Faster R-CNN将selective search替换为神经网络方法来生成边界框候选，这与Szegedy等人的[8]类似。在我们的测试中，他们最高的准确度模型可以达到7fps，而更小一些，准确度略低一些的可以达到18fps。VGG-16版本的Faster R-CNN在准确度上高了10mAP，但比YOLO慢了6倍。Zeiler-Fergus Faster R-CNN只比YOLO慢了2.5倍，但准确度更低。

Table 1: Real-Time Systems on P ASCAL VOC 2007. Comparing the performance and speed of fast detectors. Fast YOLO is the fastest detector on record for P ASCAL VOC detection and is still twice as accurate as any other real-time detector. YOLO is 10 mAP more accurate than the fast version while still well above real-time in speed.

### 4.2. VOC 2007 Error Analysis 在VOC2007上的错误分析

To further examine the differences between YOLO and state-of-the-art detectors, we look at a detailed breakdown of results on VOC 2007. We compare YOLO to Fast R-CNN since Fast R-CNN is one of the highest performing detectors on PASCAL and it’s detections are publicly available.

为进一步检验YOLO和现在最好的检测器的区别，我们查看在VOC 2007上主要的错误结果的细节。我们比较了YOLO和Fast R-CNN，因为Fast R-CNN在PASCAL上表现最好的检测器之一，其检测结果公开可用。

We use the methodology and tools of Hoiem et al. [19] For each category at test time we look at the top N predictions for that category. Each prediction is either correct or it is classified based on the type of error:

我们使用了Hoiem等[19]的方法论和工具。对于每个类别，在测试时，我们观察这个类别中得分最高的N个预测。每个预测要么是正确的，要么错误是如下类型之一：

- Correct: correct class and IOU > .5；正确：正确的类别而且IOU>0.5
- Localization: correct class, .1 < IOU < .5；定位：正确的类别，IOU在0.1与0.5之间
- Similar: class is similar, IOU > .1；相似：类别是相似的，IOU大于0.1
- Other: class is wrong, IOU > .1；其他：类别是错误的，IOU大于0.1
- Background: IOU < .1 for any object；背景：对于任何目标，IOU小于0.1。

Figure 4 shows the breakdown of each error type averaged across all 20 classes. 图4示出了20个类别中错误类别的平均情况。

Figure 4: Error Analysis: Fast R-CNN vs. YOLO These charts show the percentage of localization and background errors in the top N detections for various categories (N = # objects in that category).

YOLO struggles to localize objects correctly. Localization errors account for more of YOLO’s errors than all other sources combined. Fast R-CNN makes much fewer localization errors but far more background errors. 13.6% of it’s top detections are false positives that don’t contain any objects. Fast R-CNN is almost 3x more likely to predict background detections than YOLO.

YOLO在正确定位目标上表现较差。定位错误比其他所有错误加在一起还要多。Fast R-CNN的定位错误则少的多，但背景错误多了很多。13.6%的最高检测结果都是虚警，没有包含任何目标。Fast R-CNN与YOLO相比，将背景检测成目标的错误多了3倍。

### 4.3. Combining Fast R-CNN and YOLO 将Fast R-CNN与YOLO结合在一起

YOLO makes far fewer background mistakes than Fast R-CNN. By using YOLO to eliminate background detections from Fast R-CNN we get a significant boost in performance. For every bounding box that R-CNN predicts we check to see if YOLO predicts a similar box. If it does, we give that prediction a boost based on the probability predicted by YOLO and the overlap between the two boxes.

YOLO比Fast R-CNN的背景错误少很多。通过使用YOLO来消除Fast R-CNN中的背景检测错误，我们可以得到性能上的明显提升。对每个R-CNN预测的边界框，我们都检查看看是否YOLO也预测了一个类似的框。如果是，那么我们基于YOLO预测的概率和两个框的交叠，给这个预测一个激励。

The best Fast R-CNN model achieves a mAP of 71.8% on the VOC 2007 test set. When combined with YOLO, its mAP increases by 3.2% to 75.0%. We also tried combining the top Fast R-CNN model with several other versions of Fast R-CNN. Those ensembles produced small increases in mAP between .3 and .6%, see Table 2 for details.

最好的Fast R-CNN模型在VOC 2007测试集上取得了71.8%的mAP。与YOLO结合在一起，可以将mAP提升3.2%到达75.0%。我们还尝试了将最好的Fast R-CNN模型与几个其他版本的Fast R-CNN模型相结合。这种集成可以小幅度提升mAP，大约在0.3%到0.6%之间，详见表2。

Table 2: Model combination experiments on VOC 2007. We examine the effect of combining various models with the best version of Fast R-CNN. Other versions of Fast R-CNN provide only a small benefit while YOLO provides a significant performance boost.

The boost from YOLO is not simply a byproduct of model ensembling since there is little benefit from combining different versions of Fast R-CNN. Rather, it is precisely because YOLO makes different kinds of mistakes at test time that it is so effective at boosting Fast R-CNN’s performance.

YOLO得到的性能提升并不仅仅是模型集成的副产品，因为不同版本的Fast R-CNN的结合并没有很多性能提升。这正是因为YOLO在测试时的错误结果与Fast R-CNN是不同种类的，所以在提升Fast R-CNN性能上才会如此有效。

Unfortunately, this combination doesn’t benefit from the speed of YOLO since we run each model seperately and then combine the results. However, since YOLO is so fast it doesn’t add any significant computational time compared to Fast R-CNN.

不幸的是，这种结合在速度上没有多少优势，因为我们独立运行每个模型，然后将结果结合起来。但是，由于YOLO速度非常快，与Fast R-CNN相比，不会增加多少计算时间。

### 4.4. VOC 2012 Results

On the VOC 2012 test set, YOLO scores 57.9% mAP. This is lower than the current state of the art, closer to the original R-CNN using VGG-16, see Table 3. Our system struggles with small objects compared to its closest competitors. On categories like bottle, sheep, and tv/monitor YOLO scores 8-10% lower than R-CNN or Feature Edit. However, on other categories like cat and train YOLO achieves higher performance.

在VOC 2012测试集上，YOLO得到了57.9%的mAP。这比目前最好算法的结果略差，与原版使用VGG-16的R-CNN结果类似，见表3。我们的系统在小目标上表现略差。在瓶子、羊和电视/显示器这些类别上，YOLO比R-CNN或Feature Edit低了8到10个百分点。但是，在其他类别，如猫和火车中，YOLO得到了更好的效果。

Our combined Fast R-CNN + YOLO model is one of the highest performing detection methods. Fast R-CNN gets a 2.3% improvement from the combination with YOLO, boosting it 5 spots up on the public leaderboard.

Fast R-CNN和YOLO的结合模型是检测结果最好的模型之一。Fast R-CNN与YOLO结合后，mAP提升了2.3%，在排行榜上排名升了5位。

Table 3: P ASCAL VOC 2012 Leaderboard. YOLO compared with the full comp4 (outside data allowed) public leaderboard as of November 6th, 2015. Mean average precision and per-class average precision are shown for a variety of detection methods. YOLO is the only real-time detector. Fast R-CNN + YOLO is the forth highest scoring method, with a 2.3% boost over Fast R-CNN.

### 4.5. Generalizability: Person Detection in Artwork 泛化能力：艺术品中的人检测

Academic datasets for object detection draw the training and testing data from the same distribution. In real-world applications it is hard to predict all possible use cases and the test data can diverge from what the system has seen before [3]. We compare YOLO to other detection systems on the Picasso Dataset [12] and the People-Art Dataset [3], two datasets for testing person detection on artwork.

目标检测学术数据集的训练数据和测试数据是同分布的。在实际应用中，很难预测所有可能应用案例，测试数据可能与系统接触过的数据不一样[3]。我们将YOLO与其他检测系统在Picasso数据集[12]和People-Art数据集[3]上进行比较，这两个数据集是用于检测艺术品中的人检测。

Figure 5 shows comparative performance between YOLO and other detection methods. For reference, we give VOC 2007 detection AP on person where all models are trained only on VOC 2007 data. On Picasso models are trained on VOC 2012 while on People-Art they are trained on VOC 2010.

图5所示的是YOLO和其他检测方法的比较结果。为参考，我们给出了所有只用VOC 2007数据进行训练，在VOC 2007上对人的检测AP。在Picasso上测试的模型是在VOC 2012上训练的，在People-Art上测试的模型是在VOC 2010上训练的。

R-CNN has high AP on VOC 2007. However, R-CNN drops off considerably when applied to artwork. R-CNN uses Selective Search for bounding box proposals which is tuned for natural images. The classifier step in R-CNN only sees small regions and needs good proposals.

R-CNN在VOC 2007上AP比较高。但是，R-CNN应用于艺术品数据集上表现明显下降。R-CNN使用Selective Search来进行边界框候选，这是对自然图像调节好的。R-CNN中的分类器步骤只看到了小区域，这需要好的区域候选。

DPM maintains its AP well when applied to artwork. Prior work theorizes that DPM performs well because it has strong spatial models of the shape and layout of objects. Though DPM doesn’t degrade as much as R-CNN, it starts from a lower AP.

在应用到艺术品中时，DPM很好的保持了AP。前面的工作在理论上证明了，DPM表现不错的原因是，对于目标的形状和分布上有很强的空域模型。虽然DPM不像R-CNN一样性能下降，但是其AP本来就有些低。

YOLO has good performance on VOC 2007 and its AP degrades less than other methods when applied to artwork. Like DPM, YOLO models the size and shape of objects, as well as relationships between objects and where objects commonly appear. Artwork and natural images are very different on a pixel level but they are similar in terms of the size and shape of objects, thus YOLO can still predict good bounding boxes and detections.

YOLO在VOC 2007上表现很好，当应用于艺术品数据集中时，AP降低的也更少一些。与DPM类似，YOLO对目标的大小和形状建模，也包括与其他目标的关系，目标经常在哪里出现。艺术品和自然图像在像素层次非常不同，但是在目标大小和形状上还是类似的，所以YOLO仍然可以很好的预测边界框并检测。

## 5. Real-Time Detection In The Wild

YOLO is a fast, accurate object detector, making it ideal for computer vision applications. We connect YOLO to a webcam and verify that it maintains real-time performance, including the time to fetch images from the camera and display the detections.

YOLO是一种快速准确的目标检测器，应用于计算机视觉中非常理想。我们将YOLO应用于了网络摄像头，验证其保持了实时检测的性能，包括从摄像头取图像和显示检测内容的时间。

The resulting system is interactive and engaging. While YOLO processes images individually, when attached to a webcam it functions like a tracking system, detecting objects as they move around and change in appearance. A demo of the system and the source code can be found on our project website: http://pjreddie.com/yolo/.

虽然YOLO单独处理每一帧图像，但当接入网络摄像头时，表现的就像一个追踪系统一样，当目标到处运动并改变外观时，都可以连续检测到。系统demo和源代码都放在了我们的工程网站：http://pjreddie.com/yolo/。

## 6. Conclusion 结论

We introduce YOLO, a unified model for object detection. Our model is simple to construct and can be trained directly on full images. Unlike classifier-based approaches, YOLO is trained on a loss function that directly corresponds to detection performance and the entire model is trained jointly.

我们提出了YOLO，这是一种目标检测的统一模型。我们的模型很简洁，可以直接在整图上训练。与基于分类器的方法不同，YOLO训练的损失函数直接与检测性能相关，整个模型是共同训练的。

Fast YOLO is the fastest general-purpose object detector in the literature and YOLO pushes the state-of-the-art in real-time object detection. YOLO also generalizes well to new domains making it ideal for applications that rely on fast, robust object detection.

快速YOLO是目前文献中最快的通用目标检测器，YOLO将实时目标检测的最高性能向前推进了一大步。YOLO的泛化性能也很好，是快速稳健的目标检测应用的理想模型。

Acknowledgements: This work is partially supported by ONR N00014-13-1-0720, NSF IIS-1338054, and The Allen Distinguished Investigator Award.