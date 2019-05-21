maokuang# CornerNet: Detecting Objects as Paired Keypoints

Hei Law · Jia Deng Princeton Universtiy

## Abstract 摘要

We propose CornerNet, a new approach to object detection where we detect an object bounding box as a pair of keypoints, the top-left corner and the bottom-right corner, using a single convolution neural network. By detecting objects as paired keypoints, we eliminate the need for designing a set of anchor boxes commonly used in prior single-stage detectors. In addition to our novel formulation, we introduce corner pooling, a new type of pooling layer that helps the network better localize corners. Experiments show that CornerNet achieves a 42.2% AP on MS COCO, outperforming all existing one-stage detectors.

我们提出CornerNet，一种目标检测的新方法，我们将目标边界框检测为一对关键点，即左上角和右下角，使用的是单个卷积神经网络。通过将目标检测为成对关键点，我们不需要设计锚框，而这在单阶段检测器中经常使用。除了我们新的检测形式，我们还提出了角点池化，一种新型池化层，有助于网络更好的定位角点。试验表明，在COCO数据集上CornerNet可以取得42.2% AP，超过了之前所有的单阶段检测器。

## 1 Introduction 引言

Object detectors based on convolutional neural networks (ConvNets) (Krizhevsky et al., 2012; Simonyan and Zisserman, 2014; He et al., 2016) have achieved state-of-the-art results on various challenging benchmarks (Lin et al., 2014; Deng et al., 2009; Everingham et al., 2015). A common component of state-of-the-art approaches is anchor boxes (Ren et al., 2015; Liu et al., 2016), which are boxes of various sizes and aspect ratios that serve as detection candidates. Anchor boxes are extensively used in one-stage detectors (Liu et al., 2016; Fu et al., 2017; Redmon and Farhadi, 2016; Lin et al., 2017), which can achieve results highly competitive with two-stage detectors (Ren et al., 2015; Girshick et al., 2014; Girshick, 2015; He et al., 2017) while being more efficient. One-stage detectors place anchor boxes densely over an image and generate final box predictions by scoring anchor boxes and refining their coordinates through regression.

基于卷积神经网络目标检测器已经在各种基准检测中取得了目前最好的结果。目前最好的方法中，一种常见的部件是锚框，即各种大小、各种纵横比、用作检测候选的框。锚框在单阶段检测中使用广泛，可以取得与两阶段检测器差不多的结果，而且运行速度更快。单阶段检测器在图像中密集放置锚框，通过对锚框评分来生成最终预测框，并通过回归来对坐标值优化提炼。

But the use of anchor boxes has two drawbacks. First, we typically need a very large set of anchor boxes, e.g. more than 40k in DSSD (Fu et al., 2017) and more than 100k in RetinaNet (Lin et al., 2017). This is because the detector is trained to classify whether each anchor box sufficiently overlaps with a ground truth box, and a large number of anchor boxes is needed to ensure sufficient overlap with most ground truth boxes. As a result, only a tiny fraction of anchor boxes will overlap with ground truth; this creates a huge imbalance between positive and negative anchor boxes and slows down training (Lin et al., 2017).

但使用锚框有两个缺点。第一，一般需要非常多的锚框，如在DSSD中需要超过40k个，在RetinaNet中需要超过100k个。这是因为检测器训练的是对每个锚框是否与真值框重叠的足够多进行分类，需要大量的锚框来确保与多数真值框都有足够的重叠。结果是，只有极小一部分锚框会与真值框重叠；这样产生的正锚框和负锚框比例极为悬殊，会使训练变慢。

Second, the use of anchor boxes introduces many hyperparameters and design choices. These include how many boxes, what sizes, and what aspect ratios. Such choices have largely been made via ad-hoc heuristics, and can become even more complicated when combined with multiscale architectures where a single network makes separate predictions at multiple resolutions, with each scale using different features and its own set of anchor boxes (Liu et al., 2016; Fu et al., 2017; Lin et al., 2017).

第二，锚框的使用引入了很多超参数和设计选项。这包括，多少个锚框，大小多少，宽高比多少。这样的选择一般都是通过启发式的方法确定，当与多尺度架构结合起来时，会变的更复杂，因为单个网络要在多个分辨率上进行预测，每个尺度使用不同的特征，以及不同的锚框集。

In this paper we introduce CornerNet, a new one-stage approach to object detection that does away with anchor boxes. We detect an object as a pair of keypoints—the top-left corner and bottom-right corner of the bounding box. We use a single convolutional network to predict a heatmap for the top-left corners of all instances of the same object category, a heatmap for all bottom-right corners, and an embedding vector for each detected corner. The embeddings serve to group a pair of corners that belong to the same object—the network is trained to predict similar embeddings for them. Our approach greatly simplifies the output of the network and eliminates the need for designing anchor boxes. Our approach is inspired by the associative embedding method proposed by Newell et al. (2017), who detect and group keypoints in the context of multiperson human-pose estimation. Fig. 1 illustrates the overall pipeline of our approach.

本文中，我们提出了CornerNet，一种新的单阶段目标检测方法，不使用锚框。我们将目标检测为一对关键点，边界框的左上角和右下角。我们使用一个卷积神经网络，对相同目标类别的所有实例的左上角预测一个热力图，对右下角预测一个热力图，对每个检测到的角点预测一个嵌入向量。这个嵌入的作用是将属于同一目标的一对角点分组好，网络训练的目标就是对属于同一目标的预测类似的嵌入。我们的方法极大的简化了网络输出，消除了设计锚框的必要性。我们的方法受到相关联的嵌入方法启发(Newell et al, 2017)，他们在多人姿态估计任务中检测关键点并分类。图1所示的即我们方法的整体流程。

Fig. 1 We detect an object as a pair of bounding box corners grouped together. A convolutional network outputs a heatmap for all top-left corners, a heatmap for all bottom-right corners, and an embedding vector for each detected corner. The network is trained to predict similar embeddings for corners that belong to the same object.

Another novel component of CornerNet is corner pooling, a new type of pooling layer that helps a convolutional network better localize corners of bounding boxes. A corner of a bounding box is often outside the object—consider the case of a circle as well as the examples in Fig. 2. In such cases a corner cannot be localized based on local evidence. Instead, to determine whether there is a top-left corner at a pixel location, we need to look horizontally towards the right for the topmost boundary of the object, and look vertically towards the bottom for the leftmost boundary. This motivates our corner pooling layer: it takes in two feature maps; at each pixel location it max-pools all feature vectors to the right from the first feature map, max-pools all feature vectors directly below from the second feature map, and then adds the two pooled results together. An example is shown in Fig. 3.

CornerNet另一个新的部件是corner pooling，一种新的池化层，可以帮助卷积网络更好的定位边界框的角点。边界框的角点通常在目标之外，比如圆形目标的情况，以及图2中的目标。在这种情况下，一个角点不能基于局部证据来定位。为确定在一个像素位置是否是一个目标的左上角，我们需要从水平方向向右边看向目标的上边缘，垂直方向向下看向最左边的边缘。这启发了我们的角点池化层：它以两个特征图为输入；在每个像素点上，对所有特征向量从第一个特征图向右进行max-pool，然后从第二个特征图对所有特征向量直接向下max-pool，最后将两个池化结果相加。图3是一个示例。

Fig. 3 Corner pooling: for each channel, we take the maximum values (red dots) in two directions (red lines), each from a separate feature map, and add the two maximums together (blue dot).

We hypothesize two reasons why detecting corners would work better than bounding box centers or proposals. First, the center of a box can be harder to localize because it depends on all 4 sides of the object, whereas locating a corner depends on 2 sides and is thus easier, and even more so with corner pooling, which encodes some explicit prior knowledge about the definition of corners. Second, corners provide a more efficient way of densely discretizing the space of boxes: we just need O(wh) corners to represent $O(w^2 h^2)$ possible anchor boxes.

检测角点比检测边界框中心点或建议框效果更好，我们认为可能有两个原因。第一，框的中心更难定位，因为这取决于目标的四条边，而定位一个角点只需要两条边，所以更容易一些，而角点池化更是如此，因为其中包含角点定义的先验知识。第二，角点给出了空间离散化的更有效方式：我们只需要O(wh)个角点，就可以表示$O(w^2 h^2)$个可能的锚框。

We demonstrate the effectiveness of CornerNet on MS COCO (Lin et al., 2014). CornerNet achieves a 42.2% AP, outperforming all existing one-stage detectors. In addition, through ablation studies we show that corner pooling is critical to the superior performance of CornerNet. Code is available at https://github.com/princeton-vl/CornerNet.

我们在MS COCO上证明了CornerNet的有效性。CornerNet取得了42.2%的AP，超过了现有所有的单阶段检测器。另外，通过分离研究，我们证明了角点池化是CornerNet取得这样优异性能的关键因素。代码已开源。

## 2 Related Works

### 2.1 Two-stage object detectors 两阶段目标检测器

Two-stage approach was first introduced and popularized by R-CNN (Girshick et al., 2014). Two-stage detectors generate a sparse set of regions of interest (RoIs) and classify each of them by a network. R-CNN generates RoIs using a low level vision algorithm (Uijlings et al., 2013; Zitnick and Dollár, 2014). Each region is then extracted from the image and processed by a ConvNet independently, which creates lots of redundant computations. Later, SPP (He et al., 2014) and Fast-RCNN (Girshick, 2015) improve R-CNN by designing a special pooling layer that pools each region from feature maps instead. However, both still rely on separate proposal algorithms and cannot be trained end-to-end. Faster-RCNN (Ren et al., 2015) does away low level proposal algorithms by introducing a region proposal network (RPN), which generates proposals from a set of pre-determined candidate boxes, usually known as anchor boxes. This not only makes the detectors more efficient but also allows the detectors to be trained end-to-end. R-FCN (Dai et al., 2016) further improves the efficiency of Faster-RCNN by replacing the fully connected sub-detection network with a fully convolutional sub-detection network. Other works focus on incorporating sub-category information (Xiang et al., 2016), generating object proposals at multiple scales with more contextual information (Bell et al., 2016; Cai et al., 2016; Shrivastava et al., 2016; Lin et al., 2016), selecting better features (Zhai et al., 2017), improving speed (Li et al., 2017), cascade procedure (Cai and Vasconcelos, 2017) and better training procedure (Singh and Davis, 2017).

两阶段方法最早是R-CNN。两阶段检测器首先生成感兴趣区域的稀疏集，然后对每个都用网络进行分类。R-CNN使用底层视觉算法生成RoI。从图像中提取出每个区域，然后由一个ConvNet独立的进行处理，这带来了很多冗余计算。后来，SPP和Fast R-CNN改进了R-CNN，他们设计了一个特殊的池化层，从特征图中对每个区域进行池化。但是，这两种算法都还依赖于单独的建议算法，不能进行端到端的训练。Faster R-CNN提出了一个区域建议网络(RPN)，从预先确定的候选框中生成建议，称之为锚框。这不仅使得检测器更有效率，也使得检测器可以得到端到端的训练。R-FCN进一步改进了Faster R-CNN的效率，将全连接检测子网络替换成全卷积检测子网络。其他工作聚焦在使用子类别的信息，在多尺度上用更多上下文信息生成目标建议，选择更好的特征，改进速度，级联方法和更好的训练过程。

### 2.2 One-stage object detectors 单阶段目标检测器

On the other hand, YOLO (Redmon et al., 2016) and SSD (Liu et al., 2016) have popularized the one-stage approach, which removes the RoI pooling step and detects objects in a single network. One-stage detectors are usually more computationally efficient than two-stage detectors while maintaining competitive performance on different challenging benchmarks.

另一方面，YOLO和SSD使得单阶段方法流行起来，单阶段方法没有RoI池化步骤，用单个网络检测目标。单阶段检测器检测速度通常比两阶段检测器要块，在不同的基准测试中的性能也非常不错。

SSD places anchor boxes densely over feature maps from multiple scales, directly classifies and refines each anchor box. YOLO predicts bounding box coordinates directly from an image, and is later improved in YOLO9000 (Redmon and Farhadi, 2016) by switching to anchor boxes. DSSD (Fu et al., 2017) and RON (Kong et al., 2017) adopt networks similar to the hourglass network (Newell et al., 2016), enabling them to combine low-level and high-level features via skip connections to predict bounding boxes more accurately. However, these one-stage detectors are still outperformed by the two-stage detectors until the introduction of RetinaNet (Lin et al., 2017). In (Lin et al., 2017), the authors suggest that the dense anchor boxes create a huge imbalance between positive and negative anchor boxes during training. This imbalance causes the training to be inefficient and hence the performance to be suboptimal. They propose a new loss, Focal Loss, to dynamically adjust the weights of each anchor box and show that their one-stage detector can outperform the two-stage detectors. RefineDet (Zhang et al., 2017) proposes to filter the anchor boxes to reduce the number of negative boxes, and to coarsely adjust the anchor boxes.

SSD在多个尺度上的特征图中密集放置了很多锚框，对这些锚框进行直接分类和精炼。YOLO直接从图像中预测边界框坐标，YOLO9000进行了改进，从锚框中预测。DSSD和RON采用了与hourglass网络类似的结构，将底层特征和高层特征通过跳跃连接很好的结合起来，更准确的预测边界框。但是，这些单阶段检测器效果都没还有两阶段检测器好，直到提出了RetinaNet，作者指出，密集的锚框造成了训练中正锚框和负锚框的严重失衡，这种失衡导致训练效率低下，所以性能不好。他们提出了一种新的损失函数，focal loss，动态的调整每个锚框的权重，他们的单阶段检测器可以超过两阶段检测器的性能。RefineDet提出对锚框进行过滤，减少负锚框的数量，粗糙的调整锚框。

DeNet (Tychsen-Smith and Petersson, 2017a) is a two-stage detector which generates RoIs without using anchor boxes. It first determines how likely each location belongs to either the top-left, top-right, bottom-left or bottom-right corner of a bounding box. It then generates RoIs by enumerating all possible corner combinations, and follows the standard two-stage approach to classify each RoI. Our approach is very different from DeNet. First, DeNet does not identify if two corners are from the same objects and relies on a sub-detection network to reject poor RoIs. In contrast, our approach is a one-stage approach which detects and groups the corners using a single ConvNet. Second, DeNet selects features at manually determined locations relative to a region for classification, while our approach does not require any feature selection step. Third, we introduce corner pooling, a novel type of layer to enhance corner detection.

DeNet是一种两阶段检测器，但没有用锚框生成RoI。它首先确定一个位置有多少可能属于一个边界框的四角，然后通过枚举所有可能的角点组合来生成RoI，然后采用标准的两阶段方法来对每个RoI进行分类。我们的方法与DeNet非常不同。首先，DeNet并不识别两个角点是否是同一个目标的角点，靠一个分类子网络来拒绝不好的RoIs。相比之下，我们的方法是一种单阶段方法，用单个卷积网络对角点进行检测并分组。第二，DeNet在一个区域中手动确定的位置上选择特征进行分类，而我们的方法不需要任何特征选择步骤。第三，我们提出了角点池化，一种新型的层，可以增强角点检测。

Point Linking Network (PLN) (Wang et al., 2017) is an one-stage detector without anchor boxes. It first predicts the locations of the four corners and the center of a bounding box. Then, at each corner location, it predicts how likely each pixel location in the image is the center. Similarly, at the center location, it predicts how likely each pixel location belongs to either the top-left, top-right, bottom-left or bottom-right corner. It combines the predictions from each corner and center pair to generate a bounding box. Finally, it merges the four bounding boxes to give a bounding box. CornerNet is very different from PLN. First, CornerNet groups the corners by predicting embedding vectors, while PLN groups the corner and center by predicting pixel locations. Second, CornerNet uses corner pooling to better localize the corners.

点连接网络(PLN)是一种单阶段检测器，没有锚框。它首先预测边界框的四个角点和中心点。然后，在每个角点位置，它预测每个像素位置有多少可能是中心点。类似的，在中心点，它预测每个像素点有多少可能是四个角点。它将每个角点和中心点的预测结合起来，生成一个边界框。最后，将四个边界框融合，给出一个边界框。CornerNet与PLN也非常不同。首先，CornerNet通过预测嵌入矢量来对角点分组，而PLN通过预测像素位置对角点和中心点进行分类。第二，CornerNet使用角点池化，以更好的定位角点。

Our approach is inspired by Newell et al. (2017) on Associative Embedding in the context of multi-person pose estimation. Newell et al. propose an approach that detects and groups human joints in a single network. In their approach each detected human joint has an embedding vector. The joints are grouped based on the distances between their embeddings. To the best of our knowledge, we are the first to formulate the task of object detection as a task of detecting and grouping corners with embeddings. Another novelty of ours is the corner pooling layers that help better localize the corners. We also significantly modify the hourglass architecture and add our novel variant of focal loss (Lin et al., 2017) to help better train the network.

我们的方法受到Newell等人的多人姿态估计中的关联嵌入的启发。Newell等人提出了一种方法，用单个网络对人体关节点进行检测并分组。在他们的方法中，每个检测到的人体关节点都有一个嵌入矢量。这些关节点基于其嵌入间的距离进行分组。据我们所致，我们是第一个将目标检测表述为用嵌入进行角点检测和分组的。我们的另一个创新是角点池化层，可以更好的定位角点。我们还大幅修改了hourglass架构，增加了我们新的focal loss变体，以更好的训练网络。

## 3 CornerNet

### 3.1 Overview 概览

In CornerNet, we detect an object as a pair of keypoints — the top-left corner and bottom-right corner of the bounding box. A convolutional network predicts two sets of heatmaps to represent the locations of corners of different object categories, one set for the top-left corners and the other for the bottom-right corners. The network also predicts an embedding vector for each detected corner (Newell et al., 2017) such that the distance between the embeddings of two corners from the same object is small. To produce tighter bounding boxes, the network also predicts offsets to slightly adjust the locations of the corners. With the predicted heatmaps, embeddings and offsets, we apply a simple post-processing algorithm to obtain the final bounding boxes.

在CornerNet中，我们将目标检测为一对关键点，即边界框的左上角和右下角。一个卷积网络预测两个热力图集，以表示不同目标类别的角点位置，一个集合是左上角，另一个是右下角集合。网络还对每个检测到的角点预测一个嵌入向量，属于同一目标的两个角点间的距离很小。为得到更紧凑的边界框，网络还预测偏移，以略微调整角点的位置。有了预测到的热力图，嵌入和偏移，我们进行简单的后处理，就可以得到最后的边界框。

Fig. 4 provides an overview of CornerNet. We use the hourglass network (Newell et al., 2016) as the backbone network of CornerNet. The hourglass network is followed by two prediction modules. One module is for the top-left corners, while the other one is for the bottom-right corners. Each module has its own corner pooling module to pool features from the hourglass network before predicting the heatmaps, embeddings and offsets. Unlike many other object detectors, we do not use features from different scales to detect objects of different sizes. We only apply both modules to the output of the hourglass network.

图4给出了CornerNet的概览。我们使用hourglass网络作为CornerNet的骨干网络。Hourglass网络后是两个预测模块。一个模块检测左上角，另一个检测右下角。每个模块都有各自的角点池化模块，对hourglass网络中出来的特征进行池化，然后预测热力图、嵌入和偏移。与其他很多目标检测器不同，我们没有使用不同尺度的特征来检测不同大小的目标。我们只用两个模块对hourglass网络的输出进行处理。

Fig. 4 Overview of CornerNet. The backbone network is followed by two prediction modules, one for the top-left corners and the other for the bottom-right corners. Using the predictions from both modules, we locate and group the corners.

### 3.2 Detecting Corners 检测角点

We predict two sets of heatmaps, one for top-left corners and one for bottom-right corners. Each set of heatmaps has C channels, where C is the number of categories, and is of size H × W. There is no background channel. Each channel is a binary mask indicating the locations of the corners for a class.

我们预测两个热力图集，一个是左上角的，一个是右下角的。每个热力图集都有C个通道，其中C是类别数，热力图大小为H×W。没有背景通道。每个通道都是二值掩膜，标识了每个类别的角点的位置。

For each corner, there is one ground-truth positive location, and all other locations are negative. During training, instead of equally penalizing negative locations, we reduce the penalty given to negative locations within a radius of the positive location. This is because a pair of false corner detections, if they are close to their respective ground truth locations, can still produce a box that sufficiently overlaps the ground-truth box (Fig. 5). We determine the radius by the size of an object by ensuring that a pair of points within the radius would generate a bounding box with at least t IoU with the ground-truth annotation (we set t to 0.3 in all experiments). Given the radius, the amount of penalty reduction is given by an unnormalized 2D Gaussian, $e^{-\frac{x^2+y^2}{2σ^2}}$, whose center is at the positive location and whose σ is 1/3 of the radius.

对于每个角点，都有一个真值位置，其他所有位置都是负的。在训练过程中，我们对所有其他负位置不是进行均等的惩罚，而是减少对与正位置距离一定范围内的负位置的惩罚。这是因为如果一对负角点检测与各自的真值位置足够近，仍然可以产生与真值框重叠足够多的位置（图5）。我们根据目标大小来确定半径，以确保半径内的一对点可以生成的边界框与真值标注的IoU大于t（我们在所有试验中设t为0.3）。给定半径，惩罚的减少由于未归一化的2D高斯函数给出$e^{-\frac{x^2+y^2}{2σ^2}}$，其中心就是在正位置，其σ是半径的1/3。

Let $p_{cij}$ be the score at location (i, j) for class c in the predicted heatmaps, and let $y_{cij}$ be the “ground-truth” heatmap augmented with the unnormalized Gaussians. We design a variant of focal loss (Lin et al., 2017): 令$p_{cij}$是在位置(i,j)类别c上预测热力图的分数，$y_{cij}$是用未归一化高斯函数扩展的真值热力图。我们定义一个focal loss的变体：

$$L_{det} = -\frac{1}{N} \sum_{c=1}^C \sum_{i=1}^H \sum_{j=1}^W (1-p_{cij})^α log(p_{cij}), if y_{cij}=1$$
$$L_{det} = -\frac{1}{N} \sum_{c=1}^C \sum_{i=1}^H \sum_{j=1}^W (1-y_{cij})^β (p_{cij})^α log(1-p_{cij}), otherwise$$(1)

where N is the number of objects in an image, and α and β are the hyper-parameters which control the contribution of each point (we set α to 2 and β to 4 in all experiments). With the Gaussian bumps encoded in $y_{cij}$, the (1 − $y_{cij}$) term reduces the penalty around the ground truth locations. 其中N是图像中的目标数量，α和β是超参数，控制的是每个点的贡献（在所有试验中，α设为2，β设为4）。高斯块的参数为$y_{cij}$，项(1 − $y_{cij}$)降低真值位置附近的惩罚。

Many networks (He et al., 2016; Newell et al., 2016) involve downsampling layers to gather global information and to reduce memory usage. When they are applied to an image fully convolutionally, the size of the output is usually smaller than the image. Hence, a location (x, y) in the image is mapped to the location ([x/n], [y/n]) in the heatmaps, where n is the downsampling factor. When we remap the locations from the heatmaps to the input image, some precision may be lost, which can greatly affect the IoU of small bounding boxes with their ground truths. To address this issue we predict location offsets to slightly adjust the corner locations before remapping them to the input resolution.

很多网络都有下采样层，以获得全局信息，并降低内存使用。当对图像进行全卷积处理，输出的大小通常比原图要小。所以，原始图像中的位置(x,y)映射到了热力图的([x/n], [y/n])位置，其中n是下采样因子。当我们将热力图上的位置重新映射回原始图像，会损失一些精度，这对于小边界框与真值的IoU影响非常大。为解决这个问题，我们预测位置偏移，在重新映射回输入分辨率之前，略微调整角点位置。

$$o_k = (\frac{x_k}{n}-[\frac{x_k}{n}], \frac{y_k}{n}-[\frac{y_k}{n}])$$(2)

where $o_k$ is the offset, $x_k$ and $y_k$ are the x and y coordinate for corner k. In particular, we predict one set of offsets shared by the top-left corners of all categories, and another set shared by the bottom-right corners. For training, we apply the smooth L1 Loss (Girshick, 2015) at ground-truth corner locations: 其中$o_k$是偏移，$x_k$ and $y_k$是角点k的x坐标和y坐标。特别的，我们预测一个偏移集合，由所有类别的左上角所共用，另一个集合由右下角所共用。对于训练，我们在真值角点处使用平滑L1损失：

$$L_{off} = \frac{1}{N} \sum_{k=1}^N SmoothL1Loss(o_k, \hat o_k)$$(3)

### 3.3 Grouping Corners 角点分组

Multiple objects may appear in an image, and thus multiple top-left and bottom-right corners may be detected. We need to determine if a pair of the top-left corner and bottom-right corner is from the same bounding box. Our approach is inspired by the Associative Embedding method proposed by Newell et al. (2017) for the task of multi-person pose estimation. Newell et al. detect all human joints and generate an embedding for each detected joint. They group the joints based on the distances between the embeddings.

图像中可能出现多个目标，所以可能检测到多个左上角点和右下角点。我们需要确定一对左上和右下角点是否属于同一边界框。我们的方法是受到Newell提出的关联嵌入方法启发的，这是为多人姿态估计的任务提出的。Newell等检测所有人的关节点，为每个检测到的关节点生成一个嵌入。关节点的分组是基于嵌入间的距离。

The idea of associative embedding is also applicable to our task. The network predicts an embedding vector for each detected corner such that if a top-left corner and a bottom-right corner belong to the same bounding box, the distance between their embeddings should be small. We can then group the corners based on the distances between the embeddings of the top-left and bottom-right corners. The actual values of the embeddings are unimportant. Only the distances between the embeddings are used to group the corners.

关联嵌入的思想也可以应用于我们的任务。网络对每个检测到的角点预测一个嵌入向量，如果左上角和右下角属于同一边界框，那么嵌入间的距离就会很小。我们就可以根据左上角和右下角嵌入之间的距离，对角点进行分组。嵌入的实际值是不重要的。嵌入之间的距离用于对角点间进行分组。

We follow Newell et al. (2017) and use embeddings of 1 dimension. Let $e_{t_k}$ be the embedding for the top-left corner of object k and $e_{b_k}$ for the bottom-right corner. As in Newell and Deng (2017), we use the “pull” loss to train the network to group the corners and the “push” loss to separate the corners: 我们采用Newell等的方法，采用一维嵌入。令$e_{t_k}$是目标k左上角的嵌入，$e_{b_k}$是右下角的嵌入。我们使用pull loss来训练网络，对角点进行分组；使用push loss来对角点进行分离：

$$L_{pull} = \frac{1}{N} \sum_{k=1}^N [(e_{t_k}-e_k)^2+(e_{b_k}-e_k)^2]$$(4)

$$L_{push} = \frac{1}{N(N-1)} \sum_{k=1}^N \sum_{j=1,j!=k}^N max(0,∆-|e_k-e_j|)$$

where $e_k$ is the average of $e_{t_k}$ and $e_{b_k}$ and we set ∆ to be 1 in all our experiments. Similar to the offset loss, we only apply the losses at the ground-truth corner location. 其中$e_k$是$e_{t_k}$和$e_{b_k}$的平均，我们在所有试验中设∆为1。与偏移的损失一样，我们只将这个损失用于真值角点位置。

### 3.4 Corner Pooling 角点池化

As shown in Fig. 2, there is often no local visual evidence for the presence of corners. To determine if a pixel is a top-left corner, we need to look horizontally towards the right for the topmost boundary of an object and vertically towards the bottom for the leftmost boundary. We thus propose corner pooling to better localize the corners by encoding explicit prior knowledge.

如图2所示，角点的存在一般没有局部视觉证据。为确定是否一个点是左上角，我们需要对目标的上边缘，沿着水平方向向右看，对于左边缘，沿着竖直方向向下看。所以我们提出角点池化，通过编码先验知识，以更好的定位角点。

Suppose we want to determine if a pixel at location (i, j) is a top-left corner. Let $f_t$ and $f_l$ be the feature maps that are the inputs to the top-left corner pooling layer, and let $f_{t_{ij}}$ and $f_{l_{ij}}$ be the vectors at location (i, j) in $f_t$ and $f_l$ respectively. With H × W feature maps, the corner pooling layer first max-pools all feature vectors between (i, j) and (i, H) in $f_t$ to a feature vector $t_{ij}$, and max-pools all feature vectors between (i, j) and (W, j) in $f_l$ to a feature vector $l_{ij}$. Finally, it adds $t_{ij}$ and $l_{ij}$ together. This computation can be expressed by the following equations:

假设我们想确定位置(i,j)的像素是否是左上角。令$f_t$和$f_l$是输入到左上角点池化层的特征图，令$f_{t_{ij}}$和$f_{l_{ij}}$分别是$f_t$和$f_l$在位置(i,j)的向量。特征图大小为H×W，角点池化层首先对$f_t$中(i,j)到(i,H)上的所有特征向量进行最大池化，得到特征向量$t_{ij}$，对$f_l$中(i,j)和(W,j)上的所有特征向量进行最大池化，得到特征向量$l_{ij}$。最后，将$t_{ij}$和$l_{ij}$加到一起。这个计算表示为下式：

$$t_{ij} = \begin{cases} max(f_{t_{ij}}, t_{(i+1)j}), \quad if \space i<H \\ f_{t_{Hj}}, \quad otherwise \end{cases}$$(6)

$$l_{ij} = \begin{cases} max(f_{l_{ij}}, l_{i(j+1)}), \quad if \space j<W \\ f_{l_{iW}}, \quad otherwise \end{cases}$$(7)

where we apply an elementwise max operation. Both $t_{ij}$ and $l_{ij}$ can be computed efficiently by dynamic programming as shown Fig. 8. 其中我们进行了逐元素的最大运算。$t_{ij}$和$l_{ij}$都可以用动态规划进行有效的计算，如图8所示。

We define bottom-right corner pooling layer in a similar way. It max-pools all feature vectors between (0, j) and (i, j), and all feature vectors between (i, 0) and (i, j) before adding the pooled results. The corner pooling layers are used in the prediction modules to predict heatmaps, embeddings and offsets.

我们类似的定义右下角的角点池化层。其最大池化的特征向量在(0,j)到(i,j)范围内，和(i,0)到(i,j)范围内，然后将池化的结果加到一起。角点池化层在预测模块中用于预测热力图、嵌入和偏移。

The architecture of the prediction module is shown in Fig. 7. The first part of the module is a modified version of the residual block (He et al., 2016). In this modified residual block, we replace the first 3 × 3 convolution module with a corner pooling module, which first processes the features from the backbone network by two 3 × 3 convolution modules (Unless otherwise specified, our convolution module consists of a convolution layer, a BN layer (Ioffe and Szegedy, 2015) and a ReLU layer) with 128 channels and then applies a corner pooling layer. Following the design of a residual block, we then feed the pooled features into a 3 × 3 Conv-BN layer with 256 channels and add back the projection shortcut. The modified residual block is followed by a 3×3 convolution module with 256 channels, and 3 Conv-ReLU-Conv layers to produce the heatmaps, embeddings and offsets.

预测模块的结构如图7所示。模块中的第一部分是修改版的残差模块。在这个修改版的残差模块中，我们将第一个3×3卷积替换为角点池化模块，即，将骨干网络中出来的特征用两个3×3卷积模块（128通道）进行处理（除非另有说明，我们的卷积模块包括一个卷积层、一个BN层和一个ReLU层），然后用角点池化层进行处理。根据残差模块的设计，我们将池化的特征送入一个3×3卷积-BN层（256通道），然后加入投射捷径。修正的残差模块后，是一个3×3卷积模块（256通道），然后是3个Conv-ReLU-Conv层，分别得到热力图、嵌入和偏移。

### 3.5 Hourglass Network 沙漏网络

CornerNet uses the hourglass network (Newell et al., 2016) as its backbone network. The hourglass network was first introduced for the human pose estimation task. It is a fully convolutional neural network that consists of one or more hourglass modules. An hourglass module first downsamples the input features by a series of convolution and max pooling layers. It then upsamples the features back to the original resolution by a series of upsampling and convolution layers. Since details are lost in the max pooling layers, skip layers are added to bring back the details to the upsampled features. The hourglass module captures both global and local features in a single unified structure. When multiple hourglass modules are stacked in the network, the hourglass modules can reprocess the features to capture higher-level of information. These properties make the hourglass network an ideal choice for object detection as well. In fact, many current detectors (Shrivastava et al., 2016; Fu et al., 2017; Lin et al., 2016; Kong et al., 2017) already adopted networks similar to the hourglass network.

CornerNet使用Hourglass网络作为其骨干网络。Hourglass网络提出的时候是用于人体姿态估计任务，这是一个全卷积网络，包括一个或多个hourglass模块。一个hourglass模块首先通过一系列卷积层和最大池化层将输入特征进行降采样，然后将特征通过一系列上采样层和卷积层进行上采样，回到原始分辨率。由于在最大池化的过程中丢失了特征细节，所以添加了跳跃连接层，以将细节加入到上采样的特征中。Hourglass模块在单个统一的结构中捕捉到了全局和局部的特征。当多个hourglass模块堆叠在一个网络中时，hourglass模块可以重复处理特征，捕捉更高层的信息。这种性质使得hourglass网络也是目标检测的理想选择。实际上，很多现在的检测器已经采用了类似hourglass的网络结构。

Our hourglass network consists of two hourglasses, and we make some modifications to the architecture of the hourglass module. Instead of using max pooling, we simply use stride 2 to reduce feature resolution. We reduce feature resolutions 5 times and increase the number of feature channels along the way (256, 384, 384, 384, 512). When we upsample the features, we apply 2 residual modules followed by a nearest neighbor upsampling. Every skip connection also consists of 2 residual modules. There are 4 residual modules with 512 channels in the middle of an hourglass module. Before the hourglass modules, we reduce the image resolution by 4 times using a 7 × 7 convolution module with stride 2 and 128 channels followed by a residual block (He et al., 2016) with stride 2 and 256 channels.

我们的hourglass网络包括两个hourglass，而且我们对hourglass模块进行了一些架构的改变。我们没有使用最大池化，而只是使用了步长2来降低特征分辨率。我们将分辨率降低5次，同时逐步增加特征通道数（256,384,384,384,512）。当我们对特征上采样时，我们用两个残差模块，跟着一个最近邻上采样。每个跳跃连接也包含2个残差模块。在hourglass模块的中间，有4个512通道的残差模块。在hourglass模块之前，我们将图像分辨率降低了4倍，使用的是一个128通道的7×7卷积模块步长2，后跟着一个步长2通道数256的残差模块。

Following (Newell et al., 2016), we also add intermediate supervision in training. However, we do not add back the intermediate predictions to the network as we find that this hurts the performance of the network. We apply a 1 × 1 Conv-BN module to both the input and output of the first hourglass module. We then merge them by element-wise addition followed by a ReLU and a residual block with 256 channels, which is then used as the input to the second hourglass module. The depth of the hourglass network is 104. Unlike many other state-of-the-art detectors, we only use the features from the last layer of the whole network to make predictions.

我们还在训练中加入了中间监督。但是，我们没有没有将中间预测加回到网络中，因为我们发现这会损害网络的表现。我们对Hourglass模块的输入和输出都用1×1 Conv-BN模块进行处理。然后使用逐元素相加进行合并，后面是ReLU层和256通道的残差模块，然后用于第一个和第二个hourglass模块的输入。Hourglass网络的深度是104。与其他很多现在最好的检测器不同，我们只使用整个网络最后一层的特征进行预测。

## 4 Experiments 试验

### 4.1 Training Details 训练细节

We implement CornerNet in PyTorch (Paszke et al., 2017). The network is randomly initialized under the default setting of PyTorch with no pretraining on any external dataset. As we apply focal loss, we follow (Lin et al., 2017) to set the biases in the convolution layers that predict the corner heatmaps. During training, we set the input resolution of the network to 511 × 511, which leads to an output resolution of 128 × 128. To reduce overfitting, we adopt standard data augmentation techniques including random horizontal flipping, random scaling, random cropping and random color jittering, which includes adjusting the brightness, saturation and contrast of an image. Finally, we apply PCA (Krizhevsky et al., 2012) to the input image.

我们在PyTorch中实现CornerNet。网络在PyTorch的默认设置下进行随机初始化，没有用任何外部数据集进行预训练。我们使用focal loss时，遵循前文的方法来设置预测角点热力图的卷积层中的偏置。在训练过程中，我们设置网络的输入分辨率为511×511，这样得到的输出分辨率为128×128。为降低过拟合，我们采用标准的数据扩充技术，包括水平翻转，随机尺度，随机剪切，和随机色彩抖动，即调整亮度、饱和度和对比度。最后，我们对输入图像进行了PCA处理。

We use Adam (Kingma and Ba, 2014) to optimize the full training loss: 我们使用Adam来优化全部的训练损失：

$$L = L_{det} + αL_{pull} + βL_{push} + γL_{off}$$(8)

where α, β and γ are the weights for the pull, push and offset loss respectively. We set both α and β to 0.1 and γ to 1. We find that 1 or larger values of α and β lead to poor performance. We use a batch size of 49 and train the network on 10 Titan X (PASCAL) GPUs (4 images on the master GPU, 5 images per GPU for the rest of the GPUs). To conserve GPU resources, in our ablation experiments, we train the networks for 250k iterations with a learning rate of 2.5 × $10_{−4}$. When we compare our results with other detectors, we train the networks for an extra 250k iterations and reduce the learning rate to 2.5 × $10^{−5}$ for the last 50k iterations.

其中α, β和γ分别是pull、push和offset损失的权重。我们设α, β为1，γ为1。我们发现α, β为1或更大的话，会得到很差的结果。我们使用的batch size为49，在10个Titan X(PASCAL) GPUs上进行训练（主GPU上4幅图像，其他每个GPU 5幅图像）。为节省GPU资源，在我们的分离试验中，我们使用的学习速率为2.5 × $10_{−4}$，训练网络250k次迭代。当我们与其他检测器比较结果时，我们继续训练网络250k次，在最后50k次迭代时，把学习速率降低为2.5 × $10^{−5}$。

### 4.2 Testing Details 测试细节

During testing, we use a simple post-processing algorithm to generate bounding boxes from the heatmaps, embeddings and offsets. We first apply non-maximal suppression (NMS) by using a 3×3 max pooling layer on the corner heatmaps. Then we pick the top 100 top-left and top 100 bottom-right corners from the heatmaps. The corner locations are adjusted by the corresponding offsets. We calculate the L1 distances between the embeddings of the top-left and bottom-right corners. Pairs that have distances greater than 0.5 or contain corners from different categories are rejected. The average scores of the top-left and bottom-right corners are used as the detection scores.

在测试时，我们使用一个简单的后处理算法来从热力图、嵌入和偏移中生成边界框。我们首先使用NMS，即在角点热力图上使用3×3最大池化层。然后我们从热力图中选择最高的100个左上角和最高的100个右下角。角点位置由对应的偏移来调整。我们计算左上角和右下角嵌入的L1距离。距离大于5的角点对，或两个角点分属不同类别的，则拒绝分组到一起。左上角和右下角的平均分数被用作检测分数。

Instead of resizing an image to a fixed size, we maintain the original resolution of the image and pad it with zeros before feeding it to CornerNet. Both the original and flipped images are used for testing. We combine the detections from the original and flipped images, and apply soft-nms (Bodla et al., 2017) to suppress redundant detections. Only the top 100 detections are reported. The average inference time is 244ms per image on a Titan X (PASCAL) GPU.

我们没有把图像大小变为固定大小，而是保持图像的原始分辨率，送入CornerNet之前补零。原始图像和翻转图像都用于测试。我们将原始图像和翻转图像的检测结果进行综合，并使用soft-nms来抑制多余的检测。只给出最高的100个检测。在Titan X(PASCAL) GPU上的平均推理时间是每幅图像244ms。

### 4.3 MS COCO

We evaluate CornerNet on the very challenging MS COCO dataset (Lin et al., 2014). MS COCO contains 80k images for training, 40k for validation and 20k for testing. All images in the training set and 35k images in the validation set are used for training. The remaining 5k images in validation set are used for hyper-parameter searching and ablation study. All results on the test set are submitted to an external server for evaluation. To provide fair comparisons with other detectors, we report our main results on the test-dev set. MS COCO uses average precisions (APs) at different IoUs and APs for different object sizes as the main evaluation metrics.

我们在MS COCO数据集上评估CornerNet。MS COCO的训练集包含80k幅图像，验证集40k，测试集20k。训练集的所有图像和验证集的35k图像被用于训练。验证集剩余的5k图像用于超参数搜索和分离试验。在测试集上的所有结果提交给外部服务内进行评估。为和其他检测器进行公平比较，我们在test-dev集上给出主要结果。MS COCO使用不同IoU上的AP，和不同目标大小的AP作为主要评估标准。

### 4.4 Ablation Study 分离研究

#### 4.4.1 Corner Pooling 角点池化

Corner pooling is a key component of CornerNet. To understand its contribution to performance, we train another network without corner pooling but with the same number of parameters. 角点池化是CornerNet的关键部分。为理解其对性能的贡献，我们训练了另一个没有角点池化的网络，参数数量是一样的。

Tab. 1 shows that adding corner pooling gives significant improvement: 2.0% on AP, 2.1% on $AP^{50}$ and 2.1% on $AP^{75}$. We also see that corner pooling is especially helpful for medium and large objects, improving their APs by 2.4% and 3.6% respectively. This is expected because the topmost, bottommost, leftmost, rightmost boundaries of medium and large objects are likely to be further away from the corner locations. Fig. 8 shows four qualitative examples with and without corner pooling.

表1给出了角点池化对性能的明显改进：AP增加2.0%，$AP^{50}$增加2.1%，$AP^{75}$增加2.1%。我们还看到，角点池化对于中型和大型目标特别有用，其AP分别提高了2.4%和3.6%。这是因为，中型目标和大型目标的最上、下、左、右边缘距离角点位置都会比较远。图8给出了4个有角点池化和没有角点池化的样本。

Table 1 Ablation on corner pooling on MS COCO validation.

|  | AP | $AP^{50}$ | $AP^{75}$ | $AP^s$ | $AP^m$ | $AP^l$
--- | --- | --- | --- | --- | --- | ---
w/o corner pooling | 36.5 | 52.0 | 38.8 | 17.5 | 38.9 | 49.4
w/ corner pooling | 38.4 | 53.8 | 40.9 | 18.6 | 40.5 | 51.8
improvement | +2.0 | +2.1 | +2.1 | +1.1 | +2.4 | +3.6

#### 4.4.2 Stability of Corner Pooling over Larger Area 角点池化在较大区域上的稳定性

Corner pooling pools over different sizes of area in different quadrants of an image. For example, the top-left corner pooling pools over larger areas both horizontally and vertically in the upper-left quadrant of an image, compared to the lower-right quadrant. Therefore, the location of a corner may affect the stability of the corner pooling.

角点池化在图像的不同部分池化的区域大小不同。比如，左上角池化的区域更大，水平方向和竖直方向上都比右下角点大。所以，角点的位置可能会影响角点池化的稳定性。

We evaluate the performance of our network on detecting both the top-left and bottom-right corners in different quadrants of an image. Detecting corners can be seen as a binary classification task i.e. the ground-truth location of a corner is positive, and any location outside of a small radius of the corner is negative. We measure the performance using mAPs over all categories on the MS COCO validation set.

我们评估网络检测左上角和右下角在图像不同区域时的性能。检测角点可以看做是一个二值分类任务，即角点的真值位置是正的，以角点为中心很小半径区域之外的任何位置都是负的。我们使用mAP在所有类别上在MS COCO验证集上衡量性能。

Tab. 3 shows that without corner pooling, the top-left corner mAPs of upper-left and lower-right quadrant are 66.1% and 60.8% respectively. Top-left corner pooling improves the mAPs by 3.1% (to 69.2%) and 2.7% (to 63.5%) respectively. Similarly, bottom-right corner pooling improves the bottom-right corner mAPs of upper-left quadrant by 2.8% (from 53.4% to 56.2%), and lower-right quadrant by 2.6% (from 65.0% to 67.6%). Corner pooling gives similar improvement to corners at different quadrants, show that corner pooling is effective and stable over both small and large areas.

表3说明，没有角点池化，在左上和右下部分的左上角的mAP分别是66.1%和60.8%，而有了角点池化则将mAP分别改进了3.1%和2.7%。类似的，右下角点池化将右下角点在左上部分和右下部分的mAP改进了2.8% (from 53.4% to 56.2%)和2.6% (from 65.0% to 67.6%)。角点池化在不同部分的改进是接近的，说明角点池化对小区域和大区域都是有效和稳定的。

Table 3 Corner pooling consistently improves the network performance on detecting corners in different image quadrants, showing that corner pooling is effective and stable over both small and large areas.

|  | mAP w/o pooling | mAP w/ pooling | improvement
--- | --- | --- | ---
Top-Left Corners |
Top-Left Quad. | 66.1 | 69.2 | +3.1
Bottom-Right Quad. | 60.8 | 63.5 | +2.7
Bottom-Right Corners |
Top-Left Quad. | 53.4 | 56.2 | +2.8
Bottom-Right Quad. | 65.0 | 67.6 | +2.6

#### 4.4.3 Reducing Penalty to Negative Locations 降低负位置的惩罚

We reduce the penalty given to negative locations around a positive location, within a radius determined by the size of the object (Sec. 3.2). To understand how this helps train CornerNet, we train one network with no penalty reduction and another network with a fixed radius of 2.5. We compare them with CornerNet on the validation set.

我们降低对正位置周围的一定半径内的负位置的惩罚，半径由目标大小来决定（见3.2节）。为理解这怎样有助于CornerNet的训练，我们训练一个没有降低惩罚的，和另一个降低惩罚的，半径固定为2.5。我们在验证集上比较其结果。

Tab. 2 shows that a fixed radius improves AP over the baseline by 2.7%, $AP^m$ by 1.5% and $AP^l$ by 5.3%. Object-dependent radius further improves the AP by 2.8%, $AP^m$ by 2.0% and $AP^l$ by 5.8%. In addition, we see that the penalty reduction especially benefits medium and large objects.

表2说明，固定半径比基准的AP高了2.7%，$AP^m$高了1.5%，$AP^l$高了5.3%。与目标相关的半径进一步改进了AP 2.8%，$AP^m$ 2.0%，$AP^l$ 5.8%。另外，我们看到降低惩罚尤其对中型目标和大型目标有好处。

Table 2 Reducing the penalty given to the negative locations near positive locations helps significantly improve the performance of the network

|  | AP | $AP^{50}$ | $AP^{75}$ | $AP^s$ | $AP^m$ | $AP^l$
--- | --- | --- | --- | --- | --- | ---
w/o reducing penalty | 32.9 | 49.1 | 34.8 | 19.0 | 37.0 | 40.7
fixed radius | 35.6 | 52.5 | 37.7 | 18.7 | 38.5 | 46.0
object-dependent radius | 38.4 | 53.8 | 40.9 | 18.6 | 40.5 | 51.8

#### 4.4.4 Hourglass Network

CornerNet uses the hourglass network (Newell et al., 2016) as its backbone network. Since the hourglass network is not commonly used in other state-of-the-art detectors, we perform an experiment to study the contribution of the hourglass network in CornerNet. We train a CornerNet in which we replace the hourglass network with FPN (w/ResNet-101) (Lin et al., 2017), which is more commonly used in state-of-the-art object detectors. We only use the final output of FPN for predictions. Meanwhile, we train an anchor box based detector which uses the hourglass network as its backbone. Each hourglass module predicts anchor boxes at multiple resolutions by using features at multiple scales during upsampling stage. We follow the anchor box design in RetinaNet (Lin et al., 2017) and add intermediate supervisions during training. In both experiments, we initialize the networks from scratch and follow the same training procedure as we train CornerNet (Sec. 4.1).

CornerNet使用hourglass网络作为其骨干网络。由于hourglass网络在其他现在最好的检测器中使用并不多，我们研究了一下Hourglass网络在CornerNet中的贡献。我们训练了一个CornerNet，将Hourglass替换为FPN (w/ResNet-101)，这在目前最好的目标检测中用的较多。我们只使用FPN的最终输出进行预测。同时，我们训练了一个基于锚框检测器，使用Hourglass网络作为骨干网络。每个Hourglass模块预测多个分辨率下的锚框，在上采样阶段使用多尺度下的特征。我们使用RetinaNet中的锚框设计方法，在训练中增加了中间监督。在两个试验中，我们从头训练网络，训练方法与训练CornerNet相同。

Tab. 4 shows that CornerNet with hourglass network outperforms CornerNet with FPN by 8.2% AP, and the anchor box based detector with hourglass network by 5.5% AP. The results suggest that the choice of the backbone network is important and the hourglass network is crucial to the performance of CornerNet.

表4说明了，使用Hourglass的CornerNet超过了使用FPN的CornerNet达8.2% AP，超过了Hourglass网络作为骨干的基于锚框的检测器达5.5% AP。这个结果说明，骨干网络的选择非常重要，Hourglass网络对于CornerNet的性能非常关键。

Table 4 The hourglass network is crucial to the performance of CornerNet.

|  | AP | $AP^{50}$ | $AP^{75}$ | $AP^s$ | $AP^m$ | $AP^l$
--- | --- | --- | --- | --- | --- | ---
FPN (w/ResNet-101) + Corners | 30.2 | 44.1 | 32.0 | 13.3 | 33.3 | 42.7
Hourglass + Anchors | 32.9 | 53.1 | 35.6 | 16.5 | 38.5 | 45.0
Hourglass + Corners | 38.4 | 53.8 | 40.9 | 18.6 | 40.5 | 51.8


#### 4.4.5 Quality of the Bounding Boxes 边界框的质量

A good detector should predict high quality bounding boxes that cover objects tightly. To understand the quality of the bounding boxes predicted by CornerNet, we evaluate the performance of CornerNet at multiple IoU thresholds, and compare the results with other state-of-the-art detectors, including RetinaNet (Lin et al., 2017), Cascade R-CNN (Cai and Vasconcelos, 2017) and IoU-Net (Jiang et al., 2018).

好的检测器应当预测高质量的边界框，很紧密的与目标贴合。为理解CornerNet预测的边界框的质量，我们在多个IoU阈值上评估CornerNet的性能，与其他目前最好的检测器比较其结果，包括RetinaNet，Cascade R-CNN和IoU-Net。

Tab. 5 shows that CornerNet achieves a much higher AP at 0.9 IoU than other detectors, outperforming Cascade R-CNN + IoU-Net by 3.9%, Cascade R-CNN by 7.6% and RetinaNet by 7.3%. This suggests that CornerNet is able to generate bounding boxes of higher quality compared to other state-of-the-art detectors. 表5说明，CornerNet在0.9 IoU上比其他检测器的AP高的多，超过了Cascade R-CNN + IoU-Net 3.9%，超过了Cascade R-CNN 7.6%，RetinaNet 7.3%。这说明CornerNet能够生成比其他目前最好框架更好的边界框。

Table 5 CornerNet performs much better at high IoUs than other state-of-the-art detectors.

| | AP | $AP^{50}$ | $AP^{60}$ | $AP^{70}$ | $AP^{80}$ | $AP^{90}$
--- | --- | --- | --- | --- | --- | ---
RetinaNet (Lin et al., 2017) | 39.8 | 59.5 | 55.6 | 48.2 | 36.4 | 15.1
Cascade R-CNN (Cai and Vasconcelos, 2017) | 38.9 | 57.8 | 53.4 | 46.9 | 35.8 | 15.8
Cascade R-CNN + IoU Net (Jiang et al., 2018) | 41.4 | 59.3 | 55.3 | 49.6 | 39.4 | 19.5
CornerNet | 40.6 | 56.1 | 52.0 | 46.8 | 38.8 | 23.4

#### 4.4.6 Error Analysis 错误分析

CornerNet simultaneously outputs heatmaps, offsets, and embeddings, all of which affect detection performance. An object will be missed if either corner is missed; precise offsets are needed to generate tight bounding boxes; incorrect embeddings will result in many false bounding boxes. To understand how each part contributes to the final error, we perform an error analysis by replacing the predicted heatmaps and offsets with the ground-truth values and evaluting performance on the validation set.

CornerNet同时输出热力图，偏移和嵌入，这都影响检测性能。遗漏目标的任何一个角点，就会丢失这个目标；精确的偏移才能生成紧致的边界框；不正确的嵌入会导致很多错误的边界框。为理解每个部分对最终的错误有什么影响，我们进行了一次错误分析，将预测的热力图和偏移替换成真值，评估在验证集上的性能。

Tab. 6 shows that using the ground-truth corner heatmaps alone improves the AP from 38.4% to 73.1%. $AP^s$, $AP^m$ and $AP^l$ also increase by 42.3%, 40.7% and 30.0% respectively. If we replace the predicted offsets with the ground-truth offsets, the AP further increases by 13.0% to 86.1%. This suggests that although there is still ample room for improvement in both detecting and grouping corners, the main bottleneck is detecting corners. Fig. 9 shows some qualitative examples where the corner locations or embeddings are incorrect.

表6给出了只使用真值热力图就可以将AP从38.4%改进到73.1%。$AP^s$, $AP^m$和$AP^l$也分别提升了42.3%, 40.7%和30.0%。如果我们将预测的偏移替换成真值偏移，AP进一步提高13.0%，到86.1%。这说明，虽然在检测角点和分组角点上有充分的改进空间，但主要的瓶颈还是在检测角点上。图9给出了一些定性样本，其中的角点位置或嵌入是不正确的。

Table 6 Error analysis. We replace the predicted heatmaps and offsets with the ground-truth values. Using the ground-truth heatmaps alone improves the AP from 38.4% to 73.1%, suggesting that the main bottleneck of CornerNet is detecting corners.

|  | AP | $AP^{50}$ | $AP^{75}$ | $AP^s$ | $AP^m$ | $AP^l$
--- | --- | --- | --- | --- | --- | ---
| | 38.4 | 53.8 | 40.9 | 18.6 | 40.5 | 51.8
w/ gt heatmaps | 73.1 | 87.7 | 78.4 | 60.9 | 81.2 | 81.8
w/ gt heatmaps + offsets | 86.1 | 88.9 | 85.5 | 84.8 | 87.2 | 82.0

### 4.5 Comparisons with state-of-the-art detectors 与目前最好的检测器的对比

We compare CornerNet with other state-of-the-art detectors on MS COCO test-dev (Tab. 7). With multiscale evaluation, CornerNet achieves an AP of 42.2%, the state of the art among existing one-stage methods and competitive with two-stage methods. 我们将CornerNet与目前最好的检测器在MS COCO test-dev上进行比较（表7）。在多尺度评估上，CornerNet得到了42.2% AP，在现有的单阶段方法中是最好的，与两阶段方法也非常接近。

Table 7 CornerNet versus others on MS COCO test-dev. CornerNet outperforms all one-stage detectors and achieves results competitive to two-stage detectors

## 5 Conclusion 结论

We have presented CornerNet, a new approach to object detection that detects bounding boxes as pairs of corners. We evaluate CornerNet on MS COCO and demonstrate competitive results. 我们提出了CornerNet，一种新的目标检测方法，检测边界框的一对角点。我们在MS COCO上评估了CornerNet，证明了其很好的结果。