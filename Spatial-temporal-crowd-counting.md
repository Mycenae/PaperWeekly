# Spatio-temporal Modeling for Crowd Counting in Videos

Feng Xiong et al. Hong Kong University of Science and Technology

## Abstract 摘要

Region of Interest (ROI) crowd counting can be formulated as a regression problem of learning a mapping from an image or a video frame to a crowd density map. Recently, convolutional neural network (CNN) models have achieved promising results for crowd counting. However, even when dealing with video data, CNN-based methods still consider each video frame independently, ignoring the strong temporal correlation between neighboring frames. To exploit the otherwise very useful temporal information in video sequences, we propose a variant of a recent deep learning model called convolutional LSTM (ConvLSTM) for crowd counting. Unlike the previous CNN-based methods, our method fully captures both spatial and temporal dependencies. Furthermore, we extend the ConvLSTM model to a bidirectional ConvLSTM model which can access long-range information in both directions. Extensive experiments using four publicly available datasets demonstrate the reliability of our approach and the effectiveness of incorporating temporal information to boost the accuracy of crowd counting. In addition, we also conduct some transfer learning experiments to show that once our model is trained on one dataset, its learning experience can be transferred easily to a new dataset which consists of only very few video frames for model adaptation.

感兴趣区域(ROI)的人群计数可以视为一个回归问题，即学习一个从图像或视频帧到人群密度图的映射。近年来，卷积神经网络(CNN)模型已经在人群计数上取得了有希望的结果。但是，即使是在处理视频数据时，基于CNN的方法还是将每帧视频单独处理，忽略了相邻帧之间很强的时间关联。为发掘这些视频序列中很有用的时域信息，我们提出了一种最近的深度学习模型ConvLSTM的变体，进行人群计数。与之前基于CNN的方法不同，我们的方法完全捕捉到了空域和时域的依赖关系。而且，我们将ConvLSTM拓展到了双向ConvLSTM模型，可以在两个方向上获取长期信息。我们在4个公开数据集上进行了广泛的实验，结果表明我们方法的有效性，以及吸收了时域信息提升人群计数准确性的有效性。另外，我们还进行了一些迁移学习的实验，结果表明，我们的模型在一个数据集上训练后，其学习经验可以很容易的迁移到新数据集，即使新数据集只有很少视频帧用于改变模型。

## 1. Introduction 引言

Crowd counting is the problem of estimating the number of people in a still image or a video. It has drawn a lot of attention due to the need for solving this problem in many real-world applications such as video surveillance, traffic control, and emergency management. Proper use of crowd counting techniques can help to prevent some serious accidents such as the massive stampede that happened in Shanghai, China during the 2015 New Year’s Eve festivities, killing 35 people. Moreover, some crowd counting methods can also be applied to other object counting applications such as cell counting in microscopic images [16, 29], vehicle counting in public areas [19, 10], and animal counting in the wild [3].

人群计数是估计静止图像或视频中人数的问题。由于有很多真实世界应用的需求，如视频监控，交通控制，紧急状态管理，所以人群计数已经吸引了很多注意力。合理的使用人群计数技术可以帮助防止一些很严重的事件，如上海在2015年新年夜活动中的大规模踩踏事件，死亡35人。而且，一些人群计数方法也可以用于其他目标计数应用中，如显微图像中的细胞计数[16,29]，公共区域的车辆计数[19,10]和野外的动物计数[3]。

The methods for crowd counting in videos fall into two broad categories: (a) Region of Interest (ROI) counting, which estimates the total number of people in some region at a certain time; and (b) Line of Interest (LOI) counting, which counts the number of people crossing a detecting line in a video during a certain period of time. Since LOI counting is more restrictive in its applications and is much less studied than ROI counting, we focus on ROI counting in this paper.

视频中人群计数的方法可以分为两类：(a)感兴趣区域(ROI)计数，就是估计一定时间段内估计某区域内的总人数；(b)感兴趣线（LOI）计数，即检测视频中在一定时间段内跨过检测线的人数。由于LOI计数在应用中限制很多，比ROI计数的研究少很多，我们本文主要聚焦在ROI计数。

Many methods have been proposed in the past for crowd counting. Some methods take the approach of tackling the crowd counting problem in an unsupervised manner through grouping based on self-similarities [1] or motion similarities [22]. However, the accuracy of such fully unsupervised counting methods is limited. Thus more attention has been paid to the supervised learning approach. Supervised crowd counting methods generally fall into two categories: detection-based methods and regression-based methods. In detection-based methods, some given object detectors [13, 35, 17, 9] are used to detect people individually. They usually operate in two stages by first producing a real-valued confidence map and then locating from the map those peaks that correspond to individual people. Once the locations of all individuals have been estimated, the counting problem becomes trivial. However, object detection could be a challenging problem especially under severe occlusion in highly crowded scenes.

过去提出了很多人群计数的方法。一些方法采用无监督的方式，通过基于自相似性[1]或运动相似性[22]分组的途径来解决这个问题。但是，这种完全无监督的计数方法的准确度不高。所以，监督学习的方法更受关注。有监督人群计数的方法大致分为两类：基于检测的方法和基于回归的方法。在基于检测的方法中，使用了一些现有的检测器[13,35,17,9]来检测个体的人。检测通常分两个阶段，首先生成置信度图，然后从图中定位这些峰值，即对应人的个体。一旦所有个体的位置都估计出来之后，计数问题就是小问题了。但是，目标检测可能是个很有挑战性的问题，尤其是在高度拥挤场景中的严重遮挡的情况下。

In recent years, regression-based methods have achieved promising counting results in crowded scenes. Regression-based methods avoid solving the difficult detection problem. Instead, they regard crowd counting as a regression problem by learning a regression function or mapping from some holistic or local visual features to a crowd count or a crowd density map. Linear regression, Gaussian process regression, and neural networks are often used as the regression models. Currently, most methods which achieve state-of-the-art performance are regression-based methods [4, 7, 2, 21, 6, 32, 34, 29].

近年来，基于回归的模型在拥挤场景中取得了有希望的计数结果。基于回归的方法避免了去解决困难的检测问题，而是通过学习一个回归函数或，或从一些整体或局部视觉特征到人群数量或人群密度图的映射，从而将人群计数视作一个回归问题。线性回归，高斯过程回归，和神经网络都可以用于回归模型。目前，多数取得最好结果的方法都是基于回归的模型[4,7,2,21,6,32,34,29]。

With the recent resurgence of interest in convolutional neural network (CNN) models which have reported promising results for many computer vision tasks [15], in the recent two years some CNN-based methods [32, 34, 29, 26] have also been proposed for crowd counting, giving state-of-the-art results on the existing crowd counting datasets such as UCSD [4] and UCF [12]. Unlike traditional regression-based methods [4, 7], CNN-based methods do not need handcrafted features but can learn powerful features in an end-to-end manner. However, even when dealing with video datasets, these CNN-based methods still regard the data as individual still images and ignore the strong temporal correlation between neighboring video frames.

随着最近卷积神经网络(CNN)兴趣的复苏，在很多计算机视觉任务中都出现了很有希望的结果[15]，在最近两年提出了一些基于CNN的方法[32,34,29,26]进行人群计数，在现有的人群计数数据集上给出了目前最好的结果，如UCSD[4]和UCF[12]。与传统的基于回归的方法[4,7]不同，基于CNN的方法不需要手工设计特征，但可以以端到端的形式学习到非常强的特征。但是，即使在处理视频数据集的时候，这些基于CNN的方法仍然将数据视作单幅静止图像，忽略了相邻视频帧之间的很强的时域关联。

In this paper, we propose a variant of a recent deep learning model called convolutional LSTM (ConvLSTM) [25] for crowd counting. While CNN-based methods exploit only spatial features by ignoring the otherwise very useful temporal information in video sequences, our method fully captures both spatial and temporal dependencies. Incorporating the temporal dimension is important as it is well known that motion information can help to boost the counting accuracy in complex scenes. Thorough experimental validation using four publicly available datasets will be reported later in this paper to demonstrate the effectiveness of incorporating temporal information to boost the accuracy of crowd counting.

在本文中，我们提出了一种最近的深度学习模型的变体，即ConvLSTM[25]，进行人群计数。基于CNN的方法只利用了空间特征，忽略了视频序列中非常有用的时域信息，而我们的方法完全捕捉了基于时间和空间的依赖关系。利用了时间维度的信息非常重要，因为大家都知道，运动信息可以帮助改进复杂场景下的计数准确性。通过在4个公开数据集上的实验验证，证明了使用时域信息的有效性，可以提升人群计数的准确性。

## 2. Related Work 相关工作

### 2.1. Deep learning methods for crowd counting 基于深度学习方法的人群计数

C. Zhang et al. [32] proposed the first CNN-based method for crowd counting. Following this work, Y. Zhang et al. [34] proposed a multi-column CNN architecture which allows the input image to be of arbitrary size or resolution. The multi-column CNN architecture also uses a different method for computing the crowd density. Walach and Wolf [29] proposed a stage-wise approach by carrying out model training in stages. In the spirit of the gradient boosting approach, CNNs are added one at a time such that every new CNN is trained to estimate the residual error of the earlier prediction. After the first CNN has been trained, the second CNN is trained on the difference between the current estimate and the learning target. The third CNN is then added and the process continues. Rubio et al. [20] proposed a framework called Hydra CNN which uses a pyramid of image patches extracted at multiple scales to perform the final density prediction. All these methods have reported good results for the UCSD dataset. However, to the best of our knowledge, temporal dependencies have not been explicitly exploited by deep learning models for crowd counting. These CNN-based methods simply treat the video sequences in the UCSD dataset as a set of still images without considering their temporal dependencies.

C. Zhang等[32]提出了第一个基于CNN的人群计数方法。在这个工作以后，Y. Zhang等[34]提出了一种多列CNN架构，使得输出图像可以是任意大小或分辨率。多列CNN架构也使用了一种不同的方法计算人群密度。Walach和Wolf[29]提出了一种分阶段的方法，将模型训练分阶段进行。在梯度提升方法的思想中，一次加入一个CNN，这样每个新的CNN训练后可以估计前面预测的残差误差。在训练了第一个CNN之后，第二个CNN的训练损失函数是目前的估计与学习目标之间的差异。然后加入第三个CNN，这个过程就这样继续下去。Rubio等[20]提出一个框架称为Hydra CNN，使用了不同尺度下提取出的图像块金字塔，以进行最终的密度估计。所有这些方法都在UCSD数据集上给出了很好的结果。但是，据我们所示，利用深度学习模型的人群计数尚未加入时间依赖关系。这些基于CNN的方法只是将UCSD数据集中的视频序列作为静止图像的集合，没有考虑其时域依赖关系。

### 2.2. Density map regression for crowd counting 密度图回归进行人群计数

Lempitskey and Zisserman [16] proposed a method to change the target of regression from a single crowd count to a crowd density map. We note that crowd density is more informative than crowd count, since the former also includes location information of the crowd. With a crowd density map, the crowd count of any given region can be estimated easily. The crowd count of the whole image is simply the integral of the density function over the entire image. All CNN-based methods mentioned above have used the crowd density map as the regression target.

Lempitskey和Zisserman[16]提出了一种方法，将回归的目标从人群计数，变为人群密度图。人群密度比人群数量更富含信息量，因为前者还包含人群的位置信息。在已有人群密度图的情况下，任何区域的人群数量可以很容易的得到估计。整图的人群计数就是密度函数在整幅图像上的积分。上述提到的所有基于CNN的方法都将人群密度图作为其回归目标。

### 2.3. ConvLSTM for spatiotemporal modeling 利用ConvLSTM进行时空建模

Recurrent neural networks (RNNs) have been applied successfully to various sequence learning tasks [27]. The incorporation of long short-term memory (LSTM) cells enables RNNs to exploit longer-term temporal dependencies. By extending the fully connected LSTM (FC-LSTM) to have convolutional structures in both the input-to-state and state-to-state connections, Shi et al. [25] proposed the ConvLSTM model for precipitation nowcasting which is a spatiotemporal forecasting problem. The ConvLSTM layer not only preserves the advantages of FC-LSTM but is also suitable for spatiotemporal data due to its inherent convolutional structures.

循环神经网络(RNN)已经成功应用于各种序列学习任务[27]。使用了长短期记忆(LSTM)单元使得RNN可以利用更长期的时域依赖关系。在将input-to-state和state-to-state连接的全连接LSTM(FC-LSTM)拓展到卷积结构后，Shi等[25]提出了ConvLSTM模型，可以进行降雨量的即时预测，这是一个时空预测问题。ConvLSTM层不仅保留了FC-LSTM的优势，而且由于其内在的卷积架构，还适用于时空数据。

ConvLSTM models have also proven effective for some other spatiotemporal tasks. Finn et al. [8] employed stacked ConvLSTMs to generate motion predictions. Villegas et al. [28] proposed a ConvLSTM-based method to model the spatiotemporal dynamics for pixel-level prediction in natural videos. Also, Y. Zhang et al. [33] applied network-in-network principles, batch normalization, residual connections, and ConvLSTMs to build very deep recurrent and convolutional structures for speech recognition.

ConvLSTM模型也在一些其他时空任务中被证明有效。Finn等[8]使用了stacked ConvLSTMs以生成运动预测。Villegas等[28]提出了一种基于ConvLSTM的方法来对自然视频中像素级预测的时空变化进行建模。同时，Y. Zhang等[33]使用了network-in-network原则，批归一化，残差连接和ConvLSTM构建了非常深的循环和卷积架构以进行语音识别。

## 3. Our Crowd Counting Method 我们的人群计数方法

### 3.1. Crowd density map 人群密度图

Following the previous work [16] as reviewed above, we also formulate crowd counting as a density map estimation problem. Compared to methods that give an estimated crowd count of the whole image as output, methods that give a crowd density map also provide location information about the crowd distribution which is useful for many applications.

在[16]之后（上面进行了回顾），我们也将人群计数问题表示为一个密度图估计问题。之前很多方法都是给出整幅图像的估计人群数量作为输出，而人群密度图还给出了人群分布的位置信息，这对很多应用都是有用的。

We assume that each training image $I_i$ is annotated with a set of 2D points $P_i = \{P_1, ..., P_{C(i)}\}$, where C(i) is the total number of people annotated. We define the groundtruth density map for supervised learning as a sum of Gaussian kernels each of which is centered at the location of one person. The ground-truth density map $F_i (p)$ for image $I_i$ can be defined as follows:

我们假设每幅训练图像$I_i$的标注为2D点$P_i = \{P_1, ..., P_{C(i)}\}$的集合，其中C(i)是标注的人总数。我们定义真值密度图为高斯核的和，每个高斯核的中心都在各自人的点上。图像$I_i$的真值密度图$F_i (p)$可以定义为：

$$∀p ∈ I_i , F_i (p) = \sum_{P∈P_i} N(p; P, σ^2 I_{2×2})$$(1)

where p denotes a pixel in image $I_i$, $P_i$ is the set of annotated points (usually corresponding to the positions of the human heads), $N(p; P, σ^2 I_{2×2})$ represents a normalized 2D Gaussian kernel evaluated at the pixel position p with its mean at the head position P and an isotropic 2 × 2 covariance matrix $I_{2×2}$ with variance $σ^2$.

其中p表示图像$I_i$的一个像素，$P_i$是标注点的集合（通常对应着热门人头位置），$N(p; P, σ^2 I_{2×2})$代表归一化的2D高斯核，中心在头部位置P，协方差矩阵$I_{2×2}$为各向同性的，方差$σ^2$。

For annotated points which are close to the image boundary, part of their probability mass will reside outside the image. Consequently, integrating the ground-truth density map over the entire image will not match the crowd count exactly. Fortunately, this effect can be neglected for most applications because the differences are generally small. Moreover, in many cases, a pedestrian who lies partially outside the image boundary should not be counted as a whole person.

对于靠近图像边缘的标注点，其概率分布的一部分会在图像之外。结果是，在整幅图像上对真值密度图进行积分不会严格的等于人群数量。幸运的是，这种影响对于大多数应用都可以忽略，因为差异一般来说很小。而且，在很多情况下，如果一个行人有一部分在图像边缘之外，也不应当被认为是一整个人。

Another subtlety that is worth noticing is that the images are often not captured with a bird’s-eye view and hence leads to perspective distortion. As a result, the pixels associated with different annotated points correspond to regions of different scales in the actual 3D scene. To overcome the effects due to perspective distortion, we need to normalize the crowd density map with the perspective map M(p). The pixel value in the perspective map represents the number of pixels in the image corresponding to one meter at that location in the real scene. In our experiments, we set σ = 0.3M(p) and then normalize the whole distribution to ensure that the sum of ground-truth density is equal to the crowd count of the image.

另一个值得注意的细微差别是，图像的捕捉方式通常不是鸟眼视角的，所以通常会有视角变形。结果是，不同标注点相关的像素在实际的3D场景中对应着不同尺度的区域。为克服视角变形造成的这种影响，我们需要对人群密度图使用视角图M(p)进行归一化。视角图中的像素值代表了真实场景中那个位置的一米长度对应的图像中的像素数。在我们的实验中，我们设σ = 0.3M(p)，然后对整个分布进行归一化，以确保真值密度的和等与图像中的人群数量。

### 3.2. ConvLSTM model

FC-LSTM has proven powerful for handling temporal correlations, but it fails to maintain structural locality. To exploit temporal correlations for video crowd counting, we propose a model based on ConvLSTM [25] to learn a density map. As an extension of FC-LSTM, ConvLSTM has convolutional structures in both the input-to-state and state-to-state connections. We can regard all the inputs, cell outputs, hidden states $H_1, ..., H_t$, and gates $i_t, f_t, o_t$ of the ConvLSTM as 3D tensors whose last two dimensions are spatial dimensions. The outputs of ConvLSTM cells depend on the inputs and past states of the local neighbors. The key equations of ConvLSTM are shown in (2) below, where '*' denotes the convolution operator, '◦' denotes the
Hadamard product, and σ(·) denotes the logistic sigmoid function:

FC-LSTM已经证明了在处理时域关联时非常好用，但却不能保持结构上的局部性。为利用视频中的时域关联进行人群计数，我们提出一种基于ConvLSTM[25]的模型来学习密度图。作为FC-LSTM的拓展，ConvLSTM在input-to-state和state-to-state连接中都有卷积结构。我们可以将所有输入、单元输出、隐藏状态$H_1, ..., H_t$，和ConvLSTM的门$i_t, f_t, o_t$视作3D张量，其最后两个维度是空间维度。ConvLSTM单元的输出依赖输入的状态和局部邻域的过去状态。ConvLSTM的关键公式如式(2)所示，其中'*'表示卷积运算，'◦'表示哈达玛内积，σ(·)表示logistic sigmoid函数：

$$i_t = σ(W_{xi} ∗ X_t + W_{hi} ∗ H_{t−1} + W_{ci} ◦ C_{t−1} + b_i),$$
$$f_t = σ(W_{xf} ∗ X_t + W_{hf} ∗ H_{t−1} + W_{cf} ◦ C_{t−1} + b_f ),$$
$$C_t = f_t ◦ C_{t−1} + i_t ◦ tanh(W_{xc} ∗ X_t + W_{hc} ∗ H_{t−1} + b_c),$$
$$o_t = σ((W_{xo} ∗ X_t + W_{ho} ∗ H_{t−1} + W_{co} ◦ C_t + b_o),$$
$$H_t = o_t ◦ tanh(C_t).$$(2)

Figure 1 shows our ConvLSTM model for crowd counting where each building block involves a ConvLSTM. 图1给出了我们人群计数的ConvLSTM模型，其中每个模块都是一个ConvLSTM。

The inputs $X_{1:t} = X_1, ..., X_t$ are consecutive frames of a video and the cell outputs $C_1, ..., C_t$ are the estimated density maps of the corresponding frames. If we remove the connections between ConvLSTM cells, we can regard each ConvLSTM cell as a CNN model with gates. We set all the input-to-state and state-to-state kernels to size 5 × 5 and the number of layers to 4. To relate the feature maps to the density map, we adopt filters all of size 1 × 1. We use the Euclidean distance to measure the difference between the estimated and ground-truth density maps. So we define the loss function L(θ) between the estimated density map $F(X_{1:t}; θ)$ and the ground-truth density map $D_t$ as follows:

输入$X_{1:t} = X_1, ..., X_t$是一段视频的连续帧，单元输出$C_1, ..., C_t$为对应帧的估计密度图。如果我们将ConvLSTM单元之间的连接移除掉，我们可以将每个ConvLSTM单元视为带有门的CNN模型。我们将所有的input-to-state和state-to-state核的大小设为5×5，层数设为4。为将特征图关联到密度图，我们使用的滤波器大小都是1×1。我们使用Euclidean距离来度量密度图估计值和真值之间的差异。所以我们定义估计的密度图$F(X_{1:t}; θ)$和真值密度图$D_t$之间的损失函数L(θ)为：

$$L(θ) = \frac{1}{2T} \sum_{t=1}^T ||F(X_{1:t}; θ) − D_t||_2^2$$(2)

where T is the length of the video clip and θ denotes the parameter vector. 其中T为视频片段的长度和θ为参数向量。

### 3.3. From ConvLSTM to bidirectional ConvLSTM

Inspired by [11, 33], we further extend the ConvLSTM model to a bidirectional ConvLSTM model which can access long-range information in both directions. 受[11,33]启发，我们进一步将ConvLSTM拓展为双向ConvLSTM模型，可以在两个方向上都获取长期信息。

Figure 2 depicts the bidirectional ConvLSTM model for crowd counting. Its inputs and outputs are the same as those in the ConvLSTM model. It works by computing the forward hidden sequence $\vec h$, backward hidden sequence $\overleftarrow h$, and output sequence by iterating backward from t = T to t = 1, iterating forward from t = 1 to t = T , and then updating the output layer. If we denote the state updating function in (2) as $H_t, C_t = ConvLSTM(X_t , H_{t−1}, C_{t−1})$, the equation of bidirectional ConvLSTM can be written as follows:

图2就是人群计数的双向ConvLSTM模型，其输入和输出与ConvLSTM模型相同。模型计算的是前向隐藏序列$\vec h$，和反向隐藏序列$\overleftarrow h$，通过从t=T到t=1反向迭代，从t=1到t=T正向迭代来输出序列，然后更新输出层。如果我们将(2)中的状态更新函数表示为$H_t, C_t = ConvLSTM(X_t , H_{t−1}, C_{t−1})$，那么双向ConvLSTM方程可以写成下式：

$$\vec H_t, \vec C_t =ConvLSTM(X_t, \vec H_{t-1}, \vec C_{t-1})$$
$$\overleftarrow H_t, \overleftarrow C_t =ConvLSTM(X_t, \overleftarrow H_{t+1}, \overleftarrow C_{t+1})$$
$$y_t = concat(\vec H_t, \overleftarrow H_t)$$(4)

where $Y_t$ is the output at timestamp t. 其中$Y_t$是时间戳t时的输出。

Y. Zhang et al. [33] found that bidirectional ConvLSTM consistently outperforms its unidirectional counterpart in speech recognition. In the next section, we also compare bidirectional ConvLSTM with the original ConvLSTM for crowd counting using different datasets. Y. Zhang等[33]发现，双向ConvLSTM在语音识别中一直超过单向ConvLSTM。在下一节，我们还比较了双向ConvLSTM与原始ConvLSTM在不同数据集上进行人群计数的表现。

### 3.4. ConvLSTM-nt: a degenerate variant of ConvLSTM for comparison

To better understand the effectiveness of exploiting temporal information, we propose a degenerate variant of ConvLSTM, called ConvLSTM with no temporal information (ConvLSTM-nt), by removing all connections between the ConvLSTM cells. ConvLSTM-nt can be seen as a CNN model with gates. The parameters of ConvLSTM-nt are the same as those of ConvLSTM introduced above. The structure of ConvLSTM-nt is shown in Figure 3.

为更好的理解利用时域信息的效果，我们提出了一种ConvLSTM的蜕化变体，称为没有时域信息的ConvLSTM(ConvLSTM-nt)，去除了ConvLSTM单元之间的所有连接。ConvLSTM-nt可以视为一种带有门的CNN模型。ConvLSTM-nt的参数和上面介绍的ConvLSTM是一样的。ConvLSTM-nt的结构如图3所示。

All our three models have 4 layers, with 128, 64, 64 and 64 hidden states respectively in the four ConvLSTM layers. For the training scheme, we train all models using the TensorFlow library, optimizing to convergence using ADAM [14] with the suggested hyperparameters in TensorFlow.

我们所有的三个模型都是4层，在4个ConvLSTM层中分别包含128，64，64和64个隐藏状态。我们用TensorFlow训练所有模型，使用ADAM[14]优化至收敛，超参数使用TensorFlow默认的设置。

In the experiments to be reported in the next section, whenever the dataset consists of still images not forming video sequences, both the original ConvLSTM and our bidirectional extension cannot be used but only ConvLSTM-nt will be used.

在下一节给出的实验中，只要数据集包含了静止图像，没有形成视频序列，那么原始ConvLSTM和我们的双向拓展就都不能应用，只能用ConvLSTM-nt进行处理。

## 4. Experiments 实验

We conduct comparative study using four annotated datasets which include the UCF CC 50 dataset [12], UCSD dataset [4], Mall dataset [7], and WorldExpo’10 dataset [32,31]. Some statistics of these datasets are summarized in Table 1. We also conduct experiments in the transfer learning setting by using one of the UCSD and Mall datasets as the source domain and the other one as the target domain.

我们使用4个标注数据集进行对比实验，包括UCF CC 50数据集[12]，UCSD数据集[4]，Mall数据集[7]和WorldExpo'10数据集[32,31]。这些数据集的一些统计如表1所示。我们还在迁移学习的设置下进行了实验，使用了UCSD和Mall数据集的其中一个作为源，另一个作为目标。

### 4.1. Evaluation metric 评估标准

For crowd counting, the mean absolute error (MAE) and mean squared error (MSE) are the two most commonly used evaluation metrics. They are defined as follows: 对于人群计数，MAE和MSE是最经常使用的两种评估标准：

$$MAE = \frac {1}{N} \sum_{i=1}^N |p_i - \hat p_i|, MSE = \sqrt {\frac{1}{N} \sum_{i=1}^N (p_i - \hat p_i)^2}$$

where N is the total number of frames used for testing, $p_i$ and $p̂_i$ are the true number and estimated number of people in frame i respectively. As discussed above, $p̂_i$ is calculated by summing over the estimated density map over the entire image. 其中N是用于测试的帧的总数，$p_i$和$p̂_i$分别是帧i中的人数的真实值和估计值。如上所述，$p̂_i$的计算是在整幅图像中对估计的密度图进行积分。

Table 1. Statistics of the four datasets

Dataset | Resolution | Color | Num | FPS | Max | Min | Average | Total
--- | --- | --- | --- | --- | --- | --- | --- | ---
UCF_CC_50 | different | Gray | 50 | Images | 4543 | 94 | 1278.5 | 63974
UCSD | 158×238 | Gray | 2000 | 10 | 46 | 11 | 24.9 | 49885
Mall | 640×480 | RGB | 2000 | - | 53 | 11 | 31.2 | 62315
WorldExpo | 576×720 | RGB | 3980 | 50 | 253 | 1 | 50.2 | 199923

### 4.2. UCF CC 50 dataset

The UCF CC 50 dataset was first introduced by Idress et al. [12]. It is a very challenging dataset because it contains only 50 images of different resolutions, different scenes, and extremely high crowd density. In particular, the number of pedestrians ranges between 94 and 4,543 with an average of 1,280. Annotations of all the 63,794 people in all 50 images are available in the dataset. Since the 50 images have no temporal correlation between them, we cannot demonstrate the advantage of exploiting temporal information. So only the ConvLSTM-nt variant is applied on this dataset. The goal here is to show that our model can still give very good results for such extremely dense crowd images even though temporal information is not available.

UCF CC 50数据集由Idress等[12]首先提出，这是一个非常有挑战性的数据集，因为包含了50幅不同分辨率、不同场景、极高人群密度的图像。特别是，行人的数量从94到4543，平均1280人。50幅图像中的所有63794个人的标注都是可用的。由于这50幅图像之间没有时间关联，我们不能展现使用时域信息的优势，所以在这个数据集上只使用了ConvLSTM-nt算法。这里的目标是说明，即使是没有时域信息，我们的模型对于如此极度密集的人群图像也能给出很好的结果。

Following the setting in [12], we split the dataset randomly and perform 5-fold cross validation. To handle different resolutions, we randomly crop patches of size 72 × 72 from each image for training and testing. As for the overlapping patches in the test set, we calculate the density at each pixel by averaging the overlapping patches.

我们采用[12]中的设置，将数据集随机分割，进行5部分交叉验证。为处理不同分辨率的问题，我们随机从每幅图像中剪切出72×72大小的图像块进行训练和测试。对于测试集的重叠部分，我们通过平均重叠块的值，计算在每个像素点上计算密度。

We compare our method with six existing methods on the UCF CC 50 dataset. The results are shown in Table 2. Rodriguez et al. [23] adopted the density map estimation in detection-based methods. Lempitsky et al. [16] extracted 800 dense SIFT features from the input image and learned a density map with the proposed MESA distance (where MESA stands for Maximum Excess over SubArrays). Idress et al. [12] estimated the crowd count by multi-source features which include SIFT and head detection. The methods proposed by C. Zhang et al. [32], Y. Zhang et al. [34], and Shang et al. [24] are all CNN-based methods. Shang et al. [24] used a model pre-trained on the WorldExpo dataset as initial weights and yielded the best MAE. However, when considering only methods that do not use additional data for training, our ConvLSTM-nt model achieves the lowest MAE and MSE.

我们在UCF CC 50数据集上与6种现有的方法进行了比较。结果如表2所示。Rodriguez等[23]采用了基于检测的方法进行密度图估计。Lempitsky等[16]从输入图像中提取出了800维密集SIFT特征，利用提出的MESA距离学习了一个密度图（MESA的意思是Maximum Excess over SubArrays）。Idress等[12]通过多源特征来估计人群数量，包括SIFT和人头检测。C. Zhang等[32]、Y. Zhang等[34]和Shang等[24]提出的方法都是基于CNN的方法。Shang等[24]使用了在WorldExpo数据集上预训练的模型作为初始权重，得到了最佳的MAE。但是，如果只考虑没有使用额外数据训练的方法，我们的ConvLSTM-nt模型的MAE和MSE是最低的。

Some results obtained by ConvLSTM-nt are shown in Figure 4. Although the images have wide variations in the background and crowd density, ConvLSTM-nt is quite robust in producing reasonable density maps and hence the overall crowd counts.

ConvLSTM-nt得到的一些结果如图4所示。虽然图像在背景和人群密度上变化很大，ConvLSTM-nt都能很稳健的给出不错的密度图，也就是人群数量总数。

Table 2. Results of different methods on the UCF CC 50 dataset. It should be noticed that Shang et al. [24] used additional data for training, so it is not fair to compare its result with the others directly.

Method | MAE | MSE
--- | --- | ---
Head detection [23] | 655.7 | 697.8
Density map + MESA [16] | 493.4 | 487.1
Multi-source features [12] | 419.5 | 541.6
Crowd CNN [32] | 467.0 | 498.5
Multi-column CNN [34] | 377.6 | 509.1
ConvLSTM-nt | 284.5 | 297.1
Shang et al. [24] | 270.3 | -

Figure 4. Results for four test images from the UCF CC 50 dataset. For each example, we show the input image (left), ground-truth density map (middle), and density map obtained by ConvLSTM-nt (right).

### 4.3. UCSD dataset

The UCSD dataset [4] contains a 2,000-frame video of pedestrians on a walkway of the UCSD campus captured by a stationary camera. The video was recorded at 10 fps with dimension 238 × 158. The labeled ground truth marks the center of each pedestrian. The ROI and the perspective map are provided in the dataset.

UCSD数据集[4]包含2000帧视频的行人，由一个静止的摄像机在UCSD校园的人行道上拍摄，视频帧率为10fps，分辨率为238×158，标注的真值标记了每个行人的中心。数据集给出了ROI和视角图。

Using the same setting as in [4], we use frames 601-1400 as the training data and the remaining 1,200 frames as test data. The provided perspective map is used to adjust the ground-truth density map by setting σ = 0.3M(p). The values of the pixels outside the ROI are set to zero.

使用[4]中相同的设置，我们使用601-1400帧作为训练数据，剩下的1200帧作为测试数据。给出的视角图用于调整真值密度图，即设σ = 0.3M(p)。ROI外的像素值设为0。

The results of different methods are shown in Table 3. [4, 7, 6] are traditional methods which give the crowd count for the whole image. [16, 21] are density map regression methods using handcrafted features and regression algorithms such as linear regression and random forest regression. Most state-of-the-art methods are based on CNNs [29, 32, 20, 34]. Bidirectional ConvLSTM achieves comparable MAE and MSE with these methods. From the results of ConvLSTM-nt, unidirectional ConvLSTM, and bidirectional ConvLSTM , we can draw the conclusion that temporal information can boost the performance for this dataset.

不同方法的结果比较如表3所示。[4,7,6]是传统方法的结果，给出了整幅图的人群计数。[16,21]是密度图回归方法，使用手工设计的特征和回归算法来计算，如线性回归和随机森林回归。多数目前最好的方法是基于CNN的[29,32,20,34]。双向ConvLSTM与这些方法取得了可比的MAE和MSE结果。从ConvLSTM-nt、单向ConvLSTM和双向ConvLSTM的结果，我们可以得出结论，时域信息可以在这个数据集上提高表现。

Table 3. Results of different methods on the UCSD dataset

Method | MAE | MSE
--- | --- | ---
Gaussian process regression [4] | 2.24 | 7.97
Ridge regression [7] | 2.25 | 7.82
Cumulative attribute regression [6] | 2.07 | 6.90
Density map + MESA [16] | 1.70 | -
Count forest [21] | 1.60 | 4.40
Crowd CNN [32] | 1.60 | 3.31
Multi-column CNN [34] | 1.07 | 1.35
Hydra CNN [20] | 1.65 | -
CNN boosting [29] | 1.10 | -
ConvLSTM-nt | 1.73 | 3.52
ConvLSTM | 1.30 | 1.79
Bidirectional ConvLSTM | 1.13 | 1.43

Figure 5 shows two illustrative examples. We can see that bidirectional ConvLSTM produces density maps that are closest to the ground truth. While ConvLSTM-nt can give a rough estimation, ConvLSTM and bidirectional ConvLSTM are more accurate in the fine details. 图5给出了2个说明性的例子。我们可以看出，双向ConvLSTM得到的密度图估计是与真值最接近的。ConvLSTM-nt可以给出一个粗略的估计，但ConvLSTM和双向ConvLSTM在细节上更加准确。

Figure 5. Results for two test video frames from the UCSD dataset. For each example, we show the input video frame, ground-truth density map, and density maps obtained by the three variants of our method.

### 4.4. Mall dataset

The Mall dataset was provided by Chen et al. [7] for crowd counting. It was captured in a shopping mall using a publicly accessible surveillance camera. This video contains 2,000 annotated frames of moving and stationary pedestrians with more challenging lighting conditions and glass surface reflections. The ROI and the perspective map are also provided in the dataset.

Mall数据集由Chen等[7]提出，进行人群计数任务，这是用购物中心的公共监控摄像头捕捉的视频，视频包括2000帧标注，都是移动或静止的行人，其光照条件和镜面反光情况都非常有挑战性。数据集也给出了ROI和视角图。

Following the same setting as [7], we use the first 800 frames for training and the remaining 1,200 frames for testing. We perform comparison against Gaussian process regression [4], ridge regression [7], cumulative attribute ridge regression [6], and random forest regression [21]. Bidirectional ConvLSTM achieves state-of-the-art performance with respect to both MAE and MSE. The results are shown in Table 4, which also demonstrates the effectiveness of exploiting temporal information.

采用与[7]中同样的设置，我们使用前800帧进行训练，剩余的1200帧进行测试。我们与Gauss过程回归[4]、脊回归[7]、累计属性脊回归[6]和随机森林回归[21]进行了比较。双向ConvLSTM在MAE和MSE上取得了目前最好的性能。结果如表4所示，也表明了使用时域信息的有效性。

Table 4. Results of different methods on the Mall dataset

Method | MAE | MSE
--- | --- | ---
Gaussian process regression [4] | 3.72 | 20.1
Ridge regression [7] | 3.59 | 19.0
Cumulative attribute regression [6] | 3.43 | 17.7
Count forest [21] | 2.50 | 10.0
ConvLSTM-nt | 2.53 | 11.2
ConvLSTM | 2.24 | 8.5
Bidirectional ConvLSTM | 2.10 | 7.6

### 4.5. WorldExpo dataset

The WorldExpo dataset was introduced by C. Zhang et al. [32, 31]. This dataset contains 1,132 annotated video sequences captured by 108 surveillance cameras, all from the 2010 Shanghai World Expo. The annotations of 199,923 pedestrians in 3,980 frames include the location of the center of each human head. The test set contains five separate video sequences each of which has 120 annotated frames. The regions of interest (ROIs) are provided for these five test scenes. The perspective maps are also provided.

WorldExpo数据集由C. Zhang等[32,31]提出，这个数据集包含1132个标注的视频序列，由108个监控摄像头捕捉到，都是2010上海世博会的。标注的3980帧的199923个行人包含每个人头的中心位置。测试集包含5个单独的视频帧，每个包含120个标注的帧。在这5个测试场景中给出了感兴趣区域ROI，也给出了视角图。

For fair comparison, we follow the work of the multi-column CNN to generate the density map according to the perspective map with the relation δ = 0.2M(x), where M (x) denotes the number of pixels in the image representing one square meter at location x. Table 5 compares our model and its variants with the state-of-the-art methods. We use MAE as the evaluation metric, as suggested by the author of [32]. On average, bidirectional ConvLSTM achieves the lowest MAE. It also gives the best result for scene 5.

为公平比较，我们按照多列CNN的方式，根据视角图生成密度图，关系为δ = 0.2M(x)，其中M(x)表示在位置x上每平方米面积上图像中包含的像素数。表5比较了我们的模型及其变体，以及目前最好的方法的结果。我们使用MAE作为评估标准，[32]的作者这样建议。平均来说，双向ConvLSTM结果的MAE最低。在场景5中也得到了最好的结果。

Table 5. Results of different methods on the WorldExpo dataset

Method | Scene 1 | Scene 2 | Scene 3 | Scene 4 | Scene 5 | Average
--- | --- | --- | --- | --- | --- | ---
LBP features + ridge regression	| 13.6 | 59.8 | 37.1 | 21.8 | 23.4 | 31
Deep CNN [32] |	9.8 | 14.1 | 14.3 | 22.2 | 3.7 | 12.9
Multi-column CNN [34] |	3.4 | 20.6 | 12.9 | 13 | 8.1 | 11.6
ConvLSTM-nt | 8.6 | 16.9 | 14.6 | 15.4 | 4 | 11.9
ConvLSTM | 7.1 | 15.2 | 15.2 | 13.9 | 3.5 | 10.9
Bidirectional ConvLSTM | 6.8 | 14.5 | 14.9 | 13.5 | 3.1 | 10.6

We show the estimation results for the five test scenes obtained by our models in Figure 6. The crowd count curves are shown in different colors for the ground truth (black) and the estimation results of ConvLSTM-nt (red), ConvLSTM (green), and bidirectional ConvLSTM (blue). We note that the five scenes differ significantly in the scene type, crowd density, and change in crowd count over time.

我们在图6中给出了我们的模型在5个测试场景中的估计结果。人群计数曲线以不同的颜色显示，分别是，真值（黑色），ConvLSTM-nt估计结果（红），ConvLSTM（绿），双向ConvLSTM（蓝）。我们注意这5种场景在场景类别、人群密度和随着时间人群数量的变化上都非常不同。

Figure 6. Density map estimation examples from the WorldExpo dataset (best viewed in color). In each row, the left one shows one video frame from the test scene and the right one shows the estimation results of that scene, where the x-axis represents the frame index and the y-axis represents the crowd count.

From Table 5 and Figure 6, we can see that bidirectional ConvLSTM outperforms ConvLSTM and ConvLSTM outperforms ConvLSTM-nt in most cases (scene 1,2,4,5), which gives evidence to the effectiveness of incorporating temporal information for crowd counting. As for scene 3, a closer investigation reveals a potential problem with the labels provided in this test scene. Figure 7 illustrates the problem. There are in fact many people walking under the white ceiling of the covered walkway as we can see their moving legs clearly when playing the video, but only two red dots are provided in the frame because the heads of most of the people there are hidden. Spatiotemporal models tend to count them since motion is detected when exploiting the temporal information, but unfortunately they are not annotated in the provided labels.

从表5和图6中，我们可以看到，双向ConvLSTM超过了ConvLSTM，ConvLSTM在大部分情况下（场景1，2，4，5）超过了ConvLSTM-nt，这说明利用了时域信息进行人群计数是有效的。至于场景3，仔细研究会发现，在这个测试场景中给出的标签是有潜在问题的。图7描述了这个问题，实际上在那个白色屋檐下的走廊种走着很多人，这在播放视频的时候可以看到很多走动的腿，但是在这一帧中只给出了2个红点，因为大多数人是隐藏的。时空模型会倾向于对其计数，因为在发掘时域信息时，会检测到运动，但不幸的是在给定的标签中并没有对其进行标注。

Figure 7. A video frame from test scene 3 of the WorldExpo dataset. The region outlined in green indicates the ROI and the red dots mark the positions of the heads.

### 4.6. Transfer learning experiments 迁移学习的实验

To demonstrate the generalization capability of our model, we conduct some experiments in the transfer learning setting. Specifically, we compare with some previous methods that have also been evaluated in the transfer learning setting using the UCSD and Mall datasets, which were both captured using stationary cameras. As shown in Figure 8, the two datasets are quite different in terms of the scene type (outdoor for UCSD but indoor for Mall), crowd density, frame rate, and camera angle, among others.

为展示我们模型的泛化能力，我们在迁移学习的设置下进行了几项实验。特别的，我们与之前的方法进行了比较，也是在迁移学习的设置下进行了评估，使用的是UCSD和Mall数据集，这两个都是用静止的摄像头拍摄的视频。如图8所示，这两个数据集在场景类型（UCSD为室外，Mall为室内）、人群密度、帧率和摄像头角度以及其他方面的特征都非常不一样。

Figure 8. The UCSD and Mall datasets used for transfer learning experiments. Left column: UCSD dataset; right column: Mall dataset. Upper row: input images with annotations; lower row: density maps.

We consider two transfer learning tasks by using one dataset as the source domain and the other one as the target domain. For each task, 800 frames are used for training the model and 50 frames of the other dataset are used as the adaptation set. Following the same setting as [5, 30, 18], we use MAE as the evaluation metric. Table 6 presents the results for different methods on the two transfer learning tasks. Bidirectional ConvLSTM achieves state-of-the-art performance in both transfer learning tasks. We note that the performance of our method in the transfer learning setting is even better than many approaches tested on the standard, non-transfer-learning setting. For instance, with 800 frames of the UCSD dataset for training and 50 frames of the Mall dataset for adaptation, bidirectional ConvLSTM can achieve an MAE of 2.63, which outperforms many algorithms using 800 frames of the Mall dataset for training, according to Table 4. We can draw the conclusion that bidirectional ConvLSTM has good generalization capability. Once trained on one dataset, the learning experience can be transferred easily to a new dataset which consists of only very few video frames for adaptation.

我们还考虑了两个迁移学习任务，即使用一个数据集作为源，另一个作为目标。对每个任务，都使用了800帧视频进行训练模型，50帧另一个数据集的视频作为适应集。我们使用的设置与[5,30,18]相同，使用MAE作为评估标准。表6给出了不同方法在这两个迁移学习任务中的结果。双向ConvLSTM在两个迁移学习的任务中都取得了目前最好的结果。我们注意到，我们方法在迁移学习设置下的性能，甚至比很多方法在标准、非迁移学习设置的情况下要好。比如，在使用800帧UCSD数据集进行训练，和50帧Mall数据集进行适应的情况下，双向ConvLSTM得到的MAE为2.63，这超过了很多算法以800帧Mall数据集进行训练的结果，表4中有几个这种例子。我们可以得出结论，双向ConvLSTM有很好的泛化性能。一旦在一个数据集上得到了训练，其学习到的经验可以很容易的迁移到新的数据集中，而只需要新数据集很少的视频帧进行适应。

Table 6. Results of transfer learning across datasets with MAE as evaluation metric. FA: feature alignment; HGP: hierarchical Gaussian process; GPA: Gaussian process adaptation; GPTL: Gaussian process with transfer learning.

Method | UCSD to Mall | Mall to UCSD
--- | --- | ---
FA [5] | 7.47 | 4.44
HGP [30] | 4.36 | 3.32
GPA [18] | 4.18 | 2.79
GPTL [18] | 3.55 | 2.91
Bidirectional ConvLSTM | 2.63 | 1.82

## 5. Conclusion

In this paper, we have pursued the direction of spatiotemporal modeling for improving crowd counting in videos. By jointly capturing both spatial and temporal dependencies, we overcome a major limitation of the recent CNN-based crowd counting methods and advance the state of the art. Specifically, our models outperform existing crowd counting methods on the UCF CC 50 dataset, Mall dataset, and WorldExpo dataset, and achieve comparable results on the UCSD dataset. The superior result on the UCF CC 50 dataset shows that our model can still perform well on extremely dense crowd images even when temporal information is not available. As for the other three datasets, the results show that explicitly exploiting temporal information has a clear advantage. Finally, the last set of experiments shows that our model is robust under the transfer learning setting to generalize from previous learning experience.

在本文中，我们采用了时空网络建模来改进视频中人群计数的结果。通过同时捕捉空间和时间上的依赖关系，我们克服了最近的基于CNN的人群计数方法的一个主要局限，改进了目前最好的结果。特别的，我们的模型，在UCF CC 50数据集、Mall数据集、WorldExpo数据集上，超过了现有的人群计数的方法，在UCSD数据集上也得到了可比的结果。在UCF CC 50数据集上的优异结果说明，我们的模型即使在时域信息不可用的时候，也可以在极度密集的人群图像上得到了很好的结果。在其他三个数据集上的结果表明，利用时域信息有明显的优势。最后，最后一组实验表明，我们的模型在迁移学习设置中也非常稳健，可以从之前学习到的经验中很好的泛化。

In the future, we are going to extend our model to deal with the active learning setting for crowd counting. We will output an additional confidence map and actively query the labeler to label only the less confident regions, which would greatly alleviate the expensive labeling effort for crowd counting in videos.

将来我们会拓展我们的模型，处理主动学习设置中的人群计数。我们会输出一个额外的置信度图，主动查询标注，以标注那些置信度没那么高的区域，这会极大的缓解视频中人群计数的昂贵的标注工作量。