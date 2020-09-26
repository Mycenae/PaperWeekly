# FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks

Eddy Ilg et. al. University of Freiburg, Germany

## 0. Abstract

The FlowNet demonstrated that optical flow estimation can be cast as a learning problem. However, the state of the art with regard to the quality of the flow has still been defined by traditional methods. Particularly on small displacements and real-world data, FlowNet cannot compete with variational methods. In this paper, we advance the concept of end-to-end learning of optical flow and make it work really well. The large improvements in quality and speed are caused by three major contributions: first, we focus on the training data and show that the schedule of presenting data during training is very important. Second, we develop a stacked architecture that includes warping of the second image with intermediate optical flow. Third, we elaborate on small displacements by introducing a sub-network specializing on small motions. FlowNet 2.0 is only marginally slower than the original FlowNet but decreases the estimation error by more than 50%. It performs on par with state-of-the-art methods, while running at interactive frame rates. Moreover, we present faster variants that allow optical flow computation at up to 140fps with accuracy matching the original FlowNet.

FlowNet证明了光流估计可以视为一个学习问题。但是，光流质量的目前最好性能目前仍然是传统方法。特别是在小型偏移和真实世界数据上，FlowNet无法与变分方法相媲美。本文中，我们推动了端到端的光流估计学习，使其效果进一步提升。质量和速度的巨大改进，是因为三个因素：第一，我们聚焦在训练数据，证明了在训练过程中呈现数据的方案是非常重要的。第二，我们提出了一种层叠架构，用中间光流包含了第二幅图像的形变。第三，我们精心处理了小的偏移，引入了子网络专门处理小的运动。FlowNet 2.0只比原始FlowNet慢一点，但将估计误差降低了50%还多。与目前最好的方法性能接近，运行帧率接近交互频率。而且，我们提出了更快的变体，光流计算速度可以达到140fps，准确率与原始的FlowNet接近。

## 1. Introduction

The FlowNet by Dosovitskiy et al. [11] represented a paradigm shift in optical flow estimation. The idea of using a simple convolutional CNN architecture to directly learn the concept of optical flow from data was completely disjoint from all the established approaches. However, first implementations of new ideas often have a hard time competing with highly fine-tuned existing methods, and FlowNet was no exception to this rule. It is the successive consolidation that resolves the negative effects and helps us appreciate the benefits of new ways of thinking.

Dosovitskiy等提出的FlowNet给出光流估计的范式转换。使用一个简单的CNN来直接从数据中学习光流的想法，与现有的所有方法都没有关系。但是，新想法的首次实现，与高度精调的现有方法相比，通常都有一些困难的时光，FlowNet也没有例外。后续的巩固可以解决这些负面效果，使我们更加欣赏新的思路的好处。

At the same time, it resolves problems with small displacements and noisy artifacts in estimated flow fields. This leads to a dramatic performance improvement on real-world applications such as action recognition and motion segmentation, bringing FlowNet 2.0 to the state-of-the-art level.

同时，还解决了小的偏移和在估计的光流中含有噪声的问题。这给现实世界应用带来了很大的性能提升，如行为识别和运动分割，使FlowNet 2.0成为了目前最好的水平。

The way towards FlowNet 2.0 is via several evolutionary, but decisive modifications that are not trivially connected to the observed problems. First, we evaluate the influence of dataset schedules. Interestingly, the more sophisticated training data provided by Mayer et al. [19] leads to inferior results if used in isolation. However, a learning schedule consisting of multiple datasets improves results significantly. In this scope, we also found that the FlowNet version with an explicit correlation layer outperforms the version without such layer. This is in contrast to the results reported in Dosovitskiy et al. [11].

通向FlowNet 2.0的路是通过几个逐渐演化的，但却有决定性意义的改变形成的，与观察到的问题并不是没有关系的。首先，我们评估了数据集方案的影响。有趣的是，由Mayer等[19]给出的更加复杂的训练数据，如果单独使用，带来的效果反而要差一些。但是，包含多个数据集的学习方案，却可以显著提高效果。在这个范围内，我们还发现，带有显式的相关层的FlowNet版本，超过了没有这个层的版本。这与Dosovitskiy等[11]的结果形成了对比。

As a second contribution, we introduce a warping operation and show how stacking multiple networks using this operation can significantly improve the results. By varying the depth of the stack and the size of individual components we obtain many network variants with different size and runtime. This allows us to control the trade-off between accuracy and computational resources. We provide networks for the spectrum between 8fps and 140fps.

第二个贡献是，我们提出了一个变形运算，展示了使用这个运算叠加多个网络可以显著改进结果。通过变化叠加的深度和单个组件的大小，我们得到了很多网络变体，有着不同的大小和运行时间。这使我们可以控制准确率和计算资源的折中。我们所用的网络运行速度从8fps到140fps。

Finally, we focus on small, subpixel motion and real-world data. To this end, we created a special training dataset and a specialized network. We show that the architecture trained with this dataset performs well on small motions typical for real-world videos. To reach optimal performance on arbitrary displacements, we add a network that learns to fuse the former stacked network with the small displacement network in an optimal manner.

最后，我们聚焦在小的，亚像素的运动和真实世界数据。为此，我们创建了一种特殊的训练数据集，和一个专用的网络。我们展示了，用这个数据集训练的架构，对真实世界视频，在小的运动上表现良好。为在任意偏移上达到最佳性能，我们增加了一个网络，学习将之前叠加的小偏移网络，以一种最佳的方式进行融合。

The final network outperforms the previous FlowNet by a large margin and performs on par with state-of-the-art methods on the Sintel and KITTI benchmarks. It can estimate small and large displacements with very high level of detail while providing interactive frame rates.

最终的网络超过了FlowNet很多，在Sintel和KITTI基准测试中，与目前最好的方法性能相当。对于小的和大的偏移，都可以以很高的细节得到估计，同时保持可交互的帧率。

## 2. Related Work

End-to-end optical flow estimation with convolutional networks was proposed by Dosovitskiy et al. in [11]. Their model, dubbed FlowNet, takes a pair of images as input and outputs the flow field. Following FlowNet, several papers have studied optical flow estimation with CNNs: featuring a 3D convolutional network [31], an unsupervised learning objective [1, 34], carefully designed rotationally invariant architectures [29], or a pyramidal approach based on the coarse-to-fine idea of variational methods [21]. None of these methods significantly outperforms the original FlowNet.

Dosovitskiy等[11]提出了采用CNN的端到端的光流估计。其模型称为FlowNet，以一对图像为输入，输出光流场。根据FlowNet，几篇文章采用CNNs对光流估计进行了研究：[31]采用了一个3D CNN，[1,34]采用了一个无监督的学习目标，[29]仔细设计了旋转不变的架构，[21]采用了一种金字塔方法，基于变分方法的由粗到细的思想。这些方法都没有明显超过原始FlowNet的性能。

An alternative approach to learning-based optical flow estimation is to use CNNs for matching image patches. Thewlis et al. [30] formulate Deep Matching [32] as a convolutional network and optimize it end-to-end. Gadot & Wolf [13] and Bailer et al. [3] learn image patch descriptors using Siamese network architectures. These methods can reach good accuracy, but require exhaustive matching of patches. Thus, they are restrictively slow for most practical applications. Moreover, patch based approaches lack the possibility to use the larger context of the whole image because they operate on small image patches.

另一种基于学习的光流估计方法是，使用CNNs对图像对进行匹配。Thewlis等[30]将Deep Matching[32]表述为一个卷积网络，对其进行端到端的优化。Gadot & Wolf[13]和Bailer等[3]使用Siamese网络架构学习图像块描述子。这些方法可以达到很好的准确率，但需要图像块的穷举式匹配。因此，对于多数实际应用来说，运行太慢。而且，基于图像块的方法缺少使用更大的上下文的可能性，因此是在小的图像块上运算的。

Convolutional networks trained for per-pixel prediction tasks often produce noisy or blurry results. As a remedy, out-of-the-box optimization can be applied to the network predictions as a postprocessing operation, for example, optical flow estimates can be refined with a variational approach [11]. In some cases, this refinement can be approximated by neural networks: Chen & Pock [10] formulate reaction diffusion model as a CNN and apply it to image denoising, deblocking and superresolution. Recently, it has been shown that similar refinement can be obtained by stacking several convolutional networks on top of each other. This led to improved results in human pose estimation [18, 9] and semantic instance segmentation [23]. In this paper we adapt the idea of stacking multiple networks to optical flow estimation.

训练进行逐像素预测的CNNs，通常会产生含噪或模糊的结果。作为一个补救方法，现有的优化方法可以用到网络预测结果中，作为后处理的运算，比如，光流估计可以用变分方法进行精炼[11]。在一些情况中，这种精炼可以用神经网络近似：Chen & Pock[10]将reaction diffusion模型表述成CNN，将其应用到图像去噪，去格子和超分辨率中。最近，通过叠加几个CNNs，也证明了可以进行类似的精炼。这在人的姿态估计中也得到了改进的结果[18,9]和语义个体分割[23]。本文中，我们采用了这个思想，叠加多个网络进行光流估计。

Our network architecture includes warping layers that compensate for some already estimated preliminary motion in the second image. The concept of image warping is common to all contemporary variational optical flow methods and goes back to the work of Lucas & Kanade [17]. In Brox et al. [6] it was shown to correspond to a numerical fixed point iteration scheme coupled with a continuation method.

我们的网络架构包含变形层，对在第二幅图像中已经估计的初级运动进行补偿。图像变形的概念对所有当代光流变分方法都是很常见的，可以追溯到Lucas & Kanade的工作[17]。在Brox等[6]中，发现这对应着一个数值定点迭代与continuation方法结合到一起。

The strategy of training machine learning models on a series of gradually increasing tasks is known as curriculum learning [5]. The idea dates back at least to Elman [12], who showed that both the evolution of tasks and the network architectures can be beneficial in the language processing scenario. In this paper we revisit this idea in the context of computer vision and show how it can lead to dramatic performance improvement on a complex real-world task of optical flow estimation.

在一系列逐渐增加的任务中，训练机器学习模型的策略，被称为curriculum学习[5]。这个思想可以追溯到Elman[12]，他证明了任务的演化和网络架构的演化在语言处理的场景中都是有好处的。本文中，我们在计算机视觉中重温了这个思想，证明了在复杂的真实世界的光流估计任务中，也会带来很大的性能改进。

## 3. Dataset Schedules

High quality training data is crucial for the success of supervised training. We investigated the differences in the quality of the estimated optical flow depending on the presented training data. Interestingly, it turned out that not only the kind of data is important but also the order in which it is presented during training.

高质量的训练数据对于监督训练的成功是非常关键的。我们研究了给定训练数据的情况下，估计光流的质量的差异。有趣的是，并不只是数据类型重要，而且训练数据的顺序也重要。

The original FlowNets [11] were trained on the FlyingChairs dataset (we will call it Chairs). This rather simplistic dataset contains about 22k image pairs of chairs superimposed on random background images from Flickr. Random affine transformations are applied to chairs and background to obtain the second image and ground truth flow fields. The dataset contains only planar motions.

原始的FlowNets[11]是在FlyingChairs数据集训练的（我们称之为Chairs）。这个颇为简单的数据集包含大约22k图像对，图像内容是椅子在随机的背景图像上，背景图像是从Flickr上得到的。对椅子和背景使用了随机的仿射变换，以得到第二幅图像以及真值光流场。数据集只包含平面运动。

The FlyingThings3D (Things3D) dataset proposed by Mayer et al. [19] can be seen as a three-dimensional version of the FlyingChairs. The dataset consists of 22k renderings of random scenes showing 3D models from the ShapeNet dataset [24] moving in front of static 3D backgrounds. In contrast to Chairs, the images show true 3D motion and lighting effects and there is more variety among the object models.

由Mayer等[19]提出的FlyingThings3D (Things3D)数据集，可以认为是FlyingChairs的三维版本。这个数据集包含22k幅随机场景中展现3D模型的图像，3D模型是从ShapeNet数据集[24]中得到的，在静态的3D背景前面移动。与Chairs相比，图像展现出了真正的3D运动和光照效果，在目标模型上也有更多的可变性。

We tested the two network architectures introduced by Dosovitskiy et al. [11]: FlowNetS, which is a straightforward encoder-decoder architecture, and FlowNetC, which includes explicit correlation of feature maps. We trained FlowNetS and FlowNetC on Chairs and Things3D and an equal mixture of samples from both datasets using the different learning rate schedules shown in Figure 3. The basic schedule S_short (600k iterations) corresponds to Dosovitskiy et al. [11] except some minor changes. Apart from this basic schedule S_short, we investigated a longer schedule S_long with 1.2M iterations, and a schedule for fine-tuning S_fine with smaller learning rates. Results of networks trained on Chairs and Things3D with the different schedules are given in Table 1. The results lead to the following observations:

我们测试了Dosovitskiy等提出的两个网络架构[11]：FlowNetS，这是一个直接的编码器-解码器架构，和FlowNetC，包含特征图的直接相关。我们在Chairs、Things3D和两个数据集的等量样本混合上训练FlowNetS和FlowNetC，使用了不同的学习速率方案，如图3所示。基本方案S_short（600k次迭代）对应的是Dosovitskiy等[11]，有一些小的变化。除了这个基本方案S_short，我们研究了一个更长的方案S_long，有1.2M次迭代，以及一个精调的方案S_fine，学习速率更小。在Chairs和Things3D上采用不同方案训练的网络结果，如表1所示。从这些结果可以观察得到下面结论：

**The order of presenting training data with different properties matters**. Although Things3D is more realistic, training on Things3D alone leads to worse results than training on Chairs. The best results are consistently achieved when first training on Chairs and only then fine-tuning on Things3D. This schedule also outperforms training on a mixture of Chairs and Things3D. We conjecture that the simpler Chairs dataset helps the network learn the general concept of color matching without developing possibly confusing priors for 3D motion and realistic lighting too early. The result indicates the importance of training data schedules for avoiding shortcuts when learning generic concepts with deep networks.

**具有不同性质的数据送入训练的顺序对结果有影响**。虽然Things3D是更加真实的，在Things3D上单独训练却带来了比在Chairs上训练更坏的结果。首先在Chairs上训练，然后在Things3D上精调，总是能得到最佳的结果。这个方案也超过了在Chairs和Things3D混合训练的方案。我们推测，更简单的Chairs数据集帮助网络学习了色彩匹配的通用概念，而没有过早的产生可能的令人混淆的先验知识，包括3D运动和真实的光照。结果表明，训练数据方案的重要性，在使用深度网络学习通用概念时避免了捷径。

**FlowNetC outperforms FlowNetS**. The result we got with FlowNetS and S_short corresponds to the one reported in Dosovitskiy et al. [11]. However, we obtained much better results on FlowNetC. We conclude that Dosovitskiy et al. [11] did not train FlowNetS and FlowNetC under the exact same conditions. When done so, the FlowNetC architecture compares favorably to the FlowNetS architecture.

**FlowNetC比FlowNetS表现要好**。我们用FlowNetS和S_short得到的结果，对应着Dosovitskiy等[11]得到的结果。但是，我们用FlowNetC得到了更好的结果。我们总结，Dosovitskiy等[11]并没有在完全相同的情况下训练FlowNetS和FlowNetC。当这样做时，FlowNetC架构要比FlowNetS架构得到更好的结果。

**Improved results**. Just by modifying datasets and training schedules, we improved the FlowNetS result reported by Dosovitskiy et al. [11] by ∼ 25% and the FlowNetC result by ∼ 30%. 通过修改数据集和训练方案，我们将Dosovitskiy等[11]的FlowNetS结果改进了约～25%，FlowNetC结果改进了～30%。

In this section, we did not yet use specialized training sets for specialized scenarios. The trained network is rather supposed to be generic and to work well in various scenarios. An additional optional component in dataset schedules is fine-tuning of a generic network to a specific scenario, such as the driving scenario, which we show in Section 6.

在本节中，我们没有对专用的场景使用专用的训练集。训练的网络被认为是通用的，在各种场景中都可以很好的工作。数据集方案中另外的可选的部件是对一个具体场景精调一个通用网络，比如驾驶的场景，我们在第6节中有展示。

## 4. Stacking Networks

### 4.1. Stacking Two Networks for Flow Refinement

All state-of-the-art optical flow approaches rely on iterative methods [7, 32, 22, 2]. Can deep networks also benefit from iterative refinement? To answer this, we experiment with stacking multiple FlowNetS and FlowNetC architectures.

所有目前最好的光流方法都依赖于迭代方法[7, 32, 22, 2]。深度网络是否也能从迭代提炼中受益呢？为回答这个，我们堆叠多个FlowNetS和FlowNetC网络来进行实验。

The first network in the stack always gets the images I1 and I2 as input. Subsequent networks get I1, I2, and the previous flow estimate $w_i = (u_i, v_i)^⊤$, where i denotes the index of the network in the stack.

堆叠中的第一个网络永远将I1和I2作为输入。后续的网络的输入为I1，I2，和之前的光流估计$w_i = (u_i, v_i)^⊤$，其中i表示堆叠中网络的索引。

To make assessment of the previous error and computing an incremental update easier for the network, we also optionally warp the second image I_2(x, y) via the flow w_i and bilinear interpolation to $\tilde I_{2,i}(x, y) = I_2(x+ui, y+vi)$. This way, the next network in the stack can focus on the remaining increment between I1 and $\tilde I_{2,i}$. When using warping, we additionally provide $\tilde I_{2,i}$ and the error $e_i = ||\tilde I_{2,i} − I_1||$ as input to the next network; see Figure 2. Thanks to bilinear interpolation, the derivatives of the warping operation can be computed (see supplemental material for details). This enables training of stacked networks end-to-end.

为评估之前的误差，计算一个对于网络更容易的更新，我们还有选择的对第二幅图像I_2(x, y)通过光流w_i和双线性插值进行变形，得到$\tilde I_{2,i}(x, y) = I_2(x+ui, y+vi)$。这样，堆叠中的下一个网络可以聚焦在I1和$\tilde I_{2,i}$剩下的增量。当使用变形时，我们额外的提供了$\tilde I_{2,i}$和误差$e_i = ||\tilde I_{2,i} − I_1||$作为下一个网络的输入；见图2。多亏了双线性插值，变形操作的导数可以进行计算（详见附加材料）。这使得堆叠网络可以进行端到端的训练。

Table 2 shows the effect of stacking two networks, the effect of warping, and the effect of end-to-end training. We take the best FlowNetS from Section 3 and add another FlowNetS on top. The second network is initialized randomly and then the stack is trained on Chairs with the schedule S_long. We experimented with two scenarios: keeping the weights of the first network fixed, or updating them together with the weights of the second network. In the latter case, the weights of the first network are fixed for the first 400k iterations to first provide a good initialization of the second network. We report the error on Sintel train clean and on the test set of Chairs. Since the Chairs test set is much more similar to the training data than Sintel, comparing results on both datasets allows us to detect tendencies to over-fitting.

表2展示了堆叠两个网络的效果，变形的效果和端到端训练的效果。我们在第3节最好的FlowNetS上，增加了另一个FlowNetS。第二个网络是随机初始化的，然后这个堆叠是在Chairs上用S_long的方案进行训练的。我们在两个场景中进行了实验：保持第一个网络的权重是固定的，或与第二个网络的权重一起更新。在后者的情况下，第一个网络的权重在前400k次迭代是固定的，以给第二个网络提供好的初始化。我们在Sintel训练clean和Chairs的测试集上给出误差结果。由于Chairs测试集比Sintel与训练集更相像，在两个数据集上比较结果，使我们可以检测到过拟合的倾向。

**We make the following observations**: (1) Just stacking networks without warping improves results on Chairs but decreases performance on Sintel, i.e. the stacked network is over-fitting. (2) With warping included, stacking always improves results. (3) Adding an intermediate loss after Net1 is advantageous when training the stacked network end-to-end. (4) The best results are obtained when keeping the first network fixed and only training the second network after the warping operation.

**我嫩观察到以下现象**：(1)只叠加网络，不进行形变，在Chairs上改进了结果，但在Sintel上性能下降，即，叠加的网络是过拟合的。(2)在有形变的情况下，堆叠会一直改进结果；(3)在Net1后增加一个中间损失，在端到端的训练堆叠网络时是有好处的；(4)在形变运算后，保持第一个网络固定，只训练第二个网络，会得到最佳结果。

Clearly, since the stacked network is twice as big as the single network, over-fitting is an issue. The positive effect of flow refinement after warping can counteract this problem, yet the best of both is obtained when the stacked networks are trained one after the other, since this avoids overfitting while having the benefit of flow refinement.

很明显，由于堆叠网络是单个网络大小的两倍，过拟合就是一个问题。在形变后的光流提炼的正面效果，可以弥补这个问题，但两者最好的结果，是在两个网络逐一训练时得到的，因为这避免了过拟合，同时又有光流精炼的好处。

### 4.2. Stacking Multiple Diverse Networks

Rather than stacking identical networks, it is possible to stack networks of different type (FlowNetC and FlowNetS). Reducing the size of the individual networks is another valid option. We now investigate different combinations and additionally also vary the network size.

如果不堆叠同样的网络，则可能堆叠不同类型的网络(FlowNetC和FlowNetS)。降低单个网络的大小，是另一个可用的选项。我们现在研究不同的组合，以及变化网络大小的效果。

We call the first network the bootstrap network as it differs from the second network by its inputs. The second network could however be repeated an arbitray number of times in a recurrent fashion. We conducted this experiment and found that applying a network with the same weights multiple times and also fine-tuning this recurrent part does not improve results (see supplemental material for details). As also done in [18, 10], we therefore add networks with different weights to the stack. Compared to identical weights, stacking networks with different weights increases the memory footprint, but does not increase the runtime. In this case the top networks are not constrained to a general improvement of their input, but can perform different tasks at different stages and the stack can be trained in smaller pieces by fixing existing networks and adding new networks one-by-one. We do so by using the Chairs→Things3D schedule from Section 3 for every new network and the best configuration with warping from Section 4.1. Furthermore, we experiment with different network sizes and alternatively use FlowNetS or FlowNetC as a bootstrapping network. We use FlowNetC only in case of the bootstrap network, as the input to the next network is too diverse to be properly handeled by the Siamese structure of FlowNetC. Smaller size versions of the networks were created by taking only a fraction of the number of channels for every layer in the network. Figure 4 shows the network accuracy and runtime for different network sizes of a single FlowNetS. Factor 3/8 yields a good trade-off between speed and accuracy when aiming for faster networks.

我们称第一个网络为bootstrap网络，因为与第二个网络的输入不一样。第二个网络可以以循环的方式重复任意次数。我们进行了实验，发现对同样权重的一个网络应用多次，同时精调这个循环部分，这并不会改进结果（详见附录材料）。如[18,10]所做的，我们因此向网络中增加带有不同权重的网络。与同样权重的网络相比，堆叠不同权重的网络，增加了内存消耗，但并没有增加运行时间。在这种情况下，最上层的网络就不限于对其输入的一般改进，而是可以在不同的阶段进行不同的任务，而且这些栈可以以很小的片段进行训练，固定已有的网络，一个一个增加新的网络。我们通过对每个新网络使用第3节的Chairs→Things3D方案和4.1节的变形的最佳配置，达到这样的目的。而且，我们对不同的网络大小，以及交替使用FlowNetS或FlowNetC作为bootstrapping网络进行实验。我们只在bootstrap网络的情况下使用FlowNetC，因为下一个网络的输入太多样化，FlowNetC的Siamese架构无法处理。通过只采用每一层的一部分通道数，可以创建更小版的网络。图4展示了不同大小的FlowNetS网络的准确度和运行时间。当目标是快速网络时，因子3/8可以达到较好的速度和准确率的折中。

Notation: We denote networks trained by the Chairs→Things3D schedule from Section 3 starting with FlowNet2. Networks in a stack are trained with this schedule one-by-one. For the stack configuration we append upper- or lower-case letters to indicate the original FlowNet or the thin version with 3/8 of the channels. E.g: FlowNet2-CSS stands for a network stack consisting of one FlowNetC and two FlowNetS. FlowNet2-css is the same but with fewer channels.

注意：我们将使用第3节Chairs→Things3D方案训练的网络，表示为FlowNet2。堆叠中的网络采用这个方案逐一进行训练。对于堆叠配置，我们在FlowNet后，或3/8通道数量版的网络后，加上大写或小写字母，如，FlowNet2-CSS表示包含1个FlowNetC和2个FlowNetS的网络堆叠，FlowNet2-css是同样的配置，但是通道数更少。

Table 3 shows the performance of different network stacks. Most notably, the final FlowNet2-CSS result improves by ∼ 30% over the single network FlowNet2-C from Section 3 and by ∼ 50% over the original FlowNetC [11]. Furthermore, two small networks in the beginning always outperform one large network, despite being faster and having fewer weights: FlowNet2-ss (11M weights) over FlowNet2-S (38M weights), and FlowNet2-cs (11M weights) over FlowNet2-C (38M weights). Training smaller units step by step proves to be advantageous and enables us to train very deep networks for optical flow. At last, FlowNet2-s provides nearly the same accuracy as the original FlowNet [11], while running at 140 frames per second.

表3给出了不同网络堆叠的性能。最值得关注的是，最终的FlowNet2-CSS结果，与单个网络FlowNet2-C相比，改进了～30%，与原始的FlowNetC相比，改进了～50%。而且，开始采用两个小的网络，会一直超过一个大的网络，尽管速度会更快，权重更少：FlowNet2-ss (11M weights)超过了FlowNet2-S (38M weights)，FlowNet2-cs (11M weights)超过了FlowNet2-C (38M weights)。一步一步训练更小的单元，是有好处的，使我们可以对光流估计训练非常深的网络。最后，FlowNet2-s与原始FlowNet准确率几乎一样[11]，而运行速度可以达到140 FPS。

## 5. Small Displacements

### 5.1. Datasets

While the original FlowNet [11] performed well on the Sintel benchmark, limitations in real-world applications have become apparent. In particular, the network cannot reliably estimate small motions (see Figure 1). This is counter-intuitive, since small motions are easier for traditional methods, and there is no obvious reason why networks should not reach the same performance in this setting. Thus, we examined the training data and compared it to the UCF101 dataset [26] as one example of real-world data. While Chairs are similar to Sintel, UCF101 is fundamentally different (we refer to our supplemental material for the analysis): Sintel is an action movie and as such contains many fast movements that are difficult for traditional methods, while the displacements we see in the UCF101 dataset are much smaller, mostly smaller than 1 pixel. Thus, we created a dataset in the visual style of Chairs but with very small displacements and a displacement histogram much more like UCF101. We also added cases with a background that is homogeneous or just consists of color gradients. We call this dataset ChairsSDHom and will release it upon publication.

原始FlowNet[11]在Sintel基准测试中表现良好，但在真实世界的应用的限制非常明显。特别是，网络不能很可靠的估计小的运动（见图1）。这是反直觉的，因为对于传统方法来说，小的运动是很容易估计的，为什么网络在这个设置中无法达到相同的性能，没有明显的原因。因此，我们检查了训练数据，与UCF101数据集[26]进行了比较，作为真实世界数据的一个例子。虽然Chairs与Sintel类似，但UCF101从根本上是不一样的（见附录资料的分析）：Sintel是一个动作电影，所以包含很多快速的运动，对于传统方法来说比较困难，而我们在UCF101数据集中看到的偏移是很小的，多数小于1个像素。因此，我们按照Chairs的视觉类型创建了一个数据集，但是偏移很小，其偏移的直方图与UCF101很像。我们还增加了背景是均匀的或只包含色彩梯度的例子。我们称这个数据集为ChairsSDHom，会将其公开发布。

### 5.2. Small Displacement Network and Fusion

We fine-tuned our FlowNet2-CSS network for smaller displacements by further training the whole network stack on a mixture of Things3D and ChairsSDHom and by applying a non-linearity to the error to down-weight large displacements. We denote this network by FlowNet2-CSS-ft-sd. This increases performance on small displacements and we found that this particular mixture does not sacrifice performance on large displacements. However, in case of subpixel motion, noise still remains a problem and we conjecture that the FlowNet architecture might in general not be perfect for such motion. Therefore, we slightly modified the original FlowNetS architecture and removed the stride 2 in the first layer. We made the beginning of the network deeper by exchanging the 7×7 and 5×5 kernels in the beginning with multiple 3 × 3 kernels. Because noise tends to be a problem with small displacements, we add convolutions between the upconvolutions to obtain smoother estimates as in [19]. We denote the resulting architecture by FlowNet2-SD; see Figure 2.

我们将FlowNet2-CSS网络对小偏移进行了精调，将整个网络堆叠在Things3D和ChairsSDHom上进行进一步训练，对误差进行了一个非线性变换，以降低大的偏移的权重。我们称这个网络为FlowNet2-CSS-ft-sd。这提升了在小的偏移上的性能，我们发现这种特殊的混合并没有牺牲在大的偏移上的性能。但是，在亚像素运动的情况下，噪声仍然是一个问题，我们推测，FlowNet架构总体上对这种运动并不是完美的。因此，我们略微修改了原始的FlowNetS架构，移除了第一层的步长2。我们通过将7×7和5×5的核替换成多个3 × 3的核，从而使网络的开始更深。因为噪声倾向于是小型偏移的问题，我们在上卷积之间增加了卷积，以像[19]一样得到更光滑的估计。我们将得到的架构表示为FlowNet2-SD；见图2。

Finally, we created a small network that fuses FlowNet2-CSS-ft-sd and FlowNet2-SD (see Figure 2). The fusion network receives the flows, the flow magnitudes and the errors in brightness after warping as input. It contracts the resolution two times by a factor of 2 and expands again. Contrary to the original FlowNet architecture it expands to the full resolution. We find that this produces crisp motion boundaries and performs well on small as well as on large displacements. We denote the final network as FlowNet2.

最后，我们创建了一个小型网络，将FlowNet2-CSS-ft-sd和FlowNet2-SD融合到了一起（见图2）。融合网络以光流，光流幅度和变换后的亮度误差作为输入。它将分辨率缩小了2次，因子为2，然后再次拓展。与原始FlowNet架构相反，其扩展到完成分辨率。我们发现，这产生了洁净的运动边缘，在小的偏移和大的偏移上表现都很好。我们称最后的网络为FlowNet2。

## 6. Experiments

We compare the best variants of our network to state-of-the-art approaches on public bechmarks. In addition, we provide a comparison on application tasks, such as motion segmentation and action recognition. This allows benchmarking the method on real data.

我们在公开基准测试中，比较了我们网络最好的变体与目前最好的方法。而且，我们在应用任务中给出了一个比较，比如运动分割和行为识别。这就可以在真实数据中对各种方法进行基准测试。

### 6.1. Speed and Performance on Public Benchmarks

We evaluated all methods on a system with an Intel Xeon E5 with 2.40GHz and an Nvidia GTX 1080. Where applicable, dataset-specific parameters were used, that yield best performance. Endpoint errors and runtimes are given in Table 4.

我们评估所有方法的计算机配置为Intel Xeon E5 with 2.40GHz和Nvidia GTX 1080。在可以应用的地方，我们都使用了数据集专用的参数，可以得到最好的性能。表4中给出了EPE和运行时间。

**Sintel**: On Sintel, FlowNet2 consistently outperforms DeepFlow [32] and EpicFlow [22] and is on par with Flow-Fields. All methods with comparable runtimes have clearly inferior accuracy. We fine-tuned FlowNet2 on a mixture of Sintel clean+final training data (FlowNet2–ft-sintel). On the benchmark, in case of clean data this slightly degraded the result, while on final data FlowNet2–ft-sintel is on par with the currently published state-of-the art method Deep-DiscreteFlow [14].

**Sintel**：在Sintel上，FlowNet2一直比DeepFlow和EpicFlow要好，与Flow-Fields性能类似。运行时间类似的方法其准确率都明显要差一些。我们在Sintel clean+final的混合训练数据上精调了FlowNet2(FlowNet-ft-sintel)。在基准测试中，在clean data的情况下，性能会略微下降，而在最终的数据上，FlowNet2-ft-sintel与目前发表的最好结果Deep-DiscreteFlow[14]性能接近。

**KITTI**: On KITTI, the results of FlowNet2-CSS are comparable to EpicFlow [22] and FlowFields [2]. Fine-tuning on small displacement data degrades the result. This is probably due to KITTI containing very large displacements in general. Fine-tuning on a combination of the KITTI2012 and KITTI2015 training sets reduces the error roughly by a factor of 3 (FlowNet2-ft-kitti). Among non-stereo methods we obtain the best EPE on KITTI2012 and the first rank on the KITTI2015 benchmark. This shows how well and elegantly the learning approach can integrate the prior of the driving scenario.

**KITTI**：在KITTI上，FlowNet2-CSS的结果与EpicFlow [22]和FlowFields [2]性能接近。在小偏移数据上精调后，使得性能下降。这很可能是因为KITTI总体上包含很多大的偏移的数据。在KITTI2012和KITTI2015的训练集的组合上进行精调，将误差大致降低了3倍(FlowNet2-ft-kitti)。在non-stereo方法中，我们在KITTI2012上得到了最佳的EPE，在KITTI2015基准测试中得到了一流的结果。这说明，学习方法可以很好的整合驾驶场景中的先验知识。

**Middlebury**: On the Middlebury training set FlowNet2 performs comparable to traditional methods. The results on the Middlebury test set are unexpectedly a lot worse. Still, there is a large improvement compared to FlowNetS [11].

**Middlebury**: 在Middlebury训练集中，FlowNet2与传统方法性能类似。而在Middlebury测试集中，效果则异乎寻常的差了很多。但是，与FlowNetS比，仍然有很大进步。

Endpoint error vs. runtime evaluations for Sintel are provided in Figure 4. One can observe that the FlowNet2 family outperforms the best and fastest existing methods by large margins. Depending on the type of application, a FlowNet2 variant between 8 to 140 frames per second can be used.

EPE误差和运行时间的评估如表4所示。我们可以观察到，FlowNet2族与目前已知的最快和最好的方法相比，超出了很多。依赖于应用的类型，可以使用8-140FPS的FlowNet2变体。

### 6.2. Qualitative Results

Figures 6 and 7 show example results on Sintel and on real-world data. While the performance on Sintel is similar to FlowFields [2], we can see that on real world data FlowNet 2.0 clearly has advantages in terms of being robust to homogeneous regions (rows 2 and 5), image and compression artifacts (rows 3 and 4) and it yields smooth flow fields with sharp motion boundaries.

图6和7给出了在Sintel和真实世界数据上的结果的例子。在Sintel上的结果与FlowFields类似，但是我们可以看到在真实世界数据上FlowNet2.0有很明显的优势，在同质化区域中很稳定（第2行和第5行），在压缩的伪影中很稳定（第3行和第4行），可以得到平滑的光流场，同时有很尖锐的运动边缘。

### 6.3. Performance on Motion Segmentation and Action Recognition

To assess performance of FlowNet 2.0 in real-world applications, we compare the performance of action recognition and motion segmentation. For both applications, good optical flow is key. Thus, a good performance on these tasks also serves as an indicator for good optical flow.

为评估FlowNet 2.0在真实世界应用中的性能，我们比较了行为识别和运动分割的性能。对于两个应用来说，好的光流是关键。因此，在这些任务中的好的性能也说明有好的光流。

For motion segmentation, we rely on the well-established approach of Ochs et al. [20] to compute long term point trajectories. A motion segmentation is obtained from these using the state-of-the-art method from Keuper et al. [15]. The results are shown in Table 5. The original model in Ochs et al. [15] was built on Large Displacement Optical Flow [7]. We included also other popular optical flow methods in the comparison. The old FlowNet [11] was not useful for motion segmentation. In contrast, the FlowNet2 is as reliable as other state-of-the-art methods while being orders of magnitude faster.

对运动分割，我们用著名的Ochs等[20]的方法，来计算长期点轨迹。运动估计是从使用目前最好的方法Keuper等[15]得到的。结果如表5所示。在Ochs等[15]中的原始模型是在Large Displacement Optical Flow [7]基础上构建的。我们还在比较中包括了其他流行的光流方法。旧的FlowNet[11]对于运动分割没有用处。对比起来，FlowNet2与其他目前最好的方法一样可靠，而运算速度则快了几个数量级。

Optical flow is also a crucial feature for action recognition. To assess the performance, we trained the temporal stream of the two-stream approach from Simonyan et al. [25] with different optical flow inputs. Table 5 shows that FlowNetS [11] did not provide useful results, while the flow from FlowNet 2.0 yields comparable results to state-of-the art methods.

光流对于行为识别也是很关键的特征。为估计性能，我们用不同的光流输入，训练了Simonyan等[25]的双流方法的时间流。表5给出了FlowNetS[11]并没有给出有用的结果，而FlowNet 2.0的流给出了与目前最好的方法相似的结果。

## 7. Conclusions

We have presented several improvements to the FlowNet idea that have led to accuracy that is fully on par with state-of-the-art methods while FlowNet 2.0 runs orders of magnitude faster. We have quantified the effect of each contribution and showed that all play an important role. The experiments on motion segmentation and action recognition show that the estimated optical flow with FlowNet 2.0 is reliable on a large variety of scenes and applications. The FlowNet 2.0 family provides networks running at speeds from 8 to 140fps. This further extends the possible range of applications. While the results on Middlebury indicate imperfect performance on subpixel motion, FlowNet 2.0 results highlight very crisp motion boundaries, retrieval of fine structures, and robustness to compression artifacts. Thus, we expect it to become the working horse for all applications that require accurate and fast optical flow computation.

我们对FlowNet的思想提出了几个改进，得到的准确率与目前最好的方法接近，而且FlowNet 2.0的运行速度快了好几数量级。我们对每个贡献都进行了量化评估，表明这些都起了重要的作用。在运动分割和行为识别上的实验表明，用FlowNet 2.0估计的光流在大量场景和应用中都是可靠的。FlowNet 2.0族给出的网络的运行速度为8-140 FPS。这进一步拓展了可能的应用范围。在Middlebury上的结果表明，在亚像素的运动上性能还没有那么好，但FlowNet 2.0的确可以得到很尖锐的运动边缘，对精细结构敏感，对压缩的伪影很稳定。因此，我们期待其可以推动所有需要准确和快速的光流计算的应用。