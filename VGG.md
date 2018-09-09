# Very Deep Convolutional Networks for Large-Scale Image Recognition
# 大规模图像识别应用中非常深的卷积网络

Karen Simonyan & Andrew Zisserman Visual Geometry Group, Department of Engineering Science, University of Oxford

## ABSTRACT

In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3×3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16–19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisation and classification tracks respectively. We also show that our representations generalise well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.

本文我们研究了大规模图像识别应用中卷积神经网络的深度对准确率的影响。我们的主要贡献是使用非常小的卷积核(3×3)组成的架构彻底对网络深度进行了评估，结果显示如果带权重的层增加到16-19个，在现有配置下，网络性能会有明显提升。以这些发现为基础，我们向2014年ImageNet挑战赛提交了自己的模型，我们的团队在定位任务中得到了第一的位置，在分类任务中得到了第二的成绩。我们的模型泛化能力也非常好，实际上是目前最好的结果。我们公布了两个最好的卷积网络，为进一步在计算机视觉中研究深度视觉表示提供便利。

## 1 I NTRODUCTION

Convolutional networks (ConvNets) have recently enjoyed a great success in large-scale image and video recognition (Krizhevsky et al., 2012; Zeiler & Fergus, 2013; Sermanet et al., 2014; Simonyan & Zisserman, 2014) which has become possible due to the large public image repositories, such as ImageNet(Deng et al., 2009), and high-performance computing systems, such as GPUs or large-scale distributed clusters (Dean et al., 2012). In particular, an important role in the advance of deep visual recognition architectures has been played by the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) (Russakovsky et al., 2014), which has served as a testbed for a few generations of large-scale image classification systems, from high-dimensional shallow feature encodings (Perronnin et al., 2010) (the winner of ILSVRC-2011)to deep ConvNets (Krizhevsky et al., 2012) (the winner of ILSVRC-2012).

卷积网络(ConvNets)近些年在大规模图像和视频识别非常成功(Krizhevsky et al., 2012; Zeiler & Fergus, 2013; Sermanet et al., 2014; Simonyan & Zisserman, 2014)，这是因为有了大规模图像数据集如ImageNet(Deng et al., 2009)，和高性能计算系统如GPU和大规模分布式集群(Dean et al., 2012)。特别是ILSVRC(Russakovsky et al., 2014)在深度视觉识别的发展中起到了很重要的作用，几代大规模图像分类系统都是以它为试验台发展起来的，从ILSVRC-2011冠军的高维度浅层特征编码(Perronnin et al., 2010)到ILSVRC-2012冠军的深度卷积网络(Krizhevsky et al., 2012)。

With ConvNets becoming more of a commodity in the computer vision field, a number of attempts have been made to improve the original architecture of Krizhevsky et al. (2012) in a bid to achieve better accuracy. For instance, the best-performing submissions to the ILSVRC-2013 (Zeiler & Fergus, 2013; Sermanet et al., 2014) utilised smaller receptive window size and smaller stride of the first convolutional layer. Another line of improvements dealt with training and testing the networks densely over the whole image and over multiple scales (Sermanet et al., 2014; Howard, 2014). In this paper, we address another important aspect of ConvNet architecture design – its depth. To this end, we fix other parameters of the architecture, and steadily increase the depth of the network by adding more convolutional layers, which is feasible due to the use of very small (3 × 3) convolution filters in all layers.

随着卷积网络在计算机视觉领域变得日益普遍，有一些对原版Krizhevsky et al. (2012)网络架构进行改进的尝试，以获得更好的准确度。比如，ILSVRC-2013冠军得主(Zeiler & Fergus, 2013; Sermanet et al., 2014)使用了小一些的感受野窗口和小一些的卷积步长（在第1个卷积层中）。另一个改进的思路想要解决在整个图像上多尺度密集训练测试网络(Sermanet et al., 2014; Howard, 2014)。本文中，我们探讨了卷积网络架构设计问题的另一个重要方面，也就是深度。为了这个目的，我们解决架构的其他参数问题，然后稳步增加更多的卷积层以增加网络深度，我们在所有层中都用了很小的卷积滤波器(3×3)，所以是可行的。

As a result, we come up with significantly more accurate ConvNet architectures, which not only achieve the state-of-the-art accuracy on ILSVRC classification and localisation tasks, but are also applicable to other image recognition datasets, where they achieve excellent performance even when used as a part of a relatively simple pipelines (e.g. deep features classified by a linear SVM without fine-tuning). We have released our two best-performing models (http://www.robots.ox.ac.uk/~vgg/research/very_deep/) to facilitate further research.

结果我们得到了准确度高的多的ConvNet架构，在ILSVRC分类和定位任务中取得了目前最好的准确率，还可以用于其他识别数据集，即使在一个简单结构的一部分，也可以取得非常好的结果（如将深度特征用线性SVM分类，不精调）。我们公开了两个最好的模型为进一步的研究提供便利(http://www.robots.ox.ac.uk/~vgg/research/very_deep/)。

The rest of the paper is organised as follows. In Sect. 2, we describe our ConvNet configurations. The details of the image classification training and evaluation are then presented in Sect. 3, and the configurations are compared on the ILSVRC classification task in Sect. 4. Sect. 5 concludes the paper. For completeness, we also describe and assess our ILSVRC-2014 object localisation system in AppendixA, and discuss the generalisation of very deep features to other datasets in AppendixB. Finally, Appendix C contains the list of major paper revisions.

文章剩余部分组织如下：在第2部分中，我们列出了卷积网络的配置，第3部分给出了图像分类训练和测试的细节，第4部分在ILSVRC分类任务中进行了比较，第5部分进行了总结，附录A中评价了我们的ILSVRC目标定位系统，附录B中讨论了深度特征在其他数据集上的泛化，附录C包括了文章的主要修改过程。

## 2 CONVNET CONFIGURATIONS

To measure the improvement brought by the increased ConvNet depth in a fair setting, all our ConvNet layer configurations are designed using the same principles, inspired by Ciresan et al. (2011); Krizhevsky et al. (2012). In this section, we first describe a generic layout of our ConvNet configurations(Sect.2.1) and then detail the specific configurations used in the evaluation (Sect.2.2). Our design choices are then discussed and compared to the prior art in Sect. 2.3.