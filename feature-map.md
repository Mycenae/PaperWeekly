# Question: What is the definition of a “feature map” (aka “activation map”) in a convolutional neural network?
# 问题：卷积神经网络中的特征图（也就是激活图）怎么定义？

from [stack exchange - cross validated](https://stats.stackexchange.com/questions/291820/what-is-the-definition-of-a-feature-map-aka-activation-map-in-a-convolutio)

## Intro Background 背景介绍

Within a convolutional neural network, we usually have a general structure / flow that looks like this:

在一个卷积神经网络中，我们经常会有一个一般性的框架/流程像下面这样：

1. Input image (i.e. a 2D matrix x) 输入图像（即，2D矩阵x）

(1st Convolutional layer (Conv1) starts here...) 第1卷积层conv1从这里开始

2. convolve a set of filters (w1) along the 2D image (i.e. do the z1 = w1*x + b1 dot product multiplications), where z1 is 3D, and b1 is biases.

-  2D图像与一系列滤波器w1卷积，即z1 = w1*x + b1，这里z1是3D的，b1为偏置。

3. apply an activation function (e.g. ReLu) to make z1 non-linear (e.g. a1 = ReLu(z1)), where a1 is 3D.

- 应用激活函数（如ReLU）使z1非线性化，a1 = ReLu(z1)，这里a1是3D的。

(2nd Convolutional layer (Conv2) starts here...) 第2卷积层conv2从这里开始

4. convolve a set of filters (w2) along the newly computed activations (i.e. do the z2 = w2*a1 + b2 dot product multiplications), where z2 is 3D, and and b2 is biases.

- 将新计算出来的激活a1与一些列滤波器w2进行卷积，z2 = w2*a1 + b2，z2是3D的，b2是偏置。

5. apply an activation function (e.g. ReLu) to make z2 non-linear (e.g. a2 = ReLu(z2)), where a2 is 3D.

- 将激活函数应用于z2，a2 = ReLu(z2)，a2是3D的。

## The Question

The definition of the term "feature map" seems to vary from literature to literature. Concretely:

特征图的定义似乎很多文献定义不同，具体来说：

1. For the 1st convolutional layer, does "feature map" corresponds to the input vector x, or the output dot product z1, or the output activations a1, or the "process" converting x to a1, or something else?

- 对于第1卷积层，特征图是指输入x，还是卷积输出z1，还是激活输出a1，或是将x转化为a1的过程，或是别的什么东西？

2. Similarly, for the 2nd convolutional layer, does "feature map" corresponds to the input activations a1, or the output dot product z2, or the output activation a2, or the "process" converting a1 to a2, or something else?

- 类似的，对于第2卷积层，特征图是指输入激活a1，或是卷积输出z2，还是激活输出a2，或是将a1转换成a2的过程，或是别的什么东西？

In addition, is it true that the term "feature map" is exactly the same as "activation map"? (or do they actually mean two different thing?)

另外，特征图和激活图是不是一样的，或者它们代表两个不同的含义？

## Additional references: 参考

Snippets from Neural Networks and Deep Learning - Chapter 6: 神经网络和深度学习第6章节选

> *The nomenclature is being used loosely here. In particular, I'm using "feature map" to mean not the function computed by the convolutional layer, but rather the activation of the hidden neurons output from the layer. This kind of mild abuse of nomenclature is pretty common in the research literature.

> 命名规则这里使用的比较宽松。特别的，我使用“特征图”所指的不是卷积层计算的函数，而是那一层隐藏神经元的激活输出。这种命名法的轻微滥用在研究文献中颇为常见。

Snippets from Visualizing and Understanding Convolutional Networks by Matt Zeiler: 可视化并理解卷积神经网络论文节选

> In this paper we introduce a visualization technique that reveals the input stimuli that excite individual feature maps at any layer in the model. [...] Our approach, by contrast, provides a non-parametric view of invariance, showing which patterns from the training set activate the feature map. [...] a local contrast operation that normalizes the responses across feature maps. [...] To examine a given convnet activation, we set all other activations in the layer to zero and pass the feature maps as input to the attached deconvnet layer. [...] The convnet uses relu non-linearities, which rectify the feature maps thus ensuring the feature maps are always positive. [...] The convnet uses learned filters to convolve the feature maps from the previous layer. [...] Fig. 6, these visualizations are accurate representations of the input pattern that stimulates the given feature map in the model [...] when the parts of the original input image corresponding to the pattern are occluded, we see a distinct drop in activity within the feature map. [...]

> 本文中我们提出了一种可视化技术，可以观察输入刺激激发出的任意一层单独特征图。[...]我们的方法给出了不变性的非参数视图，展示了训练集图像的哪个模式激励出了特征图。[...]局部对比度操作，在整个特征图中归一化了响应。[...]为检查给定convnet激活，我们将层中所有其他激活置零，将特征图作为输入给连接着的deconvnet层。[...]convnet使用ReLU非线性处理，这就将特征图进行了校正，保证特征图都是正值。[...]convnet使用学习好的滤波器对前一层的特征图进行卷积。[...]图6中，这些可视化是刺激出给定特征图的输入模式精确表达。[...]当原始输入图像与这个模式对应的部分被遮挡时，我们在特征图中看到了激活明显的下降。

Remarks: also introduces the term "feature map" and "rectified feature map" in Fig 1.

注：在图1中引入了特征图和校正特征图的术语。

Snippets from Stanford CS231n Chapter on CNN: 斯坦福CS231n关于CNN的章节节选

> [...] One dangerous pitfall that can be easily noticed with this visualization is that some activation maps may be all zero for many different inputs, which can indicate dead filters, and can be a symptom of high learning rates [...] Typical-looking activations on the first CONV layer (left), and the 5th CONV layer (right) of a trained AlexNet looking at a picture of a cat. Every box shows an activation map corresponding to some filter. Notice that the activations are sparse (most values are zero, in this visualization shown in black) and mostly local.

> [...]关于这个可视化效果，很容易注意到的一个危险陷阱是，对于很多不同的输入，一些激活图可能全是零，这说明有“死滤波器”，是高学习速率的症状。[...]一只猫图像的，在第1卷积层（左）典型的激活，和训练好的AlexNet第5卷积层（右）。每个方框都是一些滤波器对应的激活图。注意激活是稀疏的（多数是零值，在这个可视化效果中，以黑色表示）而且多数是局部的。

Snippets from A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks: 理解卷积神经网络入门指南节选

> [...] Every unique location on the input volume produces a number. After sliding the filter over all the locations, you will find out that what you’re left with is a 28 x 28 x 1 array of numbers, which we call an activation map or feature map.

> [...]输入卷中每个唯一的位置都生成了一个数。在每个位置都滑动滤波器后，将会发现得到了一个28×28×1的数组，称为激活图或特征图。

## Answer one

A feature map, or activation map, is the output activations for a given filter (a1 in your case) and the definition is the same regardless of what layer you are on.

特征图，或激活图，是对给定的滤波器（如上述的a1）输出的激活，这个定义在每一层都是一样的。

Feature map and activation map mean exactly the same thing. It is called an activation map because it is a mapping that corresponds to the activation of different parts of the image, and also a feature map because it is also a mapping of where a certain kind of feature is found in the image. A high activation means a certain feature was found.

特征图和激活图实际上是一样的。称为激活图是因为，这是图像不同部分的激活的映射，称为特征图是因为，这也是这种映射可以在图像中找到其特定特征。激活值高意味着找到了特定的特征。

A "rectified feature map" is just a feature map that was created using Relu. You could possibly see the term "feature map" used for the result of the dot products (z1) because this is also really a map of where certain features are in the image, but that is not common to see.

“校正的特征图”是用ReLU生成的特征图。你可能看到了，特征图的术语用在点乘的结果z1上，因为这就是一张图上面有图像的特定特征，但这通常看不到。

## Answer two

before talk about what feature map means, let just define the term of feature vector.

在讨论特征图的意义前，先定义特征向量。

feature vector is vectorial representation of objects. For example, a car can be represented by [number of wheels, door. windows, age ..etc].

特征向量是目标的向量表示。比如，一辆车可以用[轮子数量，门数量，窗户数量，车龄等]表示。

feature map is a function that takes feature vectors in one space and transforms them into feature vectors in another. For example given a feature vector [volume ,weight, height, width] it can return [1, volume/weight, height * width] or [height * width] or even just [volume]

特征图是将特征向量整合进一个空间的函数，并转化成另一个空间的向量。比如，给定一个特征向量[容积，重量，高度，宽度]，可以转化成[1，容积/重量，高度×宽度]或[高度×宽度]或只有[容量]。