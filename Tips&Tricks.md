# Must Know Tips/Tricks in Deep Neural Networks (by Xiu-Shen Wei)
# 深度神经网络应知必会技巧

from [Xiu-Shen Wei's main page](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html)

注：翻译仅为增加个人理解，不作他用

Deep Neural Networks, especially Convolutional Neural Networks (CNN), allows computational models that are composed of multiple processing layers to learn representations of data with multiple levels of abstraction. These methods have dramatically improved the state-of-the-arts in visual object recognition, object detection, text recognition and many other domains such as drug discovery and genomics.

深度神经网络，尤其是卷积神经网络，使由多处理层组成的计算模型可以学习到多层次抽象的数据表示。这些方法极大的改善了视觉目标识别，目标检测，文字识别以及其他领域如药品发现和基因学的现有水平。

In addition, many solid papers have been published in this topic, and some high quality open source CNN software packages have been made available. There are also well-written CNN tutorials or CNN software manuals. However, it might lack a recent and comprehensive summary about the details of how to implement an excellent deep convolutional neural networks from scratch. Thus, we collected and concluded many implementation details for DCNNs. Here we will introduce these extensive implementation details, i.e., tricks or tips, for building and training your own deep networks.

另外，关于这个主题已经发表了许多论文，一些高质量开源CNN软件包已经开发出来，还有很多写的很好的CNN教程或软件手册，但关于从零开始很好的实现DCNN的细节则缺少较新和综合的总结。所以，我们收集总结了很多DCNN的实现细节，这里我们将进行介绍这些技巧或提示，以使你更好的构建和训练你自己的深度网络。

## Introduction 介绍

We assume you already know the basic knowledge of deep learning, and here we will present the implementation details (tricks or tips) in Deep Neural Networks, especially CNN for image-related tasks, mainly in eight aspects: 1) data augmentation; 2) pre-processing on images; 3) initializations of Networks; 4) some tips during training; 5) selections of activation functions; 6) diverse regularizations; 7) some insights found from figures and finally 8) methods of ensemble multiple deep networks.

我们假设你已经知道了深度学习的基本知识，这里要叙述的是其实现细节（技巧或提示），尤其是与图像相关的CNN任务，主要在8个方面：1)数据扩充;2)图像预处理;3)网络初始化;4)训练过程中的提示点;5)激活函数的选择;6)正则化;7)图示中发现的洞见;8)综合多个深度网络的方法.

Additionally, the corresponding slides are available at [slide](http://lamda.nju.edu.cn/weixs/slide/CNNTricks_slide.pdf). If there are any problems/mistakes in these materials and slides, or there are something important/interesting you consider that should be added, just feel free to contact [me](http://lamda.nju.edu.cn/weixs/).

另外，已经有了相应的[幻灯片](http://lamda.nju.edu.cn/weixs/slide/CNNTricks_slide.pdf)。如果在这些材料和幻灯片中有任何问题/错误，或者有一些重要/有趣的东西你认为应当加上去的，一定记得联系[我](http://lamda.nju.edu.cn/weixs/)。

## Sec. 1: Data Augmentation 数据扩充

Since deep networks need to be trained on a huge number of training images to achieve satisfactory performance, if the original image data set contains limited training images, it is better to do data augmentation to boost the performance. Also, data augmentation becomes the thing must to do when training a deep network.

由于深度网络训练的数据集规模要很大才能得到理想的表现，如果原始图像数据集数量有限，最好进行数据扩充来提升性能。同时，数据扩充也成了训练深度网络时必做的工作。

- There are many ways to do data augmentation, such as the popular horizontally flipping, random crops and color jittering. Moreover, you could try combinations of multiple different processing, e.g., doing the rotation and random scaling at the same time. In addition, you can try to raise saturation and value (S and V components of the HSV color space) of all pixels to a power between 0.25 and 4 (same for all pixels within a patch), multiply these values by a factor between 0.7 and 1.4, and add to them a value between -0.1 and 0.1. Also, you could add a value between [-0.1, 0.1] to the hue (H component of HSV) of all pixels in the image/patch.

- 数据扩充的方式有很多，比如水平翻转、随机剪切和色彩抖动。此外，还可以将不同的方法组合起来，比如，同时进行旋转和随机尺度变换。另外，可以尝试提升图像HSV空间中的饱和度S和明度V，至0.25到0.4（对一个图像块中的所有像素），将其像素值乘以0.7到1.4中的一个值，同时再加上一个-0.1到0.1中的值。你也可以对色调H分量加上一个[-0.1, 0.1]中的值。

- Krizhevsky et al. [1] proposed fancy PCA when training the famous Alex-Net in 2012. Fancy PCA alters the intensities of the RGB channels in training images. In practice, you can firstly perform PCA on the set of RGB pixel values throughout your training images. And then, for each training image, just add the following quantity to each RGB image pixel (i.e., $I_{xy}=[I_{xy}^R,I_{xy}^G,I_{xy}^B]^T$): $[\bf{p}_1,\bf{p}_2,\bf{p}_3][\alpha_1 \lambda_1,\alpha_2 \lambda_2,\alpha_3 \lambda_3]^T$ where, $\bf{p}_i$ and $\lambda_i$ are the *i*-th eigenvector and eigenvalue of the 3×3 covariance matrix of RGB pixel values, respectively, and $\alpha_i$ is a random variable drawn from a Gaussian with mean zero and standard deviation 0.1. Please note that, each $\alpha_i$ is drawn only once for all the pixels of a particular training image until that image is used for training again. That is to say, when the model meets the same training image again, it will randomly produce another $\alpha_i$ for data augmentation. In [1], they claimed that “fancy PCA could approximately capture an important property of natural images, namely, that object identity is invariant to changes in the intensity and color of the illumination”. To the classification performance, this scheme reduced the top-1 error rate by over 1% in the competition of ImageNet 2012.

- Krizhevsky et al. [1]在2012年训练著名的AlexNet时提出了fancy PCA，可以改变训练图像RGB通道的值。实践中，你可以首先在所有训练图像中对RGB像素值的集合进行PCA，然后对于每个训练图像，在每个RGB像素上加上下面的值，即对$I_{xy}=[I_{xy}^R,I_{xy}^G,I_{xy}^B]^T$加上$[\bf{p}_1,\bf{p}_2,\bf{p}_3][\alpha_1 \lambda_1,\alpha_2 \lambda_2,\alpha_3 \lambda_3]^T$，这里$\bf{p}_i$和$\lambda_i$分别是RGB像素值的3×3协方差矩阵的第*i*个特征向量和特征值，$\alpha_i$是一个随机变量，服从零均值标准差0.1的高斯分布。注意每个$\alpha_i$对于特定训练图像只随机取一次，直到这个图像再参加训练。这就是说，当模型再次遇到相同的图像时，会再次产生另外一个$\alpha_i$进行数据扩充。在[1]中，作者声称“fancy PCA可以近似的捕捉到自然图像的一个重要性质，即，目标对于光照的强度和颜色变化是不变的”。在ImageNet 2012中，对于分类的任务，这个方案将错误率降低了约1%。

# Sec. 2: Pre-Processing 预处理

Now we have obtained a large number of training samples (images/crops), but please do not hurry! Actually, it is necessary to do pre-processing on these images/crops. In this section, we will introduce several approaches for pre-processing.

现在我们得到了很大规模的训练样本（图像/剪切），但别着急，实际上必须对这些图像/剪切做预处理。在本节中，我们将引入集中预处理的方法。

The first and simple pre-processing approach is zero-center the data, and then normalize them, which is presented as two lines Python codes as follows:

第一个简单的预处理方法是将数据做零均值处理，并归一化，这通过下面两行python代码就可以实现：
```
>>> X -= np.mean(X, axis = 0) # zero-center
>>> X /= np.std(X, axis = 0) # normalize
```
where, X is the input data (NumIns×NumDim). Another form of this pre-processing normalizes each dimension so that the min and max along the dimension is -1 and 1 respectively. It only makes sense to apply this pre-processing if you have a reason to believe that different input features have different scales (or units), but they should be of approximately equal importance to the learning algorithm. In case of images, the relative scales of pixels are already approximately equal (and in range from 0 to 255), so it is not strictly necessary to perform this additional pre-processing step.

这里X是输入数据(NumIns×NumDim)。这种预处理的另一种形式是对每个维度进行归一化，使这个维度的最大最小值分别是1和-1。如果你相信不同输入特征有不同的尺度（或单元），但它们对于学习算法应当具有大约相同的重要性，这样预处理才有意义。在图像的例子中，像素的相对尺度已经近似相等（取值范围为0到255），严格来说不是特别必要进行这些预处理步骤。

Another pre-processing approach similar to the first one is PCA Whitening. In this process, the data is first centered as described above. Then, you can compute the covariance matrix that tells us about the correlation structure in the data:

与上面类似，另一种预处理方法是PCA白化。在这个过程中，数据首先零均值处理，然后计算其协方差矩阵，得到数据的相关结构：
```
>>> X -= np.mean(X, axis = 0) # zero-center
>>> cov = np.dot(X.T, X) / X.shape[0] # compute the covariance matrix
```
After that, you decorrelate the data by projecting the original (but zero-centered) data into the eigenbasis:

在这之后，对数据去相关处理，即将原始（零均值处理过的）数据投影到以特征向量为基形成的空间中
```
>>> U,S,V = np.linalg.svd(cov) # compute the SVD factorization of the data covariance matrix
>>> Xrot = np.dot(X, U) # decorrelate the data
```
The last transformation is whitening, which takes the data in the eigenbasis and divides every dimension by the eigenvalue to normalize the scale:

最后的变换就是白化，也就是将数据放入特征向量基空间中然后在每个维度上都除以特征值，以归一化尺度：
```
>>> Xwhite = Xrot / np.sqrt(S + 1e-5) # divide by the eigenvalues (which are square roots of the singular values)
```
Note that here it adds 1e-5 (or a small constant) to prevent division by zero. One weakness of this transformation is that it can greatly exaggerate the noise in the data, since it stretches all dimensions (including the irrelevant dimensions of tiny variance that are mostly noise) to be of equal size in the input. This can in practice be mitigated by stronger smoothing (i.e., increasing 1e-5 to be a larger number).

注意这里加上1e-5就是一个小的常数是为防止除以0，这个变换的一个缺点就是会剧烈放大数据中的噪声，因为它拉伸了输入的所有维度（包括哪些不相关的维度，方差很小的，多是噪声）成为同样的大小。实践中可以通过加强平滑来缓和这个问题（即用一个比1e-5大的数值）。

Please note that, we describe these pre-processing here just for completeness. In practice, these transformations are not used with Convolutional Neural Networks. However, it is also very important to zero-center the data, and it is common to see normalization of every pixel as well.

请注意，这里叙述的预处理只是为了完整性。在实践中，这些变换没有在CNN中使用，但数据的零均值处理还是非常重要，对每个像素进行归一化处理也非常常见。

## Sec. 3: Initializations 初始化

Now the data is ready. However, before you are beginning to train the network, you have to initialize its parameters.

现在数据准备好了，但开始训练网络之前，其参数需要初始化。

### All Zero Initialization 全零初始化

In the ideal situation, with proper data normalization it is reasonable to assume that approximately half of the weights will be positive and half of them will be negative. A reasonable-sounding idea then might be to set all the initial weights to zero, which you expect to be the “best guess” in expectation. But, this turns out to be a mistake, because if every neuron in the network computes the same output, then they will also all compute the same gradients during back-propagation and undergo the exact same parameter updates. In other words, there is no source of asymmetry between neurons if their weights are initialized to be the same.

在理想情况下，进行了适当的数据归一化后，可以假设大约半数的权重会是正值，半数为负值，所以将所有初始权重设为零是合适的，甚至可能是最合适的初始化。但这证明是个错误，因为如果每个神经元计算得到了相同的输出，那么在BP算法中将会计算得到相同的梯度，其参数更新也将会一样。换句话说，如果权重初始化都是一样的，那么神经元之间就没有不对等的来源。

### Initialization with Small Random Numbers 小随机数初始化

Thus, you still want the weights to be very close to zero, but not identically zero. In this way, you can random these neurons to small numbers which are very close to zero, and it is treated as symmetry breaking. The idea is that the neurons are all random and unique in the beginning, so they will compute distinct updates and integrate themselves as diverse parts of the full network. The implementation for weights might simply look like $weights\sim 0.001\times N(0,1)$, where *N(0,1)* is a zero mean, unit standard deviation gaussian. It is also possible to use small numbers drawn from a uniform distribution, but this seems to have relatively little impact on the final performance in practice.

所以，你仍然会希望权重很接近零，但不全是零。这样就可以将这些神经元设为小的接近零的随机数，这样就可以打破对称性。具体的想法是，每个神经元在初始时都是随机的唯一的，所以其参数更新都是不一样的，成为整个网络中多种多样的不同部分。权值的随机初始化实现为$weights\sim 0.001\times N(0,1)$，这里*N(0,1)*是零均值单位方差的高斯随机分布，用平均分布的随机数也是可以的，但实践中似乎对最后的性能有一些小的影响。

### Calibrating the Variances 校正方差

One problem with the above suggestion is that the distribution of the outputs from a randomly initialized neuron has a variance that grows with the number of inputs. It turns out that you can normalize the variance of each neuron's output to 1 by scaling its weight vector by the square root of its fan-in (i.e., its number of inputs), which is as follows:

上面方法的一个问题是随机初始化的神经元的输出的分布随着输入数量的增加而变化，结果证明你可以通过根据输入数量的平方根来改变权重向量的尺度，归一化每个神经元输出的方差为1，如下代码所示
```
>>> w = np.random.randn(n) / sqrt(n) # calibrating the variances with 1/sqrt(n)
```
where “randn” is the aforementioned Gaussian and “n” is the number of its inputs. This ensures that all neurons in the network initially have approximately the same output distribution and empirically improves the rate of convergence. The detailed derivations can be found from Page. 18 to 23 of the slides. Please note that, in the derivations, it does not consider the influence of ReLU neurons.

这里randn就是前面所说的高斯分布，n就是输入数量。这确保了网络中所有神经元初始时都有相同的输出分布，通过我们的经验，这可以改进收敛速度。推导细节可以在主页中幻灯片的18页到23页找到。请注意，推导中没有考虑ReLU神经元的影响。

### Current Recommendation 目前的建议

As aforementioned, the previous initialization by calibrating the variances of neurons is without considering ReLUs. A more recent paper on this topic by He et al. [4] derives an initialization specifically for ReLUs, reaching the conclusion that the variance of neurons in the network should be *2.0/n* as:

像前面提到的那样，校正神经元方差的初始化没有考虑ReLU的影响。最近有文章(He et al. [4])讨论这个，推导出了R针对eLU的初始化，得到结论是网络中神经元的方差应当是*2.0/n*：
```
>>> w = np.random.randn(n) * sqrt(2.0/n) # current recommendation
```
which is the current recommendation for use in practice, as discussed in [4]. 这就是目前实践中的建议[4]。

### Sec. 4: During Training 训练过程中的技巧和提示

Now, everything is ready. Let’s start to train deep networks! 现在所有都准备好了，让我们开始训练深度网络吧。

- **Filters and pooling size**. During training, the size of input images prefers to be power-of-2, such as 32 (e.g., CIFAR-10), 64, 224 (e.g., common used ImageNet), 384 or 512, etc. Moreover, it is important to employ a small filter (e.g., 3×3) and small strides (e.g., 1) with zeros-padding, which not only reduces the number of parameters, but improves the accuracy rates of the whole deep network. Meanwhile, a special case mentioned above, i.e., 3×3 filters with stride 1, could preserve the spatial size of images/feature maps. For the pooling layers, the common used pooling size is of 2×2.

- **滤波器和池化大小**。训练过程中，输入图像的大小最好是2的幂，如32（例如CIFAR-10），64,224（比如ImageNet），384,512等。要用小型滤波器如3×3，小的卷积步长如1，周围0填充，这很重要，不仅减少了参数数量，而且提高了整个深度网络的准确率。同时，刚提到的这个特殊案例，3×3的滤波器与卷积步长1，可以保持图像或特征图的空域大小。对于池化层，通常池化大小为2×2。

- **Learning rate**. In addition, as described in a blog by Ilya Sutskever [2], he recommended to divide the gradients by mini batch size. Thus, you should not always change the learning rates (LR), if you change the mini batch size. For obtaining an appropriate LR, utilizing the validation set is an effective way. Usually, a typical value of LR in the beginning of your training is 0.1. In practice, if you see that you stopped making progress on the validation set, divide the LR by 2 (or by 5), and keep going, which might give you a surprise.

- **学习速率**。另外，如Ilya Sutskever [2]的blog所说的那样，他推荐将梯度除以mini batch的大小。所以，如果改变mini batch大小的话，就不应当经常改变学习速率(LR)。通过验证集得到合适的LR是很有效的方法。通常，LR的典型值在训练开始时为0.1，在实践中，如果在验证集上没有进展了，就把LR除以2或5，可能可以继续改善性能。

- **Fine-tune on pre-trained models**. Nowadays, many state-of-the-arts deep networks are released by famous research groups, i.e., Caffe Model Zoo and VGG Group. Thanks to the wonderful generalization abilities of pre-trained deep models, you could employ these pre-trained models for your own applications directly. For further improving the classification performance on your data set, a very simple yet effective approach is to fine-tune the pre-trained models on your own data. As shown in following table, the two most important factors are the size of the new data set (small or big), and its similarity to the original data set. Different strategies of fine-tuning can be utilized in different situations. For instance, a good case is that your new data set is very similar to the data used for training pre-trained models. In that case, if you have very little data, you can just train a linear classifier on the features extracted from the top layers of pre-trained models. If your have quite a lot of data at hand, please fine-tune a few top layers of pre-trained models with a small learning rate. However, if your own data set is quite different from the data used in pre-trained models but with enough training images, a large number of layers should be fine-tuned on your data also with a small learning rate for improving performance. However, if your data set not only contains little data, but is very different from the data used in pre-trained models, you will be in trouble. Since the data is limited, it seems better to only train a linear classifier. Since the data set is very different, it might not be best to train the classifier from the top of the network, which contains more dataset-specific features. Instead, it might work better to train the SVM classifier on activations/features from somewhere earlier in the network.

- **在预训练的模型上进行精调**。现在很多最好的深度网络都是由著名研究团体发布的，比如Caffe Model Zoo和VGG Group。这些预训练深度模型的泛化能力都是很好的，可以直接将这些预训练模型用在自己的应用中。为了进一步改善在自己的数据集上的分类表现，一个简单有效的方法就是在自己的数据集上精调那些预训练好的模型。如下表所示，最重要的两个因素是新数据集的规模大小，还有与原数据集的相似程度，不同的情况下可以采用不同的精调策略。比如，如果你的新数据与预训练使用的数据很类似，那么情况就很好。如果数据量很小，那就可以在预训练模型的最上层提取出的特征的基础上直接训练一个线性分类器；如果数据量很大，那么就用小的学习速率精调几个最顶部的层。而如果你的数据与预训练的数据相差很大，但训练图像足够，那么预训练模型的很多层都需要精调，学习速率同样应当很小。而如果数据也不一样，训练数据集还很小，那么就很麻烦了。既然数据量有限，那么只训练一个线性分类器更好一些。既然数据集非常不同，如果从网络顶部开始训练分类器可能不会很好，因为包含了很多数据集相关的特征。相反，应当在网络早起的某个位置，在激活/特征上训练SVM分类器。

 | | very similar dataset | very different dataset
--- | --- | ---
very little data | Use linear classifier on top layer | You're in trouble...try linear classifier from different stages
quite a lot of data | Finetune a few layers | Finetune a large number of layers

 | | 数据集非常相似 | 数据集非常不同
 --- | --- | ---
 数据很少 | 在模型最上层训练线性分类器 | 麻烦很大，试着在不同阶段采用线性分类器
 数据很多 | 精调几层 | 精调很多层

Fine-tune your data on pre-trained models. Different strategies of fine-tuning are utilized in different situations. For data sets, Caltech-101 is similar to ImageNet, where both two are object-centric image data sets; while Place Database is different from ImageNet, where one is scene-centric and the other is object-centric.

在自己的数据集上精调预训练模型。在不同的情况下可以采用不同的精调策略。对于数据集来说，Caltech-101与ImageNet类似，两者都是以目标为中心的图像数据集；而Place数据集则与ImageNet不一样，因为一个是场景为中心的，另一个是目标为中心的。

## Sec. 5: Activation Functions 激活函数

One of the crucial factors in deep networks is activation function, which brings the non-linearity into networks. Here we will introduce the details and characters of some popular activation functions and give advices later in this section.

深度网络中的一个关键因素是激活函数，这将非线性引入到网络中。这里我们将介绍几种受欢迎的激活函数的细节和角色，后面给出建议。

![Image](http://lamda.nju.edu.cn/weixs/project/CNNTricks/imgs/neuron.png)

### Sigmoid

![Image](http://lamda.nju.edu.cn/weixs/project/CNNTricks/imgs/sigmod.png)

The sigmoid non-linearity has the mathematical form $\sigma(x)=1/(1+e^{-x})$. It takes a real-valued number and “squashes” it into range between 0 and 1. In particular, large negative numbers become 0 and large positive numbers become 1. The sigmoid function has seen frequent use historically since it has a nice interpretation as the firing rate of a neuron: from not firing at all (0) to fully-saturated firing at an assumed maximum frequency (1).

Sigmoid非线性函数的数学形式为$\sigma(x)=1/(1+e^{-x})$，以实值输入，并将输入“挤压”到0到1的输出范围：大的负值趋向于0，大的正值趋向于1。sigmoid函数历史上就频繁出现，因为可以很好的代表神经元的放电频率（即点火率）：0代表一直不放电，1代表以指定的最高频率不断放电。

In practice, the sigmoid non-linearity has recently fallen out of favor and it is rarely ever used. It has two major drawbacks:

实践中，sigmoid非线性函数逐渐没那么受欢迎了，并不怎么使用，它主要有两个主要的缺点：

1. Sigmoids saturate and kill gradients. A very undesirable property of the sigmoid neuron is that when the neuron's activation saturates at either tail of 0 or 1, the gradient at these regions is almost zero. Recall that during back-propagation, this (local) gradient will be multiplied to the gradient of this gate's output for the whole objective. Therefore, if the local gradient is very small, it will effectively “kill” the gradient and almost no signal will flow through the neuron to its weights and recursively to its data. Additionally, one must pay extra caution when initializing the weights of sigmoid neurons to prevent saturation. For example, if the initial weights are too large then most neurons would become saturated and the network will barely learn.

- Sigmoid使梯度饱和并逐渐出现梯度消失现象。sigmoid神经元的一个很不方便的性质就是，神经元的激活在0或1处就饱和了，这些区域中的梯度基本就是0。回想一下，在反向传播过程中，局部梯度会与输出的梯度相乘，所以如果局部梯度很小，梯度消失现象会导致没有信号流经神经元，权值也不会变化更新，数据也就失去了训练作用。另外，在初始化这些sigmoid神经元的权重时必须格外注意，防止饱和。比如，如果初始权重太大，那么神经元很快就会饱和，那么网络就几乎不会学习了。

2. Sigmoid outputs are not zero-centered. This is undesirable since neurons in later layers of processing in a Neural Network (more on this soon) would be receiving data that is not zero-centered. This has implications on the dynamics during gradient descent, because if the data coming into a neuron is always positive (e.g., *x*>0 element wise in $f=w^Tx+b$), then the gradient on the weights *w* will during back-propagation become either all be positive, or all negative (depending on the gradient of the whole expression *f*). This could introduce undesirable zig-zagging dynamics in the gradient updates for the weights. However, notice that once these gradients are added up across a batch of data the final update for the weights can have variable signs, somewhat mitigating this issue. Therefore, this is an inconvenience but it has less severe consequences compared to the saturated activation problem above.

- Sigmoid输出不是零均值的。这会很不方便，因为下一层的神经元要处理的数据要求是零均值的。这还对梯度下降有隐式的影响，因为如果输入神经元的数据如果一直是正值（即$f=w^Tx+b$中*x*>0，每个元素都是正值），那么在反向传播的过程中对权重*w*的梯度会要么都是正值，要么都是负值（看整个表达式*f*的梯度而定）。这可能会在对权重的梯度更新时引入锯齿（z形）前进的不良效应。而一旦这些梯度在一批数据上叠加，权重的最后更新会有不同的符号，这会在某种程度上消除这些问题。所以，这个影响会不太方便，但严重程度与上面的饱和激活相比要低一些。

### tanh(x)

![Image](http://lamda.nju.edu.cn/weixs/project/CNNTricks/imgs/tanh.png)

The tanh non-linearity squashes a real-valued number to the range [-1, 1]. Like the sigmoid neuron, its activations saturate, but unlike the sigmoid neuron its output is zero-centered. Therefore, in practice the tanh non-linearity is always preferred to the sigmoid nonlinearity.

tanh非线性函数将实值输入压缩到[-1,1]的范围。和sigmoid神经元类似，其激活是饱和的，但与sigmoid不同的是其输出是零均值的。所以，在实践中tanh非线性函数一直比sigmoid非线性函数受欢迎。

### Rectified Linear Unit

![Image](http://lamda.nju.edu.cn/weixs/project/CNNTricks/imgs/relu.png)

The Rectified Linear Unit (ReLU) has become very popular in the last few years. It computes the function $f(x)=max(0,x)$, which is simply thresholded at zero.

整流线性单元ReLU在过去几年中变得非常流行，其函数形式是$f(x)=max(0,x)$，就是一个简单的0阈值函数。

There are several pros and cons to using the ReLUs: 采用ReLU有几个优缺点

1. (Pros) Compared to sigmoid/tanh neurons that involve expensive operations (exponentials, etc.), the ReLU can be implemented by simply thresholding a matrix of activations at zero. Meanwhile, ReLUs does not suffer from saturating.

- （优点）与sigmoid/tanh神经元相比，实现简单，计算量小，同时没有饱和现象。

2. (Pros) It was found to greatly accelerate (e.g., a factor of 6 in [1]) the convergence of stochastic gradient descent compared to the sigmoid/tanh functions. It is argued that this is due to its linear, non-saturating form.

- （优点）采用随机梯度下降法时，与sigmoid/tanh函数相比，可以极大加速收敛（比如在[1]中是6倍加速），有观点认为是由于其线性非饱和的形式。

3. (Cons) Unfortunately, ReLU units can be fragile during training and can “die”. For example, a large gradient flowing through a ReLU neuron could cause the weights to update in such a way that the neuron will never activate on any datapoint again. If this happens, then the gradient flowing through the unit will forever be zero from that point on. That is, the ReLU units can irreversibly die during training since they can get knocked off the data manifold. For example, you may find that as much as 40% of your network can be “dead” (i.e., neurons that never activate across the entire training dataset) if the learning rate is set too high. With a proper setting of the learning rate this is less frequently an issue.

- （缺点）不幸的是，ReLU单元在训练中可能很脆弱，会死掉。比如，很大的梯度经过ReLU神经元可能会导致权重更新后，神经元在任何数据点上永远也不会再激活了。如果发生这样的情况，那么流经这个单元的梯度从这个点开始会一直是0。也就是说，这个ReLU单元会不可逆的死去，因为它们已经从数据流形上脱落了。比如，如果学习速率过高，可能会发现网络的40%都死掉了（即，神经元在整个训练数据集上都不会激活）。如果学习速率设定合理，那么这个问题不会经常出现。

### Leaky ReLU

![Image](http://lamda.nju.edu.cn/weixs/project/CNNTricks/imgs/leaky.png)

Leaky ReLUs are one attempt to fix the “dying ReLU” problem. Instead of the function being zero when  *x*<0, a leaky ReLU will instead have a small negative slope (of 0.01, or so). That is, the function computes $f(x)=\alpha x$ if *x*<0 and $f(x)=x$ if $x\geq 0$, where $\alpha$ is a small constant. Some people report success with this form of activation function, but the results are not always consistent.

Leaky ReLU是解决ReLU死掉的问题的一种尝试。Leaky ReLU函数在*x*<0时不是0，而是有一个小的负斜率，比如0.01。即，$f(x)=\alpha x$当*x*<0，而$f(x)=x$当$x\geq 0$，这里$\alpha$是一个小的常数。一些人用这种激活函数得出了成功的结果，但并不总是这样。

### Parametric ReLU

Nowadays, a broader class of activation functions, namely the rectified unit family, were proposed. In the following, we will talk about the variants of ReLU.

现在提出了更广的一类激活函数，即整流单元函数族，下面我们讨论这种ReLU的变种函数。

![Image](http://lamda.nju.edu.cn/weixs/project/CNNTricks/imgs/relufamily.png)

ReLU, Leaky ReLU, PReLU and RReLU. In these figures, for PReLU, $\alpha_i$ is learned and for Leaky ReLU $\alpha_i$ is fixed. For RReLU, $\alpha_{ji}$ is a random variable keeps sampling in a given range, and remains fixed in testing.

ReLU, Leaky ReLU, PReLU和RReLU。在这些图中，对于PReLU，$\alpha_i$是学习得到的；对于Leaky ReLU，$\alpha_i$是固定的；对于RReLU，$\alpha_{ji}$是一随机变量，满足一定分布，在测试中保持固定。

The first variant is called parametric rectified linear unit (PReLU) [4]. In PReLU, the slopes of negative part are learned from data rather than pre-defined. He et al. [4] claimed that PReLU is the key factor of surpassing human-level performance on ImageNet classification task. The back-propagation and updating process of PReLU is very straightforward and similar to traditional ReLU, which is shown in Page. 43 of the slides.

第一个变种称为PReLU[4]。在PReLU中，负值部分的斜率是从数据中学习得到的，而不是预定义的。He et al. [4]声称PReLU是在ImageNet分类任务中超越人眼表现的关键因素。PReLU的反向传播和更新过程很直接，与传统ReLU很类似，这在幻灯片的43页有介绍。

### Randomized ReLU

The second variant is called randomized rectified linear unit (RReLU). In RReLU, the slopes of negative parts are randomized in a given range in the training, and then fixed in the testing. As mentioned in [5], in a recent Kaggle National Data Science Bowl (NDSB) competition, it is reported that RReLU could reduce overfitting due to its randomized nature. Moreover, suggested by the NDSB competition winner, the random $\alpha_i$ in training is sampled from 1/*U*(3,8) and in test time it is fixed as its expectation, i.e., 2/(*l+u*)=2/11.

第二种变体称为Randomized ReLU，在RReLU中，负值部分的斜率在训练过程中是一定范围内的随机数，在测试过程中固定下来。如[5]中提到的，在最近的Kaggle National Data Science Bowl(NDSB)比赛中，RReLU由于其随机的特性，据称可以减少过拟合现象。而且，据NDSB比赛赢家建议，训练过程中随机的$\alpha_i$是从1/U(3,8)中取样得到，在测试时固定为其期望值，即2/(*l+u*)=2/11。

In [5], the authors evaluated classification performance of two state-of-the-art CNN architectures with different activation functions on the CIFAR-10, CIFAR-100 and NDSB data sets, which are shown in the following tables. Please note that, for these two networks, activation function is followed by each convolutional layer. And the *a* in these tables actually indicates $1/\alpha$, where $\alpha$ is the aforementioned slopes.

在[5]中，作者将两个当时最好的CNN架构在CIFAR-10,CIFAR-100和NDSB数据集上进行分类性能比较，采用了不同的激活函数，如下表所示。请注意，对于这两个网络来说，激活函数就在每个卷积层后。表格中的*a*实际上代表$1/\alpha$，$\alpha$为前面所说的斜率。

![Image](http://lamda.nju.edu.cn/weixs/project/CNNTricks/imgs/relures.png)

From these tables, we can find the performance of ReLU is not the best for all the three data sets. For Leaky ReLU, a larger slope $\alpha$ will achieve better accuracy rates. PReLU is easy to overfit on small data sets (its training error is the smallest, while testing error is not satisfactory), but still outperforms ReLU. In addition, RReLU is significantly better than other activation functions on NDSB, which shows RReLU can overcome overfitting, because this data set has less training data than that of CIFAR-10/CIFAR-100. In conclusion, three types of ReLU variants all consistently outperform the original ReLU in these three data sets. And PReLU and RReLU seem better choices. Moreover, He et al. also reported similar conclusions in [4].

从这些表格里，我们可以发现ReLU的表现在所有数据集中都不是最好的。对于Leaky ReLU来说，大的斜率$\alpha$可以得到更好的准确率。PReLU在小数据集上容易过拟合（其训练错误率最小，但测试错误率没那么理想），但仍然比ReLU表现要好。另外，RReLU在NDSB数据集上明显比其他激活函数要好，这说明RReLU可以克服过拟合，因为这个数据集训练数据比CIFAR-10/CIFAR-100都要少。总结一下，三种ReLU变体都在三个数据集上一致要比ReLU激活函数要好，而且PReLU和RReLU似乎是更好的选择。而且，He et al.[4]也有类似的结论。

##Sec. 6: Regularizations

There are several ways of controlling the capacity of Neural Networks to prevent overfitting: 有几种方法可以控制神经网络防止过拟合

- **L2 regularization** is perhaps the most common form of regularization. It can be implemented by penalizing the squared magnitude of all parameters directly in the objective. That is, for every weight *w* in the network, we add the term $\frac{1}{2}\lambda w^2$ to the objective, where $\lambda$ is the regularization strength. It is common to see the factor of 1/2 in front because then the gradient of this term with respect to the parameter *w* is simply $\lambda w$ instead of $2\lambda w$. The L2 regularization has the intuitive interpretation of heavily penalizing peaky weight vectors and preferring diffuse weight vectors.

- **L2正则化**可能是正则化最常见的形式，通过在目标函数中直接惩罚所有参数的幅度平方来实现，即，对于网络中的每个权重*w*，我们在目标函数中增加$\frac{1}{2}\lambda w^2$项，这里$\lambda$为正则化强度。L2正则化的直观解释是，严重惩罚峰值的权重矢量，希望能权重矢量能扩散开来。

- **L1 regularization** is another relatively common form of regularization, where for each weight *w* we add the term $\lambda |w|$ to the objective. It is possible to combine the L1 regularization with the L2 regularization: $\lambda_1 |w|+\lambda_2 w^2$ (this is called Elastic net regularization). The L1 regularization has the intriguing property that it leads the weight vectors to become sparse during optimization (i.e. very close to exactly zero). In other words, neurons with L1 regularization end up using only a sparse subset of their most important inputs and become nearly invariant to the “noisy” inputs. In comparison, final weight vectors from L2 regularization are usually diffuse, small numbers. In practice, if you are not concerned with explicit feature selection, L2 regularization can be expected to give superior performance over L1.

- **L1正则化**是另一个相对常见的正则化形式，这里对于权重*w*，我们在目标函数中增加$\lambda |w|$项。可以将L1正则化项和L2正则化项组合起来：$\lambda_1 |w|+\lambda_2 w^2$，这称为弹性(Elastic)网络正则化。L1正则化吸引人的性质是可以引导权值矢量在优化过程中变稀疏（即，一些分量接近0），换句话说，L1正则化的神经元最后会只用输入最重要的一部分稀疏子集，从而对“含噪”输入具有不变性。对比起来，L2正则化得到的权值矢量通常会扩散为很小的数。实际中，如果不是特别关注显式的特征选择，L2正则化会比L1正则化有更好的期待效果。

- **Max norm constraints**. Another form of regularization is to enforce an absolute upper bound on the magnitude of the weight vector for every neuron and use projected gradient descent to enforce the constraint. In practice, this corresponds to performing the parameter update as normal, and then enforcing the constraint by clamping the weight vector $\vec{w}$ of every neuron to satisfy $\parallel \vec{w} \parallel_2 <c$. Typical values of *c* are on orders of 3 or 4. Some people report improvements when using this form of regularization. One of its appealing properties is that network cannot “explode” even when the learning rates are set too high because the updates are always bounded.

- **最大范数约束**。另一种形式的正则化是对每个神经元的权重矢量的幅度强加一个上限，通过投影梯度下降来强加这个约束。在实践中，参数更新与正常一样，通过限制权重矢量$\vec{w}$的范数，即要求$\parallel \vec{w} \parallel_2 <c$，来实施限制，*c*的典型值为3或4。一些人用了这种正则化后得到了改进的结果。其中一个吸引人的特性是，即使学习速率过高，网络也不会“爆炸”，因为权值更新一直受限。

- **Dropout** is an extremely effective, simple and recently introduced regularization technique by Srivastava et al. in [6] that complements the other methods (L1, L2, maxnorm). During training, dropout can be interpreted as sampling a Neural Network within the full Neural Network, and only updating the parameters of the sampled network based on the input data. (However, the exponential number of possible sampled networks are not independent because they share the parameters.) During testing there is no dropout applied, with the interpretation of evaluating an averaged prediction across the exponentially-sized ensemble of all sub-networks (more about ensembles in the next section). In practice, the value of dropout ratio *p*=0.5 is a reasonable default, but this can be tuned on validation data.

- **Dropout**是一种极其高效、简单的正则化方法，近年由Srivastava et al. 在[6]中提出，文中是作为其他正则化方法(L1, L2, maxnorm)的补充。训练过程中，dropout可以解释为对全神经网络的一种采样，只对采样到的网络根据输入进行参数更新。（但对网络进行的采样的可能性数目为指数级的，它们并不独立，因为一些参数是共享的）在测试过程中没有dropout。实践中，dropout率*p*=0.5是合理的默认值，但这也可以根据验证集调整。

![Image](http://lamda.nju.edu.cn/weixs/project/CNNTricks/imgs/dropout.png)

The most popular used regularization technique dropout [6]. While training, dropout is implemented by only keeping a neuron active with some probability *p* (a hyper-parameter), or setting it to zero otherwise. In addition, Google applied for a US patent for dropout in 2014.

最受欢迎的正则化技术dropout[6]。当训练时，实现dropout的原则是，以某概率*p*（超参数）保持一个神经元是活跃的，或将其设为0。另外，Google在2014年注册了一个dropout的美国专利。

## Sec. 7: Insights from Figures

Finally, from the tips above, you can get the satisfactory settings (e.g., data processing, architectures choices and details, etc.) for your own deep networks. During training time, you can draw some figures to indicate your networks’ training effectiveness.

最后，从上面的提示和技巧中，你使自己的深度网络得到了满意的设置（如，数据处理，架构选择和细节等）。在训练期间，你还可以画一些图，来指示网络训练的有效性。

- As we have known, the learning rate is very sensitive. From Fig. 1 in the following, a very high learning rate will cause a quite strange loss curve. A low learning rate will make your training loss decrease very slowly even after a large number of epochs. In contrast, a high learning rate will make training loss decrease fast at the beginning, but it will also drop into a local minimum. Thus, your networks might not achieve a satisfactory results in that case. For a good learning rate, as the red line shown in Fig. 1, its loss curve performs smoothly and finally it achieves the best performance.

- 我们都知道，学习速率是非常敏感的。从下图1中可以看出，很高的学习速率会导致奇怪的损失曲线。低学习速率会使训练损失降低的很缓慢，过了很多个epoch也没有降低多少。而高的学习速率会使训练损失在开始降低的很快，但可能会掉入局部极值。所以，网络在这些情况下不能得到理想的结果。对于一个好的学习速率，如图1中的红线，其损失曲线非常平滑，最后得到最佳的性能。

- Now let’s zoom in the loss curve. The epochs present the number of times for training once on the training data, so there are multiple mini batches in each epoch. If we draw the classification loss every training batch, the curve performs like Fig. 2. Similar to Fig. 1, if the trend of the loss curve looks too linear, that indicates your learning rate is low; if it does not decrease much, it tells you that the learning rate might be too high. Moreover, the “width” of the curve is related to the batch size. If the “width” looks too wide, that is to say the variance between every batch is too large, which points out you should increase the batch size.

- 现在我们对损失曲线进行放大。epochs是在训练数据集上的训练次数，所以在每个epoch中有多个mini batch。如果我们对每个训练batch都画出其分类损失，这个曲线可能会像图2。与图1类似，如果损失曲线的趋向看起来太线性，那就说明学习速率太小；如果下降不多，那说明学习速率可能过大。而且，曲线的“宽度”与batch size相关；如果宽度过宽，那么每个batch之间的变化太大，这说明应当增加batch size。

- Another tip comes from the accuracy curve. As shown in Fig. 3, the red line is the training accuracy, and the green line is the validation one. When the validation accuracy converges, the gap between the red line and the green one will show the effectiveness of your deep networks. If the gap is big, it indicates your network could get good accuracy on the training data, while it only achieve a low accuracy on the validation set. It is obvious that your deep model overfits on the training set. Thus, you should increase the regularization strength of deep networks. However, no gap meanwhile at a low accuracy level is not a good thing, which shows your deep model has low learnability. In that case, it is better to increase the model capacity for better results.

- 从准确率曲线也可以得到一些提示。如图3所示，红线是训练准确率，绿线是验证数据准确率。当验证准确率收敛时，红线和绿线之间的差距能说明深度模型的有效性。如果差距很大，说明网络在训练时能得到很好的结果，而在验证集上效果则没那么好，这明显说明模型在训练集上过拟合了，所以应当增加模型正则化的强度。但没有差距同时正确率还很低，也不是一件好事，这说明模型的学习能力不足，这种情况下，最好增加模型的容量来学习到更好的结果。

![Image](http://lamda.nju.edu.cn/weixs/project/CNNTricks/imgs/trainfigs.png)

## Sec. 8: Ensemble 集成/组合

In machine learning, ensemble methods [8] that train multiple learners and then combine them for use are a kind of state-of-the-art learning approach. It is well known that an ensemble is usually significantly more accurate than a single learner, and ensemble methods have already achieved great success in many real-world tasks. In practical applications, especially challenges or competitions, almost all the first-place and second-place winners used ensemble methods.

在机器学习中，组合方法[8]，也就是训练多个学习器，然后组合起来使用，这是一种最新的学习方法。大家都知道，组合方法通常比单独的学习器准确度有明显提升，组合方法已经在现实世界任务中取得了很大成功。在实际应用中，尤其是挑战赛或比赛中，几乎所有第一名和第二名的赢家都使用组合方法。

Here we introduce several skills for ensemble in the deep learning scenario. 这里我们介绍深度学习场景中的几种组合技能。

- **Same model, different initialization**. Use cross-validation to determine the best hyperparameters, then train multiple models with the best set of hyperparameters but with different random initialization. The danger with this approach is that the variety is only due to initialization.

- **同样的模型，不同的初始化**。用交叉验证来确定最佳超参数，然后用最佳超参数集训练多个不同随机初始化的模型。这种方法的危险之处在于仅仅由于初始化导致的多样性。

- **Top models discovered during cross-validation**. Use cross-validation to determine the best hyperparameters, then pick the top few (e.g., 10) models to form the ensemble. This improves the variety of the ensemble but has the danger of including suboptimal models. In practice, this can be easier to perform since it does not require additional retraining of models after cross-validation. Actually, you could directly select several state-of-the-art deep models from Caffe Model Zoo to perform ensemble.

- **交叉验证发现的最佳模型**。用交叉验证来确定最佳超参数，然后选出最佳的几个（如10个）模型来形成组合。这改进了组合的多样性，但危险在于包括了次佳模型。在实践中这会比较容易实施，因为在交叉验证后不需要额外的重新训练。实际上，可以直接从Caffe Model Zoo中选择几个最好的模型进行组合。

- **Different checkpoints of a single model**. If training is very expensive, some people have had limited success in taking different checkpoints of a single network over time (for example after every epoch) and using those to form an ensemble. Clearly, this suffers from some lack of variety, but can still work reasonably well in practice. The advantage of this approach is that is very cheap.

- **单个模型的不同checkpoints**。如果训练代价非常高，那么可以利用单个网络的在不同时间（比如每个epoch结束）的checkpoints进行组合，这有过几个成功的案例。很明显，这缺少一些多样性，但在实践中仍然可以工作的不错。这种方法的优点在于代价低。

- **Some practical examples**. If your vision tasks are related to high-level image semantic, e.g., event recognition from still images, a better ensemble method is to employ multiple deep models trained on different data sources to extract different and complementary deep representations. For example in the Cultural Event Recognition challenge in associated with ICCV’15, we utilized five different deep models trained on images of ImageNet, Place Database and the cultural images supplied by the competition organizers. After that, we extracted five complementary deep features and treat them as multi-view data. Combining “early fusion” and “late fusion” strategies described in [7], we achieved one of the best performance and ranked the 2nd place in that challenge. Similar to our work, [9] presented the Stacked NN framework to fuse more deep networks at the same time.

- **一些实际例子**。如果视觉任务与高层图像语义相关，如，静止图像中的事件识别，那么更好的组合方法是采用多个深度模型，它们分别在不同的数据源训练，提取不同而且互补的深度表示。比如，在ICCV'15中的文化事件识别挑战赛中，我们利用5种不同的深度模型，分别在ImageNet, Place数据集和竞赛组织者提供的文化图像集上训练。此后，我们提取了5种互补的深度特征，将其作为多视图数据。结合[7]中“早期融合”和“晚期融合”的策略，我们取得了最佳表现之一，在挑战赛中排名第2。与我们的工作类似，[9]提出了Stacked NN框架来同时融合更多的深度网络。

## Miscellaneous

In real world applications, the data is usually class-imbalanced: some classes have a large number of images/training instances, while some have very limited number of images. As discussed in a recent technique report [10], when deep CNNs are trained on these imbalanced training sets, the results show that imbalanced training data can potentially have a severely negative impact on overall performance in deep networks. For this issue, the simplest method is to balance the training data by directly up-sampling and down-sampling the imbalanced data, which is shown in [10]. Another interesting solution is one kind of special crops processing in our challenge solution [7]. Because the original cultural event images are imbalanced, we merely extract crops from the classes which have a small number of training images, which on one hand can supply diverse data sources, and on the other hand can solve the class-imbalanced problem. In addition, you can adjust the fine-tuning strategy for overcoming class-imbalance. For example, you can divide your own data set into two parts: one contains the classes which have a large number of training samples (images/crops); the other contains the classes of limited number of samples. In each part, the class-imbalanced problem will be not very serious. At the beginning of fine-tuning on your data set, you firstly fine-tune on the classes which have a large number of training samples (images/crops), and secondly, continue to fine-tune but on the classes with limited number samples.

在真实世界应用中，数据通常在类别间不均衡：一些类的图像或训练样本很多，一些类的图像很少。最近的科技报告[10]中提出，当深度CNN在这些不均衡训练集上进行训练时，得到的结果说明，不均衡的训练数据可能对深度网络的总体表现有严重的负面作用。对这个问题，最简单的方法是通过直接对不均衡数据进行上采样和下采样，来使训练数据均衡下来，这在[10]中有叙述。另外一个有趣的解决方案是我们挑战赛方案[7]中的一种特殊剪切过程。由于开始的文化事件图像不均衡，我们仅仅从含有训练数目少的类中选取了那些剪切，这一方面可以提供多样的数据源，另一方面可以解决类别不均衡问题。另外，可以调整精调的策略来解决类别不均衡问题。比如，你可以将自己的数据分成两部分：一种包括很多训练样本（图像/剪切），另一种包括样本数量有限的类。在每一部分中，类别不均衡的问题都不会太严重。在自己的数据集上进行精调的开始时，首先精调训练样本多的类，然后继续精调那些样本少的类。

# References & Source Links

1. A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet Classification with Deep Convolutional Neural Networks. In NIPS, 2012
2. A Brief Overview of Deep Learning, which is a guest post by Ilya Sutskever.
3. CS231n: Convolutional Neural Networks for Visual Recognition of Stanford University, held by Prof. Fei-Fei Li and Andrej Karpathy.
4. K. He, X. Zhang, S. Ren, and J. Sun. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. In ICCV, 2015.
5. B. Xu, N. Wang, T. Chen, and M. Li. Empirical Evaluation of Rectified Activations in Convolution Network. In ICML Deep Learning Workshop, 2015.
6. N. Srivastava, G. E. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR, 15(Jun):1929−1958, 2014.
7. X.-S. Wei, B.-B. Gao, and J. Wu. Deep Spatial Pyramid Ensemble for Cultural Event Recognition. In ICCV ChaLearn Looking at People Workshop, 2015.
8. Z.-H. Zhou. Ensemble Methods: Foundations and Algorithms. Boca Raton, FL: Chapman & HallCRC/, 2012. (ISBN 978-1-439-830031)
9. M. Mohammadi, and S. Das. S-NN: Stacked Neural Networks. Project in Stanford CS231n Winter Quarter, 2015.
10. P. Hensman, and D. Masko. The Impact of Imbalanced Training Data for Convolutional Neural Networks. Degree Project in Computer Science, DD143X, 2015.