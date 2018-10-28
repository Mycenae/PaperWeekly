# Layers 层

To create a Caffe model you need to define the model architecture in a protocol buffer definition file (prototxt). 要生成caffe模型，需要在一个protocol buffer定义文件(prototxt)中定义模型架构。

Caffe layers and their parameters are defined in the protocol buffer definitions for the project in caffe.proto. Caffe层与其参数定义在工程的caffe.proto中prototol buffer定义内。

## Data Layers 数据层

Data enters Caffe through data layers: they lie at the bottom of nets. Data can come from efficient databases (LevelDB or LMDB), directly from memory, or, when efficiency is not critical, from files on disk in HDF5 or common image formats.

数据从数据层进入Caffe，数据层在网络最底部。数据可以从高效的数据库(LevelDB or LMDB)中来，或直接从内存中来，或者在效率没那么重要的时候，从磁盘的文件以HDF5或普通图像格式来。

Common input preprocessing (mean subtraction, scaling, random cropping, and mirroring) is available by specifying TransformationParameters by some of the layers. The bias, scale, and crop layers can be helpful with transforming the inputs, when TransformationParameter isn’t available.

普通输入预处理（减去均值，尺度变换，随机裁剪，镜像）可以在一些层中通过指定TransformationParameters完成。当TransformationParameters不可用时候，Bias, scale和crop层在输入变换中很有用。

Layers:

- Image Data - read raw images.
- Database - read data from LEVELDB or LMDB.
- HDF5 Input - read HDF5 data, allows data of arbitrary dimensions.
- HDF5 Output - write data as HDF5.
- Input - typically used for networks that are being deployed.
- Window Data - read window data file.
- Memory Data - read data directly from memory.
- Dummy Data - for static data and debugging.

Note that the Python Layer can be useful for create custom data layers. 注意Python的Layer可用于创建定制数据层。

## Vision Layers 视觉层

Vision layers usually take images as input and produce other images as output, although they can take data of other types and dimensions. A typical “image” in the real-world may have one color channel (c=1), as in a grayscale image, or three color channels (c=3) as in an RGB (red, green, blue) image. But in this context, the distinguishing characteristic of an image is its spatial structure: usually an image has some non-trivial height h>1 and width w>1. This 2D geometry naturally lends itself to certain decisions about how to process the input. In particular, most of the vision layers work by applying a particular operation to some region of the input to produce a corresponding region of the output. In contrast, other layers (with few exceptions) ignore the spatial structure of the input, effectively treating it as “one big vector” with dimension chw.

视觉层通常以图像为输入，并生成其他图像作为输出，但是也可以以其他类型和维数的数据为输入。真实世界的典型“图像”可能只有一个颜色通道c=1，这就是灰度图像，或三个颜色通道c=3，这就是RGB彩色图像。但在这个上下文中，图像的显著特征是其空间结构：通常一幅图像有高度h>1和宽度w>1。这种2D几何形状自然的带来了一些如何处理输入的方式。特别的，大多数视觉层的工作方式都是在输入的某区域中进行特定操作，生成对应的输出中的区域。而对比起来，其他层（有一些特例）忽略输入的空间结构，将其视为一个维度为chw的大型矢量。

Layers:

- Convolution Layer - convolves the input image with a set of learnable filters, each producing one feature map in the output image. 将输入图像与一个可学习的滤波器集卷积，每个生成输出图像中的一个特征图。
- Pooling Layer - max, average, or stochastic pooling.
- Spatial Pyramid Pooling (SPP)
- Crop - perform cropping transformation.
- Deconvolution Layer - transposed convolution.

- Im2Col - relic helper layer that is not used much anymore.

## Recurrent Layers 循环层

Layers: Recurrent / RNN / Long-Short Term Memory (LSTM)

## Common Layers 普通层

Layers: Inner Product - fully connected layer. / Dropout / Embed - for learning embeddings of one-hot encoded vector (takes index as input).

## Normalization Layers 归一化层

- Local Response Normalization (LRN) - performs a kind of “lateral inhibition” by normalizing over local input regions.
- Mean Variance Normalization (MVN) - performs contrast normalization / instance normalization.
- Batch Normalization - performs normalization over mini-batches.

The bias and scale layers can be helpful in combination with normalization.

## Activation / Neuron Layers 激活层/神经元层

In general, activation / Neuron layers are element-wise operators, taking one bottom blob and producing one top blob of the same size. In the layers below, we will ignore the input and out sizes as they are identical:

一般来说，激活/神经元层是逐元素的运算符，下层blob输入和生成的上层blob维数一样。在下面的层中，我们将忽略输入输出维数，因为是一样的：

Input & Output dimensions: n * c * h * w

Layers:

- ReLU / Rectified-Linear and Leaky-ReLU - ReLU and Leaky-ReLU rectification.
- PReLU - parametric ReLU.
- ELU - exponential linear rectification.
- Sigmoid
- TanH
- Absolute Value
- Power - f(x) = (shift + scale * x) ^ power.
- Exp - f(x) = base ^ (shift + scale * x).
- Log - f(x) = log(x).
- BNLL - f(x) = log(1 + exp(x)).
- Threshold - performs step function at user defined threshold.
- Bias - adds a bias to a blob that can either be learned or fixed.
- Scale - scales a blob by an amount that can either be learned or fixed.

## Utility Layers 工具层

Layers: Flatten / Reshape / Batch Reindex / Split / Concat / Slicing / Eltwise - element-wise operations such as product or sum between two blobs. / Filter or Mask - mask or select output using last blob. / Parameter - enable parameters to be shared between layers. / Reduction - reduce input blob to scalar blob using operations such as sum or mean. / Silence - prevent top-level blobs from being printed during training. / ArgMax / Softmax / Python - allows custom Python layers.

## Loss Layers 损失层

Loss drives learning by comparing an output to a target and assigning cost to minimize. The loss itself is computed by the forward pass and the gradient w.r.t. to the loss is computed by the backward pass.

损失驱动学习，通过将输出与目标比较，指定进行最小化的代价函数，实现学习过程。损失本身是由前向过程计算得到的，损失的梯度则是由后向过程计算得到的。

Layers:

- Multinomial Logistic Loss
- Infogain Loss - a generalization of MultinomialLogisticLossLayer.
- Softmax with Loss - computes the multinomial logistic loss of the softmax of its inputs. It’s conceptually identical to a softmax layer followed by a multinomial logistic loss layer, but provides a more numerically stable gradient.
- Sum-of-Squares / Euclidean - computes the sum of squares of differences of its two inputs.
- Hinge / Margin - The hinge loss layer computes a one-vs-all hinge (L1) or squared hinge loss (L2).
- Sigmoid Cross-Entropy Loss - computes the cross-entropy (logistic) loss, often used for predicting targets interpreted as probabilities.
- Accuracy / Top-k layer - scores the output as an accuracy with respect to target – it is not actually a loss and has no backward step.
- Contrastive Loss