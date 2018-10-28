# Loss 损失函数

In Caffe, as in most of machine learning, learning is driven by a loss function (also known as an error, cost, or objective function). A loss function specifies the goal of learning by mapping parameter settings (i.e., the current network weights) to a scalar value specifying the “badness” of these parameter settings. Hence, the goal of learning is to find a setting of the weights that minimizes the loss function.

在Caffe中，和在大多数机器学习中一样，学习是由损失函数驱动的（也称为错误函数、代价函数或目标函数）。损失函数将参数设置（即网络目前的参数）映射到一个标量值，这个值是这些参数设置“坏”的程度的度量，这样从而指定了学习的目的。所以，学习的目标是找到一种权值设置来最小化损失函数。

The loss in Caffe is computed by the Forward pass of the network. Each layer takes a set of input (bottom) blobs and produces a set of output (top) blobs. Some of these layers’ outputs may be used in the loss function. A typical choice of loss function for one-versus-all classification tasks is the SoftmaxWithLoss function, used in a network definition as follows, for example:

Caffe中的损失函数是由网络的前向过程来计算的。每一层都有输入（下层）blob集合，都会产生一个输出（上层）blob集合。一些层的输出可能用在损失函数中。一对多分类任务损失函数的典型选择是SoftmaxWithLoss函数，用在网络定义中的例子如下所示：

```
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "pred"
  bottom: "label"
  top: "loss"
}
```

In a SoftmaxWithLoss function, the top blob is a scalar (empty shape) which averages the loss (computed from predicted labels pred and actuals labels label) over the entire mini-batch.

在SoftmaxWithLoss函数中，上层blob是一个标量，其值为整个mini-batch的平均损失（由预测出的标签和真实标签计算得到）。

## Loss weights 损失权重

For nets with multiple layers producing a loss (e.g., a network that both classifies the input using a SoftmaxWithLoss layer and reconstructs it using a EuclideanLoss layer), loss weights can be used to specify their relative importance.

对于多层产生输出的网络来说（如，一个网络既对输入用SoftmaxWithLoss层进行分类，也用Euclidean层进行重建），可以用损失权重来指定其相对重要性。

By convention, Caffe layer types with the suffix Loss contribute to the loss function, but other layers are assumed to be purely used for intermediate computations. However, any layer can be used as a loss by adding a field `loss_weight: <float>` to a layer definition for each top blob produced by the layer. Layers with the suffix Loss have an implicit loss_weight: 1 for the first top blob (and loss_weight: 0 for any additional tops); other layers have an implicit loss_weight: 0 for all tops. So, the above SoftmaxWithLoss layer could be equivalently written as:

按照惯例，Caffe的层中，以Loss为后缀的，都对损失函数有贡献，其他层认为只纯粹作中间计算。但是，任何层都可以用作损失，方法是在层定义中对该层产生的每个top blob加一个域`loss_weight: <float>`。含有Loss后缀的层都有个隐含的loss_weight: 1（对任何另外的top有隐含的loss_weight: 0），其他层的所有top都有隐含的loss_weight: 0。所以，上述SoftmaxWithLoss层可以等价的写成如下形式：

```
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "pred"
  bottom: "label"
  top: "loss"
  loss_weight: 1
}
```

However, any layer able to backpropagate may be given a non-zero loss_weight, allowing one to, for example, regularize the activations produced by some intermediate layer(s) of the network if desired. For non-singleton outputs with an associated non-zero loss, the loss is computed simply by summing over all entries of the blob.

但是，任何可以反向传播的层都可以给定非零的loss_weight，比如，可以对网络某些中间层产生的激活进行正则化。对于非单值的输出，如果有关联的非零损失权重，那么loss的计算就是简单的将blob所有元素相加。

The final loss in Caffe, then, is computed by summing the total weighted loss over the network, as in the following pseudo-code:

最后在Caffe中的损失函数，就是整个网络中所有损失的加权求和，如下面的伪代码所示：

```
loss := 0
for layer in layers:
  for top, loss_weight in layer.tops, layer.loss_weights:
    loss += loss_weight * sum(top)
```