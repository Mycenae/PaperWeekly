# Forward and Backward 前向与后向

The forward and backward passes are the essential computations of a Net. 前向计算和后向计算是Net对象计算的本质。

![Image](http://caffe.berkeleyvision.org/tutorial/fig/forward_backward.png)

Let’s consider a simple logistic regression classifier. 我们考虑一个简单的logistic回归分类器。

The **forward** pass computes the output given the input for inference. In forward Caffe composes the computation of each layer to compute the “function” represented by the model. This pass goes from bottom to top.

前向传播是在推理时，给定输入，计算输出。前向的过程，Caffe的计算由每层的计算组成，得到整个模型代表的“函数”。前向传播是从下往上。

![Image](http://caffe.berkeleyvision.org/tutorial/fig/forward.jpg)

The data x is passed through an inner product layer for g(x) then through a softmax for h(g(x)) and softmax loss to give $f_W(x)$.

数据x经过内积层得到g(x)然后经过softmax层得到h(g(x))，其softmax损失函数为$f_W(x)$。

The **backward** pass computes the gradient given the loss for learning. In backward Caffe reverse-composes the gradient of each layer to compute the gradient of the whole model by automatic differentiation. This is back-propagation. This pass goes from top to bottom.

反向传播计算的是给定学习的损失函数的梯度。在反向传播过程中，Caffe也相反的用每层计算梯度来组成整个梯度，整个过程是自动计算求导的。这就是反向传播，过程是由上至下。

![Image](http://caffe.berkeleyvision.org/tutorial/fig/backward.jpg)

The backward pass begins with the loss and computes the gradient with respect to the output $∂fW/∂h$. The gradient with respect to the rest of the model is computed layer-by-layer through the chain rule. Layers with parameters, like the `INNER_PRODUCT` layer, compute the gradient with respect to their parameters $∂fW/∂W_ip$ during the backward step.

反向传播开始于损失函数，计算对输出的梯度$∂fW/∂h$。对模型其他部分的梯度逐层根据链式法则计算。有参数的层，如`INNER_PRODUCT`层，也在反向传播过程中计算对其参数的梯度$∂fW/∂W_ip$。

These computations follow immediately from defining the model: Caffe plans and carries out the forward and backward passes for you.

这些计算在定义模型之后立刻就进行了：Caffe负责为你计划并执行正向和反向传播的过程。

- The `Net::Forward()` and `Net::Backward()` methods carry out the respective passes while `Layer::Forward()` and `Layer::Backward()` compute each step.
- `Net::Forward()`和`Net::Backward()`方法分别执行每个方向的计算，而`Layer::Forward()`和`Layer::Backward()`计算每个步骤。
- Every layer type has `forward_{cpu,gpu}()` and `backward_{cpu,gpu}()` methods to compute its steps according to the mode of computation. A layer may only implement CPU or GPU mode due to constraints or convenience.
- 每个种类的层都有`forward_{cpu,gpu}()`和`backward_{cpu,gpu}()`方法根据计算模式来计算每个步骤。由于限制或便捷的原因，一个层可以只实现CPU或GPU方法。

The **Solver** optimizes a model by first calling forward to yield the output and loss, then calling backward to generate the gradient of the model, and then incorporating the gradient into a weight update that attempts to minimize the loss. Division of labor between the Solver, Net, and Layer keep Caffe modular and open to development.

**Solver**优化模型的方法是，首先调用前向函数来产生输出和损失，然后调用反向函数来产生模型梯度，将用梯度进行权值更新，以最小化损失函数。Solver, Net和Layer的分工使Caffe模块化，也一直在开发过程中。

For the details of the forward and backward steps of Caffe’s layer types, refer to the layer catalogue.

Caffe各种类层的前向和后向步骤，详见layer目录。