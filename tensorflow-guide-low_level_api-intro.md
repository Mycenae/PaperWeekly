# Introduction 简介

This guide gets you started programming in the low-level TensorFlow APIs (TensorFlow Core), showing you how to:

这篇指南使你开始用TensorFlow底层API（TensorFlow核心）进行编程，让你怎样去：

- Manage your own TensorFlow program (a `tf.Graph`) and TensorFlow runtime (a `tf.Session`), instead of relying on Estimators to manage them.
- 管理自己的TensorFlow程序(`tf.Graph`)和TensorFlow运行时(`tf.Session`)，而不是依靠Estimator管理它们。
- Run TensorFlow operations, using a `tf.Session`.
- 用`tf.Session`运行TensorFlow操作。
- Use high level components (`datasets`, `layers`, and `feature_columns`) in this low level environment.
- 在底层环境中使用高层组件(`datasets`, `layers`, `feature_columns`)。
- Build your own training loop, instead of using the one provided by Estimators.
- 构建自己的训练循环，而不使用Estimator提供的训练循环。

We recommend using the higher level APIs to build models when possible. Knowing TensorFlow Core is valuable for the following reasons:

我们推荐使用高层API来构建模型，但知道TensorFlow核心有如下好处：

- Experimentation and debugging are both more straight forward when you can use low level TensorFlow operations directly.
- 当可以直接使用TensorFlow底层操作时，试验和调试都会更加简单。
- It gives you a mental model of how things work internally when using the higher level APIs.
- 在使用高层API时，脑中会有程序怎样工作的一个模型。

## Setup 设置

Before using this guide, install TensorFlow. 使用这个指南之前，安装TensorFlow。

To get the most out of this guide, you should know the following: 为能从这个指南学到尽可能多的东西，你应当知道下面的东西。

- How to program in Python. 怎样用Python编程
- At least a little bit about arrays. 至少一点关于数组的知识
- Ideally, something about machine learning. 一些关于机器学习的知识

Feel free to launch python and follow along with this walkthrough. Run the following lines to set up your Python environment:

打开Python，跟随指南一起走。运行下面的行来设置你的Python环境。

```
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
```

## Tensor Values 张量值

The central unit of data in TensorFlow is the tensor. A tensor consists of a set of primitive values shaped into an array of any number of dimensions. A tensor's rank is its number of dimensions, while its shape is a tuple of integers specifying the array's length along each dimension. Here are some examples of tensor values:

TensorFlow中的数据中心单元是张量tensor。一个张量就是一组原始数值，组成任意维数的阵列的形状。张量的rank是维度的数目，其shape是一个整数元组，指定了每个维度的阵列长度。下面是张量值的一些例子。

```
3. # a rank 0 tensor; a scalar with shape [],
[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```

TensorFlow uses numpy arrays to represent tensor values.

TensorFlow使用numpy阵列来代表张量值。

## TensorFlow Core Walkthrough TensorFlow核心纵览

You might think of TensorFlow Core programs as consisting of two discrete sections: TensorFlow核心程序包括以下两个具体的部分：

1. Building the computational graph (a `tf.Graph`). 构建计算图(`tf.Graph`)
2. Running the computational graph (using a `tf.Session`). 运行计算图(使用`tf.Session`))

### Graph 图

A computational graph is a series of TensorFlow operations arranged into a graph. The graph is composed of two types of objects.

计算图是排列成图状的一系列TensorFlow运算，这个图由两种对象构成。

- Operations (or "ops"): The nodes of the graph. Operations describe calculations that consume and produce tensors.
- 运算(或ops)：图的节点。运算描述了消耗和产生张量的计算。
- Tensors: The edges in the graph. These represent the values that will flow through the graph. Most TensorFlow functions return `tf.Tensors`.
- 张量：图的边。代表了流经这个图的值，多数TensorFlow函数返回`tf.Tensors`。

> `tf.Tensors` do not have values, they are just handles to elements in the computation graph.

> `tf.Tensors`没有值，它们只是计算图中元素的句柄。

Let's build a simple computational graph. The most basic operation is a constant. The Python function that builds the operation takes a tensor value as input. The resulting operation takes no inputs. When run, it outputs the value that was passed to the constructor. We can create two floating point constants a and b as follows:

我们来构建一个简单的计算图。最基本的操作是常数。构建这个操作的Python函数以张量为输入，得到的操作没有输入。当运行时，输出的值传递给构建函数constructor，我们可以生成两个浮点数常数a和b如下：

```
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)
```

The print statements produce: 打印语句输出：

```
Tensor("Const:0", shape=(), dtype=float32)
Tensor("Const_1:0", shape=(), dtype=float32)
Tensor("add:0", shape=(), dtype=float32)
```

Notice that printing the tensors does not output the values 3.0, 4.0, and 7.0 as you might expect. The above statements only build the computation graph. These `tf.Tensor` objects just represent the results of the operations that will be run.

注意打印张量没有输出其值3.0, 4.0和7.0，上述语句只构建了计算图。这些`tf.Tensor`对象只代表这些运算的结果。

Each operation in a graph is given a unique name. This name is independent of the names the objects are assigned to in Python. Tensors are named after the operation that produces them followed by an output index, as in "add:0" above.

图中的每个运算都有一个唯一的名字。这个名字与Python中指定的对象名无关。张量是以产生张量的运算来命名的，后面加上输出索引号，如上例中的“add:0”。

### TensorBoard

TensorFlow provides a utility called TensorBoard. One of TensorBoard's many capabilities is visualizing a computation graph. You can easily do this with a few simple commands.

TensorFlow有一个工具称为TensorBoard。TensorBoard众多功能中的一项就是将计算图可视化。用一些简单的命令就可以很容易做到。

First you save the computation graph to a TensorBoard summary file as follows: 首先将计算图保存到TensorBoard的摘要文件中，如下所示：

```
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
```

This will produce an event file in the current directory with a name in the following format: 这将会在当前目录中产生一个event文件，文件名格式如下：

```
events.out.tfevents.{timestamp}.{hostname}
```

Now, in a new terminal, launch TensorBoard with the following shell command: 现在，在一个新的终端上，用如下shell命令启动TensorBoard：

```
tensorboard --logdir .
```

Then open TensorBoard's graphs page in your browser, and you should see a graph similar to the following:

在浏览器中打开TensorBoard的图页面(链接)，你可以看到一个图与下面类似：

![Image](https://www.tensorflow.org/images/getting_started_add.png)

For more about TensorBoard's graph visualization tools see TensorBoard: Graph Visualization. 更详细的TensorBoard图可视化工具见TensorBoard: Graph Visualization。

### Session 会话

To evaluate tensors, instantiate a `tf.Session` object, informally known as a session. A session encapsulates the state of the TensorFlow runtime, and runs TensorFlow operations. If a `tf.Graph` is like a .py file, a `tf.Session` is like the python executable.

为求的张量的值，实例化一个`tf.Session`对象，就是一个session。一个session封装了TensorFlow运行时的状态，运行TensorFlow的计算。如果`tf.Graph`像一个py文件，那么`tf.Session`就像python可执行文件。

The following code creates a `tf.Session` object and then invokes its run method to evaluate the total tensor we created above:

如下的代码生成了一个`tf.Session`对象，然后调用其run方法来求的我们上面创建的total张量的值：

```
sess = tf.Session()
print(sess.run(total))
```

When you request the output of a node with `Session.run` TensorFlow backtracks through the graph and runs all the nodes that provide input to the requested output node. So this prints the expected value of 7.0:

当用`Session.run`求一个节点的输出时，TensorFlow沿着计算图回溯，运行所有给这个节点输入的节点。所以这打印出了期待的值7.0：

```
7.0
```

You can pass multiple tensors to `tf.Session.run`. The run method transparently handles any combination of tuples or dictionaries, as in the following example:

可以将多个张量送入`tf.Session.run`。run方法透明的处理所有元组或字典的任何组合，如下例所示：

```
print(sess.run({'ab':(a, b), 'total':total}))
```

which returns the results in a structure of the same layout: 返回结果布局结构是一样的：

```
{'total': 7.0, 'ab': (3.0, 4.0)}
```

During a call to `tf.Session.run` any `tf.Tensor` only has a single value. For example, the following code calls `tf.random_uniform` to produce a `tf.Tensor` that generates a random 3-element vector (with values in [0,1)):

调用`tf.Session.run`时，任何`tf.Tensor`都只有一个值。比如，如下代码调用`tf.random_uniform`来产生一个`tf.Tensor`，这是一个随机的3元素矢量（取值范围为[0,1)）:

```
vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))
```

The result shows a different random value on each call to run, but a consistent value during a single run (out1 and out2 receive the same random input): 结果显示每次调用run时都会产生不同的随机值，但一次run时的值是一致的（out1和out2得到了相同的随机输入）：

```
[ 0.52917576  0.64076328  0.68353939]
[ 0.66192627  0.89126778  0.06254101]
(
  array([ 1.88408756,  1.87149239,  1.84057522], dtype=float32),
  array([ 2.88408756,  2.87149239,  2.84057522], dtype=float32)
)
```

Some TensorFlow functions return `tf.Operations` instead of `tf.Tensors`. The result of calling run on an Operation is None. You run an operation to cause a side-effect, not to retrieve a value. Examples of this include the initialization, and training ops demonstrated later.

一些TensorFlow函数返回`tf.Operations`，对一个Operation调用run得到的返回是None。运行一个操作是为了产生连带作用，不是为了得到一个值。这样的例子包括在下面初始化的例子，还有训练操作的例子。

### Feeding

As it stands, this graph is not especially interesting because it always produces a constant result. A graph can be parameterized to accept external inputs, known as placeholders. A placeholder is a promise to provide a value later, like a function argument.

这个图没那么有趣，因为永远产生一个常数结果。一个图可以有接受外部输入的参数，就是placeholder，即承诺后面会提供一个值，像一个函数参数。

```
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y
```

The preceding three lines are a bit like a function in which we define two input parameters (x and y) and then an operation on them. We can evaluate this graph with multiple inputs by using the `feed_dict` argument of the run method to feed concrete values to the placeholders:

前面的三行程序有点像一个函数，定义了两个输入参数x和y，然后对其进行操作。我们可以用run方法的`feed_dict`参数，对placeholder输入具体的值，对这种多输入的图求值。

```
print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))
```

This results in the following output: 结果如下所示：

```
7.5
[ 3.  7.]
```

Also note that the feed_dict argument can be used to overwrite any tensor in the graph. The only difference between placeholders and other `tf.Tensors` is that placeholders throw an error if no value is fed to them.

注意feed_dict参数可以用来重写图中的任意张量。placeholders和其他`tf.Tensors`唯一的区别是，如果不给placeholder供给一个值，那么就会抛出一个错误。

## Datasets 数据集

Placeholders work for simple experiments, but `Datasets` are the preferred method of streaming data into a model.

Placeholder可以用在简单的试验中，但通常使用`Datasets`将数据流式输入到模型中。

To get a runnable `tf.Tensor` from a `Dataset` you must first convert it to a `tf.data.Iterator`, and then call the Iterator's `get_next` method.

为从`Dataset`中得到可以运行的`tf.Tensor`，必须首先将其转化成`tf.data.Iterator`，然后调用Iterator的`get_next`方法。

The simplest way to create an Iterator is with the `make_one_shot_iterator` method. For example, in the following code the next_item tensor will return a row from the my_data array on each run call:

生成一个Iterator最简单的方法是用`make_one_shot_iterator`方法。比如，在下面的代码中next_item张量在每次run调用的时候都会从my_data阵列中返回一行。

```
my_data = [
    [0, 1,],
    [2, 3,],
    [4, 5,],
    [6, 7,],
]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()
```

Reaching the end of the data stream causes `Dataset` to throw an `OutOfRangeError`. For example, the following code reads the next_item until there is no more data to read:

到达数据流的最后`Dataset`将会抛出一个`OutOfRangeError`。比如，下面的代码读取next_item直到没有数据可读。

```
while True:
  try:
    print(sess.run(next_item))
  except tf.errors.OutOfRangeError:
    break
```

If the `Dataset` depends on stateful operations you may need to initialize the iterator before using it, as shown below:

如果`Dataset`依赖一些有状态的操作，那就必须初始化迭代器然后使用，如下所示：

```
r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)
while True:
  try:
    print(sess.run(next_row))
  except tf.errors.OutOfRangeError:
    break
```

For more details on Datasets and Iterators see: Importing Data. 更多Datasets和Iterator的细节见Importing Data文档。

## Layers 层

A trainable model must modify the values in the graph to get new outputs with the same input. Layers are the preferred way to add trainable parameters to a graph.

可训练的模型用相同的输入可以修改计算图中的值，得到新的输出。为给计算图增加可训练的参数，图是一种首选的方法。

Layers package together both the variables and the operations that act on them. For example a densely-connected layer performs a weighted sum across all inputs for each output and applies an optional activation function. The connection weights and biases are managed by the layer object.

层将变量和在变量上的运算打包在一起。比如，一个全连接层的每个输出，都对所有输入计算加权求和，然后再进行可选的激活运算。连接权值和偏置由层对象来管理。

### Creating Layers 生成层

The following code creates a Dense layer that takes a batch of input vectors, and produces a single output value for each. To apply a layer to an input, call the layer as if it were a function. For example:

下面的代码生成一个全连接层，接受一批向量作为输入，产生一个单独的输出值。要将层应用到输入上，调用这个层，就像它是个函数。比如：

```
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)
```

The layer inspects its input to determine sizes for its internal variables. So here we must set the shape of the x placeholder so that the layer can build a weight matrix of the correct size.

层会检查输入确定形状，所以我们必须为placeholder x设定形状，这样层才会构建出正确形状的权值矩阵。

Now that we have defined the calculation of the output, y, there is one more detail we need to take care of before we run the calculation.

注意我们定义了输出y的运算，在run这个运算之前，没有什么需要注意的细节了。

### Initializing Layers 初始化层

The layer contains variables that must be initialized before they can be used. While it is possible to initialize variables individually, you can easily initialize all the variables in a TensorFlow graph as follows:

层包括必须初始化的变量，初始化之后才可以使用，在TensorFlow计算图中，可以很轻松的初始化所有参数如下：

```
init = tf.global_variables_initializer()
sess.run(init)
```

> Important: Calling `tf.global_variables_initializer` only creates and returns a handle to a TensorFlow operation. That op will initialize all the global variables when we run it with `tf.Session.run`.

> 重要：调用`tf.global_variables_initializer`仅仅生成并返回一个TensorFlow操作句柄。当用`tf.Session.run`运行的时候，这个op将初始化所有全局变量。

Also note that this `global_variables_initializer` only initializes variables that existed in the graph when the initializer was created. So the initializer should be one of the last things added during graph construction.

还要注意到，这个`global_variables_initializer`只会初始化那些在计算图中存在的变量，所以初始化器是构建计算图最后需要做的事。

### Executing Layers 现有的层

Now that the layer is initialized, we can evaluate the linear_model's output tensor as we would any other tensor. For example, the following code:

现在层已经初始化了，我们可以求线性模型的输出张量的值。例如下面的代码

```
print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))
```

will generate a two-element output vector such as the following: 会产生一个两元素输出矢量，像下面这样：

```
[[-3.41378999]
 [-9.14999008]]
```

### Layer Function shortcuts  层函数的捷径

For each layer class (like `tf.layers.Dense`) TensorFlow also supplies a shortcut function (like `tf.layers.dense`). The only difference is that the shortcut function versions create and run the layer in a single call. For example, the following code is equivalent to the earlier version:

对于每个层的类（像`tf.layers.Dense`），TensorFlow还都提供了捷径函数（像`tf.layers.dense`）。唯一的不同是，捷径函数调用一次就生成并运行了这个层。比如，下面的代码与上面的是等价的：

```
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.layers.dense(x, units=1)

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
```

While convenient, this approach allows no access to the `tf.layers.Layer` object. This makes introspection and debugging more difficult, and layer reuse impossible.

这很方便，但是这种方法不允许访问`tf.layers.Layer`对象。这使得调试更难，而且层也不能复用。

## Feature columns 特征列

The easiest way to experiment with feature columns is using the `tf.feature_column.input_layer` function. This function only accepts dense columns as inputs, so to view the result of a categorical column you must wrap it in an `tf.feature_column.indicator_column`. For example:

试验特征列最简单的方法是使用`tf.feature_column.input_layer`函数。这个函数只接受稠密列作为输入，所以要观察类别列的结果，需要将其包装在`tf.feature_column.indicator_column`中。比如：

```
features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)
```

Running the `inputs` tensor will parse the `features` into a batch of vectors.

运行张量`inputs`会将`features`解析成一批向量。

Feature columns can have internal state, like layers, so they often need to be initialized. Categorical columns use lookup tables internally and these require a separate initialization op, `tf.tables_initializer`.

特征列可以有内部状态，和层类似，所以经常需要初始化。类别列内部使用查找表，这需要单独进行初始化操作，`tf.tables_initializer`。

```
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))
```

Once the internal state has been initialized you can run `inputs` like any other `tf.Tensor`: 内部状态初始化之后，可以像其他任何`tf.Tensor`一样运行`inputs`张量：

```
print(sess.run(inputs))
```

This shows how the feature columns have packed the input vectors, with the one-hot "department" as the first two indices and "sales" as the third.

下面展示了特征列如何打包输入矢量，前两列为独热的`department`，第三列为`sales`特征。

```
[[  1.   0.   5.]
 [  1.   0.  10.]
 [  0.   1.   8.]
 [  0.   1.   9.]]
```

## Training 训练

Now that you're familiar with the basics of core TensorFlow, let's train a small regression model manually.现在你熟悉了TensorFlow核心的基础。让我们来手动训练一个小的回归模型。

### Define the data 定义数据

First let's define some inputs, x, and the expected output for each input, y_true: 首先我们定义输入x，对每个输入有一个期望输出，y_true：

```
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
```

### Define the model 定义模型

Next, build a simple linear model, with 1 output: 下一步，创建一个简单的线性模型，只有一个输出：

```
linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
```

You can evaluate the predictions as follows: 可以像下面的代码一样求得预测值：

```
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y_pred))
```

The model hasn't yet been trained, so the four "predicted" values aren't very good. Here's what we got; your own output will almost certainly differ:

我们的模型还没有经过训练，所以预测的值不是很好。这是我们得到的结果，你自己的输出肯定与这个不一样：

```
[[ 0.02631879]
 [ 0.05263758]
 [ 0.07895637]
 [ 0.10527515]]
 ```

### Loss 损失函数

To optimize a model, you first need to define the loss. We'll use the mean square error, a standard loss for regression problems.

为优化模型，首先需要定义损失函数。我们用均方误差函数，这是回归模型的标准损失函数。

While you could do this manually with lower level math operations, the `tf.losses` module provides a set of common loss functions. You can use it to calculate the mean square error as follows:

虽然可以用底层数学运算手工实现，但`tf.losses`模块提供了一系列普通损失函数。可以使用这些函数来计算均方误差函数，如下：

```
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

print(sess.run(loss))
```

This will produce a loss value, something like: 这会产生一个损失值，比如： `2.23962`

### Training 训练

TensorFlow provides optimizers implementing standard optimization algorithms. These are implemented as sub-classes of `tf.train.Optimizer`. They incrementally change each variable in order to minimize the loss. The simplest optimization algorithm is gradient descent, implemented by `tf.train.GradientDescentOptimizer`. It modifies each variable according to the magnitude of the derivative of loss with respect to that variable. For example:

TensorFlow还提供了优化器，实现了标准优化算法，是作为`tf.train.Optimizer`的子类实现的。算法一点点改变每个变量，逐渐最小化损失函数。最简单的优化算法是梯度下降算法，由`tf.train.GradientDescentOptimizer`实现。算法根据损失函数对变量的导数的幅度来改变这个变量。比如：

```
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```

This code builds all the graph components necessary for the optimization, and returns a training operation. When run, the training op will update variables in the graph. You might run it as follows:

这个代码构建了优化函数需要的所有计算图的组件，返回了一个训练操作。当运行时，训练op会在计算中更新变量值。可以按如下方式运行：

```
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)
```

Since `train` is an op, not a tensor, it doesn't return a value when run. To see the progression of the loss during training, we run the loss tensor at the same time, producing output like the following:

由于`train`是一个op，不是一个张量，运行时不会返回值。为观察训练过程中损失函数值的变化过程，我们同时运行loss张量，产生的输入如下所示：

```
1.35659
1.00412
0.759167
0.588829
0.470264
0.387626
0.329918
0.289511
0.261112
0.241046
...
```

### Complete program 完整程序
```
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

print(sess.run(y_pred))
```

## Next steps 下一步

To learn more about building models with TensorFlow consider the following:

- Custom Estimators, to learn how to build customized models with TensorFlow. Your knowledge of TensorFlow Core will help you understand and debug your own models.

If you want to learn more about the inner workings of TensorFlow consider the following documents, which go into more depth on many of the topics discussed here:

- Graphs and Sessions
- Tensors
- Variables
