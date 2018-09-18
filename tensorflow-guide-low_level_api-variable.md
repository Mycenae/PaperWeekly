# Variables 变量

A TensorFlow variable is the best way to represent shared, persistent state manipulated by your program.

一个TensorFlow变量是程序表示共享、持续的状态的最好方式。

Variables are manipulated via the `tf.Variable` class. A `tf.Variable` represents a tensor whose value can be changed by running ops on it. Unlike `tf.Tensor` objects, a `tf.Variable` exists outside the context of a single `session.run` call.

变量是通过`tf.Variable`类操作。一个`tf.Variable`代表一个张量，其值在对其运行一些操作时可以改变。与`tf.Tensor`对象不同，`tf.Variable`对象在`session.run`调用上下文之外也存在。

Internally, a `tf.Variable` stores a persistent tensor. Specific ops allow you to read and modify the values of this tensor. These modifications are visible across multiple `tf.Session`s, so multiple workers can see the same values for a `tf.Variable`.

`tf.Variable`内部存储的一个持续张量。特定的操作可以读取并改变这个张量的值。这种改变在不同的`tf.Session`之间都可见，所以多个工作者可以看到一个`tf.Variable`的同样的值。

## Creating a Variable 创建一个变量

The best way to create a variable is to call the `tf.get_variable` function. This function requires you to specify the Variable's name. This name will be used by other replicas to access the same variable, as well as to name this variable's value when checkpointing and exporting models. `tf.get_variable` also allows you to reuse a previously created variable of the same name, making it easy to define models which reuse layers.

创建一个变量的最佳方式是调用`tf.get_variable`函数。这个函数需要指定变量的名称。这个名称会被用于其他副本，来访问同一个变量，在checkpoint和导出模型时，也用这个名称代表响应变量的值。`tf.get_variable`还可以复用相同名称的变量，使模型中复用层的定义更容易。

To create a variable with `tf.get_variable`, simply provide the name and shape 要用`tf.get_variable`, 只需要提供名称和形状

```
my_variable = tf.get_variable("my_variable", [1, 2, 3])
```

This creates a variable named "my_variable" which is a three-dimensional tensor with shape [1, 2, 3]. This variable will, by default, have the dtype `tf.float32` and its initial value will be randomized via `tf.glorot_uniform_initializer`.

这会创建一个变量，名称为"my_variable"，是一个三维张量，形状为[1,2,3]。这个变量默认数据类型为`tf.float32`，其初始值会由`tf.glorot_uniform_initializer`随机初始化。

You may optionally specify the dtype and initializer to `tf.get_variable`. For example: 可以选择性的指定`tf.get_variable`的数据类型和初始化器，比如：

```
my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32,
  initializer=tf.zeros_initializer)
```

TensorFlow provides many convenient initializers. Alternatively, you may initialize a `tf.get_variable` to have the value of a `tf.Tensor`. For example:

TensorFlow提供了很多方便的初始化器。一种选择是，你可以将`tf.get_variable`初始化为一个`tf.Tensor`的值，比如：

```
other_variable = tf.get_variable("other_variable", dtype=tf.int32,
  initializer=tf.constant([23, 42]))
```

Note that when the initializer is a `tf.Tensor` you should not specify the variable's shape, as the shape of the initializer tensor will be used.

注意当初始化器为`tf.Tensor`时，不用指定变量的形状，因为会用这个初始化张量的形状。

### Variable collections 变量集合

Because disconnected parts of a TensorFlow program might want to create variables, it is sometimes useful to have a single way to access all of them. For this reason TensorFlow provides collections, which are named lists of tensors or other objects, such as `tf.Variable` instances.

因为TensorFlow程序的不同部分都会创建变量，如果有种方法访问所有变量将会比较有用。因为这个原因，TensorFlow提供了collections的功能，就是命名过的张量或其他对象的列表，其他对象比如`tf.Variable`实例。

By default every `tf.Variable` gets placed in the following two collections: 默认每个`tf.Variable`都会在下面两个集合中：

- `tf.GraphKeys.GLOBAL_VARIABLES` --- variables that can be shared across multiple devices, 可以跨设备分享的变量
- `tf.GraphKeys.TRAINABLE_VARIABLES` --- variables for which TensorFlow will calculate gradients. TensorFlow将会计算这些变量的梯度

If you don't want a variable to be trainable, add it to the `tf.GraphKeys.LOCAL_VARIABLES` collection instead. For example, the following snippet demonstrates how to add a variable named `my_local` to this collection:

如果不想将变量设为trainable，那么可以将其加入`tf.GraphKeys.LOCAL_VARIABLES`集合。比如，下面的代码片段展示了怎样将一个名为`my_local`的变量加入这个集合：

```
my_local = tf.get_variable("my_local", shape=(),
collections=[tf.GraphKeys.LOCAL_VARIABLES])
```

Alternatively, you can specify `trainable=False` as an argument to `tf.get_variable`: 或者可以指定`tf.get_variable`的一个参数为`trainable=False`：

```
my_non_trainable = tf.get_variable("my_non_trainable",
                                   shape=(),
                                   trainable=False)
```

You can also use your own collections. Any string is a valid collection name, and there is no need to explicitly create a collection. To add a variable (or any other object) to a collection after creating the variable, call `tf.add_to_collection`. For example, the following code adds an existing variable named `my_local` to a collection named `my_collection_name`:

你还可以使用自己的集合collections。任何字符串都是有效的集合名称，没有必要显示的创建一个集合。创建变量后，要将其加入一个集合，调用`tf.add_to_collection`。比如，下面的代码将已有的名为`my_local`的变量加入了`my_collection_name`集合：

`tf.add_to_collection("my_collection_name", my_local)`

And to retrieve a list of all the variables (or other objects) you've placed in a collection you can use: 为检索所有放在集合中的变量列表，可以使用：

`tf.get_collection("my_collection_name")`

### Device placement 设备布置

Just like any other TensorFlow operation, you can place variables on particular devices. For example, the following snippet creates a variable named v and places it on the second GPU device: 和其他任何TensorFlow操作一样，你可以将变量放在特定的设备中。比如，下面的代码片段创建了一个变量，名称为v，放置在第二个GPU设备上：

```
with tf.device("/device:GPU:1"):
  v = tf.get_variable("v", [1])
```

It is particularly important for variables to be in the correct device in distributed settings. Accidentally putting variables on workers instead of parameter servers, for example, can severely slow down training or, in the worst case, let each worker blithely forge ahead with its own independent copy of each variable. For this reason we provide `tf.train.replica_device_setter`, which can automatically place variables in parameter servers. For example:

特别重要的是，在分布式设置中，变量要在正确的设备中。比如，意外的将变量放在工作机上，而没有放在参数服务器上，可能使训练严重减速，或在最坏的情况下，使每个工作机很容易的伪造自己独立的变量副本。因为这个原因，我们提供了`tf.train.replica_device_setter`，这可以自动将变量放置在参数服务器上，比如：

```
cluster_spec = {
    "ps": ["ps0:2222", "ps1:2222"],
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
  v = tf.get_variable("v", shape=[20, 20])  # this variable is placed
                                            # in the parameter server
                                            # by the replica_device_setter
```

## Initializing variables 变量初始化

Before you can use a variable, it must be initialized. If you are programming in the low-level TensorFlow API (that is, you are explicitly creating your own graphs and sessions), you must explicitly initialize the variables. Most high-level frameworks such as `tf.contrib.slim`, `tf.estimator.Estimator` and `Keras` automatically initialize variables for you before training a model.

在使用变量之前，必须初始化。如果用TensorFlow底层API编程（也就是显式的创建自己的图和会话），就必须显式的初始化变量。大多数高层框架，比如`tf.contrib.slim`, `tf.estimator.Estimator`和`Keras`在训练模型前会自动初始化变量。

Explicit initialization is otherwise useful because it allows you not to rerun potentially expensive initializers when reloading a model from a checkpoint as well as allowing determinism when randomly-initialized variables are shared in a distributed setting.

显式的初始化在其他方面也很有用，因为在从checkpoint重载模型时，不用重新运行可能很耗费的初始化器；目前在分布式环境中随机初始化变量很多，显式初始化可以有一定的确定性。

To initialize all trainable variables in one go, before training starts, call `tf.global_variables_initializer()`. This function returns a single operation responsible for initializing all variables in the `tf.GraphKeys.GLOBAL_VARIABLES` collection. Running this operation initializes all variables. For example:

要一次性初始化所有可训练变量，调用`tf.global_variables_initializer()`。这个函数返回一个操作，会初始化所有在`tf.GraphKeys.GLOBAL_VARIABLES`集合中的变量。运行这个操作，会初始化所有变量。比如：

```
session.run(tf.global_variables_initializer())
# Now all variables are initialized.
```

If you do need to initialize variables yourself, you can run the variable's initializer operation. For example:

如果你确实需要自己初始化变量，可以运行变量的初始化器，比如：

`session.run(my_variable.initializer)`

You can also ask which variables have still not been initialized. For example, the following code prints the names of all variables which have not yet been initialized:

你还可以查询哪些变量还没有被初始化。比如，下面的代码会打印出所有未初始化的变量名：

`print(session.run(tf.report_uninitialized_variables()))`

Note that by default `tf.global_variables_initializer` does not specify the order in which variables are initialized. Therefore, if the initial value of a variable depends on another variable's value, it's likely that you'll get an error. Any time you use the value of a variable in a context in which not all variables are initialized (say, if you use a variable's value while initializing another variable), it is best to use `variable.initialized_value()` instead of `variable`:

注意默认的`tf.global_variables_initializer`不会指定初始化变量的顺序，所以，如果变量的初始化值与另一个变量的值有关，那么很可能得到运行错误。使用变量的值的时候，如果不是所有的变量都初始化了（比如，如果使用变量值的时候初始化另外一个变量），最好使用`variable.initialized_value()`，而不要使用`variable`。

```
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = tf.get_variable("w", initializer=v.initialized_value() + 1)
```

## Using variables 使用变量

To use the value of a `tf.Variable` in a TensorFlow graph, simply treat it like a normal `tf.Tensor`: 在TensorFlow图中使用`tf.Variable`的值时，只需要像一个正常的`tf.Tensor`一样使用就可以了。

```
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = v + 1  # w is a tf.Tensor which is computed based on the value of v.
           # Any time a variable is used in an expression it gets automatically
           # converted to a tf.Tensor representing its value.
```

To assign a value to a variable, use the methods `assign`, `assign_add`, and friends in the `tf.Variable` class. For example, here is how you can call these methods:

给变量赋值，可以使用方法`assign`, `assign_add`。比如，可以像下面这样调用这些方法：

```
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
tf.global_variables_initializer().run()
sess.run(assignment)  # or assignment.op.run(), or assignment.eval()
```

Most TensorFlow optimizers have specialized ops that efficiently update the values of variables according to some gradient descent-like algorithm. See `tf.train.Optimizer` for an explanation of how to use optimizers.

多数TensorFlow优化器都有特殊的操作，可以根据一些梯度下降类的算法高效的更新变量值。如何使用优化器请参考`tf.train.Optimizer`文档。

Because variables are mutable it's sometimes useful to know what version of a variable's value is being used at any point in time. To force a re-read of the value of a variable after something has happened, you can use `tf.Variable.read_value`. For example:

因为变量值是可变的，有时候知道正在使用的变量值是哪个版本就非常有用。要强制重新读取变量值，可以使用`tf.Variable.read_value`，比如：

```
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
with tf.control_dependencies([assignment]):
  w = v.read_value()  # w is guaranteed to reflect v's value after the
                      # assign_add operation.
```

## Sharing variables 分享变量

TensorFlow supports two ways of sharing variables: TensorFlow支持两种分享变量的方式：

- Explicitly passing `tf.Variable` objects around. 显式的传递`tf.Variable`对象
- Implicitly wrapping `tf.Variable` objects within `tf.variable_scope` objects. 隐式的将`tf.Variable`对象包装在`tf.variable_scope`对象中

While code which explicitly passes variables around is very clear, it is sometimes convenient to write TensorFlow functions that implicitly use variables in their implementations. Most of the functional layers from `tf.layers` use this approach, as well as all `tf.metrics`, and a few other library utilities.

显式传递变量的代码非常清晰，但有时候写一些TensorFlow函数在实现中隐式的使用变量也很方便。多数`tf.layers`中的层都使用这种方法，也包括`tf.metrics`中的，还有其他一些工具库中的。

Variable scopes allow you to control variable reuse when calling functions which implicitly create and use variables. They also allow you to name your variables in a hierarchical and understandable way.

在调用的函数中有隐式的创建并使用变量的情况中，变量范围可以控制变量的重用。还可以以一种层次式的可理解的方式为变量命名。

For example, let's say we write a function to create a convolutional / relu layer: 比如，我们写一个函数创建一个卷积/ReLU层：

```
def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)
```

This function uses short names `weights` and `biases`, which is good for clarity. In a real model, however, we want many such convolutional layers, and calling this function repeatedly would not work:

这个函数使用了变量名`weights`和`biases`的简称，这用起来非常简洁。但在真实模型中，我们会有很多这种卷积层，如果重复调用这个函数，将会出问题：

```
input1 = tf.random_normal([1,10,10,32])
input2 = tf.random_normal([1,20,20,32])
x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape = [32])  # This fails.
```

Since the desired behavior is unclear (create new variables or reuse the existing ones?) TensorFlow will fail. Calling `conv_relu` in different scopes, however, clarifies that we want to create new variables:

由于想要的行为不明确（是要创建新变量，还是要重用旧的变量？），TensorFlow将会执行失败。在不同的范围内调用`conv_relu`，可以澄清我们是想要创建新变量：

```
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])
```

If you do want the variables to be shared, you have two options. First, you can create a scope with the same name using `reuse=True`: 如果确实想要分享变量，那么有两个选项。首先，可以创建一个同名的范围，同时指定`reuse=True`：

```
with tf.variable_scope("model"):
  output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
  output2 = my_image_filter(input2)
```

You can also call `scope.reuse_variables()` to trigger a reuse: 也可以调用`scope.reuse_variables()`来触发重用：

```
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
  scope.reuse_variables()
  output2 = my_image_filter(input2)
```

Since depending on exact string names of scopes can feel dangerous, it's also possible to initialize a variable scope based on another one: 由于依靠严格的字符串名字可能觉得很危险，也可以用用另一个范围名称来初始化另一个：

```
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
with tf.variable_scope(scope, reuse=True):
  output2 = my_image_filter(input2)
```