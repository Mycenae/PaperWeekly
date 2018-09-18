# Graphs and Sessions 计算图和会话

TensorFlow uses a dataflow graph to represent your computation in terms of the dependencies between individual operations. This leads to a low-level programming model in which you first define the dataflow graph, then create a TensorFlow session to run parts of the graph across a set of local and remote devices.

TensorFlow使用数据流图，按照操作间的从属关系来表示计算。这样形成了一种底层编程模型，首先要定义数据流图，然后创建TensorFlow会话在一些本地或远程设备上运行图的某部分。

This guide will be most useful if you intend to use the low-level programming model directly. Higher-level APIs such as `tf.estimator.Estimator` and `Keras` hide the details of graphs and sessions from the end user, but this guide may also be useful if you want to understand how these APIs are implemented.

如果你打算直接使用底层编程模型，这个指南就非常有用了。高层API比如`tf.estimator.Estimator`和`Keras`隐藏了图和会话的细节，但如果你想理解这些API是怎样实现的，那么这个指南也很有用。

## Why dataflow graphs? 为什么使用数据流图？

![Image](https://www.tensorflow.org/images/tensors_flowing.gif)

Dataflow is a common programming model for parallel computing. In a dataflow graph, the nodes represent units of computation, and the edges represent the data consumed or produced by a computation. For example, in a TensorFlow graph, the `tf.matmul` operation would correspond to a single node with two incoming edges (the matrices to be multiplied) and one outgoing edge (the result of the multiplication).

数据流是并行计算的一种普通编程模型。在数据流图中，节点表示计算单元，边表示计算消耗或产生的数据。比如，在TensorFlow图中，`tf.matmul`操作对应一个节点，有两个输入的边（进行乘法的矩阵）和一个输出的边（乘法计算结果）。

Dataflow has several advantages that TensorFlow leverages when executing your programs: 在程序运行时，TensorFlow会利用几个数据流的优势：

- **Parallelism**. By using explicit edges to represent dependencies between operations, it is easy for the system to identify operations that can execute in parallel.

- **并行性**。通过使用显式的边来表示操作间的从属关系，系统很容易识别需要并行执行的操作。

- **Distributed execution**. By using explicit edges to represent the values that flow between operations, it is possible for TensorFlow to partition your program across multiple devices (CPUs, GPUs, and TPUs) attached to different machines. TensorFlow inserts the necessary communication and coordination between devices.

- **分布式运行**。通过使用显式的边来表示操作间流动的数据值，TensorFlow可以将程序划分在不同机器的几个设备上运行(CPUs, GPUs, and TPUs)。TensorFlow添加设备间必须的通信和协调。

- **Compilation**. TensorFlow's XLA compiler can use the information in your dataflow graph to generate faster code, for example, by fusing together adjacent operations.

- **编译程序**。TensorFlow的XLA编译器可以使用数据流图中的信息生成更快的代码，比如，通过融合相邻的操作。

- **Portability**. The dataflow graph is a language-independent representation of the code in your model. You can build a dataflow graph in Python, store it in a `SavedModel`, and restore it in a C++ program for low-latency inference.

- **可移植性**。模型的数据流图是与代码语言无关的表示。你可以用Python构建数据流图，保存在`SavedModel`中，在C++程序中恢复出来进行低延迟推理。

## What is a `tf.Graph`? 什么是`tf.Graph`

A `tf.Graph` contains two relevant kinds of information: `tf.Graph`包括两种相关的信息：

- Graph structure. The nodes and edges of the graph, indicating how individual operations are composed together, but not prescribing how they should be used. The graph structure is like assembly code: inspecting it can convey some useful information, but it does not contain all of the useful context that source code conveys.

- 图结构。图中的节点和边，表明了单个的操作是怎样组合到一起的，但没有规定怎样使用。图结构像是汇编代码：检查这些代码可以传递一些有用的信息，但没有包括所有源代码想传递的有用的上下文。

- Graph collections. TensorFlow provides a general mechanism for storing collections of metadata in a `tf.Graph`. The `tf.add_to_collection` function enables you to associate a list of objects with a key (where `tf.GraphKeys` defines some of the standard keys), and `tf.get_collection` enables you to look up all objects associated with a key. Many parts of the TensorFlow library use this facility: for example, when you create a `tf.Variable`, it is added by default to collections representing "global variables" and "trainable variables". When you later come to create a `tf.train.Saver` or `tf.train.Optimizer`, the variables in these collections are used as the default arguments.

- 图集合。TensorFlow提供了`tf.Graph`中存储元数据集合的一般性机制。`tf.add_to_collection`函数可以将对象列表与一个键联系到一起（这里`tf.GraphKeys`定义了一些标准键），`tf.get_collection`可以查找联系到一个键的所有对象。TensorFlow库的很多部分使用：比如，当创建了一个`tf.Variable`，默认就加入到代表全局变量和可训练变量的集合里。当后来创建一个`tf.train.Saver`或`tf.train.Optimizer`，这些集合中的变量作为默认参数使用。

## Building a `tf.Graph` 构建一个`tf.Graph`

Most TensorFlow programs start with a dataflow graph construction phase. In this phase, you invoke TensorFlow API functions that construct new `tf.Operation` (node) and `tf.Tensor` (edge) objects and add them to a `tf.Graph` instance. TensorFlow provides a default graph that is an implicit argument to all API functions in the same context. For example:

大多TensorFlow程序以数据流图的创建阶段开始。在这个阶段，调用TensorFlow API函数来构建新的`tf.Operation`即节点，和`tf.Tensor`，即边，将其加入一个`tf.Graph`实例。TensorFlow提供了默认图，这是所有API函数在同样的上下文里的隐式参数。比如：

- Calling `tf.constant(42.0)` creates a single `tf.Operation` that produces the value 42.0, adds it to the default graph, and returns a `tf.Tensor` that represents the value of the constant.

- 调用`tf.constant(42.0)`创建了一个`tf.Operation`，产生值42.0，将其加入默认图，返回一个`tf.Tensor`，代表常数的值。

- Calling `tf.matmul(x, y)` creates a single `tf.Operation` that multiplies the values of `tf.Tensor` objects x and y, adds it to the default graph, and returns a `tf.Tensor` that represents the result of the multiplication.

- 调用`tf.matmul(x, y)`创建了一个`tf.Operation`，使`tf.Tensor`对象x和y相乘，将其加入默认图，返回一个`tf.Tensor`对象，代表乘法运算的结果。

- Executing `v = tf.Variable(0)` adds to the graph a `tf.Operation` that will store a writeable tensor value that persists between `tf.Session.run` calls. The `tf.Variable` object wraps this operation, and can be used like a tensor, which will read the current value of the stored value. The `tf.Variable` object also has methods such as `assign` and `assign_add` that create `tf.Operation` objects that, when executed, update the stored value. (See Variables for more information about variables.)

- 执行`v = tf.Variable(0)`将一个`tf.Operation`加入图中，这个操作将储存一个可写入的张量值，这个值在不同的`tf.Session.run`中可存续。`tf.Variable`对象包装了这个操作，可以像张量一样使用，可以读取现在存储的值。`tf.Variable`对象还有一些方法，比如`assign`和`assign_add`，这些方法也会产生`tf.Operation`对象，在执行的时候更新存储的值。（详见Variable文档）

- Calling `tf.train.Optimizer.minimize` will add operations and tensors to the default graph that calculates gradients, and return a `tf.Operation` that, when run, will apply those gradients to a set of variables.

- 调用`tf.train.Optimizer.minimize`会将操作和张量加入到默认图，会计算梯度，返回一个`tf.Operation`对象，在运行时会将这些梯度应用在一个变量集合上。

Most programs rely solely on the default graph. However, see Dealing with multiple graphs for more advanced use cases. High-level APIs such as the `tf.estimator.Estimator` API manage the default graph on your behalf, and--for example--may create different graphs for training and evaluation.

大多程序都只依赖默认图。但也可有高级用例，详见Dealing with multiple graphs文档。高级API如`tf.estimator.Estimator`代用户管理默认图，也可能创建不同的图来进行训练和评估。

> Note: Calling most functions in the TensorFlow API merely adds operations and tensors to the default graph, but does not perform the actual computation. Instead, you compose these functions until you have a `tf.Tensor` or `tf.Operation` that represents the overall computation--such as performing one step of gradient descent--and then pass that object to a `tf.Session` to perform the computation. See the section "Executing a graph in a `tf.Session`" for more details.

> 注意：调用大多TensorFlow API函数仅仅将操作和张量加入到默认图，但没有执行实际计算。这些函数一起，然后有一个`tf.Tensor`或`tf.Operation`代表全体运算，比如执行一步梯度下降，然后将这个对象传入`tf.Session`来执行计算。详见"Executing a graph in a `tf.Session`"文档。

## Naming operations 命名操作

A `tf.Graph` object defines a namespace for the `tf.Operation` objects it contains. TensorFlow automatically chooses a unique name for each operation in your graph, but giving operations descriptive names can make your program easier to read and debug. The TensorFlow API provides two ways to override the name of an operation:

一个`tf.Graph`对象对其中包含的`tf.Operation`对象定义了一个命名空间。TensorFlow为其中的每个操作自动选择一个唯一的名称，但给一些操作描述性的名称可以使程序易读易调试。TensorFlow API有两种方法重写一个操作的名称：

- Each API function that creates a new `tf.Operation` or returns a new `tf.Tensor` accepts an optional name argument. For example, `tf.constant(42.0, name="answer")` creates a new `tf.Operation` named "answer" and returns a `tf.Tensor` named "answer:0". If the default graph already contains an operation named "answer", then TensorFlow would append "_1", "_2", and so on to the name, in order to make it unique.

- 每个创建新的`tf.Operation`对象或返回新`tf.Tensor`对象的API函数接受一个可选的命名参数。比如，`tf.constant(42.0, name="answer")`创建一个新的`tf.Operation`名称为"answer"，返回一个`tf.Tensor`名称为"answer:0"。如果默认图里已经有一个操作名称为"answer"，那么TensorFlow会在名称后加上"_1", "_2"等等，使其名称唯一。

- The `tf.name_scope` function makes it possible to add a name scope prefix to all operations created in a particular context. The current name scope prefix is a "/"-delimited list of the names of all active `tf.name_scope` context managers. If a name scope has already been used in the current context, TensorFlow appends "_1", "_2", and so on. For example:

- `tf.name_scope`函数可以在一个特定上下文中的所有操作都加上一个命名范围的前缀。现有的命名范围是一个"/"分隔的所有活跃的`tf.name_scope`上下文管理器的名称列表。如果一个命名范围已经在当前上下文中被使用，TensorFlow会自动加上"_1", "_2"等等。比如：

```
c_0 = tf.constant(0, name="c")  # => operation named "c"

# Already-used names will be "uniquified".
c_1 = tf.constant(2, name="c")  # => operation named "c_1"

# Name scopes add a prefix to all operations created in the same context.
with tf.name_scope("outer"):
  c_2 = tf.constant(2, name="c")  # => operation named "outer/c"

  # Name scopes nest like paths in a hierarchical file system.
  with tf.name_scope("inner"):
    c_3 = tf.constant(3, name="c")  # => operation named "outer/inner/c"

  # Exiting a name scope context will return to the previous prefix.
  c_4 = tf.constant(4, name="c")  # => operation named "outer/c_1"

  # Already-used name scopes will be "uniquified".
  with tf.name_scope("inner"):
    c_5 = tf.constant(5, name="c")  # => operation named "outer/inner_1/c"
```

The graph visualizer uses name scopes to group operations and reduce the visual complexity of a graph. See Visualizing your graph for more information.

图可视化器使用命名范围来将操作分组，减少图的视觉复杂度。详见Visualizing your graph文档。

Note that `tf.Tensor` objects are implicitly named after the `tf.Operation` that produces the tensor as output. A tensor name has the form "<OP_NAME>:< i>" where:

注意`tf.Tensor`对象隐式的根据产生张量输出的`tf.Operation`命名，张量名称格式为"<OP_NAME>:< i>"，这里：

- "<OP_NAME>" is the name of the operation that produces it.
- "<OP_NAME>"是产生张量的操作名称。
- "< i>" is an integer representing the index of that tensor among the operation's outputs.
- "< i>"是一个整数，表示操作输出的张量索引。

## Placing operations on different devices 将操作置于不同的设备上

If you want your TensorFlow program to use multiple different devices, the `tf.device` function provides a convenient way to request that all operations created in a particular context are placed on the same device (or type of device).

如果你想TensorFlow程序使用数个不同的设备，`tf.device`提供了方便的方法，可以要求特定上下文里的所有操作置于相同设备中（或设备类型）。

A device specification has the following form: 设备规格格式如下：

`/job:<JOB_NAME>/task:<TASK_INDEX>/device:<DEVICE_TYPE>:<DEVICE_INDEX>`

where: 这里

- <JOB_NAME> is an alpha-numeric string that does not start with a number.
- <JOB_NAME>是一个字母数字字符串，不能以数字开头。
- <DEVICE_TYPE> is a registered device type (such as GPU or CPU).
- <DEVICE_TYPE>是一个注册的设备类型，如GPU或CPU。
- <TASK_INDEX> is a non-negative integer representing the index of the task in the job named <JOB_NAME>. See `tf.train.ClusterSpec` for an explanation of jobs and tasks.
- <TASK_INDEX>是非负整数，表示在<JOB_NAME>工作中的任务索引，详见`tf.train.ClusterSpec`文档。
- <DEVICE_INDEX> is a non-negative integer representing the index of the device, for example, to distinguish between different GPU devices used in the same process.
- <DEVICE_INDEX>是非负整数，代表设备索引，比如，用来区分同一进程中使用的不同GPU设备。

You do not need to specify every part of a device specification. For example, if you are running in a single-machine configuration with a single GPU, you might use `tf.device` to pin some operations to the CPU and GPU:

你不需要指定所有的设备规格格式。比如，如果在单GPU的单机配置下运行，可以使用`tf.device`指定CPU和GPU分别运行哪些程序：

```
# Operations created outside either context will run on the "best possible"
# device. For example, if you have a GPU and a CPU available, and the operation
# has a GPU implementation, TensorFlow will choose the GPU.
weights = tf.random_normal(...)

with tf.device("/device:CPU:0"):
  # Operations created in this context will be pinned to the CPU.
  img = tf.decode_jpeg(tf.read_file("img.jpg"))

with tf.device("/device:GPU:0"):
  # Operations created in this context will be pinned to the GPU.
  result = tf.matmul(weights, img)
```

If you are deploying TensorFlow in a typical distributed configuration, you might specify the job name and task ID to place variables on a task in the parameter server job ("/job:ps"), and the other operations on task in the worker job ("/job:worker"):

如果将TensorFlow部署在典型的分布式配置中，可能会指定工作名称和任务名称，将任务中的变量放在参数服务器工作中("/job:ps")，其他的操作放在worker工作中("/job:worker")：

```
with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(tf.truncated_normal([784, 100]))
  biases_1 = tf.Variable(tf.zeroes([100]))

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(tf.truncated_normal([100, 10]))
  biases_2 = tf.Variable(tf.zeroes([10]))

with tf.device("/job:worker"):
  layer_1 = tf.matmul(train_batch, weights_1) + biases_1
  layer_2 = tf.matmul(train_batch, weights_2) + biases_2
```

`tf.device` gives you a lot of flexibility to choose placements for individual operations or broad regions of a TensorFlow graph. In many cases, there are simple heuristics that work well. For example, the `tf.train.replica_device_setter` API can be used with `tf.device` to place operations for data-parallel distributed training. For example, the following code fragment shows how `tf.train.replica_device_setter` applies different placement policies to `tf.Variable` objects and other operations:

`tf.device`使得在选择计算图中单个操作或广大区域的放置选择上很有灵活性。在很多情况下，会有简单的启发式想法，可以工作的很好。比如，`tf.train.replica_device_setter` API 可以和`tf.device`一起使用，放置数据并行分布式训练的操作。比如，下面的代码展示了`tf.train.replica_device_setter`如何应用不同的放置策略来处理`tf.Variable`和其他操作的：

```
with tf.device(tf.train.replica_device_setter(ps_tasks=3)):
  # tf.Variable objects are, by default, placed on tasks in "/job:ps" in a
  # round-robin fashion.
  w_0 = tf.Variable(...)  # placed on "/job:ps/task:0"
  b_0 = tf.Variable(...)  # placed on "/job:ps/task:1"
  w_1 = tf.Variable(...)  # placed on "/job:ps/task:2"
  b_1 = tf.Variable(...)  # placed on "/job:ps/task:0"

  input_data = tf.placeholder(tf.float32)     # placed on "/job:worker"
  layer_0 = tf.matmul(input_data, w_0) + b_0  # placed on "/job:worker"
  layer_1 = tf.matmul(layer_0, w_1) + b_1     # placed on "/job:worker"
```

## Tensor-like objects 类似张量的对象

Many TensorFlow operations take one or more `tf.Tensor` objects as arguments. For example, `tf.matmul` takes two `tf.Tensor` objects, and `tf.add_n` takes a list of n `tf.Tensor` objects. For convenience, these functions will accept a tensor-like object in place of a `tf.Tensor`, and implicitly convert it to a `tf.Tensor` using the `tf.convert_to_tensor` method. Tensor-like objects include elements of the following types:

很多TensorFlow操作会有一个或多个`tf.Tensor`对象的参数。比如，`tf.matmul`有2个`tf.Tensor`对象参数，而`tf.add_n`有一个`tf.Tensor`对象列表作为参数。为方便起见，这些函数会以类张量对象为参数，并隐式的将其转化成张量，使用的是`tf.convert_to_tensor`方法。类张量对象包括以下类型的元素：

- `tf.Tensor`
- `tf.Variable`
- numpy.ndarray
- list (and lists of tensor-like objects)
- Scalar Python types: bool, float, int, str

You can register additional tensor-like types using `tf.register_tensor_conversion_function`. 可以使用函数来注册另外的类张量类型。

> Note: By default, TensorFlow will create a new `tf.Tensor` each time you use the same tensor-like object. If the tensor-like object is large (e.g. a numpy.ndarray containing a set of training examples) and you use it multiple times, you may run out of memory. To avoid this, manually call `tf.convert_to_tensor` on the tensor-like object once and use the returned `tf.Tensor` instead.

> 注意：默认的话，每次使用类张量对象，TensorFlow会创建一个新的`tf.Tensor`对象。如果类张量对象很大，比如训练集样本组成的numpy.ndarray，并使用很多次，可能内存会不够用。为避免这种情况，对类张量对象手动调用`tf.convert_to_tensor`，使用返回的`tf.Tensor`对象。

## Executing a graph in a `tf.Session` 在`tf.Session`中执行计算图

TensorFlow uses the `tf.Session` class to represent a connection between the client program---typically a Python program, although a similar interface is available in other languages---and the C++ runtime. A `tf.Session` object provides access to devices in the local machine, and remote devices using the distributed TensorFlow runtime. It also caches information about your `tf.Graph` so that you can efficiently run the same computation multiple times.

TensorFlow使用`tf.Session`类代表客户程序和运行时的连接，客户程序通常是Python程序，其他语言也有类似的接口，运行时为C++的。`tf.Session`对象提供了对本地机器的设备的访问，和对使用分布式TensorFlow运行时的远程设备的访问，还缓存了`tf.Graph`的信息，这样可以高效的多次运行同样的计算任务。

### Creating a `tf.Session` 创建一个`tf.Session`

If you are using the low-level TensorFlow API, you can create a `tf.Session` for the current default graph as follows: 如果你使用底层TensorFlow API，可以这样为目前默认图创建一个`tf.Session`：

```
# Create a default in-process session.
with tf.Session() as sess:
  # ...

# Create a remote session.
with tf.Session("grpc://example.org:2222"):
  # ...
```

Since a `tf.Session` owns physical resources (such as GPUs and network connections), it is typically used as a context manager (in a with block) that automatically closes the session when you exit the block. It is also possible to create a session without using a with block, but you should explicitly call `tf.Session.close` when you are finished with it to free the resources.

由于一个`tf.Session`拥有物理资源（如GPU，网络连接），所以一般用作上下文管理器（在一个with块中），在退出这个块时自动关闭session。也可以不用with块创建session，那就需要在结束的时候显式的调用`tf.Session.close`，以释放资源。

> Note: Higher-level APIs such as `tf.train.MonitoredTrainingSession` or `tf.estimator.Estimator` will create and manage a `tf.Session` for you. These APIs accept optional target and config arguments (either directly, or as part of a `tf.estimator.RunConfig` object), with the same meaning as described below.

> 注意：高层API如`tf.train.MonitoredTrainingSession`或`tf.estimator.Estimator`会为你创建并管理一个`tf.Session`对象。这些API接收可选目标和配置参数，就是一个`tf.estimator.RunConfig`对象，或其一部分，其意义如下所述。

`tf.Session.init` accepts three optional arguments: 接收以下三个可选参数：

1. target. If this argument is left empty (the default), the session will only use devices in the local machine. However, you may also specify a grpc:// URL to specify the address of a TensorFlow server, which gives the session access to all devices on machines that this server controls. See `tf.train.Server` for details of how to create a TensorFlow server. For example, in the common between-graph replication configuration, the `tf.Session` connects to a `tf.train.Server` in the same process as the client. The distributed TensorFlow deployment guide describes other common scenarios.

- target. 如果这个参数为空（默认就是这样），session将只使用本地机器的设备。但也可以指定一个grpc:// URL来指定一个TensorFlow服务器，这使得session可以访问服务器控制的所有机器的设备。`tf.train.Server`详述了如何创建一个TensorFlow服务器。比如，在常见的图间复制配置中，`tf.Session`与客户端一样在同一进程中连接`tf.train.Server`。分布式TensorFlow部署文档里详述了其他常见情景。

2. graph. By default, a new `tf.Session` will be bound to---and only able to run operations in---the current default graph. If you are using multiple graphs in your program (see Programming with multiple graphs for more details), you can specify an explicit `tf.Graph` when you construct the session.

- graph. 默认情况下，一个新的`tf.Session`会绑定到目前的默认图上，只能在这个session上运行操作。如果程序中用了多图（详见多图编程文档），在创建session时可以显式的指定`tf.Graph`。

3. config. This argument allows you to specify a `tf.ConfigProto` that controls the behavior of the session. For example, some of the configuration options include:

- config. 这个参数可以指定一个`tf.ConfigProto`，控制session的行为。比如，一些配置选项包括：

- allow_soft_placement. Set this to True to enable a "soft" device placement algorithm, which ignores `tf.device` annotations that attempt to place CPU-only operations on a GPU device, and places them on the CPU instead.

- allow_soft_placement. 将这个设成True，可以启用一种软的设备放置算法，会忽略`tf.device`注解中将CPU操作放到GPU上的语句，并将其放回CPU。

- cluster_def. When using distributed TensorFlow, this option allows you to specify what machines to use in the computation, and provide a mapping between job names, task indices, and network addresses. See `tf.train.ClusterSpec.as_cluster_def` for details.

- cluster_def. 当使用分布式TensorFlow时，这个选项可以指定在计算中使用哪台机器，并为job names, task indices和network addresses提供映射。详见`tf.train.ClusterSpec.as_cluster_def`文档。

- graph_options.optimizer_options. Provides control over the optimizations that TensorFlow performs on your graph before executing it.

- graph_options.optimizer_options. 包括了优化器的控制选项，在TensorFlow执行图之前，配置好优化器。

- gpu_options.allow_growth. Set this to True to change the GPU memory allocator so that it gradually increases the amount of memory allocated, rather than allocating most of the memory at startup.

- gpu_options.allow_growth. 将这个设成True，可以改变GPU内存分配器，可以逐渐增加分配的内存，而不是一开始就把大部分内存分配掉。

### Using `tf.Session.run` to execute operations 用`tf.Session.run`来执行操作

The `tf.Session.run` method is the main mechanism for running a `tf.Operation` or evaluating a `tf.Tensor`. You can pass one or more `tf.Operation` or `tf.Tensor` objects to `tf.Session.run`, and TensorFlow will execute the operations that are needed to compute the result.

`tf.Session.run`方法是运行一个`tf.Operation`或对`tf.Tensor`求值的主要方法。可以传递一个或多个`tf.Operation` or `tf.Tensor`给`tf.Session.run`，TensorFlow会执行需要的操作来求得结果。

`tf.Session.run` requires you to specify a list of fetches, which determine the return values, and may be a `tf.Operation`, a `tf.Tensor`, or a tensor-like type such as `tf.Variable`. These fetches determine what subgraph of the overall `tf.Graph` must be executed to produce the result: this is the subgraph that contains all operations named in the fetch list, plus all operations whose outputs are used to compute the value of the fetches. For example, the following code fragment shows how different arguments to `tf.Session.run` cause different subgraphs to be executed:

`tf.Session.run`需要指定一个fetches列表，这确定了返回值，可以是`tf.Operation`或`tf.Tensor`或类张量类型如`tf.Variable`。这些fetches确定了执行整个图的哪些子图，以得到结果：fetch列表中命名的所有操作组成的图，以及那些结果用来计算fetches的值的操作。比如，下面的代码片段展示了`tf.Session.run`不同的参数怎样导致执行不同的子图的：

```
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
output = tf.nn.softmax(y)
init_op = w.initializer

with tf.Session() as sess:
  # Run the initializer on `w`.
  sess.run(init_op)

  # Evaluate `output`. `sess.run(output)` will return a NumPy array containing
  # the result of the computation.
  print(sess.run(output))

  # Evaluate `y` and `output`. Note that `y` will only be computed once, and its
  # result used both to return `y_val` and as an input to the `tf.nn.softmax()`
  # op. Both `y_val` and `output_val` will be NumPy arrays.
  y_val, output_val = sess.run([y, output])
```

`tf.Session.run` also optionally takes a dictionary of feeds, which is a mapping from `tf.Tensor` objects (typically `tf.placeholder` tensors) to values (typically Python scalars, lists, or NumPy arrays) that will be substituted for those tensors in the execution. For example:

`tf.Session.run`还有字典feed的可选参数，这是`tf.Tensor`，一般为`tf.placeholder`，到值的映射，一般是Python标量、列表或Numpy array，在执行程序时进行相应的替代。比如：

```
# Define a placeholder that expects a vector of three floating-point values,
# and a computation that depends on it.
x = tf.placeholder(tf.float32, shape=[3])
y = tf.square(x)

with tf.Session() as sess:
  # Feeding a value changes the result that is returned when you evaluate `y`.
  print(sess.run(y, {x: [1.0, 2.0, 3.0]}))  # => "[1.0, 4.0, 9.0]"
  print(sess.run(y, {x: [0.0, 0.0, 5.0]}))  # => "[0.0, 0.0, 25.0]"

  # Raises <a href="../api_docs/python/tf/errors/InvalidArgumentError"> 
  # <code>tf.errors.InvalidArgumentError</code></a>, because you must feed a value for
  # a `tf.placeholder()` when evaluating a tensor that depends on it.
  sess.run(y)

  # Raises `ValueError`, because the shape of `37.0` does not match the shape
  # of placeholder `x`.
  sess.run(y, {x: 37.0})
```

`tf.Session.run` also accepts an optional options argument that enables you to specify options about the call, and an optional run_metadata argument that enables you to collect metadata about the execution. For example, you can use these options together to collect tracing information about the execution:

`tf.Session.run`还接收可选参数option，可以指定调用的option，可选的run_metadata参数使得程序可以收集执行时的元数据。比如，可以使用这些选项来收集运行时的追踪信息：

```
y = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
  # Define options for the `sess.run()` call.
  options = tf.RunOptions()
  options.output_partition_graphs = True
  options.trace_level = tf.RunOptions.FULL_TRACE

  # Define a container for the returned metadata.
  metadata = tf.RunMetadata()

  sess.run(y, options=options, run_metadata=metadata)

  # Print the subgraphs that executed on each device.
  print(metadata.partition_graphs)

  # Print the timings of each operation that executed.
  print(metadata.step_stats)
```

## Visualizing your graph 图可视化

TensorFlow includes tools that can help you to understand the code in a graph. The graph visualizer is a component of TensorBoard that renders the structure of your graph visually in a browser. The easiest way to create a visualization is to pass a `tf.Graph` when creating the `tf.summary.FileWriter`:

```
# Build your graph.
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
# ...
loss = ...
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
  # `sess.graph` provides access to the graph used in a <a href="../api_docs/python/tf/Session"><code>tf.Session</code></a>.
  writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)

  # Perform your computation...
  for i in range(1000):
    sess.run(train_op)
    # ...

  writer.close()
```

> Note: If you are using a `tf.estimator.Estimator`, the graph (and any summaries) will be logged automatically to the model_dir that you specified when creating the estimator.

You can then open the log in tensorboard, navigate to the "Graph" tab, and see a high-level visualization of your graph's structure. Note that a typical TensorFlow graph---especially training graphs with automatically computed gradients---has too many nodes to visualize at once. The graph visualizer makes use of name scopes to group related operations into "super" nodes. You can click on the orange "+" button on any of these super nodes to expand the subgraph inside.

![Image](https://www.tensorflow.org/images/mnist_deep.png)

For more information about visualizing your TensorFlow application with TensorBoard, see the TensorBoard guide.

## Programming with multiple graphs 多图编程

> Note: When training a model, a common way of organizing your code is to use one graph for training your model, and a separate graph for evaluating or performing inference with a trained model. In many cases, the inference graph will be different from the training graph: for example, techniques like dropout and batch normalization use different operations in each case. Furthermore, by default utilities like `tf.train.Saver` use the names of `tf.Variable` objects (which have names based on an underlying `tf.Operation`) to identify each variable in a saved checkpoint. When programming this way, you can either use completely separate Python processes to build and execute the graphs, or you can use multiple graphs in the same process. This section describes how to use multiple graphs in the same process.

As noted above, TensorFlow provides a "default graph" that is implicitly passed to all API functions in the same context. For many applications, a single graph is sufficient. However, TensorFlow also provides methods for manipulating the default graph, which can be useful in more advanced use cases. For example:

- A `tf.Graph` defines the namespace for `tf.Operation` objects: each operation in a single graph must have a unique name. TensorFlow will "uniquify" the names of operations by appending "_1", "_2", and so on to their names if the requested name is already taken. Using multiple explicitly created graphs gives you more control over what name is given to each operation.

- The default graph stores information about every `tf.Operation` and `tf.Tensor` that was ever added to it. If your program creates a large number of unconnected subgraphs, it may be more efficient to use a different `tf.Graph` to build each subgraph, so that unrelated state can be garbage collected.

You can install a different `tf.Graph` as the default graph, using the `tf.Graph.as_default` context manager:

```
g_1 = tf.Graph()
with g_1.as_default():
  # Operations created in this scope will be added to `g_1`.
  c = tf.constant("Node in g_1")

  # Sessions created in this scope will run operations from `g_1`.
  sess_1 = tf.Session()

g_2 = tf.Graph()
with g_2.as_default():
  # Operations created in this scope will be added to `g_2`.
  d = tf.constant("Node in g_2")

# Alternatively, you can pass a graph when constructing a <a href="../api_docs/python/tf/Session"><code>tf.Session</code></a>:
# `sess_2` will run operations from `g_2`.
sess_2 = tf.Session(graph=g_2)

assert c.graph is g_1
assert sess_1.graph is g_1

assert d.graph is g_2
assert sess_2.graph is g_2
```

To inspect the current default graph, call `tf.get_default_graph`, which returns a `tf.Graph` object:

```
# Print all of the operations in the default graph.
g = tf.get_default_graph()
print(g.get_operations())
```