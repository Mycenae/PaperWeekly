# Blobs, Layers, and Nets: anatomy of a Caffe model Caffe模型解析

Deep networks are compositional models that are naturally represented as a collection of inter-connected layers that work on chunks of data. Caffe defines a net layer-by-layer in its own model schema. The network defines the entire model bottom-to-top from input data to loss. As data and derivatives flow through the network in the forward and backward passes Caffe stores, communicates, and manipulates the information as blobs: the blob is the standard array and unified memory interface for the framework. The layer comes next as the foundation of both model and computation. The net follows as the collection and connection of layers. The details of blob describe how information is stored and communicated in and across layers and nets.

深度网络可以很自然的表示成作用于大量数据的互联层的集合。Caffe用自己的模型概要格式来逐层定义网络。网络自下而上的从输入数据到损失函数定义整个网络。当数据和导数在网络中正向和反向传播时，Caffe将这些信息以blob的方式进行存储、通信、操作：blob是标准阵列和框架统一的内存接口。模型和计算的下一个基础是layer。网络就是层的集合与连接。Blob的细节描述了信息是怎样存储的，信息是怎样在层内层间、网络内通信的。

Solving is configured separately to decouple modeling and optimization. 解决方案分别进行配置来使模型和优化分离。

We will go over the details of these components in more detail. 我们将详述这些组件。

## Blob storage and communication Blob存储和通信

A Blob is a wrapper over the actual data being processed and passed along by Caffe, and also under the hood provides synchronization capability between the CPU and the GPU. Mathematically, a blob is an N-dimensional array stored in a C-contiguous fashion.

Blob是Caffe进行处理和传递的真实数据的包装，也在后台提供CPU和GPU之间的同步功能。数学上来讲，blob是一个以C语言样式连续的N维阵列。

Caffe stores and communicates data using blobs. Blobs provide a unified memory interface holding data; e.g., batches of images, model parameters, and derivatives for optimization.

Caffe用blob存储数据并进行通信。Blob提供了统一的内存接口来保存数据，即，批次图像，模型参数和用于优化的导数。

Blobs conceal the computational and mental overhead of mixed CPU/GPU operation by synchronizing from the CPU host to the GPU device as needed. Memory on the host and device is allocated on demand (lazily) for efficient memory usage.

混合CPU/GPU操作在需要时进行CPU主机到GPU设备的同步操作，Blob隐藏了这种在计算和脑力上的经常性耗费工作。主机和设备上的内存按照要求（惰性的）进行分配，以有效使用内存。

The conventional blob dimensions for batches of image data are number N x channel K x height H x width W. Blob memory is row-major in layout, so the last / rightmost dimension changes fastest. For example, in a 4D blob, the value at index (n, k, h, w) is physically located at index ((n * K + k) * H + h) * W + w.

批次图像数据的传统blob维数为数量N × 通道数K × 高度H × 宽度W。Blob内存布局是以行为主的，所以最后/最右的维度变化最快。例如，在4D blob中，索引(n, k, h, w)的值所在的物理内存地址索引为((n * K + k) * H + h) * W + w。

- Number / N is the batch size of the data. Batch processing achieves better throughput for communication and device processing. For an ImageNet training batch of 256 images N = 256.
- 数量N是数据的批次规模。批处理使通信和设备处理有更好的吞吐率。在ImageNet数据集上，一般每批训练256幅图像，N = 256。
- Channel / K is the feature dimension e.g. for RGB images K = 3.
- 通道数K是特征维度，如对于RGB图像来说，K = 3。

Note that although many blobs in Caffe examples are 4D with axes for image applications, it is totally valid to use blobs for non-image applications. For example, if you simply need fully-connected networks like the conventional multi-layer perceptron, use 2D blobs (shape (N, D)) and call the InnerProductLayer (which we will cover soon).

注意虽然Caffe例子中的很多blob都是图像应用中的4D数据，但完全可以在非图像应用中使用blob。例如，如果你像传统多层感知机一样仅需要全连接层网络，使用2D blob（形状为(N ,D)），调用函数InnerProductLayer就可以了（后面会讲到）。

Parameter blob dimensions vary according to the type and configuration of the layer. For a convolution layer with 96 filters of 11 x 11 spatial dimension and 3 inputs the blob is 96 x 3 x 11 x 11. For an inner product / fully-connected layer with 1000 output channels and 1024 input channels the parameter blob is 1000 x 1024.

参数blob维数随着层的类型和配置不同而不同。若卷积层包含96个的滤波器，卷积核大小11×11，输入为3通道，那么blob维数为96 x 3 x 11 x 11。对于内积层即全连接层来说，若包含1000个输出通道，1024个输入通道，那么参数blob为1000×1024。

For custom data it may be necessary to hack your own input preparation tool or data layer. However once your data is in your job is done. The modularity of layers accomplishes the rest of the work for you.

对于定制数据，则需要弄明白数据准备工具或数据层。但一旦任务数据准备好，你的任务就完成了。网络层的模块性会为你完成剩下的任务。

### Implementation Details 实现细节

As we are often interested in the values as well as the gradients of the blob, a Blob stores two chunks of memories, data and diff. The former is the normal data that we pass along, and the latter is the gradient computed by the network.

由于我们经常对blob的数值以及导数感兴趣，一个Blob存储两块内存，分别是data和diff。前者是传递的正常数据，后者是网络计算得到的梯度。

Further, as the actual values could be stored either on the CPU and on the GPU, there are two different ways to access them: the const way, which does not change the values, and the mutable way, which changes the values:

进一步，由于实际值可能在CPU或/和GPU上存储，就有两种访问方式：const常数方式，不改变数值，和mutable可变方式，改变其中的数值。

```
const Dtype* cpu_data() const;
Dtype* mutable_cpu_data();
```
(similarly for gpu and diff). 对于GPU上的数据和梯度数据diff是类似的。

The reason for such design is that, a Blob uses a SyncedMem class to synchronize values between the CPU and GPU in order to hide the synchronization details and to minimize data transfer. A rule of thumb is, always use the const call if you do not want to change the values, and never store the pointers in your own object. Every time you work on a blob, call the functions to get the pointers, as the SyncedMem will need this to figure out when to copy data.

这样设计的原因在于，Blob使用SyncedMen类在CPU和GPU之间同步数值，以隐藏同步的细节并最小化数据传输量。首要规则是，如果不想改变数值，就永远使用const常数方式，而且永远不要在自己的对象中存储指针。每次你在blob上操作，调用函数以获得指针，因为SyncedMen将需要这个来弄清楚什么时候复制数据。

In practice when GPUs are present, one loads data from the disk to a blob in CPU code, calls a device kernel to do GPU computation, and ferries the blob off to the next layer, ignoring low-level details while maintaining a high level of performance. As long as all layers have GPU implementations, all the intermediate data and gradients will remain in the GPU.

在实践中，当GPU存在的时候，需要将数据从磁盘载入到CPU代码的blob中，调用一个device kernel来做GPU计算，将blob运送到下一层去，忽略底层细节的同时还可以保持高层的操作。只要所有层都有GPU实现，所有中间数据和梯度都会一直待在GPU内。

If you want to check out when a Blob will copy data, here is an illustrative example: 如果想检查Blob什么时候复制数据，下面是一个说明性的例子：

```
// Assuming that data are on the CPU initially, and we have a blob.
const Dtype* foo;
Dtype* bar;
foo = blob.gpu_data(); // data copied cpu->gpu.
foo = blob.cpu_data(); // no data copied since both have up-to-date contents.
bar = blob.mutable_gpu_data(); // no data copied.
// ... some operations ...
bar = blob.mutable_gpu_data(); // no data copied when we are still on GPU.
foo = blob.cpu_data(); // data copied gpu->cpu, since the gpu side has modified the data
foo = blob.gpu_data(); // no data copied since both have up-to-date contents
bar = blob.mutable_cpu_data(); // still no data copied.
bar = blob.mutable_gpu_data(); // data copied cpu->gpu.
bar = blob.mutable_cpu_data(); // data copied gpu->cpu.
```

## Layer computation and connections 层的计算与连接

The layer is the essence of a model and the fundamental unit of computation. Layers convolve filters, pool, take inner products, apply nonlinearities like rectified-linear and sigmoid and other elementwise transformations, normalize, load data, and compute losses like softmax and hinge. See the layer catalogue for all operations. Most of the types needed for state-of-the-art deep learning tasks are there.

层是模型的本质，也是计算的基本单元。层的内容包括与滤波器卷积，pool，内积，非线性操作如ReLU或sigmoid，还有其他逐元素的变换，如归一化，载入数据，计算损失函数如softmax或hinge。参考层的目录中的所有操作，最新的深度学习任务所需的多数类型都在那里。

bottom blob [data]->[conv1(convolution)]->[conv1] top blob

A layer takes input through bottom connections and makes output through top connections. 层的输入是通过下层的连接，输出是通过上层的连接。

Each layer type defines three critical computations: setup, forward, and backward. 每个层的类型定义了三种关键计算：

- Setup: initialize the layer and its connections once at model initialization.
- 设置：在模型初始化时，进行一次层及其连接的初始化。
- Forward: given input from bottom compute the output and send to the top.
- 前向：从下部给定输入，计算输出并送入上层。
- Backward: given the gradient w.r.t. the top output compute the gradient w.r.t. to the input and send to the bottom. A layer with parameters computes the gradient w.r.t. to its parameters and stores it internally.
- 后向：给定对上层输出的梯度，计算对输入的梯度，然后送入下层。带有参数的层计算对参数的梯度，并在内部存储起来。

More specifically, there will be two Forward and Backward functions implemented, one for CPU and one for GPU. If you do not implement a GPU version, the layer will fall back to the CPU functions as a backup option. This may come handy if you would like to do quick experiments, although it may come with additional data transfer cost (its inputs will be copied from GPU to CPU, and its outputs will be copied back from CPU to GPU).

特别的，有两个前向和后向的函数需要实现，一个是CPU版的，一个是GPU版的。如果不实现GPU版，层会以备份的选项使用CPU版函数。如果想进行快速试验，这可能比较方便，但会带来额外的数据传输代价（输入会从GPU复制到CPU，输出从CPU复制到GPU）。

Layers have two key responsibilities for the operation of the network as a whole: a forward pass that takes the inputs and produces the outputs, and a backward pass that takes the gradient with respect to the output, and computes the gradients with respect to the parameters and to the inputs, which are in turn back-propagated to earlier layers. These passes are simply the composition of each layer’s forward and backward.

整体来讲，网络操作中层有两个关键责任：前向计算从输入计算输出，后向计算用对输出的梯度，计算对参数的梯度和对输入的梯度，然后依次反向传播到更前面的层去。

Developing custom layers requires minimal effort by the compositionality of the network and modularity of the code. Define the setup, forward, and backward for the layer and it is ready for inclusion in a net.

定制层也很容易，因为网络和代码都是模块性的。定义层的设置、前向和后向计算，然后就可以成为网络的一个模块了。

## Net definition and operation 网络定义与操作

The net jointly defines a function and its gradient by composition and auto-differentiation. The composition of every layer’s output computes the function to do a given task, and the composition of every layer’s backward computes the gradient from the loss to learn the task. Caffe models are end-to-end machine learning engines.

网络通过其组成和自动求导，定义了一个函数和其梯度。每个层的输出组成了这个函数来完成给定任务，每层的后向组成计算了损失函数的梯度来学习任务。Caffe模型是端到端的机器学习引擎。

The net is a set of layers connected in a computation graph – a directed acyclic graph (DAG) to be exact. Caffe does all the bookkeeping for any DAG of layers to ensure correctness of the forward and backward passes. A typical net begins with a data layer that loads from disk and ends with a loss layer that computes the objective for a task such as classification or reconstruction.

网络是连接为计算图的层集合，确切的说，计算图是一个有向无环图(DAG)。对任何层的DAG，Caffe完成所有的簿记工作，以确保正向和反向计算的正确性。典型的网络从数据层开始，从磁盘将数据载入，以损失层结束，计算分类或重建任务的目标函数。

The net is defined as a set of layers and their connections in a plaintext modeling language. A simple logistic regression classifier 网络定义为层的集合以及其连接，采用纯文本建模语言。如下图的简单logistic回归分类器

![Image](http://caffe.berkeleyvision.org/tutorial/fig/logreg.jpg)

is defined by 定义为

```
name: "LogReg"
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  data_param {
    source: "input_leveldb"
    batch_size: 64
  }
}
layer {
  name: "ip"
  type: "InnerProduct"
  bottom: "data"
  top: "ip"
  inner_product_param {
    num_output: 2
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip"
  bottom: "label"
  top: "loss"
}
```

Model initialization is handled by Net::Init(). The initialization mainly does two things: scaffolding the overall DAG by creating the blobs and layers (for C++ geeks: the network will retain ownership of the blobs and layers during its lifetime), and calls the layers’ SetUp() function. It also does a set of other bookkeeping things, such as validating the correctness of the overall network architecture. Also, during initialization the Net explains its initialization by logging to INFO as it goes:

模型初始化由Net::Init()函数处理。初始化主要做两件事：生成blob和层以形成整体的DAG（对于C++程序员来说，网络会记住blob和层在其整个生命周期的的归属），然后调用层的SetUp()函数。还包括一系列其他簿记工作，如验证网络结构整体的正确性。在初始化过程中，Net通过记录信息到INFO来解释初始化：

```
I0902 22:52:17.931977 2079114000 net.cpp:39] Initializing net from parameters:
name: "LogReg"
[...model prototxt printout...]
# construct the network layer-by-layer
I0902 22:52:17.932152 2079114000 net.cpp:67] Creating Layer mnist
I0902 22:52:17.932165 2079114000 net.cpp:356] mnist -> data
I0902 22:52:17.932188 2079114000 net.cpp:356] mnist -> label
I0902 22:52:17.932200 2079114000 net.cpp:96] Setting up mnist
I0902 22:52:17.935807 2079114000 data_layer.cpp:135] Opening leveldb input_leveldb
I0902 22:52:17.937155 2079114000 data_layer.cpp:195] output data size: 64,1,28,28
I0902 22:52:17.938570 2079114000 net.cpp:103] Top shape: 64 1 28 28 (50176)
I0902 22:52:17.938593 2079114000 net.cpp:103] Top shape: 64 (64)
I0902 22:52:17.938611 2079114000 net.cpp:67] Creating Layer ip
I0902 22:52:17.938617 2079114000 net.cpp:394] ip <- data
I0902 22:52:17.939177 2079114000 net.cpp:356] ip -> ip
I0902 22:52:17.939196 2079114000 net.cpp:96] Setting up ip
I0902 22:52:17.940289 2079114000 net.cpp:103] Top shape: 64 2 (128)
I0902 22:52:17.941270 2079114000 net.cpp:67] Creating Layer loss
I0902 22:52:17.941305 2079114000 net.cpp:394] loss <- ip
I0902 22:52:17.941314 2079114000 net.cpp:394] loss <- label
I0902 22:52:17.941323 2079114000 net.cpp:356] loss -> loss
# set up the loss and configure the backward pass
I0902 22:52:17.941328 2079114000 net.cpp:96] Setting up loss
I0902 22:52:17.941328 2079114000 net.cpp:103] Top shape: (1)
I0902 22:52:17.941329 2079114000 net.cpp:109]     with loss weight 1
I0902 22:52:17.941779 2079114000 net.cpp:170] loss needs backward computation.
I0902 22:52:17.941787 2079114000 net.cpp:170] ip needs backward computation.
I0902 22:52:17.941794 2079114000 net.cpp:172] mnist does not need backward computation.
# determine outputs
I0902 22:52:17.941800 2079114000 net.cpp:208] This network produces output loss
# finish initialization and report memory usage
I0902 22:52:17.941810 2079114000 net.cpp:467] Collecting Learning Rate and Weight Decay.
I0902 22:52:17.941818 2079114000 net.cpp:219] Network initialization done.
I0902 22:52:17.941824 2079114000 net.cpp:220] Memory required for data: 201476
```

Note that the construction of the network is device agnostic - recall our earlier explanation that blobs and layers hide implementation details from the model definition. After construction, the network is run on either CPU or GPU by setting a single switch defined in Caffe::mode() and set by Caffe::set_mode(). Layers come with corresponding CPU and GPU routines that produce identical results (up to numerical errors, and with tests to guard it). The CPU / GPU switch is seamless and independent of the model definition. For research and deployment alike it is best to divide model and implementation.

注意网络的构建是与设备无关的，回忆一下我们前面的解释，blob和层的实现细节是对模型定义隐藏的。构建好之后，通过设置一个定义在Caffe::mode()中开关，具体操作是Caffe::set_mode()，网络就会相应运行在CPU或GPU上。不论是CPU子程序实现的层，还是GPU子程序实现的层，其计算结果都是一样的（数值误差进行过测试来保证）。CPU/GPU开关是无缝的，与模型定义无关。对研究和部署，最好将模型与实现分开。

### Model format 模型格式

The models are defined in plaintext protocol buffer schema (prototxt) while the learned models are serialized as binary protocol buffer (binaryproto) .caffemodel files.

模型定义采用纯文本的protocol buffer格式(prototxt)，学习好的模型序列化为二进制protocol buffer (binaryproto) .caffemodel文件。

The model format is defined by the protobuf schema in caffe.proto. The source file is mostly self-explanatory so one is encouraged to check it out.

模型格式由caffe.proto中的protobuf概要定义。源文件大多是自解释的，所以可以去查看。

Caffe speaks Google Protocol Buffer for the following strengths: minimal-size binary strings when serialized, efficient serialization, a human-readable text format compatible with the binary version, and efficient interface implementations in multiple languages, most notably C++ and Python. This all contributes to the flexibility and extensibility of modeling in Caffe.

Caffe采用Google Protocol Buffer由于以下原因：在序列化时二进制字符串是最小的大小，高效序列化，与二进制版本兼容的可读文本格式，多语言实现的高效接口，最著名的是C++和Python。这些都有助于Caffe模型的灵活性和可扩展性。