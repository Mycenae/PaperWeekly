# Caffe: Convolutional Architecture for Fast Feature Embedding

Yangqing Jia et al. UC Berkeley

## Abstract

Caffe provides multimedia scientists and practitioners with a clean and modifiable framework for state-of-the-art deep learning algorithms and a collection of reference models. The framework is a BSD-licensed C++ library with Python and MATLAB bindings for training and deploying general-purpose convolutional neural networks and other deep models efficiently on commodity architectures. Caffe fits industry and internet-scale media needs by CUDA GPU computation, processing over 40 million images a day on a single K40 or Titan GPU (≈ 2.5 ms per image). By separating model representation from actual implementation, Caffe allows experimentation and seamless switching among platforms for ease of development and deployment from prototyping machines to cloud environments.

Caffe为多媒体科学家和专业人员进行目前最好的深度学习算法研究提供了一个干净可修改的框架，以及参考模型集合。这个框架是一个BSD-licenced C++库，有Python和Matlab接口，可以再商用架构上进行通用卷积神经网络和其他深度模型的高效训练和部署。Caffe通过采用CUDA GPU计算，适用于工业和互联网级的多媒体需求，在单个K40或Titan GPU上每天处理超过40 million图像（每幅图像约2.5ms）。通过将模型表示与实际实现分离开来，Caffe可以进行试验，在不同平台上无缝切换，在原型机和云平台上都可以进行很容易的开发和部署。

Caffe is maintained and developed by the Berkeley Vision and Learning Center (BVLC) with the help of an active community of contributors on GitHub. It powers on-going research projects, large-scale industrial applications, and startup prototypes in vision, speech, and multimedia.

Caffe由Berkeley视觉和学习中心(BVLC)开发和维护，在Github上有活跃的群体，为视觉、语音和多媒体类进行的研究项目，大规模工业化应用和初始的原型提供帮助。

**Keywords** Open Source, Computer Vision, Neural Networks, Parallel Computation, Machine Learning

## 1 Introduction

A key problem in multimedia data analysis is discovery of effective representations for sensory inputs—images, soundwaves, haptics, etc. While performance of conventional, handcrafted features has plateaued in recent years, new developments in deep compositional architectures have kept performance levels rising [8]. Deep models have outperformed hand-engineered feature representations in many domains, and made learning possible in domains where engineered features were lacking entirely.

多媒体数据分析的一个关键问题是为传感器输入发现有效的表示，包括图像，音频，触觉等。传统的手工设计的特征的性能在最近几年已经达到了稳定停滞期，但深度架构的新发展在持续提升性能。深度模型在很多领域中已经超过了手工设计特征的表示，在手工设计特征无法学习的领域，深度学习则非常活跃。

We are particularly motivated by large-scale visual recognition, where a specific type of deep architecture has achieved a commanding lead on the state-of-the-art. These Convolutional Neural Networks, or CNNs, are discriminatively trained via back-propagation through layers of convolutional filters and other operations such as rectification and pooling. Following the early success of digit classification in the 90’s, these models have recently surpassed all known methods for large-scale visual recognition, and have been adopted by industry heavyweights such as Google, Facebook, and Baidu for image understanding and search.

我们的工作由大规模视觉识别推动，其中一种特定类型的深度架构取得了目前最好的结果。这些CNNs通过反向传播进行训练，还包括其他运算，如Rect和池化。在上世纪90年代有过在数字分类中的成功，这些模型在大规模视觉识别中最近超过了所有已知的方法，已经在工业巨头中得到采用，如Google，Facebook和Baidu，进行图像理解和搜索。

While deep neural networks have attracted enthusiastic interest within computer vision and beyond, replication of published results can involve months of work by a researcher or engineer. Sometimes researchers deem it worthwhile to release trained models along with the paper advertising their performance. But trained models alone are not sufficient for rapid research progress and emerging commercial applications, and few toolboxes offer truly off-the-shelf deployment of state-of-the-art models—and those that do are often not computationally efficient and thus unsuitable for commercial deployment.

DNN在计算机视觉和其他领域非常受欢迎，但一位研究者或工程师复现已发表的成果可能需要几个月的工作。有时研究者认为放出训练模型是值得的，可以为其论文及性能提高名声。但只有训练好的模型是不足以进行快速研究的，也不足以进行商业应用，很少有工具可以为目前最好的模型进行真正的开箱即用的部署，已有的一些工具不能进行高效的计算，所以也不适于商业部署。

To address such problems, we present Caffe, a fully opensource framework that affords clear access to deep architectures. The code is written in clean, efficient C++, with CUDA used for GPU computation, and nearly complete, well-supported bindings to Python/Numpy and MATLAB. Caffe adheres to software engineering best practices, providing unit tests for correctness and experimental rigor and speed for deployment. It is also well-suited for research use, due to the careful modularity of the code, and the clean separation of network definition (usually the novel part of deep learning research) from actual implementation.

为解决上述问题，我们提出了Caffe，一个完全开源的框架，可以很清楚的访问深度架构。代码是用高效的C++写成的，使用CUDA进行GPU计算，几乎完全支持Python/Numpy和Matlab接口。Caffe坚持了软件工程的最佳实践，有着单元测试的正确性，和部署上的严谨性和速度试验。同时还非常适合于研究使用，因为代码的模块性非常好，网络定义和实际实现也有着清晰的分隔（这通常是深度学习研究的创新部分）。

In Caffe, multimedia scientists and practitioners have an orderly and extensible toolkit for state-of-the-art deep learning algorithms, with reference models provided out of the box. Fast CUDA code and GPU computation fit industry needs by achieving processing speeds of more than 40 million images per day on a single K40 or Titan GPU. The same models can be run in CPU or GPU mode on a variety of hardware: Caffe separates the representation from the actual implementation, and seamless switching between heterogeneous platforms furthers development and deployment—Caffe can even be run in the cloud.

在Caffe中，多媒体科学家和专业人员有了可扩展的工具箱进行目前最好的深度学习算法研究，还有现成的参考模型。CUDA快速代码和GPU计算符合工业需要，在K40或Titan GPU上每天可以处理40 million幅图像。相同的模型可以在CPU或GPU上运行：Caffe将表示与实际实现进行了分离，在不同的平台上达到了无缝衔接，Caffe甚至还可以在云上运行。

While Caffe was first designed for vision, it has been adopted and improved by users in speech recognition, robotics, neuroscience, and astronomy. We hope to see this trend continue so that further sciences and industries can take advantage of deep learning.

Caffe是为视觉任务设计的，但也可以改进用于语言处理，机器人，神经科学和天文学。我们希望这个趋势可以进行下去，这样更多的科学界和工业界也可以利用深度学习。

Caffe is maintained and developed by the BVLC with the active efforts of several graduate students, and welcomes open-source contributions at http://github.com/BVLC/caffe. We thank all of our contributors for their work!

## 2. Highlights OF Caffe

Caffe provides a complete toolkit for training, testing, finetuning, and deploying models, with well-documented examples for all of these tasks. As such, it’s an ideal starting point for researchers and other developers looking to jump into state-of-the-art machine learning. At the same time, it’s likely the fastest available implementation of these algorithms, making it immediately useful for industrial deployment.

Caffe提供了完备的工具，可以进行模型训练，测试，精调和部署，这些任务的案例文档丰富。这样，这是研究者和其他开发者的理想起点，可以进行目前最好的机器学习的工作。同时，这很可能是这些算法目前最快的实现，立即可以用于工业部署。

**Modularity**. The software is designed from the beginning to be as modular as possible, allowing easy extension to new data formats, network layers, and loss functions. Lots of layers and loss functions are already implemented, and plentiful examples show how these are composed into trainable recognition systems for various tasks.

**模块性**。软件最初的设计就是尽可能的模块化，新的数据格式、新的网络层、损失函数容易拓展。很多层和损失函数是已经实现的，很多例子展示了这是怎样组成各种任务中的可训练的识别系统的。

**Separation of representation and implementation**. Caffe model definitions are written as config files using the Protocol Buffer language. Caffe supports network architectures in the form of arbitrary directed acyclic graphs. Upon instantiation, Caffe reserves exactly as much memory as needed for the network, and abstracts from its underlying location in host or GPU. Switching between a CPU and GPU implementation is exactly one function call.

**表示和实现的分离**。Caffe模型的定义是作为配置文件形式的，使用的是Protocol Buffer语言。Caffe支持的网络架构是任意的有向无环图。在实例化时，Caffe为网络保留了尽可能多的空间，从其潜在位置中尽可能抽闲出来，不论是主机或是GPU。CPU和GPU的调用切换实际上是一个函数调用。

**Test coverage**. Every single module in Caffe has a test, and no new code is accepted into the project without corresponding tests. This allows rapid improvements and refactoring of the codebase, and imparts a welcome feeling of peacefulness to the researchers using the code.

**测试覆盖**。Caffe中的每个模块都有测试，没有对应的测试，新代码加入不到工程中来。这使得代码库可以快速改进和重构，为使用这些代码的研究者传递了欢迎的感觉。

**Python and MATLAB bindings**. For rapid prototyping and interfacing with existing research code, Caffe provides Python and MATLAB bindings. Both languages may be used to construct networks and classify inputs. The Python bindings also expose the solver module for easy prototyping of new training procedures.

**Python和Matlab接口**。为进行快速原型实现和与现有的研究代码进行对接，Caffe提供了Python和Matlab接口。这两种语言可以用于构建网络和对输入分类。Python接口也对solver模块开放，可以快速开发新的训练方法。

**Pre-trained reference models**. Caffe provides (for academic and non-commercial use—not BSD license) reference models for visual tasks, including the landmark “AlexNet” ImageNet model [8] with variations and the R-CNN detection model [3]. More are scheduled for release. We are strong proponents of reproducible research: we hope that a common software substrate will foster quick progress in the search over network architectures and applications.

**预训练的参考模型**。Caffe为视觉任务给出了参考模型，包括标志性的AlexNet ImageNet模型，和R-CNN检测模型，还会按计划放出更多。我们积极推动可复现的研究：我们希望这个通用的软件基础会使网络架构和应用的研究更加快速。

### 2.1 Comparison to related software

We summarize the landscape of convolutional neural network software used in recent publications in Table 1. While our list is incomplete, we have included the toolkits that are most notable to the best of our knowledge. Caffe differs from other contemporary CNN frameworks in two major ways:

我们在表1中总结了最近文献中用到的CNN软件。虽然我们的列表不完整，但已经包括了我们所知道的最著名的工具。Caffe与其他CNN架构有两点主要不同：

(1) The implementation is completely C++ based, which eases integration into existing C++ systems and interfaces common in industry. The CPU mode removes the barrier of specialized hardware for deployment and experiments once a model is trained. 实现是完全基于C++的，这与现有的工业系统中的C++系统和接口更容易集成。CPU模式没有专用硬件部署和试验的困难。

(2) Reference models are provided off-the-shelf for quick experimentation with state-of-the-art results, without the need for costly re-learning. By finetuning for related tasks, such as those explored by [2], these models provide a warmstart to new research and applications. Crucially, we publish not only the trained models but also the recipes and code to reproduce them. 参考模型库可以马上对目前最好的结果进行快速试验，不需要复杂的重新训练。通过对相关任务的精调，如[2]中的例子，这些模型为新的研究和应用提供了预热。关键是，我们放出的不仅有训练好的模型，还有重新训练的代码和方法。

Table 1: Comparison of popular deep learning frameworks. Core language is the main library language, while bindings have an officially supported library interface for feature extraction, training, etc. CPU indicates availability of host-only computation, no GPU usage (e.g., for cluster deployment); GPU indicates the GPU computation capability essential for training modern CNNs.

Caffe, cuda-convnet, Decaf, OverFeat, Theano/Pylearn2, Torch7

## 3 Architecture

### 3.1 Data Storage

Caffe stores and communicates data in 4-dimensional arrays called blobs. Caffe中的数据存储和通信都是以4维阵列形式的，称为blobs。

Blobs provide a unified memory interface, holding batches of images (or other data), parameters, or parameter updates. Blobs conceal the computational and mental overhead of mixed CPU/GPU operation by synchronizing from the CPU host to the GPU device as needed. In practice, one loads data from the disk to a blob in CPU code, calls a CUDA kernel to do GPU computation, and ferries the blob off to the next layer, ignoring low-level details while maintaining a high level of performance. Memory on the host and device is allocated on demand (lazily) for efficient memory usage.

Blobs是统一的内存接口，保存了图像批数据（或其他数据），参数，或参数更新。Blobs隐藏了CPU/GPU混合的计算和脑力耗费，在需要的时候从CPU到GPU同步数据。实践中，数据从磁盘载入到blob，调用CUDA核进行GPU计算，将blob送入下一层，都是在CPU代码中进行的，忽略了底层细节同时保持了高性能。主机和设备上的内存分配是按需的，以提高内存使用效率。

Models are saved to disk as Google Protocol Buffers, which have several important features: minimal-size binary strings when serialized, efficient serialization, a human-readable text format compatible with the binary version, and efficient interface implementations in multiple languages, most notably C++ and Python.

模型用Google Protocol Buffer保存到磁盘，有几个重要的特点：序列化时二值字符串规模最小，序列化效率高，人类可读的文本格式，与二进制版本兼容，可用多种语言进行接口实现，主要是C++和Python。

Large-scale data is stored in LevelDB databases. In our test program, LevelDB and Protocol Buffers provide a throughput of 150MB/s on commodity machines with minimal CPU impact. Thanks to layer-wise design (discussed below) and code modularity, we have recently added support for other data sources, including some contributed by the open source community. 大规模数据存储在LevelDB数据库中。在我们的测试程序中，LevelDB和Protocol Buffer在商用机器上可以达到150MB/s的吞吐量，对CPU影响很小。多亏了有分层的设计和代码模块化，我们最近增加了其他数据源的支持，包括一些开源团体提供的贡献。

### 3.2 Layers

A Caffe layer is the essence of a neural network layer: it takes one or more blobs as input, and yields one or more blobs as output. Layers have two key responsibilities for the operation of the network as a whole: a forward pass that takes the inputs and produces the outputs, and a backward pass that takes the gradient with respect to the output, and computes the gradients with respect to the parameters and to the inputs, which are in turn back-propagated to earlier layers.

Caffe中的层实质上是一个神经网络层：以一个或多个blobs作为输入，产生一个或多个blobs作为输出。在网络整体的运算中，层有两个关键责任：前向过程从输入产生输出，反向过程以输出的梯度作为输入，计算对参数的梯度和对输入的梯度，向早期的层进行反向传播。

Caffe provides a complete set of layer types including: convolution, pooling, inner products, nonlinearities like rectified linear and logistic, local response normalization, elementwise operations, and losses like softmax and hinge. These are all the types needed for state-of-the-art visual tasks. Coding custom layers requires minimal effort due to the compositional construction of networks.

Caffe给出了很多层的类型，包括：卷积，池化，全连接，非线性（如ReLU），LRN，逐元素运算，损失函数层（如softmax和hinge）。这些都是目前最好的视觉模型中所需要的。定制层的代码也很容易，因为网络构建本来就是模块式的。

### 3.3 Networks and Run Mode

Caffe does all the bookkeeping for any directed acyclic graph of layers, ensuring correctness of the forward and backward passes. Caffe models are end-to-end machine learning systems. A typical network begins with a data layer that loads from disk and ends with a loss layer that computes the objective for a task such as classification or reconstruction.

Caffe完成任何有向无环图式的层的记录工作，确保前向和反向过程的正确性。Caffe模型是端到端的机器学习系统。典型的网络以数据层开始，从磁盘读取数据，以损失层结束，计算一个任务的目标函数，如分类或重建。

The network is run on CPU or GPU by setting a single switch. Layers come with corresponding CPU and GPU routines that produce identical results (with tests to prove it). The CPU/GPU switch is seamless and independent of the model definition.

网络在CPU或GPU上运行，只需要设置一个参数开关。每个层都有CPU和GPU实现，会得到相同的结果（有测试结果证明）。CPU/GPU开关是无缝的，与模型定义无关。

### 3.4 Training a Network

Caffe trains models by the fast and standard stochastic gradient descent algorithm. Figure 1 shows a typical example of a Caffe network (for MNIST digit classification) during training: a data layer fetches the images and labels from disk, passes it through multiple layers such as convolution, pooling and rectified linear transforms, and feeds the final prediction into a classification loss layer that produces the loss and gradients which train the whole network. This example is found in the Caffe source code at examples/lenet/lenet_train.prototxt. Data are processed in mini-batches that pass through the network sequentially. Vital to training are learning rate decay schedules, momentum, and snapshots for stopping and resuming, all of which are implemented and documented.

Caffe使用快速标准的SGD算法训练模型。图1所示的是一个典型的Caffe网络训练过程的例子（MNIST数字分类）：数据层从磁盘中读取图像和标签，经过许多层，如卷积，池化和ReLU变换，将其送入最终的分类损失层，生成损失和梯度，可以训练这个网络。这个例子可以在Caffe源码中找到。数据以小批次进行处理，按顺序经过网络处理。对训练最关键的是学习速率衰减方案，动量，和停止、恢复时的快照，所有这些都已经实现并存档。

Figure 1: An MNIST digit classification example of a Caffe network, where blue boxes represent layers and yellow octagons represent data blobs produced by or fed into the layers.

Finetuning, the adaptation of an existing model to new architectures or data, is a standard method in Caffe. From a snapshot of an existing network and a model definition for the new network, Caffe finetunes the old model weights for the new task and initializes new weights as needed. This capability is essential for tasks such as knowledge transfer [2], object detection [3], and object retrieval [5].

精调，是已有模型对新架构或数据的适应改变，是Caffe中的标准方法。从现有网络的快照，和新网络的模型定义中，Caffe可以将老模型的权重为新任务精调，并为新的权重按照需要进行初始化。这种任务对于knowledge transfer、目标检测和object retrieval是基本的。

## 4 Applications and Examples

In its first six months since public release, Caffe has already been used in a large number of research projects at UC Berkeley and other universities, achieving state-of-the-art performance on a number of tasks. Members of Berkeley EECS have also collaborated with several industry partners such as Facebook [11] and Adobe [6], using Caffe or its direct precursor (Decaf) to obtain state-of-the-art results.

在公开后的前6个月里，Caffe在UC Berkeley和其他大学中已经得到了广泛的应用，在很多任务中得到了目前最好的效果。Berkeley EECS的成员与一些工业合作伙伴联合起来，如Facebook和Adobe，使用Caffe得到目前最好的结果。

**Object Classification**. Caffe has an online demo showing state-of-the-art object classification on images provided by the users, including via mobile phone. The demo takes the image and tries to categorize it into one of the 1,000 ImageNet categories. A typical classification result is shown in Figure 2. Caffe有在线的demo，展示了目前最好的图像目标分类结果，也可以从手机端访问。这个demo输入为图像，按照1000 ImageNet类别进行分类。典型的分类结果如图2所示。

Figure 2: An example of the Caffe object classification demo. Try it out yourself online!

Furthermore, we have successfully trained a model with all 10,000 categories of the full ImageNet dataset by finetuning this network. The resulting network has been applied to open vocabulary object retrieval [5].

而且，我们通过精调成功了训练了一个模型，包含完整ImageNet数据集的10000个类别。这个网络已经用于open vocabulary object retrieval。

**Learning Semantic Features**. In addition to end-to-end training, Caffe can also be used to extract semantic features from images using a pre-trained network. These features can be used “downstream” in other vision tasks with great success [2]. Figure 3 shows a two-dimensional embedding of all the ImageNet validation images, colored by a coarse category that they come from. The nice separation testifies to a successful semantic embedding.

**学习语义特征**。除了端到端的训练，Caffe也可以使用预训练网络从图像中提取语义特征。这些特征可以成功的用于其他视觉任务中。图3所示的是所有ImageNet验证集图像的二维嵌入，以不同的类别用颜色标注出来。分类的结果很好，说明了语义嵌入效果非常成功。

Figure 3: Features extracted from a deep network, visualized in a 2-dimensional space. Note the clear separation between categories, indicative of a successful embedding.

Intriguingly, this learned feature is useful for a lot more than object categories. For example, Karayev et al. have shown promising results finding images of different styles such as “Vintage” and “Romantic” using Caffe features (Figure 4) [6].

非常有趣的是，这种学习到的特征可用与很多其他任务中。比如，Karayev等使用Caffe特征在发现图像不同的风格上也发现非常好用，如Vintage或Romantic。

Figure 4: Top three most-confident positive predictions on the Flickr Style dataset, using a Caffe-trained classifier.

**Object Detection**. Most notably, Caffe has enabled us to obtain by far the best performance on object detection, evaluated on the hardest academic datasets: the PASCAL VOC 2007-2012 and the ImageNet 2013 Detection challenge [3].

**目标检测**。最值得注意的是，Caffe在目标检测中得到了目前最好的结果，在目前最难的学术数据集上进行的评估，PASCAL VOC 2007-2012和ImageNet 2013 检测挑战赛。

Girshick et al. [3] have combined Caffe together with techniques such as Selective Search [10] to effectively perform simultaneous localization and recognition in natural images. Figure 5 shows a sketch of their approach.

Girshick等将Caffe与Selective Search结合起来，进行自然图像的同时定位和识别。图5给出这种方法的梗概。

Figure 5: The R-CNN pipeline that uses Caffe for object detection.

**Beginners’ Guides**. To help users get started with installing, using, and modifying Caffe, we have provided instructions and tutorials on the Caffe webpage. The tutorials range from small demos (MNIST digit recognition) to serious deployments (end-to-end learning on ImageNet). 为帮助用户安装、使用、修改Caffe，我们提供了指南和教程。教程从小型demo（MNIST数字识别）到严肃的部署（ImageNet上端到端的学习）

Although these tutorials serve as effective documentation of the functionality of Caffe, the Caffe source code additionally provides detailed inline documentation on all modules. This documentation will be exposed in a standalone web interface in the near future.

虽然这些教程是Caffe功能的文档，Caffe的源代码在各个模块中也提供了详细的内在文档。这些文档在将来会给出独立的网络界面。

## 5 Availability

Source code is published BSD-licensed on GitHub. Project details, step-wise tutorials, and pre-trained models are on the homepage. Development is done in Linux and OS X, and users have reported Windows builds. A public Caffe Amazon EC2 instance is coming soon.