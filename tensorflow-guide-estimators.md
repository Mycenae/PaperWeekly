# Estimators

This document introduces `Estimator`s--a high-level TensorFlow API that greatly simplifies machine learning programming. Estimators encapsulate the following actions:

本文档介绍了`Estimator`s，它是一个高层TensorFlow API，可以极大的简化机器学习编程。Estimators封装了下面的功能：

- training 训练
- evaluation 评价
- prediction 预测
- export for serving 导出作其他用处

You may either use the pre-made Estimators we provide or write your own custom Estimators. All Estimators--whether pre-made or custom--are classes based on the `tf.estimator.Estimator` class.

你可以使用我们提供的预制Estimators，也可以写自己的定制Estimators。所有的Estimators都以`tf.estimator.Estimator`为基类，不管是预制的还是定制的。

> TensorFlow also includes a deprecated Estimator class at `tf.contrib.learn.Estimator`, which you should not use.

> TensorFlow还包含了一个弃用的Estimator类，是`tf.contrib.learn.Estimator`，不要使用这个类。

## Advantages of Estimators 优势

Estimators provide the following benefits: Estimators有以下几点优势：

1. You can run Estimator-based models on a local host or on a distributed multi-server environment without changing your model. Furthermore, you can run Estimator-based models on CPUs, GPUs, or TPUs without recoding your model.
- 你可以在本机上运行基于Estimator的模型，也可以在分布式多服务器的环境运行，而不需要改变模型。进一步，你可以在CPU、GPU或TPU上运行基于Estimator的模型，不用改变代码。
2. Estimators simplify sharing implementations between model developers.
- Estimator简化了模型开发者间的实现分享。
3. You can develop a state of the art model with high-level intuitive code. In short, it is generally much easier to create models with Estimators than with the low-level TensorFlow APIs.
- 可以用高层的直观的代码开发最新的模型。简而言之，比起用底层TensorFlow API，用Estimator可以更容易的开发模型。
4. Estimators are themselves built on `tf.layers`, which simplifies customization.
- Estimator建立在`tf.layers`之上，简化了定制过程。
5. Estimators build the graph for you. Estimator为你建立计算图。
6. Estimators provide a safe distributed training loop that controls how and when to:
- Estimator提供了安全的分布式训练循环，控制了怎样及什么时候做下面的事：
- build the graph 建立计算图
- initialize variables 初始化变量
- start queues 开始队列
- handle exceptions 处理异常
- create checkpoint files and recover from failures 生成checkpoint文件，从运行故障中恢复
- save summaries for TensorBoard 为TensorBoard保存摘要信息

When writing an application with Estimators, you must separate the data input pipeline from the model. This separation simplifies experiments with different data sets.

当使用Estimator写一个应用时，必须将数据输入管道与模型分开，这将简化用不同的数据集进行试验。

## Pre-made Estimators 预制Estimator

Pre-made Estimators enable you to work at a much higher conceptual level than the base TensorFlow APIs. You no longer have to worry about creating the computational graph or sessions since Estimators handle all the "plumbing" for you. That is, pre-made Estimators create and manage `Graph` and `Session` objects for you. Furthermore, pre-made Estimators let you experiment with different model architectures by making only minimal code changes. `DNNClassifier`, for example, is a pre-made Estimator class that trains classification models based on dense, feed-forward neural networks.

预制Estimator与基础的TensorFlow API相比，可以使人工作在很高的概念层次上。不需要担心创建计算图或会话，因为Estimator为你处理所有这些细节。也就是说，预制Estimator为你生成并管理`Graph`和`Session`对象。更进一步，利用预制Estimator对不同的模型架构进行试验，只需要修改很少的代码。比如，`DNNClassifier`就是一个预制Estimator类，可以训练基于全连接、全向神经网络的分类模型。

### Structure of a pre-made Estimators program 预制Estimator程序结构

A TensorFlow program relying on a pre-made Estimator typically consists of the following four steps:

TensorFlow的预制Estimator程序一般包括以下4个步骤：

1. **Write one or more dataset importing functions**. For example, you might create one function to import the training set and another function to import the test set. Each dataset importing function must return two objects:

**一个或多个数据集导入函数**。比如，你可能会创建一个函数来导入训练集，另一个函数来导入测试集。每个数据集导入函数都要返回2个对象：

- a dictionary in which the keys are feature names and the values are Tensors (or SparseTensors) containing the corresponding feature data
- 一个字典，其中key为特征名称，value为包含对应特征数据的张量（或稀疏张量）
- a Tensor containing one or more labels
- 包含一个或多个标签数据的张量

For example, the following code illustrates the basic skeleton for an input function:

比如，下面的代码就是输入函数的基本骨架结构：

```
def input_fn(dataset):
   ...  # manipulate dataset, extracting the feature dict and the label
   return feature_dict, label
```

2. **Define the feature columns**. Each `tf.feature_column` identifies a feature name, its type, and any input pre-processing. For example, the following snippet creates three feature columns that hold integer or floating-point data. The first two feature columns simply identify the feature's name and type. The third feature column also specifies a lambda the program will invoke to scale the raw data:

**定义特征列**。每个`tf.feature_column`可以确定一个特征名称，其类型，和任何输入预处理。比如，下列代码片段生成了三个特征列，类型是整型或浮点数。前两个特征列仅仅确定了特征的名称和类型，第三个特征列还指定了一个lambda函数，用来处理原始数据。

```
# Define three numeric feature columns.
population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
median_education = tf.feature_column.numeric_column('median_education',
                    normalizer_fn=lambda x: x - global_education_mean)
```

3. **Instantiate the relevant pre-made Estimator**. For example, here's a sample instantiation of a pre-made Estimator named LinearClassifier:

**实例化相关的预制Estimator**。比如，这里有一个预制Estimator实例化的例子，名称为LinearClassifier：

```
# Instantiate an estimator, passing the feature columns.
estimator = tf.estimator.LinearClassifier(
    feature_columns=[population, crime_rate, median_education],
    )
```

4. **Call a training, evaluation, or inference method**. For example, all Estimators provide a train method, which trains a model.

**调用训练、评价或推理方法**。比如，所有的Estimator都会有一个训练方法，来训练模型。

```
# my_training_set is the function created in Step 1
estimator.train(input_fn=my_training_set, steps=2000)
```

### Benefits of pre-made Estimators 预制Estimator的好处

Pre-made Estimators encode best practices, providing the following benefits:

预制Estimator内包括了最好的代码，提供了以下好处：

- Best practices for determining where different parts of the computational graph should run, implementing strategies on a single machine or on a cluster.
- 确定计算图的不同部分哪部分该运行的最优方法，在单机上或集群上的最好实现策略。
- Best practices for event (summary) writing and universally useful summaries.
- 最好的事件（摘要）记录及统一摘要。

If you don't use pre-made Estimators, you must implement the preceding features yourself.

如果你不用预制Estimators，那就必须自己实现以上的代码。

## Custom Estimators 定制Estimators

The heart of every Estimator--whether pre-made or custom--is its model function, which is a method that builds graphs for training, evaluation, and prediction. When you are using a pre-made Estimator, someone else has already implemented the model function. When relying on a custom Estimator, you must write the model function yourself. A companion document explains how to write the model function.

每个Estimator的中心，不论是预制的或定制的，都是其模型函数，就是构建训练、评估和预测的计算图的方法。当使用预制Estimator时，其他人已经实现了这些模型函数。当定制Estimator时，必须自己实现模型函数。这里是怎样写模型函数的文档。

## Recommended workflow 推荐的工作流

We recommend the following workflow: 我们推荐如下的工作流：

1. Assuming a suitable pre-made Estimator exists, use it to build your first model and use its results to establish a baseline.
- 如果有合适的预制Estimator，那么就使用建立第一个模型，使用其结果来确定一个基准。
2. Build and test your overall pipeline, including the integrity and reliability of your data with this pre-made Estimator.
- 构建并测试整个管道，包括数据与此预制模型结合的完整性和依赖性。
3. If suitable alternative pre-made Estimators are available, run experiments to determine which pre-made Estimator produces the best results.
- 如果还有另外合适的预制Estimator可用，进行试验并确定哪个预制Estimator得到最佳结果。
4. Possibly, further improve your model by building your own custom Estimator.
- 如果可能的话，通过构建自己的定制Estimator来进一步改进模型。

## Creating Estimators from Keras models 从Keras模型创建Estimator

You can convert existing Keras models to Estimators. Doing so enables your Keras model to access Estimator's strengths, such as distributed training. Call `tf.keras.estimator.model_to_estimator` as in the following sample:

可以将现有的Keras模型转换到Estimator。这样Keras的模型可以利用Estimator的便利，比如分布式训练。可以像下面的例子中一样调用`tf.keras.estimator.model_to_estimator`:

```
# Instantiate a Keras inception v3 model.
keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)
# Compile model with the optimizer, loss, and metrics you'd like to train with.
keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metric='accuracy')
# Create an Estimator from the compiled Keras model. Note the initial model
# state of the keras model is preserved in the created Estimator.
est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_inception_v3)

# Treat the derived Estimator as you would with any other Estimator.
# First, recover the input name(s) of Keras model, so we can use them as the
# feature column name(s) of the Estimator input function:
keras_inception_v3.input_names  # print out: ['input_1']
# Once we have the input name(s), we can create the input function, for example,
# for input(s) in the format of numpy ndarray:
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_1": train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=False)
# To train, we call Estimator's train function:
est_inception_v3.train(input_fn=train_input_fn, steps=2000)
```

Note that the names of feature columns and labels of a keras estimator come from the corresponding compiled keras model. For example, the input key names for `train_input_fn` above can be obtained from `keras_inception_v3.input_names`, and similarly, the predicted output names can be obtained from `keras_inception_v3.output_names`.

注意特征列和标签的名字。比如，`train_input_fn`的输入key名称可以从`keras_inception_v3.input_names`得到，类似的，预测的输出名称可以从`keras_inception_v3.output_names`得到。

For more details, please refer to the documentation for `tf.keras.estimator.model_to_estimator`.

更多细节请参考`tf.keras.estimator.model_to_estimator`的文档。