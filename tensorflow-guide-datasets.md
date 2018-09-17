# Importing Data 导入数据

The `tf.data` API enables you to build complex input pipelines from simple, reusable pieces. For example, the pipeline for an image model might aggregate data from files in a distributed file system, apply random perturbations to each image, and merge randomly selected images into a batch for training. The pipeline for a text model might involve extracting symbols from raw text data, converting them to embedding identifiers with a lookup table, and batching together sequences of different lengths. The `tf.data` API makes it easy to deal with large amounts of data, different data formats, and complicated transformations.

`tf.data` API使我们可以从简单的可重用的数据片段中构建复杂的输入管道。比如，图像模型的管道可能从分布式文件系统的文件中聚集数据，对每幅图像进行随机的处理，并将随机选择的图像合并到一个批次中进行训练。一个文本模型的管道可能需要从原始文本数据中提取符号，将其转化成带有查询表的嵌套标识符，将不同长度的序列合并到一个批次中。`tf.data` API使处理大规模数据、不同的数据格式、复杂的变换变得容易。

The `tf.data` API introduces two new abstractions to TensorFlow:

`tf.data` API 向Tensorflow引入了两种新的抽象：

1. A `tf.data.Dataset` represents a sequence of elements, in which each element contains one or more Tensor objects. For example, in an image pipeline, an element might be a single training example, with a pair of tensors representing the image data and a label. There are two distinct ways to create a dataset:

- 一个`tf.data.Dataset`代表一个元素序列，其中每个元素包含一个或多个Tensor对象。比如，在一个图像管道中，一个元素可能是单个训练样本，包括两个tensor，一个是图像数据，一个是标签。有两种不同的创建dataset的方法：

- Creating a source (e.g. `Dataset.from_tensor_slices()`) constructs a dataset from one or more `tf.Tensor` objects.

- 创建一个源（例如`Dataset.from_tensor_slices()`）从一个或多个`tf.Tensor`对象构建一个数据集。

- Applying a transformation (e.g. `Dataset.batch()`) constructs a dataset from one or more `tf.data.Dataset` objects.

- 应用一个变换（比如`Dataset.batch()`）从一个或多个`tf.data.Dataset`对象中构建一个数据集。

2. A `tf.data.Iterator` provides the main way to extract elements from a dataset. The operation returned by `Iterator.get_next()` yields the next element of a `Dataset` when executed, and typically acts as the interface between input pipeline code and your model. The simplest iterator is a "one-shot iterator", which is associated with a particular `Dataset` and iterates through it once. For more sophisticated uses, the `Iterator.initializer` operation enables you to reinitialize and parameterize an iterator with different datasets, so that you can, for example, iterate over training and validation data multiple times in the same program.

- 一个`tf.data.Iterator`提供了从数据集中提取数据的主要方法。`Iterator.get_next()`返回这个操作，执行的时候产生`Dataset`的下一个对象，一般作为输入管道代码和你的模型之间的接口。最简单的迭代器是one-shot迭代器，即与一个特定的`Dataset`关联，并将其中元素迭代一遍。对于更多复杂的应用，`Iterator.initializer`操作可以针对不同的数据集重新初始化并赋予迭代器参数，比如可以在同样的程序中对训练数据和验证数据迭代很多次。

## Basic mechanics 基本机制

This section of the guide describes the fundamentals of creating different kinds of `Dataset` and `Iterator` objects, and how to extract data from them.

本节指南描述创建不同种类的`Dataset`和`Iterator`对象的基本方法，以及怎样从中提取数据。

To start an input pipeline, you must define a source. For example, to construct a `Dataset` from some tensors in memory, you can use `tf.data.Dataset.from_tensors()` or `tf.data.Dataset.from_tensor_slices()`. Alternatively, if your input data are on disk in the recommended TFRecord format, you can construct a `tf.data.TFRecordDataset`.

为了构建输入管道，必须定义源。比如，为从内存中的张量构建一个`Dataset`，可以使用`tf.data.Dataset.from_tensors()`或`tf.data.Dataset.from_tensor_slices()`。还有一种选择，如果输入数据以推荐的TFRecord格式存储在硬盘上，那么可以构建一个`tf.data.TFRecordDataset`。

Once you have a `Dataset` object, you can transform it into a new `Dataset` by chaining method calls on the `tf.data.Dataset` object. For example, you can apply per-element transformations such as `Dataset.map()` (to apply a function to each element), and multi-element transformations such as `Dataset.batch()`. See the documentation for `tf.data.Dataset` for a complete list of transformations.

有了`Dataset`对象后，可以用那些以`tf.data.Dataset`作为操作对象的方法，将其转换成新的`Dataset`。比如，可以应用一些对每个元素进行变换的操作，如`Dataset.map()`，和多元素变换的操作，如`Dataset.batch()`。阅读`tf.data.Dataset`的文档查看变换的完整列表。

The most common way to consume values from a `Dataset` is to make an iterator object that provides access to one element of the dataset at a time (for example, by calling `Dataset.make_one_shot_iterator()`). A `tf.data.Iterator` provides two operations: `Iterator.initializer`, which enables you to (re)initialize the iterator's state; and `Iterator.get_next()`, which returns `tf.Tensor` objects that correspond to the symbolic next element. Depending on your use case, you might choose a different type of iterator, and the options are outlined below.

从`Dataset`提取值的最普通方式就是创建一个迭代器对象，这样就可以一次从数据集得到一个元素（比如，调用`Dataset.make_one_shot_iterator()`）。`tf.data.Iterator`提供两种操作，`Iterator.initializer`可以（重新）初始化迭代器的状态，`Iterator.get_next()`返回对应的下一个符号元素`tf.Tensor`对象。根据不同的使用情况，你可以选择不同类型的迭代器，选项罗列如下。

### Dataset structure 数据集结构

A dataset comprises elements that each have the same structure. An element contains one or more `tf.Tensor` objects, called components. Each component has a `tf.DType` representing the type of elements in the tensor, and a `tf.TensorShape` representing the (possibly partially specified) static shape of each element. The `Dataset.output_types` and `Dataset.output_shapes` properties allow you to inspect the inferred types and shapes of each component of a dataset element. The nested structure of these properties map to the structure of an element, which may be a single tensor, a tuple of tensors, or a nested tuple of tensors. For example:

一个数据集由很多同样结构的元素组成。一个元素包含一个或多个`tf.Tensor`对象，称为部件。每个部件有一个`tf.DType`的属性，表示张量中元素的类型，还有一个`tf.TensorShape`的属性，表示每个元素的形状（可能只指定的一部分值）。通过`Dataset.output_types`和`Dataset.output_shapes`属性可以查看一个数据集元素的推测类型和每个部件的形状。这些属性的嵌套结构对应着一个元素的结构，可能是单个张量，或张量的元组，或张量的嵌套元组。比如：

```
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"
```

It is often convenient to give names to each component of an element, for example if they represent different features of a training example. In addition to tuples, you can use `collections.namedtuple` or a dictionary mapping strings to tensors to represent a single element of a `Dataset`.

给每个组件或元素都进行命名，会给工作带来便利，比如它们可能代表一个训练样本的不同特征。除了元组，还可以使用`collections.namedtuple`或字符串到张量的映射字典来代表`Dataset`的元素。

```
dataset = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
```

The `Dataset` transformations support datasets of any structure. When using the `Dataset.map()`, `Dataset.flat_map()`, and `Dataset.filter()` transformations, which apply a function to each element, the element structure determines the arguments of the function:

`Dataset`变换支持任何结构的数据集。当使用`Dataset.map()`, `Dataset.flat_map()`和`Dataset.filter()`这些作用于每个元素的变换时，元素的结构决定了函数的参数：

```
dataset1 = dataset1.map(lambda x: ...)

dataset2 = dataset2.flat_map(lambda x, y: ...)

# Note: Argument destructuring is not available in Python 3.
dataset3 = dataset3.filter(lambda x, (y, z): ...)
```

### Creating an iterator 创建一个迭代器

Once you have built a `Dataset` to represent your input data, the next step is to create an `Iterator` to access elements from that dataset. The `tf.data` API currently supports the following iterators, in increasing level of sophistication:

创建了`Dataset`来代表输入数据后，下一步就是要创建一个`Iterator`来访问数据集中的元素。`tf.data` API 目前支持以下迭代器，其复杂程度逐渐增加

- one-shot, 一次性的
- initializable, 可以初始化
- reinitializable, and 可以重新初始化
- feedable. 可供给

A one-shot iterator is the simplest form of iterator, which only supports iterating once through a dataset, with no need for explicit initialization. One-shot iterators handle almost all of the cases that the existing queue-based input pipelines support, but they do not support parameterization. Using the example of `Dataset.range()`:

one-shot迭代器是形式最简单的迭代器，只支持迭代访问数据集一次，不需要显式的初始化。目前基于队列的输入管道所支持的情况，one-shot迭代器几乎都可以处理，但不支持参数化。以'Dataset.range()`为例：

```
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(100):
  value = sess.run(next_element)
  assert i == value
```

> Note: Currently, one-shot iterators are the only type that is easily usable with an Estimator

An initializable iterator requires you to run an explicit `iterator.initializer` operation before using it. In exchange for this inconvenience, it enables you to parameterize the definition of the dataset, using one or more `tf.placeholder()` tensors that can be fed when you initialize the iterator. Continuing the `Dataset.range()` example:

可以初始化的迭代器需要在使用前显式的运行`iterator.initalizer`操作。虽然有这点不便，但可以使数据集的定义含有参数，当初始化迭代器时，使用一个或多个可以feed的`tf.placeholder()`张量。继续使用`Dataset.range()`为例子：

```
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Initialize an iterator over a dataset with 10 elements.
sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
  value = sess.run(next_element)
  assert i == value

# Initialize the same iterator over a dataset with 100 elements.
sess.run(iterator.initializer, feed_dict={max_value: 100})
for i in range(100):
  value = sess.run(next_element)
  assert i == value
```

A reinitializable iterator can be initialized from multiple different `Dataset` objects. For example, you might have a training input pipeline that uses random perturbations to the input images to improve generalization, and a validation input pipeline that evaluates predictions on unmodified data. These pipelines will typically use different `Dataset` objects that have the same structure (i.e. the same types and compatible shapes for each component).

可以重新初始化的迭代器可以从多个不同的`Dataset`对象初始化。比如，你可能有一个训练输入管道，使用的是对输入图像进行随机处理，这样可以改进泛化能力，还有一个验证输入管道，评估未修正数据的预测结果。这些管道一般会使用不同的`Dataset`对象，但有着相同的结构（即，每个部件都有相同的数据类型和兼容的数据形状）。

```
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `training_dataset`
# or `validation_dataset` here, because they are compatible.
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# Run 20 epochs in which the training dataset is traversed, followed by the
# validation dataset.
for _ in range(20):
  # Initialize an iterator over the training dataset.
  sess.run(training_init_op)
  for _ in range(100):
    sess.run(next_element)

  # Initialize an iterator over the validation dataset.
  sess.run(validation_init_op)
  for _ in range(50):
    sess.run(next_element)
```

A feedable iterator can be used together with `tf.placeholder` to select what `Iterator` to use in each call to `tf.Session.run`, via the familiar `feed_dict` mechanism. It offers the same functionality as a reinitializable iterator, but it does not require you to initialize the iterator from the start of a dataset when you switch between iterators. For example, using the same training and validation example from above, you can use `tf.data.Iterator.from_string_handle` to define a feedable iterator that allows you to switch between the two datasets:

可以feed的迭代器与`tf.placeholder`一起使用，在每次调用`tf.Session.run`时，通过熟悉的`feed_dict`机制选择使用哪个`Iterator`。这与可重新初始化迭代器的功能一样，但在不同迭代器之间切换时，不需要从一个数据集开始处初始化迭代器。比如，使用和上面相同的训练和验证样本，可以使用`tf.data.Iterator.from_string_handele`来定义一个可feed的迭代器，在两个数据集之间切换：

```
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50)

# A feedable iterator is defined by a handle placeholder and its structure. We
# could use the `output_types` and `output_shapes` properties of either
# `training_dataset` or `validation_dataset` here, because they have
# identical structure.
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# You can use feedable iterators with a variety of different kinds of iterator
# (such as one-shot and initializable iterators).
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

# The `Iterator.string_handle()` method returns a tensor that can be evaluated
# and used to feed the `handle` placeholder.
training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

# Loop forever, alternating between training and validation.
while True:
  # Run 200 steps using the training dataset. Note that the training dataset is
  # infinite, and we resume from where we left off in the previous `while` loop
  # iteration.
  for _ in range(200):
    sess.run(next_element, feed_dict={handle: training_handle})

  # Run one pass over the validation dataset.
  sess.run(validation_iterator.initializer)
  for _ in range(50):
    sess.run(next_element, feed_dict={handle: validation_handle})
```

### Consuming values from an iterator 使用迭代器产生的数据值

The `Iterator.get_next()` method returns one or more `tf.Tensor` objects that correspond to the symbolic next element of an iterator. Each time these tensors are evaluated, they take the value of the next element in the underlying dataset. (Note that, like other stateful objects in TensorFlow, calling `Iterator.get_next()` does not immediately advance the iterator. Instead you must use the returned `tf.Tensor` objects in a TensorFlow expression, and pass the result of that expression to `tf.Session.run()` to get the next elements and advance the iterator.)

`Iterator.get_next()`方法返回一个或多个`tf.Tensor`对象，对应着迭代器的下一个符号元素。每次使用这些张量，都会从数据集下一个元素取值。（注意，像tensorflow其他有状态的对象，调用`Iterator.get_next()`不会立刻使迭代器步进。必须在一个tensorflow表达式中先使用返回的`tf.Tensor`对象，将表达式的结果传递给`tf.Session.run()`来得到下一个元素并使迭代器步进）

If the iterator reaches the end of the dataset, executing the `Iterator.get_next()` operation will raise a `tf.errors.OutOfRangeError`. After this point the iterator will be in an unusable state, and you must initialize it again if you want to use it further.

如果迭代器到达了数据集的结尾，执行`Iterator.get_next()`操作将抛出一个`tf.errors.OutOfRangeError`。在这点之后，迭代器将会处于不可用状态，必须重新进行初始化，才能继续使用。

```
dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Typically `result` will be the output of a model, or an optimizer's
# training operation.
result = tf.add(next_element, next_element)

sess.run(iterator.initializer)
print(sess.run(result))  # ==> "0"
print(sess.run(result))  # ==> "2"
print(sess.run(result))  # ==> "4"
print(sess.run(result))  # ==> "6"
print(sess.run(result))  # ==> "8"
try:
  sess.run(result)
except tf.errors.OutOfRangeError:
  print("End of dataset")  # ==> "End of dataset"
```

A common pattern is to wrap the "training loop" in a `try-except` block:

通常会将训练循环包装在一个`try-except`块中：

```
sess.run(iterator.initializer)
while True:
  try:
    sess.run(result)
  except tf.errors.OutOfRangeError:
    break
```

If each element of the dataset has a nested structure, the return value of `Iterator.get_next()` will be one or more `tf.Tensor` objects in the same nested structure:

如果数据集中每个元素含有嵌套结构，那么`Iterator.get_next()`的返回值将会是同样嵌套结构的一个或多个`tf.Tensor`对象。

```
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()

sess.run(iterator.initializer)
next1, (next2, next3) = iterator.get_next()
```

Note that `next1`, `next2`, and `next3` are tensors produced by the same op/node (created by `Iterator.get_next()`). Therefore, evaluating any of these tensors will advance the iterator for all components. A typical consumer of an iterator will include all components in a single expression.

注意`next1`, `next2`和`next3`是同一个操作/节点（由`Iterator.get_next()`生成的）产生的张量。所以，使用其中任何一个张量都将会推进迭代器。一般使用迭代器产生的数据会在一个表达式中包含所有部件。

### Saving iterator state 保存迭代器状态

The `tf.contrib.data.make_saveable_from_iterator` function creates a `SaveableObject` from an iterator, which can be used to save and restore the current state of the iterator (and, effectively, the whole input pipeline). A saveable object thus created can be added to `tf.train.Saver` variables list or the `tf.GraphKeys.SAVEABLE_OBJECTS` collection for saving and restoring in the same manner as a `tf.Variable`. Refer to `Saving and Restoring` for details on how to save and restore variables.

`tf.contrib.data.make_saveable_from_iterator`函数会从迭代器中生成一个`SaveableObject`对象，可以保存并恢复迭代器的目前状态（实际上就是整个输入管道的状态）。这样产生的可保存对象可以加入`tf.train.Saver`变量列表或`tf.GraphKeys.SAVEABLE_OBJECTS`集合进行保存和恢复，模式和`tf.Variable`一样。如何进行保存和恢复变量的细节请参考"保存和恢复"一节。

```
# Create saveable object from iterator.
saveable = tf.contrib.data.make_saveable_from_iterator(iterator)

# Save the iterator state by adding it to the saveable objects collection.
tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
saver = tf.train.Saver()

with tf.Session() as sess:

  if should_checkpoint:
    saver.save(path_to_checkpoint)

# Restore the iterator state.
with tf.Session() as sess:
  saver.restore(sess, path_to_checkpoint)
```

## Reading input data 读取输入数据

### Consuming NumPy arrays 使用Numpy数组

If all of your input data fit in memory, the simplest way to create a `Dataset` from them is to convert them to `tf.Tensor` objects and use `Dataset.from_tensor_slices()`.

如果内存可以存储所有的输入数据，那么从这些数据中生成`Dataset`最简单的方法就是将其转化成`tf.Tensor`对象，并使用`Dataset.from_tensor_slices()`。

```
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
```

Note that the above code snippet will embed the `features` and `labels` arrays in your TensorFlow graph as `tf.constant()` operations. This works well for a small dataset, but wastes memory---because the contents of the array will be copied multiple times---and can run into the 2GB limit for the `tf.GraphDef` protocol buffer.

注意上面的代码片段会将`features`和`labels`数组嵌入到tensorflow图中，成为`tf.constant()`操作。这对于小数据集来说很好，但浪费内存，因为这个数组将会被复制很多次，会达到`tf.GraphDef`协议缓存的2GB限制。

As an alternative, you can define the `Dataset` in terms of `tf.placeholder()` tensors, and feed the NumPy arrays when you initialize an `Iterator` over the dataset.

换一种方法，可以定义`Dataset`作为`tf.placeholder()`张量，在整个数据集上初始化一个`Iterator`对象时，再feed给Numpy数组。

```
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
dataset = ...
iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})
```

### Consuming TFRecord data 使用TFRecord数据

The `tf.data` API supports a variety of file formats so that you can process large datasets that do not fit in memory. For example, the TFRecord file format is a simple record-oriented binary format that many TensorFlow applications use for training data. The `tf.data.TFRecordDataset` class enables you to stream over the contents of one or more TFRecord files as part of an input pipeline.

`tf.data` API支持很多文件格式，所以可以处理内存存储不了的大规模数据集。比如，TFRecord文件格式是一种简单的面向记录的二进制格式，很多TensorFlow应用都用这种格式作为训练数据。`tf.data.TFRecordDataset`类可以将一个或多个TFRecord文件作为输入管道。

```
# Creates a dataset that reads all of the examples from two files.
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
```

The `filenames` argument to the `TFRecordDataset` initializer can either be a string, a list of strings, or a `tf.Tensor` of strings. Therefore if you have two sets of files for training and validation purposes, you can use a `tf.placeholder(tf.string)` to represent the filenames, and initialize an iterator from the appropriate filenames:

`TFRecordDataset`初始化器参数`filenames`可以是字符串，字符串列表，或`tf.Tensor`个字符串。所以如果有训练和验证目的的两套文件，可以使用`tf.placeholder(tf.string)`来代表文件名，用适当的文件名来初始化迭代器：

```
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)  # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

# You can feed the initializer with the appropriate filenames for the current
# phase of execution, e.g. training vs. validation.

# Initialize `iterator` with training data.
training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

# Initialize `iterator` with validation data.
validation_filenames = ["/var/data/validation1.tfrecord", ...]
sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
```

### Consuming text data 使用文本数据

Many datasets are distributed as one or more text files. The `tf.data.TextLineDataset` provides an easy way to extract lines from one or more text files. Given one or more filenames, a `TextLineDataset` will produce one string-valued element per line of those files. Like a `TFRecordDataset`, `TextLineDataset` accepts filenames as a `tf.Tensor`, so you can parameterize it by passing a `tf.placeholder(tf.string)`.

很多数据集是一个或多个文本文件，`tf.data.TextLineDataset`可以很容易的从一个或多个文本文件中提取出文字行。给定一个或多个文件名，`TextLineDataset`将对这些文件的每一行都产生一个字符串元素。就像`TFRecordDataset`一样，`TextLineDataset`也接受`tf.Tensor`格式的文件名，所以可以用`tf.placeholder(tf.string)`作为其参数。

```
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]
dataset = tf.data.TextLineDataset(filenames)
```

By default, a `TextLineDataset` yields every line of each file, which may not be desirable, for example if the file starts with a header line, or contains comments. These lines can be removed using the `Dataset.skip()` and `Dataset.filter()` transformations. To apply these transformations to each file separately, we use `Dataset.flat_map()` to create a nested `Dataset` for each file.

`TextLineDataset`默认每个文件每一行都产生一个元素，我们可能不想这样，比如如果文件以一个标题行开始，或包括注释，这些行可以用`Dataset.skip()`和`Dataset.filter()`去掉。为了对每个文件进行这样的处理，我们使用`Dataset.flat_map()`对每个文件产生一个嵌套的`Dataset`。

```
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]

dataset = tf.data.Dataset.from_tensor_slices(filenames)

# Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
# and then concatenate their contents sequentially into a single "flat" dataset.
# * Skip the first line (header row).
# * Filter out lines beginning with "#" (comments).
dataset = dataset.flat_map(
    lambda filename: (
        tf.data.TextLineDataset(filename)
        .skip(1)
        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))
```

### Consuming CSV data 使用CSV数据

The CSV file format is a popular format for storing tabular data in plain text. The `tf.contrib.data.CsvDataset` class provides a way to extract records from one or more CSV files that comply with RFC 4180. Given one or more filenames and a list of defaults, a `CsvDataset` will produce a tuple of elements whose types correspond to the types of the defaults provided, per CSV record. Like `TFRecordDataset` and `TextLineDataset`, `CsvDataset` accepts filenames as a `tf.Tensor`, so you can parameterize it by passing a `tf.placeholder(tf.string)`.

CSV文件格式以文本存储表格数据，是一种很受欢迎的数据格式。`tf.contrib.data.CsvDataset`类可以从一个或多个符合RFC 4180标准的CSV文件中提取记录。给定一个或多个文件名，以及一些默认值的列表，`CsvDataset`将会生成一个元组，其元素类型与给定的默认值对应，对每个CSV记录都是这样。与`TFRecordDataset`和`TextLineDataset`类似，`CsvDataset`接受`tf.Tensor`类的文件名，可以用`tf.placeholder(tf.string)`作为参数。

```
# Creates a dataset that reads all of the records from two CSV files, each with
# eight float columns
filenames = ["/var/data/file1.csv", "/var/data/file2.csv"]
record_defaults = [tf.float32] * 8   # Eight required float columns
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)
```

If some columns are empty, you can provide defaults instead of types.

如果一些列是空的，那么可以不给出类型。

```
# Creates a dataset that reads all of the records from two CSV files, each with
# four float columns which may have missing values
record_defaults = [[0.0]] * 8
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)
```

By default, a `CsvDataset` yields every column of every line of the file, which may not be desirable, for example if the file starts with a header line that should be ignored, or if some columns are not required in the input. These lines and fields can be removed with the `header` and `select_cols` arguments respectively.

`CsvDataset`默认对每个文件每一行都产生数据，这可能不是理想状态，比如如果文件开头是个标题行，需要忽略掉，或如果一些行不作为输入。那么这些行和字段可以去掉，方法是分别使用`header`和`select_cols`参数。

```
# Creates a dataset that reads all of the records from two CSV files with
# headers, extracting float data from columns 2 and 4.
record_defaults = [[0.0]] * 2  # Only provide defaults for the selected columns
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[2,4])
```

## Preprocessing data with Dataset.map() 数据预处理

The `Dataset.map(f)` transformation produces a new dataset by applying a given function f to each element of the input dataset. It is based on the `map()` function that is commonly applied to lists (and other structures) in functional programming languages. The function f takes the `tf.Tensor` objects that represent a single element in the input, and returns the `tf.Tensor` objects that will represent a single element in the new dataset. Its implementation uses standard TensorFlow operations to transform one element into another.

`Dataset.map(f)`变换将函数f应用于输入数据集的每个元素上，从而生成一个新的数据集。在函数式编程语言中，函数`map()`一般应用在列表（或其他结构上），这与之类似。函数f以`tf.Tensor`对象作为输入，代表输入中的单个元素，返回`tf.Tensor`对象，代表新数据集中的单个元素。其实现使用标准TensorFlow操作，将一个元素转换成另一个。

This section covers common examples of how to use `Dataset.map()`.

本节涵盖了`Dataset.map()`使用的普通例子。

### Parsing tf.Example protocol buffer messages 解析tf.Example协议缓冲区信息

Note: What is [protocol buffer messages](https://www.ibm.com/developerworks/cn/linux/l-cn-gpb/index.html)?

Many input pipelines extract `tf.train.Example` protocol buffer messages from a TFRecord-format file (written, for example, using `tf.python_io.TFRecordWriter`). Each `tf.train.Example` record contains one or more "features", and the input pipeline typically converts these features into tensors.

很多输入管道从一个TFRecord格式的文件中提取`tf.train.Example`协议缓冲区数据（比如，用`tf.python_io.TFRecordWriter`写）。每个`tf.train.Example`记录包括一个或多个“特征”，输入管道一般将这些特征转换成张量。

```
# Transforms a scalar string `example_proto` into a pair of a scalar string and
# a scalar integer, representing an image and its label, respectively.
def _parse_function(example_proto):
  features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
              "label": tf.FixedLenFeature((), tf.int32, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["image"], parsed_features["label"]

# Creates a dataset that reads all of the examples from two files, and extracts
# the image and label features.
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
```

### Decoding image data and resizing it 解码图像数据并调整大小

When training a neural network on real-world image data, it is often necessary to convert images of different sizes to a common size, so that they may be batched into a fixed size.

当在真实世界图像数据上训练一个神经网络时，经常需要将不同大小的图像转换成通用大小，这样它们才能编入一个固定大小的批次。

```
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# A vector of filenames.
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0, 37, ...])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
```

### Applying arbitrary Python logic with tf.py_func() 用tf.py_func()实现任意Python代码逻辑

For performance reasons, we encourage you to use TensorFlow operations for preprocessing your data whenever possible. However, it is sometimes useful to call upon external Python libraries when parsing your input data. To do so, invoke, the `tf.py_func()` operation in a `Dataset.map()` transformation.

为了程序性能，我们鼓励采用TensorFlow操作进行数据预处理。但解析输入数据时，有时候需要调用外部Python库。为了实现这个功能，在`Dataset.map()`变换中可以调用`tf.py_fun()`。

```
import cv2

# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(filename, label):
  image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
  return image_decoded, label

# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
labels = [0, 37, 29, 1, ...]

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
    lambda filename, label: tuple(tf.py_func(
        _read_py_function, [filename, label], [tf.uint8, label.dtype])))
dataset = dataset.map(_resize_function)
```

## Batching dataset elements 数据集元素分批次

### Simple batching 简单批次

The simplest form of batching stacks n consecutive elements of a dataset into a single element. The `Dataset.batch()` transformation does exactly this, with the same constraints as the `tf.stack()` operator, applied to each component of the elements: i.e. for each component i, all elements must have a tensor of the exact same shape.

分批次最简单的方法是将数据集中n个连续的元素堆叠成单个元素。`Dataset.batch()`变换就是这样做的，与`tf.stack()`操作的限制相同，应用在每个元素的每个部件上：即，对于每个部件i，所有元素的张量形状必须一样。

```
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])
```

### Batching tensors with padding 填充张量分批次

The above recipe works for tensors that all have the same size. However, many models (e.g. sequence models) work with input data that can have varying size (e.g. sequences of different lengths). To handle this case, the `Dataset.padded_batch()` transformation enables you to batch tensors of different shape by specifying one or more dimensions in which they may be padded.

上面的方法适用于张量尺寸相同的情况。但是，很多模型（如序列模型）的输入数据可能大小不一（如，不同长度的序列）。为了处理这种情况，`Dataset.padded_batch()`变换可以对不同形状的张量进行分批次，需要指定一个或多个维度大小来进行数据填充。

```
dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                               #      [5, 5, 5, 5, 5, 0, 0],
                               #      [6, 6, 6, 6, 6, 6, 0],
                               #      [7, 7, 7, 7, 7, 7, 7]]
```

The `Dataset.padded_batch()` transformation allows you to set different padding for each dimension of each component, and it may be variable-length (signified by None in the example above) or constant-length. It is also possible to override the padding value, which defaults to 0.

`Dataset.padded_batch()`变换可以对不同部件的不同维度进行不同填充，可能是变长的（上例中用None表示）或固定长度的。也可以指定填充数值，默认是0。

## Training workflows 训练工作流程

### Processing multiple epochs 处理多轮数据

The `tf.data` API offers two main ways to process multiple epochs of the same data.

`tf.data` API 有两种主要方式对同一数据进行多轮处理。

The simplest way to iterate over a dataset in multiple epochs is to use the `Dataset.repeat()` transformation. For example, to create a dataset that repeats its input for 10 epochs:

最简单的额方法是用`Dataset.repeat()`方法在同一数据集上进行多轮处理。比如，将数据集重复10次输入：

```
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.repeat(10)
dataset = dataset.batch(32)
```

Applying the `Dataset.repeat()` transformation with no arguments will repeat the input indefinitely. The `Dataset.repeat()` transformation concatenates its arguments without signaling the end of one epoch and the beginning of the next epoch.

不带参数执行`Dataset.repeat()`变换将无限次重复输入。`Dataset.repeat()`变换将参数连接到一起，并不指示一轮的结束和下一轮的开始。

If you want to receive a signal at the end of each epoch, you can write a training loop that catches the `tf.errors.OutOfRangeError` at the end of a dataset. At that point you might collect some statistics (e.g. the validation error) for the epoch.

如果想在一轮结束的时候收到信号，可以写一个训练循环，在数据集结束的时候接受`tf.errors.OutOfRangeError`错误。在这个节点可以收集一些这一轮的统计数字（比如验证错误率）。

```
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Compute for 100 epochs.
for _ in range(100):
  sess.run(iterator.initializer)
  while True:
    try:
      sess.run(next_element)
    except tf.errors.OutOfRangeError:
      break

  # [Perform end-of-epoch calculations here.]
```

### Randomly shuffling input data 随机打乱输入数据顺序

The `Dataset.shuffle()` transformation randomly shuffles the input dataset using a similar algorithm to `tf.RandomShuffleQueue`: it maintains a fixed-size buffer and chooses the next element uniformly at random from that buffer.

`Dataset.shuffle()`变换用`tf.RandomShuffleQueue`类似的算法来随机打乱输入数据集顺序，它维护了一个固定大小的缓冲区，从这个缓冲区中均匀随机的选择下一个元素。

```
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.repeat()
```

### Using high-level APIs 使用高层API

The `tf.train.MonitoredTrainingSession` API simplifies many aspects of running TensorFlow in a distributed setting. `MonitoredTrainingSession` uses the `tf.errors.OutOfRangeError` to signal that training has completed, so to use it with the `tf.data` API, we recommend using `Dataset.make_one_shot_iterator()`. For example:

`tf.train.MonitoredTrainingSession` API 简化了分布式设置中运行TensorFlow的很多方面。`MonitoredTrainingSession`使用`tf.errors.OutOfRangeError`表示训练已经结束，所以为了与`tf.data` API一起使用，我们推荐使用`Dataset.make_one_shot_iterator()`。例如：

```
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.repeat(num_epochs)
iterator = dataset.make_one_shot_iterator()

next_example, next_label = iterator.get_next()
loss = model_function(next_example, next_label)

training_op = tf.train.AdagradOptimizer(...).minimize(loss)

with tf.train.MonitoredTrainingSession(...) as sess:
  while not sess.should_stop():
    sess.run(training_op)
```

To use a `Dataset` in the `input_fn` of a `tf.estimator.Estimator`, we also recommend using `Dataset.make_one_shot_iterator()`. For example:

为在`tf.estimator.Estimator`中的`input_fn`使用`Dataset`，我们还推荐使用`Dataset.make_one_shot_iterator()`。比如：

```
def dataset_input_fn():
  filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
  dataset = tf.data.TFRecordDataset(filenames)

  # Use `tf.parse_single_example()` to extract data from a `tf.Example`
  # protocol buffer, and perform any additional per-record preprocessing.
  def parser(record):
    keys_to_features = {
        "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
        "date_time": tf.FixedLenFeature((), tf.int64, default_value=""),
        "label": tf.FixedLenFeature((), tf.int64,
                                    default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    image = tf.image.decode_jpeg(parsed["image_data"])
    image = tf.reshape(image, [299, 299, 1])
    label = tf.cast(parsed["label"], tf.int32)

    return {"image_data": image, "date_time": parsed["date_time"]}, label

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  # tensor for each example.
  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(32)
  dataset = dataset.repeat(num_epochs)
  iterator = dataset.make_one_shot_iterator()

  # `features` is a dictionary in which each value is a batch of values for
  # that feature; `labels` is a batch of labels.
  features, labels = iterator.get_next()
  return features, labels
```